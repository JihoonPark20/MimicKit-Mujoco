"""Reproducible local port audit; no GPU-simulator equivalence is implied.

Run cases serially for comparable throughput. Timings include device sync at
interval boundaries; compilation/construction and warm-up are excluded.
"""

import argparse
import cProfile
import hashlib
import importlib.util
import io
import json
import os
import platform
import pstats
import resource
import statistics
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimickit_mujoco import bootstrap  # noqa: E402

bootstrap()

import torch  # noqa: E402
import yaml  # noqa: E402
from envs.env_builder import build_env  # noqa: E402
from learning.agent_builder import build_agent  # noqa: E402
from util import mp_util  # noqa: E402
from util.util import set_rand_seed  # noqa: E402

from mimickit_mujoco.cli import ALGORITHMS, read_yaml, select_device  # noqa: E402


def synchronize(device):
    if device.startswith("mps"):
        torch.mps.synchronize()
    elif device.startswith("cuda"):
        torch.cuda.synchronize()


def measure(fn, device):
    synchronize(device)
    start = time.perf_counter()
    result = fn()
    synchronize(device)
    return time.perf_counter() - start, result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("env", "train"), default="env")
    parser.add_argument("--algorithm", choices=ALGORITHMS, default="deepmimic")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-threads", type=int, default=4)
    parser.add_argument("--steps", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--prior-dir", type=Path)
    parser.add_argument(
        "--engine-source", type=Path, help="Diagnostic baseline engine .py; default is the installed engine"
    )
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.prior_dir:
        args.prior_dir = args.prior_dir.resolve()
    if args.engine_source:
        args.engine_source = args.engine_source.resolve()
    os.chdir(ROOT)
    device = select_device(args.device)
    set_rand_seed(42)
    mp_util.init(0, 1, device, 0)
    torch.set_num_threads(args.num_threads)
    from engines import mujoco_engine

    engine_source = args.engine_source or Path(mujoco_engine.__file__)
    if args.engine_source:
        spec = importlib.util.spec_from_file_location("benchmark_baseline_engine", engine_source)
        baseline = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(baseline)
        mujoco_engine.MujocoEngine = baseline.MujocoEngine
    env_name, agent_name = ALGORITHMS[args.algorithm]
    ec = read_yaml(f"data/envs/{env_name}_g1_env.yaml")
    ac = read_yaml(f"data/agents/{agent_name}_agent.yaml")
    sc = read_yaml("data/engines/mujoco_engine.yaml")
    sc["num_threads"] = args.num_threads
    if args.prior_dir:
        ac["smp_prior_cfg"] = str(args.prior_dir / "diffusion_config.yaml")
        ac["smp_prior_model"] = str(args.prior_dir / "model.pt")
    with tempfile.TemporaryDirectory(prefix="mk-benchmark-") as tmp:
        cfg_dir = Path(tmp)
        for name, config in (("env", ec), ("agent", ac), ("engine", sc)):
            (cfg_dir / f"{name}.yaml").write_text(yaml.safe_dump(config))
        init_start = time.perf_counter()
        env = build_env(str(cfg_dir / "env.yaml"), str(cfg_dir / "engine.yaml"), args.num_envs, device, False)
        try:
            report = {
                "date": time.strftime("%Y-%m-%d %H:%M:%S %z"),
                "platform": platform.platform(),
                "python": platform.python_version(),
                "torch": torch.__version__,
                "device": device,
                "mode": args.mode,
                "algorithm": args.algorithm,
                "num_envs": args.num_envs,
                "threads": args.num_threads,
                "engine_init_seconds": time.perf_counter() - init_start,
                "seed": 42,
                "engine_source_sha256": hashlib.sha256(engine_source.read_bytes()).hexdigest(),
                "env_config": ec,
                "engine_config": sc,
            }
            elapsed = []
            if args.mode == "env":
                env.reset()
                action = torch.zeros((args.num_envs, 29), device=device)

                def step_many(n):
                    resets = torch.zeros((), device=device, dtype=torch.long)
                    for _ in range(n):
                        obs, reward, done, _ = env.step(action)
                        ids = (done != 0).nonzero(as_tuple=False).flatten()
                        resets += len(ids)
                        if len(ids):
                            env.reset(ids)
                    return obs, reward, resets

                step_many(args.warmup)
                for _ in range(args.repeats):
                    seconds, (obs, reward, resets) = measure(lambda: step_many(args.steps), device)
                    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                    elapsed.append(seconds)
                report.update(
                    steps_per_repeat=args.steps,
                    resets_last_repeat=int(resets.item()),
                    warmup_steps=args.warmup,
                    action_policy="zero joint targets, reset on done",
                )
                samples = args.steps * args.num_envs

                def profile_fn():
                    return step_many(32)
            else:
                agent = build_agent(str(cfg_dir / "agent.yaml"), env, device)
                agent._curr_obs, agent._curr_info = agent._reset_envs()
                agent._init_train()
                agent._train_iter()  # allocate replay/optimizer state, JIT warmup
                agent._iter += 1
                stage_records = {}
                for name in ("_rollout_train", "_build_train_data", "_update_model", "_update_normalizers"):
                    original = getattr(agent, name)

                    def timed(*a, _original=original, _name=name, **kw):
                        seconds, result = measure(lambda: _original(*a, **kw), device)
                        stage_records.setdefault(_name, []).append(seconds)
                        return result

                    setattr(agent, name, timed)
                for _ in range(args.repeats):
                    seconds, info = measure(agent._train_iter, device)
                    assert all(torch.isfinite(p).all() for p in agent.parameters())
                    elapsed.append(seconds)
                    agent._iter += 1
                # The later profiling iteration must not contaminate timed samples.
                report["stages_seconds"] = {k: list(v) for k, v in stage_records.items()}
                report["agent_config"] = ac
                report["network_parameters"] = sum(p.numel() for p in agent._model.parameters())
                report["rollout_steps"] = ac["steps_per_iter"]
                report["losses_last_iteration"] = {k: float(v) for k, v in info.items() if "loss" in k.lower()}
                report["buffer_bytes"] = {}
                for name in ("_exp_buffer", "_disc_buffer"):
                    buffer = getattr(agent, name, None)
                    if buffer is not None:
                        report["buffer_bytes"][name] = {
                            k: v.numel() * v.element_size() for k, v in buffer._buffers.items()
                        }
                samples = args.num_envs * ac["steps_per_iter"]
                profile_fn = agent._train_iter
            report["repeat_seconds"] = elapsed
            report["median_seconds"] = statistics.median(elapsed)
            report["env_transitions_per_second"] = samples / statistics.median(elapsed)
            # Profile is a separate interval, never included in throughput.
            if args.profile:
                prof = cProfile.Profile()
                prof.enable()
                profile_fn()
                synchronize(device)
                prof.disable()
                output = io.StringIO()
                pstats.Stats(prof, stream=output).strip_dirs().sort_stats("cumtime").print_stats(35)
                report["profile"] = output.getvalue()
            report["peak_rss_bytes"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * (
                1 if sys.platform == "darwin" else 1024
            )
            if device == "mps":
                report["mps_allocated_bytes"] = torch.mps.current_allocated_memory()
                report["mps_driver_bytes"] = torch.mps.driver_allocated_memory()
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        "output": str(args.output),
                        "transitions_s": report["env_transitions_per_second"],
                        "seconds": elapsed,
                    }
                ),
                flush=True,
            )
        finally:
            env._engine.close()


if __name__ == "__main__":
    main()
