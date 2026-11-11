"""G1 train, play, motion inspection and SMP prior entry points."""

import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import yaml

from . import ROOT, bootstrap

ALGORITHMS = {
    "deepmimic": ("deepmimic", "deepmimic_g1_ppo"),
    "awr": ("deepmimic", "deepmimic_g1_awr"),
    "amp": ("amp", "amp_g1"),
    "ase": ("ase", "ase_g1"),
    "lcp": ("deepmimic", "lcp_g1"),
    "add": ("add", "add_g1"),
    "smp": ("smp", "smp_g1"),
}


def select_device(value, *, cpu_physics=True):
    if value == "auto":
        # Native CPU physics plus small MPS environment operations was slower
        # on all seven measured G1 workloads. Explicit MPS remains available.
        if torch.cuda.is_available():
            return "cuda:0"
        # Standalone prior training has no simulator round trips; preserve its
        # accelerator preference rather than applying policy-rollout timings.
        if not cpu_physics and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    dev = torch.device(value)
    if dev.type == "mps" and not torch.backends.mps.is_available():
        raise ValueError("MPS is unavailable; select --device cpu")
    if dev.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable; install CUDA PyTorch on an NVIDIA machine")
    if dev.type not in ("cpu", "mps", "cuda"):
        raise ValueError("Choose cpu, mps, cuda:N or auto")
    return str(dev)


def read_yaml(path):
    with open(path) as f:
        return yaml.safe_load(f)


def smoke_config(config):
    """A small real optimizer update, explicitly not a learning-quality preset."""
    config = dict(config)
    config["model"] = dict(config["model"])
    for k in config["model"]:
        if k.endswith("_net"):
            config["model"][k] = "fc_2layers_128units"
    config.update(steps_per_iter=8, iters_per_output=1, test_episodes=1)
    for k in list(config):
        if k.endswith("_epochs"):
            config[k] = 1
        elif k.endswith("_batch_size") and k != "smp_eval_batch_size":
            config[k] = 1
    if "disc_buffer_size" in config:
        config.update(disc_buffer_size=256, disc_replay_samples=16)
    return config


def parser():
    p = argparse.ArgumentParser(description="MimicKit + native MuJoCo for Unitree G1 (29 DoF)")
    p.add_argument("command", choices=("train", "play", "view-motion", "train-prior", "fetch-data", "doctor"))
    p.add_argument("--algorithm", choices=ALGORITHMS, default="deepmimic")
    p.add_argument(
        "--device",
        default="auto",
        help="auto: CUDA if available, otherwise CPU; MPS is opt-in except train-prior; physics always CPU",
    )
    p.add_argument("--num-envs", type=int, default=None)
    p.add_argument("--num-threads", type=int, default=4)
    p.add_argument("--iterations", type=int, default=None)
    p.add_argument("--steps", type=int, default=600, help="Playback or motion-view steps")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--motion", type=Path, help="MimicKit motion .pkl or dataset .yaml")
    p.add_argument("--archive", type=Path, help="Local official ZIP for fetch-data")
    p.add_argument("--checkpoint", type=Path, help="Policy weights; play also restores adjacent saved configs")
    p.add_argument("--out-dir", type=Path)
    p.add_argument("--prior-dir", type=Path, help="SMP directory containing model.pt and diffusion_config.yaml")
    p.add_argument("--visualize", action="store_true", help="Native viewer; use mjpython on macOS")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--smoke", action="store_true", help="Small networks and a short real update; no convergence claim")
    p.add_argument("--logger", choices=("txt", "tb", "wandb"), default="txt")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for key in ("motion", "checkpoint", "out_dir", "prior_dir", "archive"):
        val = getattr(args, key)
        if val is not None:
            setattr(args, key, val.resolve())
    if not (ROOT / "data/engines/mujoco_engine.yaml").exists():
        raise SystemExit("Run from a source checkout installed with pip install -e . (data/ is required).")
    os.chdir(ROOT)
    bootstrap()
    if args.command == "fetch-data":
        from .data import fetch_data

        fetch_data(args.archive)
        return
    device = select_device(args.device, cpu_physics=args.command != "train-prior")
    if args.command == "doctor":
        import mujoco

        print(
            json.dumps(
                {
                    "python": sys.version.split()[0],
                    "torch": torch.__version__,
                    "mujoco": mujoco.__version__,
                    "learner": device,
                    "physics": "cpu",
                    "mps": torch.backends.mps.is_available(),
                    "cuda": torch.cuda.is_available(),
                    "g1_model": (ROOT / "data/assets/g1/g1.xml").exists(),
                    "official_walk": (ROOT / "data/motions/g1/g1_walk.pkl").exists(),
                },
                indent=2,
            )
        )
        return
    if args.num_threads < 1 or (args.num_envs is not None and args.num_envs < 1):
        raise SystemExit("--num-threads and --num-envs must be positive")
    if (args.iterations is not None and args.iterations < 1) or args.steps < 1:
        raise SystemExit("--iterations and --steps must be positive")
    import util.mp_util as mp_util
    from util.util import set_rand_seed

    mp_util.init(0, 1, device, 0)
    set_rand_seed(args.seed)
    torch.set_num_threads(args.num_threads)
    if args.command == "train-prior":
        from .prior import train_prior

        train_prior(args, device)
        return
    if args.command == "play" and args.checkpoint is None:
        raise SystemExit("play requires --checkpoint output/RUN/model.pt")
    env_name, agent_name = ALGORITHMS[args.algorithm]
    env_cfg = read_yaml(f"data/envs/{env_name}_g1_env.yaml")
    agent_cfg = read_yaml(f"data/agents/{agent_name}_agent.yaml")
    engine_cfg = read_yaml("data/engines/mujoco_engine.yaml")
    if args.checkpoint is not None:
        for filename in ("env_config.yaml", "agent_config.yaml", "engine_config.yaml"):
            if not (args.checkpoint.parent / filename).exists():
                raise SystemExit(f"Keep saved {filename} next to the checkpoint")
        env_cfg = read_yaml(args.checkpoint.parent / "env_config.yaml")
        agent_cfg = read_yaml(args.checkpoint.parent / "agent_config.yaml")
        engine_cfg = read_yaml(args.checkpoint.parent / "engine_config.yaml")
    if args.command == "view-motion":
        env_cfg = read_yaml("data/envs/view_motion_g1_env.yaml")
    if args.motion is not None:
        env_cfg["motion_file"] = str(args.motion)
    if args.prior_dir is not None:
        agent_cfg["smp_prior_cfg"] = str(args.prior_dir / "diffusion_config.yaml")
        agent_cfg["smp_prior_model"] = str(args.prior_dir / "model.pt")
    engine_cfg["num_threads"] = args.num_threads
    if args.smoke and args.checkpoint is None:
        agent_cfg = smoke_config(agent_cfg)
        env_cfg["episode_length"] = 0.5
    if not Path(env_cfg["char_file"]).exists() or not Path(env_cfg["motion_file"]).exists():
        raise SystemExit("Missing official G1 assets. Run: python -m mimickit_mujoco fetch-data")
    if args.command != "view-motion" and agent_cfg["agent_name"] == "SMP":
        if not Path(agent_cfg["smp_prior_model"]).exists():
            raise SystemExit("Train a G1 prior first: python -m mimickit_mujoco train-prior; then use --prior-dir")
    visualize = (args.visualize or args.command in ("play", "view-motion")) and not args.headless
    if visualize and sys.platform == "darwin" and not os.environ.get("MJPYTHON_BIN"):
        # The native viewer itself gives the definitive mjpython check. Keep
        # imports headless until this point so CPU training needs no display.
        print("macOS native viewer: launch this command with .venv/bin/mjpython")
    n = args.num_envs or (4 if args.smoke else 64 if args.command == "train" else 1)
    out_dir = args.out_dir or ROOT / "output" / f"{args.algorithm}_g1_{time.strftime('%Y%m%d_%H%M%S')}"
    if args.command == "train":
        if (out_dir / "model.pt").exists():
            raise SystemExit("Output already contains model.pt; select a new --out-dir to preserve it")
        out_dir.mkdir(parents=True, exist_ok=True)
    # Builders take filenames; save effective configs for reproducible playback.
    import tempfile

    from envs.env_builder import build_env
    from learning.agent_builder import build_agent

    with tempfile.TemporaryDirectory(prefix="mimickit-config-") as tmp:
        cfg_dir = out_dir if args.command == "train" else Path(tmp)
        for name, cfg in (("env", env_cfg), ("agent", agent_cfg), ("engine", engine_cfg)):
            (cfg_dir / f"{name}_config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
        print(f"G1 | {agent_cfg['agent_name']} | physics=cpu | learner={device} | environments={n}")
        env = build_env(str(cfg_dir / "env_config.yaml"), str(cfg_dir / "engine_config.yaml"), n, device, visualize)
        try:
            if args.command == "view-motion":
                env.reset()
                for _ in range(args.steps):
                    _, _, done, _ = env.step(torch.zeros((n, 29), device=device))
                    ids = (done != 0).nonzero(as_tuple=False).flatten()
                    if len(ids):
                        env.reset(ids)
                print(f"Viewed {args.steps} motion frames")
                return
            agent = build_agent(str(cfg_dir / "agent_config.yaml"), env, device)
            if args.checkpoint:
                agent.load(str(args.checkpoint))
            if args.command == "train":
                iterations = args.iterations or (1 if args.smoke else 10001)
                metadata = dict(
                    algorithm=agent_cfg["agent_name"],
                    learner=device,
                    num_envs=n,
                    seed=args.seed,
                    smoke=args.smoke,
                    iterations=iterations,
                    upstream="fbe1a36e878f036d6bf5521defb8eb3b8ab4170d",
                    initial_weights=str(args.checkpoint) if args.checkpoint else None,
                )
                (out_dir / "run_config.json").write_text(json.dumps(metadata, indent=2) + "\n")
                agent.train_model(iterations * n * agent_cfg["steps_per_iter"], str(out_dir), True, args.logger)
                if not all(torch.isfinite(p).all() for p in agent.parameters()):
                    raise FloatingPointError("Nonfinite model parameters after training")
                print(f"Saved policy and configs: {out_dir}")
            else:
                from learning.base_agent import AgentMode

                agent.eval()
                agent.set_mode(AgentMode.TEST)
                obs, info = agent._reset_envs()
                total_reward = 0.0
                with torch.no_grad():
                    for _ in range(args.steps):
                        action, _ = agent._decide_action(obs, info)
                        obs, r, done, info = agent._step_env(action)
                        if not torch.isfinite(obs).all():
                            raise FloatingPointError("Nonfinite playback observations")
                        total_reward += r.mean().item()
                        obs, info = agent._reset_done_envs(done)
                print(json.dumps({"steps": args.steps, "mean_step_reward": total_reward / args.steps}))
        except KeyboardInterrupt:
            print("Stopped; previously saved checkpoints are preserved.")
        finally:
            env._engine.close()
