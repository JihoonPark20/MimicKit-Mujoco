"""Instrument the backend and test a substep-batching hypothesis in isolation.

No production engine code is changed. The candidate is limited to the shipped
position-control mode. Numerical checks precede timing comparisons.
"""

import argparse
import collections
import json
import os
import statistics
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimickit_mujoco import bootstrap  # noqa: E402

bootstrap()
from engines import mujoco_engine  # noqa: E402
from envs.env_builder import build_env  # noqa: E402
from util.util import set_rand_seed  # noqa: E402


def make_env(device, n=64):
    set_rand_seed(42)
    env = build_env("data/envs/deepmimic_g1_env.yaml", "data/engines/mujoco_engine.yaml", n, device, False)
    env.reset()
    return env


def instrument(device):
    env = make_env(device)
    engine = env._engine
    counts = collections.Counter()
    sizes = collections.Counter()
    orig_tensor, orig_numpy = engine._tensor, mujoco_engine.as_numpy

    def tensor(value):
        caller = sys._getframe(1)
        name = caller.f_locals.get("name", caller.f_code.co_name)
        counts[f"tensor:{name}"] += 1
        sizes["returned_float32_bytes"] += np.asarray(value).size * 4
        return orig_tensor(value)

    def numpy(value):
        if isinstance(value, torch.Tensor):
            counts["tensor_to_cpu_calls"] += 1
            sizes["tensor_to_cpu_bytes"] += value.numel() * value.element_size()
        return orig_numpy(value)

    engine._tensor = tensor
    mujoco_engine.as_numpy = numpy
    original_refresh = mujoco_engine._Object.refresh

    def refresh(obj):
        if obj.dirty:
            mask = getattr(obj, "_dirty_ids", np.ones(len(obj.qpos), dtype=bool))
            counts["batch_forward_calls"] += 1
            counts["forwarded_environments"] += int(mask.sum())
            counts["full_batch_forward_calls" if mask.all() else "partial_batch_forward_calls"] += 1
        return original_refresh(obj)

    mujoco_engine._Object.refresh = refresh
    try:
        env.step(torch.zeros((64, 29), device=device))
        normal = {"calls": dict(counts), "sizes": dict(sizes)}
        counts.clear()
        sizes.clear()
        env.reset(torch.tensor([0], device=device))
        partial = {"calls": dict(counts), "sizes": dict(sizes)}
        return {
            "device": device,
            "one_control_step": normal,
            "one_environment_reset": partial,
            "num_sensors": int(engine.objects[0].model.nsensor),
            "sensor_scalar_outputs": int(engine.objects[0].model.nsensordata),
        }
    finally:
        mujoco_engine.as_numpy = orig_numpy
        mujoco_engine._Object.refresh = original_refresh
        env._engine.close()


def fixed_target_step(engine, grouped):
    """Compare fixed targets with one native call versus eight Python calls."""
    for obj in engine.objects:
        for _ in range(1 if grouped else engine.substeps):
            obj.ctrl[:] = obj.cmd
            obj.batch.step(nstep=engine.substeps if grouped else 1, stage=mujoco.mjtStage.mjSTAGE_NONE)
        bad = [
            int(mujoco.mjtWarning.mjWARN_BADQPOS),
            int(mujoco.mjtWarning.mjWARN_BADQVEL),
            int(mujoco.mjtWarning.mjWARN_BADQACC),
        ]
        if np.any(obj.batch.warning[:, bad, 1] > 0) or not np.isfinite(obj.qpos).all():
            raise FloatingPointError("Invalid candidate dynamics")
        obj.dirty = True
        obj.refresh()
        obj.applied[:] = 0


def compare_substeps():
    env_a, env_b = make_env("cpu"), make_env("cpu")
    a, b = env_a._engine, env_b._engine
    oa, ob = a.objects[0], b.objects[0]
    states = {k: getattr(oa, k).copy() for k in ("qpos", "qvel")}
    command = states["qpos"][:, oa.qids].copy()

    def reset(obj):
        obj.batch.reset(stage=mujoco.mjtStage.mjSTAGE_NONE)
        for k, value in states.items():
            getattr(obj, k)[:] = value
        obj.cmd[:] = command
        obj.dirty = True
        obj.refresh()

    try:
        reset(oa)
        reset(ob)
        maximum = collections.defaultdict(float)
        for _ in range(120):
            fixed_target_step(a, False)
            fixed_target_step(b, True)
            for key in ("qpos", "qvel", "xpos", "xquat", "sensors", "forces"):
                error = float(np.max(np.abs(getattr(oa, key) - getattr(ob, key))))
                maximum[key] = max(maximum[key], error)
        assert max(maximum.values()) < 1e-9, maximum
        timings = {"baseline": [], "grouped": []}
        # Alternate order to reduce thermal/order bias; resets precede timings.
        for repeat in range(5):
            order = ("baseline", "grouped") if repeat % 2 == 0 else ("grouped", "baseline")
            for name in order:
                reset(oa if name == "baseline" else ob)
                engine, grouped = (a, False) if name == "baseline" else (b, True)
                start = time.perf_counter()
                for _ in range(120):
                    fixed_target_step(engine, grouped)
                timings[name].append(time.perf_counter() - start)
        return {
            "num_envs": 64,
            "control_steps": 120,
            "max_abs_state_difference": dict(maximum),
            "repeat_seconds": timings,
            "physics_only_speedup": statistics.median(timings["baseline"]) / statistics.median(timings["grouped"]),
        }
    finally:
        a.close()
        b.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output = args.output.resolve()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    devices = ["cpu"]
    if torch.backends.mps.is_available():
        devices.append("mps")
    report = {"instrumentation": [instrument(device) for device in devices], "substep_batching": compare_substeps()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
