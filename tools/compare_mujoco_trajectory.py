"""Compare seeded environment trajectories against a saved pre-optimization engine."""

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimickit_mujoco import bootstrap  # noqa: E402

bootstrap()
from engines import mujoco_engine  # noqa: E402
from envs.env_builder import build_env  # noqa: E402
from util.util import set_rand_seed  # noqa: E402


def trace(engine_class, device, steps):
    original = mujoco_engine.MujocoEngine
    mujoco_engine.MujocoEngine = engine_class
    env = None
    try:
        set_rand_seed(42)
        torch.set_num_threads(4)
        env = build_env("data/envs/deepmimic_g1_env.yaml", "data/engines/mujoco_engine.yaml", 8, device, False)
        rng = np.random.default_rng(27)
        records = {k: [] for k in ("obs", "reward", "done", "qpos", "qvel", "body", "contacts")}
        env.reset()
        for step in range(steps):
            action = torch.tensor(rng.normal(0, 0.2, (8, 29)), dtype=torch.float32, device=device)
            obs, reward, done, _ = env.step(action)
            values = (
                obs.cpu().numpy(),
                reward.cpu().numpy(),
                done.cpu().numpy(),
                env._engine.objects[0].qpos,
                env._engine.objects[0].qvel,
                env._engine.get_body_pos(0).cpu().numpy(),
                env._engine.get_ground_contact_forces(0).cpu().numpy(),
            )
            for key, value in zip(records, values):
                records[key].append(value.copy())
            ids = (done != 0).nonzero(as_tuple=False).flatten()
            if step % 13 == 0:
                ids = torch.tensor([1, 5], device=device)
            env.reset(ids)
        return {key: np.array(value) for key, value in records.items()}
    finally:
        if env is not None:
            env._engine.close()
        mujoco_engine.MujocoEngine = original


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-engine", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--steps", type=int, default=120)
    args = parser.parse_args()
    args.baseline_engine = args.baseline_engine.resolve()
    args.output = args.output.resolve()
    os.chdir(ROOT)
    spec = importlib.util.spec_from_file_location("baseline_engine", args.baseline_engine)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    original = trace(baseline.MujocoEngine, args.device, args.steps)
    optimized = trace(mujoco_engine.MujocoEngine, args.device, args.steps)
    report = {
        "device": args.device,
        "num_envs": 8,
        "control_steps": args.steps,
        "baseline_engine_sha256": hashlib.sha256(args.baseline_engine.read_bytes()).hexdigest(),
        "optimized_engine_sha256": hashlib.sha256(Path(mujoco_engine.__file__).read_bytes()).hexdigest(),
        "max_abs_difference": {key: float(np.max(np.abs(original[key] - optimized[key]))) for key in original},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    for key in original:
        np.testing.assert_array_equal(original[key], optimized[key], err_msg=key)


if __name__ == "__main__":
    main()
