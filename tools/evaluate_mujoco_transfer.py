"""Evaluate an upstream policy on MuJoCo; a transfer diagnostic, not retraining."""

import argparse
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
from envs.env_builder import build_env  # noqa: E402
from learning.agent_builder import build_agent  # noqa: E402
from learning.base_agent import AgentMode  # noqa: E402
from util import mp_util  # noqa: E402
from util.util import set_rand_seed  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-envs", type=int, default=8)
    parser.add_argument("--episodes-per-env", type=int, default=4)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--env-config", default="data/envs/deepmimic_g1_env.yaml")
    parser.add_argument("--agent-config", default="data/agents/lcp_g1_agent.yaml")
    args = parser.parse_args()
    args.checkpoint, args.output = args.checkpoint.resolve(), args.output.resolve()
    os.chdir(ROOT)
    torch.set_num_threads(4)
    mp_util.init(0, 1, "cpu", 0)
    env = build_env(args.env_config, "data/engines/mujoco_engine.yaml", args.num_envs, "cpu", False)
    try:
        agent = build_agent(args.agent_config, env, "cpu")
        agent.load(str(args.checkpoint))
        agent.eval()
        agent.set_mode(AgentMode.TEST)
        records = []
        with torch.no_grad():
            for seed in args.seeds:
                set_rand_seed(seed)
                obs, info = agent._reset_envs()
                counts = np.zeros(args.num_envs, dtype=int)
                returns = torch.zeros(args.num_envs)
                lengths = torch.zeros(args.num_envs, dtype=torch.long)
                for step in range(10000):
                    action, _ = agent._decide_action(obs, info)
                    obs, reward, done, info = agent._step_env(action)
                    assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
                    returns += reward
                    lengths += 1
                    failure_details = {}
                    if (done == 1).any() and env._global_obs and env._track_global_root():
                        # Match the shipped global-reference G1 termination rules.
                        ground = env._engine.get_ground_contact_forces(0).clone()
                        ground[:, env._contact_body_ids] = 0
                        contact_failure = (ground.abs() > 0.1).any(dim=-1).any(dim=-1)
                        body_error = (
                            torch.linalg.vector_norm(env._engine.get_body_pos(0) - env._ref_body_pos, dim=-1)
                            .max(dim=-1)
                            .values
                        )
                        for idx in (done == 1).nonzero(as_tuple=False).flatten().tolist():
                            failure_details[idx] = {
                                "contact_failure": bool(contact_failure[idx]),
                                "pose_failure": bool(body_error[idx] > env._pose_termination_dist),
                                "max_body_position_error_m": body_error[idx].item(),
                            }
                    for idx in (done != 0).nonzero(as_tuple=False).flatten().tolist():
                        if counts[idx] < args.episodes_per_env:
                            records.append(
                                {
                                    "seed": seed,
                                    "env": idx,
                                    "return": returns[idx].item(),
                                    "length_steps": lengths[idx].item(),
                                    "done": done[idx].item(),
                                    **failure_details.get(idx, {}),
                                }
                            )
                            counts[idx] += 1
                        returns[idx] = 0
                        lengths[idx] = 0
                    obs, info = agent._reset_done_envs(done)
                    if min(counts) == args.episodes_per_env:
                        break
                else:
                    raise RuntimeError("Episode budget not reached")
                print(f"seed={seed} completed {sum(counts)} episodes in {step + 1} vector steps", flush=True)
        result = {
            "checkpoint": str(args.checkpoint),
            "env_config": args.env_config,
            "agent_config": args.agent_config,
            "num_envs": args.num_envs,
            "episodes": len(records),
            "mean_return": float(np.mean([r["return"] for r in records])),
            "mean_length_seconds": float(np.mean([r["length_steps"] / 30 for r in records])),
            "failure_fraction": float(np.mean([r["done"] == 1 for r in records])),
            "timeout_fraction": float(np.mean([r["done"] == 3 for r in records])),
            "records": records,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps({k: v for k, v in result.items() if k != "records"}))
    finally:
        env._engine.close()


if __name__ == "__main__":
    main()
