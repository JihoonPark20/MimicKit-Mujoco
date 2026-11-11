"""Run real G1 updates, checkpoint reloads and SMP prior training for all methods.

Example: python tools/verify_mujoco.py --device mps --out-dir output/verify_mps
These are integration smoke tests, not learned-motion quality benchmarks.
"""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimickit_mujoco.cli import ALGORITHMS  # noqa: E402


def verify_updates(run_dir):
    """Check that consecutive iterations changed real network weights."""
    import torch

    snapshots = sorted((run_dir / "int_models").glob("*.pt"))
    if len(snapshots) != 2:
        raise AssertionError("Expected checkpoints from both optimizer iterations")
    first, second = [torch.load(p, map_location="cpu", weights_only=True) for p in snapshots]
    keys = [k for k in first if k.startswith("_model.") and k.endswith(("weight", "bias"))]
    if not keys or not any(not torch.equal(first[k], second[k]) for k in keys):
        raise AssertionError("No network parameter changed between iterations")
    if not all(torch.isfinite(v).all() for v in second.values()):
        raise AssertionError("Checkpoint contains nonfinite state")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def run(name, argv):
        start = time.monotonic()
        log_path = out / f"{name}.log"
        with log_path.open("w") as log:
            result = subprocess.run(
                [sys.executable, "-m", "mimickit_mujoco", *argv, "--device", args.device],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        entry = {
            "name": name,
            "device": args.device,
            "exit_code": result.returncode,
            "seconds": round(time.monotonic() - start, 3),
            "log": str(log_path),
        }
        results.append(entry)
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(json.dumps(entry), flush=True)
        if result.returncode:
            print(log_path.read_text()[-8000:])
            raise SystemExit(result.returncode)

    prior = out / "prior"
    run("prior", ["train-prior", "--smoke", "--out-dir", str(prior)])
    for algorithm in ALGORITHMS:
        run_dir = out / algorithm
        run(
            f"train_{algorithm}",
            [
                "train",
                "--algorithm",
                algorithm,
                "--smoke",
                "--iterations",
                "2",
                "--out-dir",
                str(run_dir),
                "--prior-dir",
                str(prior),
            ],
        )
        verify_updates(run_dir)
        run(f"play_{algorithm}", ["play", "--checkpoint", str(run_dir / "model.pt"), "--headless", "--steps", "40"])
    run("view_motion", ["view-motion", "--headless", "--steps", "60"])


if __name__ == "__main__":
    main()
