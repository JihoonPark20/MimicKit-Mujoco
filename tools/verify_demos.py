"""Exercise selected G1 demos through reference, prior, policy training and reload.

These are execution checks with small networks, not trained skill evaluations.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimickit_mujoco.cli import ALGORITHMS, read_yaml  # noqa: E402
from mimickit_mujoco.demos import DEMOS, apply_demo, read_demo  # noqa: E402
from tools.verify_mujoco import verify_updates  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--demos", choices=DEMOS, nargs="+", default=list(DEMOS))
    parser.add_argument("--algorithms", choices=ALGORITHMS, nargs="+", default=list(ALGORITHMS))
    args = parser.parse_args()
    out = args.out_dir.resolve()
    if out.exists() and any(out.iterdir()):
        raise SystemExit("Choose a new, empty output directory to preserve previous results.")
    out.mkdir(parents=True, exist_ok=True)
    results = []

    def run(name, argv):
        log_path = out / f"{name}.log"
        with log_path.open("w") as log:
            process = subprocess.run(
                [sys.executable, "-m", "mimickit_mujoco", *argv, "--device", args.device],
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        results.append({"name": name, "device": args.device, "exit_code": process.returncode})
        (out / "results.json").write_text(json.dumps(results, indent=2) + "\n")
        print(f"{name}: exit {process.returncode}", flush=True)
        if process.returncode:
            print(log_path.read_text()[-4000:])
            raise SystemExit(process.returncode)

    for demo in args.demos:
        run(f"{demo}_reference", ["view-motion", "--demo", demo, "--headless", "--steps", "300"])
        prior = out / demo / "prior"
        if "smp" in args.algorithms:
            run(f"{demo}_prior", ["train-prior", "--demo", demo, "--smoke", "--out-dir", str(prior)])
            assert read_yaml(prior / "diffusion_config.yaml")["motion_file"] == read_demo(demo)["motion_file"]
            assert read_yaml(prior / "env_config.yaml")["demo_name"] == demo
        for algorithm in args.algorithms:
            dest = out / demo / algorithm
            argv = [
                "train",
                "--algorithm",
                algorithm,
                "--demo",
                demo,
                "--smoke",
                "--iterations",
                "2",
                "--out-dir",
                str(dest),
            ]
            if algorithm == "smp":
                argv += ["--prior-dir", str(prior)]
            run(f"{demo}_{algorithm}_train", argv)
            verify_updates(dest)
            saved = read_yaml(dest / "env_config.yaml")
            expected = apply_demo(read_yaml(ROOT / f"data/envs/{ALGORITHMS[algorithm][0]}_g1_env.yaml"), demo)
            for key in ("motion_file", "demo_name", "contact_bodies", "enable_early_termination"):
                assert saved[key] == expected[key], (demo, algorithm, key)
            # Do not supply --demo: exercise the configuration saved with the policy.
            run(
                f"{demo}_{algorithm}_play",
                ["play", "--checkpoint", str(dest / "model.pt"), "--headless", "--steps", "60"],
            )


if __name__ == "__main__":
    main()
