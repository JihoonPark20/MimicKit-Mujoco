# G1 MuJoCo validation

Functional validation uses the official G1 model and sample motions. The latest checks passed on **2026-11-12**, with macOS Apple Silicon, Python **3.12.4**, PyTorch **2.15.0** and MuJoCo **3.17.0**. Resolved dependencies are recorded in [uv.lock](../uv.lock).

## Run the checks

Install the test dependencies from the repository root:

```bash
python -m pip install -e '.[test]'
python -m pytest -q
python tools/verify_mujoco.py --device cpu --out-dir output/verify_cpu
```

Use `--device mps` on a Mac with MPS support. Choose a new output directory for each integration run to preserve existing checkpoints. The script writes results and logs to that directory.

## Coverage

The unit tests cover:

- G1 forward kinematics, joint ordering and quaternion conventions.
- World-space root/body velocities and independent partial resets.
- Ground and total contact forces against independently computed contact wrenches.
- Actuator limits, control modes and invalid-action handling.
- State consistency across stepping and resets, retained observation values, and engine lifecycle.
- Reference-motion playback and device selection.
- Demo contact rules, checkpoint preset handling and matching SMP priors.

The integration script trains a small G1 prior, runs two actual updates for **DeepMimic/PPO, AWR, AMP, ASE, LCP, ADD and SMP**, reloads every saved policy, and plays reference motion. It checks that model weights change between updates and that checkpoints contain finite values.

The current unit suite passes **23 tests**. The demo integration coverage is described below. Smoke runs use four environments, small networks and short episodes. Native viewer execution and official asset checksum verification have also been checked.

## Motion demo checks

The [G1 demos](G1_DEMOS.md) cover `spinkick`, `kick_combo` and `cartwheel` with all seven algorithms. On **2026-11-12**, all **96 integration commands** passed across the supported Mac CPU and MPS execution paths: for each device, three 300-step reference runs, three matching SMP prior runs, and training/reload/playback for 21 motion/algorithm combinations.

Every policy completed two small updates with changed, finite model weights. Saved motion names and contact settings matched the selected preset, and playback restored those settings from the checkpoint. Unit tests also confirm that the cartwheel preset permits wrist contact while retaining head-contact termination.

```bash
python tools/verify_demos.py --device cpu --out-dir output/verify_demos
```

Use `--device mps` on a compatible Mac. Select a subset with `--demos cartwheel --algorithms amp smp`. These runs establish execution; the policies and priors used for verification are not converged skill demonstrations.

## Validation scope

These checks establish functional execution, checkpoint handling and simulator-state consistency. They do not establish policy convergence, paper-result reproduction or readiness for deployment on a physical robot.

Linux/NVIDIA CUDA, distributed MuJoCo training, other robots and obstacle/object-interaction tasks have not been validated. Motion quality and transfer of policies trained with other simulators require task-specific evaluation.
