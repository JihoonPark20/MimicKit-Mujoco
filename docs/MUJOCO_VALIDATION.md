# G1 MuJoCo validation

Functional validation uses the official G1 model and sample motions. The latest checks passed on **2026-11-11**, with macOS Apple Silicon, Python **3.12.4**, PyTorch **2.15.0** and MuJoCo **3.17.0**. Resolved dependencies are recorded in [uv.lock](../uv.lock).

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

The integration script trains a small G1 prior, runs two actual updates for **DeepMimic/PPO, AWR, AMP, ASE, LCP, ADD and SMP**, reloads every saved policy, and plays reference motion. It checks that model weights change between updates and that checkpoints contain finite values.

The latest validation completed **15 unit tests and 32 integration commands** across the supported Mac execution paths. Smoke runs use four environments, small networks and short episodes. Native viewer execution and official asset checksum verification have also been checked.

## Validation scope

These checks establish functional execution, checkpoint handling and simulator-state consistency. They do not establish policy convergence, paper-result reproduction or readiness for deployment on a physical robot.

Linux/NVIDIA CUDA, distributed MuJoCo training, other robots and obstacle/object-interaction tasks have not been validated. Motion quality and transfer of policies trained with other simulators require task-specific evaluation.
