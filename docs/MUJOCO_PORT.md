# MuJoCo port scope and provenance

## Sources

- [MimicKit](https://github.com/xbpeng/MimicKit), commit `fbe1a36e878f036d6bf5521defb8eb3b8ab4170d`: learning methods, kinematics, motion formats, tools and original configurations. Its [Apache-2.0 license](../LICENSE) and [citation](../CITATION.cff) are retained.
- [legged-gym-mujoco](https://github.com/JihoonPark20/legged-gym-mujoco): the CPU batch simulation framework, including native MuJoCo state handling, actuator-based PD control, contact sensors and viewer lifecycle. Its [BSD license](../licenses/legged-gym-mujoco-BSD-3-Clause.txt) and third-party notices are retained.
- Official MimicKit data bundle: the G1 model, meshes and sample motions. Extracted files are recorded in [official_g1_manifest.json](../data/official_g1_manifest.json). The [Unitree license](../licenses/unitree-g1-BSD-3-Clause.txt) and [bundle license](../data/LICENSE) are included.

The MuJoCo backend is in [mujoco_engine.py](../mimickit/engines/mujoco_engine.py). The [convenience CLI](../mimickit_mujoco/cli.py) provides G1 training, prior training, playback and data installation.

## Representation and control

The G1 model has 29 independent hinge joints. Joint names determine the MuJoCo DOF mapping. Fixed bodies remain in the body list so the kinematic model and simulator agree.

MimicKit uses `xyzw` quaternions; MuJoCo uses `wxyz`. The backend exposes root and body velocities in world coordinates and converts the free joint's local angular velocity internally. Getter results are read-only float32 tensors on the selected device; clone them before modification and use setters to change simulator state.

The default configuration uses position actuators with the official asset's stiffness, damping and torque limits, the `implicitfast` integrator, **240 Hz physics** and **30 Hz control**. Velocity, torque, disabled-control and explicit PD modes are also available through the engine configuration.

The scene includes a flat floor and frictional robot contacts (`condim=3`). Ground-contact and total-contact sensors report world-space forces on the robot. Each environment has independent state. Visual reference characters are nonphysical and displayed in the native viewer.

## Algorithms and execution

DeepMimic/PPO, AWR, AMP, ASE, LCP, ADD and SMP retain their upstream learning implementations. G1 AWR uses the upstream AWR recipe with G1 observations. ASE uses the official G1 walk/run clips. SMP requires a separately trained G1 TinyMDM prior.

Physics runs on the CPU. Automatic learner selection uses CUDA when available and CPU otherwise; MPS can be selected explicitly. Standalone prior training selects CUDA, then MPS, then CPU. The convenience CLI uses one learner process. The original distributed launcher remains available, but distributed MuJoCo training has not been validated.

Checkpoints contain model weights and normalizers. The CLI also saves the effective environment, engine and agent configurations. Loading a checkpoint starts a new optimizer, replay buffer, random-number stream and simulation state; it does not restore an exact interrupted training session.

## Supported scope

- One G1 robot per independent flat-ground environment, with optional visual reference characters.
- Seven learning methods; this does not include every upstream robot or scene. Other robots, spherical joints, fixed-root tasks, interacting objects and obstacle tasks require additional backend work.
- Official G1 sample motions. The short walk clip is a starting point for experiments; vault clips require additional scene setup.
- Source-based installation with `pip install -e .` or `uv sync`. Keep the repository's assets and configuration tree alongside the Python packages.

Functional checks cover Apple Silicon CPU and MPS execution. Linux/NVIDIA CUDA and multi-GPU execution require hardware validation. Policies trained with other simulators may require adaptation to MuJoCo's contact, actuator and solver behavior.

See the [README](../README.md) for commands and the [validation guide](MUJOCO_VALIDATION.md) for functional checks.
