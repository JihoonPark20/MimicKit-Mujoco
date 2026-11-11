"""Native CPU MuJoCo backend for G1 motion imitation.

Batch ownership, persistent MjData, and substep control follow the supplied
legged-gym-mujoco framework (BSD-3-Clause; see licenses/). Public state follows
MimicKit: xyzw rotations, world-space root/body velocities, DFS joint order.
MuJoCo stores wxyz rotations and body-local free-joint angular velocity.
"""

import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
import torch
from engines.engine import ControlMode, Engine, ObjType
from mujoco import batch


def as_numpy(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def rotate(q, v):
    """Rotate vectors by unit wxyz quaternions."""
    t = 2 * np.cross(q[..., 1:], v)
    return v + q[..., :1] * t + np.cross(q[..., 1:], t)


def inverse_rotate(q, v):
    inv = q.copy()
    inv[..., 1:] *= -1
    return rotate(inv, v)


class _Object:
    def __init__(self, config, asset_file, num_envs, visual, start_pos, start_rot, color, collisions, disable_motors):
        path = Path(asset_file).resolve()
        root = ET.parse(path).getroot()
        compiler = root.find("compiler")
        if compiler is None:
            compiler = ET.SubElement(root, "compiler")
        for key in ("meshdir", "texturedir"):
            compiler.set(key, str(path.parent / compiler.get(key, "")))
        compiler.set("fusestatic", "false")
        compiler.set("alignfree", "false")
        world = root.find("worldbody")
        bodies = list(world.iter("body"))
        if len(world.findall("body")) != 1:
            raise ValueError("G1 backend expects one articulated root per asset")
        for b in bodies:
            if len(b.findall("joint")) > 1:
                raise ValueError("This backend supports G1 hinge joints, not spherical humanoid joints")
        self.names = [b.get("name") for b in bodies]
        if any(n is None for n in self.names):
            raise ValueError("All articulated bodies must be named")
        joints = list(world.iter("joint"))
        joints = [j for j in joints if j.get("type", "hinge") != "free"]
        if any(j.get("type", "hinge") != "hinge" for j in joints):
            raise ValueError("Only hinge joints are supported by the G1 backend")
        # Read resolved defaults before converting drive metadata into actuators.
        source = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        joint_names = [j.get("name") for j in joints]
        ids = np.array([source.joint(n).id for n in joint_names])
        self.kp = source.jnt_stiffness[ids].copy()
        self.kd = source.dof_damping[source.jnt_dofadr[ids]].copy()
        self.limits = source.jnt_range[ids].copy()
        self.effort = np.max(np.abs(source.jnt_actfrcrange[ids]), axis=1)
        if np.any(self.effort <= 0):
            raise ValueError("Each G1 joint must define a positive actuatorfrcrange")
        self.mode = ControlMode[config.get("control_mode", "pos")]
        for old in list(root.findall("actuator")) + list(root.findall("sensor")) + list(root.findall("keyframe")):
            root.remove(old)
        option = root.find("option")
        if option is None:
            option = ET.SubElement(root, "option")
        option.attrib.update(
            timestep=str(1 / config["sim_freq"]),
            integrator="implicitfast",
            gravity="0 0 -9.81",
            solver="Newton",
            iterations=str(config.get("iterations", 50)),
        )
        actuators = ET.SubElement(root, "actuator")
        for i, j in enumerate(joints):
            j.set("stiffness", "0")
            j.set("damping", "0")
            attrs = dict(
                name=f"drive_{i}",
                joint=joint_names[i],
                forcelimited="true",
                forcerange=f"{-self.effort[i]} {self.effort[i]}",
                ctrllimited="false",
            )
            if self.mode == ControlMode.pos:
                ET.SubElement(actuators, "position", **attrs, kp=str(self.kp[i]), kv=str(self.kd[i]))
            elif self.mode == ControlMode.vel:
                ET.SubElement(actuators, "velocity", **attrs, kv=str(self.kd[i]))
            else:
                ET.SubElement(actuators, "motor", **attrs, gear="1")
        if not visual:
            ET.SubElement(
                world,
                "geom",
                name="mimickit_floor",
                type="plane",
                size="0 0 0.1",
                friction="1 0.005 0.0001",
                rgba="0.22 0.25 0.3 1",
            )
            ET.SubElement(world, "light", pos="0 -3 5", dir="0 0 -1", directional="true")
        for geom in bodies[0].iter("geom"):
            if visual:
                geom.attrib.update(contype="0", conaffinity="0")
            elif geom.get("contype", "1") != "0" or geom.get("conaffinity", "1") != "0":
                # Upstream assets use condim=1 for Isaac import. Native MuJoCo
                # interprets that as frictionless contact; locomotion requires
                # tangential constraints, as in legged-gym-mujoco's feet.
                geom.set("condim", "3")
                if not collisions:
                    geom.attrib.update(contype="1", conaffinity="0")
            if color is not None:
                geom.set("rgba", " ".join(map(str, [*color[:3], 0.5 if visual else 1])))
        sensors = ET.SubElement(root, "sensor")
        self.sensor_names = {k: [] for k in ("lin", "ang", "contact", "ground")}
        for i, name in enumerate(self.names):
            for kind, tag in (("lin", "framelinvel"), ("ang", "frameangvel")):
                sn = f"mk_{kind}_{i}"
                ET.SubElement(sensors, tag, name=sn, objtype="xbody", objname=name)
                self.sensor_names[kind].append(sn)
            if not visual:
                for kind in ("contact", "ground"):
                    sn = f"mk_{kind}_{i}"
                    attrs = dict(name=sn, body2=name, data="force", reduce="netforce", num="1")
                    if kind == "ground":
                        attrs["geom1"] = "mimickit_floor"
                    ET.SubElement(sensors, "contact", **attrs)
                    self.sensor_names[kind].append(sn)
        self.model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
        self.batch = batch.Batch(self.model, nsim=num_envs, nthread=config.get("num_threads", 4), persistent=True)
        self.batch.reset()
        self.qpos = self.batch.bind("qpos")
        self.qvel = self.batch.bind("qvel")
        self.ctrl = self.batch.bind("ctrl")
        self.xpos = self.batch.bind("xpos")
        self.xquat = self.batch.bind("xquat")
        self.sensors = self.batch.bind("sensordata")
        self.applied = self.batch.bind("xfrc_applied")
        self.forces = self.batch.bind("qfrc_actuator")
        self.warmstart = self.batch.bind("qacc_warmstart")
        self._dirty_ids = np.ones(num_envs, dtype=bool)
        self.snapshots = {}
        self.body_ids = np.array([self.model.body(n).id for n in self.names])
        self.joint_ids = np.array([self.model.joint(n).id for n in joint_names])
        self.qids = self.model.jnt_qposadr[self.joint_ids]
        self.vids = self.model.jnt_dofadr[self.joint_ids]
        if self.model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError("G1 requires a free root joint")
        self.visual = visual
        self.kinematic = visual or disable_motors
        self.cmd = np.zeros((num_envs, len(joints)))
        self.columns = {}
        for kind, names in self.sensor_names.items():
            self.columns[kind] = np.array(
                [np.arange(self.model.sensor(n).adr[0], self.model.sensor(n).adr[0] + 3) for n in names], dtype=int
            )
        if start_pos is not None:
            self.qpos[:, :3] = start_pos
        if start_rot is not None:
            self.qpos[:, 3:7] = np.asarray(start_rot)[[3, 0, 1, 2]]
        self.dirty = True
        self.refresh()

    @property
    def dirty(self):
        return bool(self._dirty_ids.any())

    @dirty.setter
    def dirty(self, value):
        # Also supports explicit full-state invalidation by diagnostic tools.
        self._dirty_ids[:] = value
        if value:
            self.snapshots.clear()

    def invalidate(self, ids):
        self._dirty_ids[ids] = True
        self.snapshots.clear()

    def refresh(self):
        if self.dirty:
            ids = None if self._dirty_ids.all() else np.flatnonzero(self._dirty_ids)
            self.batch.forward(ids=ids)
            self._dirty_ids[:] = False

    def sensor(self, kind):
        self.refresh()
        if self.visual and kind in ("contact", "ground"):
            return np.zeros((len(self.qpos), len(self.names), 3))
        return self.sensors[:, self.columns[kind]]


class MujocoEngine(Engine):
    # Generalized-coordinate setters suffice; MuJoCo owns body FK/velocities.
    owns_derived_body_state = True

    def __init__(self, config, num_envs, device, visualize, record_video=False):
        super().__init__(visualize)
        self.config = dict(config)
        self.num_envs = num_envs
        self.device = device
        self.visualize = visualize
        self.mode = ControlMode[config.get("control_mode", "pos")]
        sim_freq, control_freq = config["sim_freq"], config["control_freq"]
        if num_envs < 1 or sim_freq <= 0 or control_freq <= 0 or sim_freq % control_freq:
            raise ValueError("Positive frequencies with integral sim_freq/control_freq are required")
        self.substeps = int(sim_freq / control_freq)
        self.objects = []
        self.env_objects = []
        self.viewer = None
        self.renderer = None
        self.callbacks = {}
        self.camera_pos = np.array([0.0, -4.0, 2.0])
        self.camera_target = np.array([0.0, 0.0, 0.8])
        self.record_video = record_video
        self.recording = False
        self.video = None
        self.closed = False

    def get_name(self):
        return "mujoco"

    def get_timestep(self):
        return 1 / self.config["control_freq"]

    def get_num_envs(self):
        return self.num_envs

    def get_control_mode(self):
        return self.mode

    def create_env(self):
        self.env_objects.append([])
        return len(self.env_objects) - 1

    def create_obj(
        self,
        env_id,
        obj_type,
        asset_file,
        name,
        is_visual=False,
        enable_self_collisions=True,
        fix_root=False,
        start_pos=None,
        start_rot=None,
        color=None,
        disable_motors=False,
    ):
        if obj_type != ObjType.articulated or fix_root:
            raise NotImplementedError("This G1 backend supports a free articulated robot and visual references")
        obj_id = len(self.env_objects[env_id])
        if env_id == 0:
            if not is_visual and any(not o.visual for o in self.objects):
                raise NotImplementedError("Multiple interacting robots/objects are not supported")
            self.objects.append(
                _Object(
                    self.config,
                    asset_file,
                    self.num_envs,
                    is_visual,
                    start_pos,
                    start_rot,
                    color,
                    enable_self_collisions,
                    disable_motors,
                )
            )
        self.env_objects[env_id].append(obj_id)
        return obj_id

    def initialize_sim(self):
        if len(self.env_objects) != self.num_envs or any(len(v) != len(self.objects) for v in self.env_objects):
            raise ValueError("All vector environments must have identical objects")
        self._view_data = self.objects[0].batch.get_data(0)

    def _obj(self, obj_id):
        if self.closed:
            raise RuntimeError("Engine is closed")
        return self.objects[obj_id]

    def _tensor(self, value):
        return torch.as_tensor(np.array(value, dtype=np.float32), device=self.device)

    def _snapshot(self, obj_id, name, read):
        """Read-only snapshots, reused until state changes.

        Never refill a previously returned tensor: retained rollout values must
        survive later physics steps and partial resets unchanged.
        """
        obj = self._obj(obj_id)
        if name not in obj.snapshots:
            obj.snapshots[name] = self._tensor(read(obj))
        return obj.snapshots[name]

    def _ids(self, env_ids):
        return np.arange(self.num_envs) if env_ids is None else np.atleast_1d(as_numpy(env_ids)).astype(int)

    def set_cmd(self, obj_id, cmd):
        obj = self._obj(obj_id)
        value = as_numpy(cmd)
        if value.shape != obj.cmd.shape or not np.isfinite(value).all():
            raise ValueError(f"Expected finite command of shape {obj.cmd.shape}")
        obj.cmd[:] = value

    def step(self):
        if self.closed:
            raise RuntimeError("Engine is closed")
        for obj in self.objects:
            if obj.kinematic:
                continue
            if self.mode == ControlMode.pd_explicit:
                # Explicit feedback must be recomputed at every physics substep.
                for _ in range(self.substeps):
                    effort = obj.kp * (obj.cmd - obj.qpos[:, obj.qids]) - obj.kd * obj.qvel[:, obj.vids]
                    obj.ctrl[:] = np.clip(effort, -obj.effort, obj.effort)
                    obj.batch.step(stage=mujoco.mjtStage.mjSTAGE_NONE)
            else:
                # Native position/velocity actuators still evaluate feedback
                # each substep; only their targets (or applied torques) are fixed.
                obj.ctrl[:] = 0 if self.mode == ControlMode.none else obj.cmd
                obj.batch.step(nstep=self.substeps, stage=mujoco.mjtStage.mjSTAGE_NONE)
            bad = [
                int(mujoco.mjtWarning.mjWARN_BADQPOS),
                int(mujoco.mjtWarning.mjWARN_BADQVEL),
                int(mujoco.mjtWarning.mjWARN_BADQACC),
            ]
            if np.any(obj.batch.warning[:, bad, 1] > 0) or not np.isfinite(obj.qpos).all():
                raise FloatingPointError("MuJoCo reported invalid dynamics")
            # Expose positions, velocities and contacts from the same post-step state.
            obj.dirty = True
            obj.refresh()
            obj.applied[:] = 0
        if self.recording:
            if self.renderer is None:
                self.renderer = mujoco.Renderer(self.objects[0].model, height=480, width=640)
            self.objects[0].batch.get_data(0, self._view_data)
            self.renderer.update_scene(self._view_data, camera=self._camera())
            self.video.add_frame(self.renderer.render().copy())

    def get_root_pos(self, obj_id):
        return self._snapshot(obj_id, "root_pos", lambda o: o.qpos[:, :3])

    def get_root_rot(self, obj_id):
        return self._snapshot(obj_id, "root_rot", lambda o: o.qpos[:, [4, 5, 6, 3]])

    def get_root_vel(self, obj_id):
        return self._snapshot(obj_id, "root_vel", lambda o: o.qvel[:, :3])

    def get_root_ang_vel(self, obj_id):
        return self._snapshot(obj_id, "root_ang_vel", lambda o: rotate(o.qpos[:, 3:7], o.qvel[:, 3:6]))

    def get_dof_pos(self, obj_id):
        return self._snapshot(obj_id, "dof_pos", lambda o: o.qpos[:, o.qids])

    def get_dof_vel(self, obj_id):
        return self._snapshot(obj_id, "dof_vel", lambda o: o.qvel[:, o.vids])

    def _derived_snapshot(self, obj_id, name, read):
        self._obj(obj_id).refresh()
        return self._snapshot(obj_id, name, read)

    def get_dof_forces(self, obj_id):
        return self._derived_snapshot(obj_id, "dof_forces", lambda o: o.forces[:, o.vids])

    def get_body_pos(self, obj_id):
        return self._derived_snapshot(obj_id, "body_pos", lambda o: o.xpos[:, o.body_ids])

    def get_body_rot(self, obj_id):
        return self._derived_snapshot(obj_id, "body_rot", lambda o: o.xquat[:, o.body_ids][..., [1, 2, 3, 0]])

    def get_body_vel(self, obj_id):
        return self._derived_snapshot(obj_id, "body_vel", lambda o: o.sensor("lin"))

    def get_body_ang_vel(self, obj_id):
        return self._derived_snapshot(obj_id, "body_ang_vel", lambda o: o.sensor("ang"))

    def get_contact_forces(self, obj_id):
        return self._derived_snapshot(obj_id, "contact", lambda o: o.sensor("contact"))

    def get_ground_contact_forces(self, obj_id):
        return self._derived_snapshot(obj_id, "ground", lambda o: o.sensor("ground"))

    def _set(self, env_id, obj_id, field, columns, value):
        o = self._obj(obj_id)
        ids = self._ids(env_id)
        array = getattr(o, field)
        array[np.ix_(ids, np.asarray(columns))] = as_numpy(value)
        if field in ("qpos", "qvel"):
            # Discard solver warm starts after teleports and partial resets.
            o.warmstart[ids] = 0
        o.invalidate(ids)

    def set_root_pos(self, env_id, obj_id, value):
        self._set(env_id, obj_id, "qpos", [0, 1, 2], value)

    def set_root_rot(self, env_id, obj_id, value):
        q = as_numpy(value)[..., [3, 0, 1, 2]]
        norm = np.linalg.norm(q, axis=-1, keepdims=True)
        if np.any(norm < 1e-8):
            raise ValueError("Root quaternion must be nonzero")
        self._set(env_id, obj_id, "qpos", [3, 4, 5, 6], q / norm)

    def set_root_vel(self, env_id, obj_id, value):
        self._set(env_id, obj_id, "qvel", [0, 1, 2], value)

    def set_root_ang_vel(self, env_id, obj_id, value):
        o = self._obj(obj_id)
        q = o.qpos[self._ids(env_id), 3:7]
        self._set(env_id, obj_id, "qvel", [3, 4, 5], inverse_rotate(q, np.broadcast_to(as_numpy(value), (len(q), 3))))

    def set_dof_pos(self, env_id, obj_id, value):
        self._set(env_id, obj_id, "qpos", self._obj(obj_id).qids, value)

    def set_dof_vel(self, env_id, obj_id, value):
        self._set(env_id, obj_id, "qvel", self._obj(obj_id).vids, value)

    # MimicKit writes derived rigid-body buffers after resetting generalized
    # state. MuJoCo owns those buffers: FK refreshes them from qpos/qvel instead.
    def set_body_pos(self, env_id, obj_id, value):
        self._obj(obj_id).invalidate(self._ids(env_id))

    def set_body_rot(self, env_id, obj_id, value):
        self._obj(obj_id).invalidate(self._ids(env_id))

    def set_body_vel(self, env_id, obj_id, value):
        self._obj(obj_id).invalidate(self._ids(env_id))

    def set_body_ang_vel(self, env_id, obj_id, value):
        self._obj(obj_id).invalidate(self._ids(env_id))

    def set_body_forces(self, env_id, obj_id, body_id, forces):
        o = self._obj(obj_id)
        ids = self._ids(env_id)
        o.applied[ids, o.body_ids[body_id], :3] = as_numpy(forces)
        o.invalidate(ids)

    def get_obj_type(self, obj_id):
        return ObjType.articulated

    def get_obj_num_dofs(self, obj_id):
        return len(self._obj(obj_id).qids)

    def get_obj_num_bodies(self, obj_id):
        return len(self._obj(obj_id).names)

    def get_obj_body_names(self, obj_id):
        return list(self._obj(obj_id).names)

    def find_obj_body_id(self, obj_id, body_name):
        names = self._obj(obj_id).names
        return names.index(body_name) if body_name in names else -1

    def get_obj_torque_limits(self, env_id, obj_id):
        return self._obj(obj_id).effort.copy()

    def get_obj_dof_limits(self, env_id, obj_id):
        limits = self._obj(obj_id).limits
        return limits[:, 0].copy(), limits[:, 1].copy()

    def get_obj_pd_gains(self, env_id, obj_id):
        o = self._obj(obj_id)
        return o.kp.copy(), o.kd.copy()

    def calc_obj_mass(self, env_id, obj_id):
        return float(self._obj(obj_id).model.body_mass.sum())

    def set_camera_pose(self, pos, look_at):
        self.camera_pos, self.camera_target = np.array(pos), np.array(look_at)

    def get_camera_pos(self):
        return self.camera_pos.copy()

    def get_camera_dir(self):
        d = self.camera_target - self.camera_pos
        return d / max(np.linalg.norm(d), 1e-8)

    def _camera(self):
        cam = mujoco.MjvCamera()
        delta = self.camera_pos - self.camera_target
        cam.lookat[:] = self.camera_target
        cam.distance = np.linalg.norm(delta)
        cam.azimuth = np.degrees(np.arctan2(delta[1], delta[0]))
        cam.elevation = -np.degrees(np.arctan2(delta[2], np.linalg.norm(delta[:2])))
        return cam

    def register_keyboard_callback(self, key_str, callback_func):
        self.callbacks[{"ENTER": 257, "SPACE": 32}.get(key_str, key_str)] = callback_func

    def render(self):
        import mujoco.viewer

        o = self._obj(0)
        o.refresh()
        o.batch.get_data(0, self._view_data)
        if self.viewer is None:
            self.viewer = mujoco.viewer.launch_passive(
                o.model, self._view_data, key_callback=lambda key: self.callbacks.get(key, lambda: None)()
            )
            cam = self._camera()
            self.viewer.cam.lookat[:] = cam.lookat
            self.viewer.cam.distance = cam.distance
            self.viewer.cam.azimuth = cam.azimuth
            self.viewer.cam.elevation = cam.elevation
        if not self.viewer.is_running():
            raise KeyboardInterrupt
        with self.viewer.lock():
            self.viewer.cam.lookat[:] = self.camera_target
            self.viewer.user_scn.ngeom = 0
            # Reference objects use independent nonintegrating batches. Add
            # their geometry to the viewer without introducing physical bodies.
            for ref in self.objects[1:]:
                ref.refresh()
                if not hasattr(ref, "scene"):
                    ref.scene = mujoco.MjvScene(ref.model, maxgeom=1000)
                    ref.view_data = ref.batch.get_data(0)
                ref.batch.get_data(0, ref.view_data)
                mujoco.mjv_updateScene(
                    ref.model,
                    ref.view_data,
                    mujoco.MjvOption(),
                    None,
                    self.viewer.cam,
                    mujoco.mjtCatBit.mjCAT_ALL,
                    ref.scene,
                )
                for i in range(ref.scene.ngeom):
                    scene = self.viewer.user_scn
                    if scene.ngeom >= scene.maxgeom:
                        break
                    dst, src = scene.geoms[scene.ngeom], ref.scene.geoms[i]
                    # Mesh IDs belong to the same asset model; generated floor
                    # and sensors do not change mesh indexing.
                    for attr in (
                        "type",
                        "dataid",
                        "objtype",
                        "objid",
                        "category",
                        "matid",
                        "texcoord",
                        "segid",
                        "emission",
                        "specular",
                        "shininess",
                        "reflectance",
                    ):
                        setattr(dst, attr, getattr(src, attr))
                    for attr in ("size", "pos", "mat", "rgba"):
                        getattr(dst, attr)[:] = getattr(src, attr)
                    scene.ngeom += 1
        self.viewer.sync()
        super().render()

    def enabled_record_video(self):
        return self.record_video

    def start_video_recording(self):
        from util.video import Video

        self.video = Video(self.config["control_freq"])
        self.recording = True

    def stop_video_recording(self):
        self.recording = False

    def get_video_recording(self):
        return self.video

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        self.objects.clear()
        self.closed = True
