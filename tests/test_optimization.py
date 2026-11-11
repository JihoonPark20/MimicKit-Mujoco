"""Numerical safeguards for batching, snapshot lifetime and partial forwards."""

import mujoco
import numpy as np
import pytest
import torch
from engines.engine import ControlMode, ObjType
from engines.mujoco_engine import MujocoEngine


def make_engine(mode="pos", device="cpu"):
    engine = MujocoEngine(dict(control_mode=mode, sim_freq=240, control_freq=30, num_threads=2), 4, device, False)
    for _ in range(4):
        engine.create_obj(
            engine.create_env(), ObjType.articulated, "data/assets/g1/g1.xml", "g1", start_pos=[0, 0, 0.8]
        )
    engine.initialize_sim()
    return engine


@pytest.mark.parametrize("mode", ["pos", "vel", "torque", "none", "pd_explicit"])
def test_batched_control_matches_substep_feedback(mode):
    """Compare to independent single-substep integration, including external forces."""
    actual, reference = make_engine(mode), make_engine(mode)
    rng = np.random.default_rng(13)
    try:
        for step in range(40):
            cmd = rng.uniform(-0.15, 0.15, (4, 29))
            for engine in (actual, reference):
                engine.set_cmd(0, cmd)
                if step % 7 == 0:
                    engine.set_body_forces([1, 3], 0, 0, [0.7, -0.2, 0.3])
                if step == 20:
                    engine.set_root_pos([1, 3], 0, [0.1, -0.1, 0.9])
                    engine.set_root_vel([1, 3], 0, 0.0)
            actual.step()
            obj = reference.objects[0]
            for _ in range(reference.substeps):
                if reference.mode == ControlMode.pd_explicit:
                    torque = obj.kp * (obj.cmd - obj.qpos[:, obj.qids]) - obj.kd * obj.qvel[:, obj.vids]
                    obj.ctrl[:] = np.clip(torque, -obj.effort, obj.effort)
                else:
                    obj.ctrl[:] = 0 if reference.mode == ControlMode.none else obj.cmd
                obj.batch.step(stage=mujoco.mjtStage.mjSTAGE_NONE)
            obj.dirty = True
            obj.refresh()
            obj.applied[:] = 0
            for key in ("qpos", "qvel", "xpos", "xquat", "sensors", "forces"):
                np.testing.assert_array_equal(getattr(actual.objects[0], key), getattr(obj, key), err_msg=key)
    finally:
        actual.close()
        reference.close()


@pytest.mark.parametrize("device", ["cpu", "mps"])
def test_snapshots_survive_mutations_and_partial_forward(device):
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    engine = make_engine(device=device)
    try:
        obj = engine.objects[0]
        body = engine.get_body_pos(0)
        root = engine.get_root_pos(0)
        saved_body, saved_root = body.clone(), root.clone()
        assert engine.get_body_pos(0) is body  # no second allocation/transfer
        forwarded = []
        original = obj.batch.forward

        def track_forward(ids=None, **kwargs):
            forwarded.append(None if ids is None else np.array(ids))
            return original(ids=ids, **kwargs)

        obj.batch.forward = track_forward
        engine.set_root_pos([1], 0, [0.3, 0.2, 1.1])
        engine.set_root_pos([3], 0, [-0.3, 0.1, 0.9])
        updated_body = engine.get_body_pos(0)
        assert len(forwarded) == 1
        np.testing.assert_array_equal(forwarded[0], [1, 3])
        torch.testing.assert_close(updated_body[[0, 2]], body[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(body, saved_body, rtol=0, atol=0)
        torch.testing.assert_close(root, saved_root, rtol=0, atol=0)
        assert not torch.equal(engine.get_root_pos(0)[[1, 3]], root[[1, 3]])
        # A full forward is an independent oracle for the partial calculation.
        partial = {k: getattr(obj, k).copy() for k in ("xpos", "xquat", "sensors", "forces")}
        original()
        for k, value in partial.items():
            np.testing.assert_array_equal(value, getattr(obj, k), err_msg=k)
        angular = torch.tensor([0.4, -0.1, 0.3], device=device)
        engine.set_root_rot(None, 0, torch.tensor([0.0, 0.0, 0.6, 0.8], device=device))
        engine.set_root_ang_vel(None, 0, angular)
        torch.testing.assert_close(engine.get_root_ang_vel(0), angular.expand(4, 3))
        before_step = engine.get_body_pos(0)
        saved_before_step = before_step.clone()
        engine.set_cmd(0, torch.zeros((4, 29), device=device))
        engine.step()
        assert engine.get_body_pos(0) is not before_step
        torch.testing.assert_close(before_step, saved_before_step, rtol=0, atol=0)
        torch.testing.assert_close(body, saved_body, rtol=0, atol=0)
    finally:
        engine.close()


def test_native_reset_skips_redundant_body_kinematics(env, monkeypatch):
    def unexpected(*args):
        raise AssertionError("Native reset must derive body state in MuJoCo")

    monkeypatch.setattr(env._kin_char_model, "forward_kinematics", unexpected)
    env._reset_char_rigid_body_state(torch.tensor([1]))


def test_auto_device_preserves_explicit_mps(monkeypatch):
    from mimickit_mujoco.cli import select_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: True)
    assert select_device("auto") == "cpu"
    assert select_device("auto", cpu_physics=False) == "mps"
    assert select_device("mps") == "mps"
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert select_device("auto") == "cuda:0"
    assert select_device("auto", cpu_physics=False) == "cuda:0"
