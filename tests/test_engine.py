import mujoco
import numpy as np
import pytest
import torch


def test_g1_forward_kinematics_matches_mimickit(env):
    """Catch silent DFS ordering and quaternion/frame convention errors."""
    eng, kin = env._engine, env._kin_char_model
    rng = np.random.default_rng(41)
    dof = torch.tensor(rng.uniform(-0.3, 0.3, (4, 29)), dtype=torch.float32)
    quat = torch.tensor(rng.normal(size=(4, 4)), dtype=torch.float32)
    quat /= torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    pos = torch.tensor(rng.normal(size=(4, 3)), dtype=torch.float32)
    eng.set_root_pos(None, 0, pos)
    eng.set_root_rot(None, 0, quat)
    eng.set_dof_pos(None, 0, dof)
    expected_pos, expected_rot = kin.forward_kinematics(pos, quat, kin.dof_to_rot(dof))
    torch.testing.assert_close(eng.get_body_pos(0), expected_pos, atol=2e-6, rtol=2e-5)
    dots = torch.sum(eng.get_body_rot(0) * expected_rot, dim=-1).abs()
    torch.testing.assert_close(dots, torch.ones_like(dots), atol=2e-6, rtol=0)
    assert eng.get_obj_num_dofs(0) == 29


def test_world_velocity_and_partial_reset(env):
    eng = env._engine
    quat = torch.tensor([0.0, 0.0, np.sin(0.6), np.cos(0.6)], dtype=torch.float32)
    eng.set_root_rot(None, 0, quat)
    velocity = torch.tensor([0.7, -0.2, 0.3])
    angular = torch.tensor([0.4, 0.3, -0.2])
    eng.set_root_vel(None, 0, velocity)
    eng.set_root_ang_vel(None, 0, angular)
    torch.testing.assert_close(eng.get_root_ang_vel(0), angular.expand(4, 3))
    torch.testing.assert_close(eng.get_body_ang_vel(0)[:, 0], angular.expand(4, 3))
    torch.testing.assert_close(eng.get_body_vel(0)[:, 0], velocity.expand(4, 3))
    before = eng.objects[0].qpos.copy(), eng.objects[0].qvel.copy()
    env.reset(torch.tensor([1, 3]))
    np.testing.assert_array_equal(eng.objects[0].qpos[[0, 2]], before[0][[0, 2]])
    np.testing.assert_array_equal(eng.objects[0].qvel[[0, 2]], before[1][[0, 2]])


def test_contacts_match_independent_contact_wrench_sum(env):
    eng = env._engine
    obj = eng.objects[0]
    eng.set_root_pos(None, 0, [0.0, 0.0, 0.65])
    eng.set_dof_vel(None, 0, 0.0)
    eng.set_root_vel(None, 0, 0.0)
    eng.set_root_ang_vel(None, 0, 0.0)
    obj.refresh()
    data = obj.batch.get_data(0)
    floor = obj.model.geom("mimickit_floor").id
    expected = np.zeros((obj.model.nbody, 3))
    ground = np.zeros_like(expected)
    wrench = np.empty(6)
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == floor or c.geom2 == floor:
            assert c.dim == 3, "Foot-ground contacts must have tangential friction"
        mujoco.mj_contactForce(obj.model, data, i, wrench)
        force = c.frame.reshape(3, 3).T @ wrench[:3]
        b1, b2 = obj.model.geom_bodyid[c.geom1], obj.model.geom_bodyid[c.geom2]
        expected[b1] -= force
        expected[b2] += force
        if c.geom1 == floor:
            ground[b2] += force
        if c.geom2 == floor:
            ground[b1] -= force
    assert np.linalg.norm(ground) > 0
    np.testing.assert_allclose(eng.get_contact_forces(0)[0], expected[obj.body_ids], atol=1e-2, rtol=2e-5)
    np.testing.assert_allclose(eng.get_ground_contact_forces(0)[0], ground[obj.body_ids], atol=1e-2, rtol=2e-5)


def test_pd_force_limits_and_invalid_actions(env):
    eng = env._engine
    obj = eng.objects[0]
    eng.set_cmd(0, torch.full((4, 29), 20.0))
    eng.step()
    forces = eng.get_dof_forces(0).numpy()
    assert np.all(np.abs(forces) <= obj.effort + 1e-4)
    with pytest.raises(ValueError, match="finite command"):
        eng.set_cmd(0, torch.full((4, 29), float("nan")))
    with pytest.raises(ValueError):
        eng.set_cmd(0, torch.zeros((1, 29)))


def test_step_reset_and_close(env):
    obs, _ = env.reset()
    for _ in range(20):
        obs, reward, done, _ = env.step(torch.zeros((4, 29)))
        assert torch.isfinite(obs).all() and torch.isfinite(reward).all()
        env.reset((done != 0).nonzero(as_tuple=False).flatten())
    env._engine.close()
    env._engine.close()
    with pytest.raises(RuntimeError, match="closed"):
        env._engine.step()


def test_motion_playback_does_not_integrate():
    from envs.env_builder import build_env

    env = build_env("data/envs/view_motion_g1_env.yaml", "data/engines/mujoco_engine.yaml", 1, "cpu", False)
    try:
        env.reset()
        initial = env._engine.objects[0].qpos.copy()
        env._engine.step()
        np.testing.assert_array_equal(initial, env._engine.objects[0].qpos)
        for _ in range(35):
            obs, _, _, _ = env.step(torch.zeros((1, 29)))
            assert torch.isfinite(obs).all()
        assert not np.array_equal(initial, env._engine.objects[0].qpos)
    finally:
        env._engine.close()
