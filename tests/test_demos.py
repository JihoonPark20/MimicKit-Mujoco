"""Demo contact semantics and checkpoint/prior configuration boundaries."""

import xml.etree.ElementTree as ET

import pytest
import torch

from mimickit_mujoco import ROOT
from mimickit_mujoco.cli import main, parser, read_yaml
from mimickit_mujoco.demos import DEMOS, apply_demo, validate_demo_prior


@pytest.mark.parametrize("demo", DEMOS)
def test_contact_rules_allow_cartwheel_hands_but_reject_head(demo):
    from envs.deepmimic_env import compute_done

    base = read_yaml(ROOT / "data/envs/deepmimic_g1_env.yaml")
    cfg = apply_demo(base, demo)
    assert base["motion_file"].endswith("g1_walk.pkl")
    assert cfg["enable_early_termination"] and cfg["pose_termination"]
    bodies = [b.get("name") for b in ET.parse(ROOT / cfg["char_file"]).getroot().find("worldbody").iter("body")]
    allowed = torch.tensor([bodies.index(name) for name in cfg["contact_bodies"]])
    forces = torch.zeros((3, len(bodies), 3))
    for row, name in enumerate(["left_wrist_yaw_link", "head_link", "left_ankle_roll_link"]):
        forces[row, bodies.index(name), 2] = 10
    rotations = torch.tensor([[0.0, 0.0, 0.0, 1.0]]).expand(3, -1)
    positions = torch.zeros((3, len(bodies), 3))
    done = compute_done(
        torch.zeros(3, dtype=torch.int),
        torch.ones(3),
        10.0,
        rotations,
        positions,
        rotations,
        positions,
        forces,
        allowed,
        True,
        1.0,
        True,
        True,
        torch.zeros(3),
        torch.full((3,), 5.0),
        torch.zeros(3, dtype=torch.bool),
        True,
    )
    assert done.tolist() == ([0, 1, 0] if demo == "cartwheel" else [1, 1, 0])


def test_reference_demo_does_not_require_contact_config():
    cfg = apply_demo(read_yaml(ROOT / "data/envs/view_motion_g1_env.yaml"), "cartwheel")
    assert "contact_bodies" not in cfg
    assert cfg["motion_file"].endswith("g1_cartwheel.pkl")


def test_demo_and_custom_motion_are_mutually_exclusive():
    with pytest.raises(SystemExit):
        parser().parse_args(["train", "--demo", "spinkick", "--motion", "custom.pkl"])


@pytest.mark.parametrize("command", ["train", "play"])
def test_checkpoint_cannot_silently_replace_demo(command):
    with pytest.raises(SystemExit, match="checkpoints restore saved presets"):
        main([command, "--demo", "cartwheel", "--checkpoint", "model.pt"])


def test_smp_demo_requires_matching_prior():
    cfg = apply_demo(read_yaml(ROOT / "data/envs/smp_g1_env.yaml"), "cartwheel")
    with pytest.raises(ValueError, match="train-prior --demo cartwheel"):
        validate_demo_prior(cfg, {"motion_file": "data/motions/g1/g1_walk.pkl"})
    validate_demo_prior(cfg, {"motion_file": str(ROOT / cfg["motion_file"])})
