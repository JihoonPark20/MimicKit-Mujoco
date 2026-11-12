"""Official G1 motion presets shared by policy, prior and reference playback."""

from copy import deepcopy
from pathlib import Path

import yaml

from . import ROOT

DEMOS = ("spinkick", "kick_combo", "cartwheel")


def read_demo(name):
    if name not in DEMOS:
        raise ValueError(f"Unknown G1 demo: {name}")
    with (ROOT / "data/demos" / f"g1_{name}.yaml").open() as stream:
        return yaml.safe_load(stream)


def apply_demo(env_config, name):
    preset = read_demo(name)
    config = deepcopy(env_config)
    config["motion_file"] = preset["motion_file"]
    config["demo_name"] = name
    if config["env_name"] != "view_motion":
        config["contact_bodies"] = list(dict.fromkeys([*config["contact_bodies"], *preset["extra_contact_bodies"]]))
    return config


def validate_demo_prior(env_config, prior_config):
    """Catch accidental reuse of a walk or different-motion prior in a demo."""
    name = env_config.get("demo_name")
    if name is None:
        return

    def resolve(path):
        path = Path(path)
        return (path if path.is_absolute() else ROOT / path).resolve()

    prior_motion = prior_config.get("motion_file")
    if not prior_motion or resolve(prior_motion) != resolve(env_config["motion_file"]):
        raise ValueError(
            f"The {name} demo requires a prior trained on {env_config['motion_file']}. "
            f"Run train-prior --demo {name}, then pass its directory with --prior-dir."
        )
