import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimickit_mujoco import bootstrap  # noqa: E402

bootstrap()
os.chdir(ROOT)


@pytest.fixture
def env():
    from envs.env_builder import build_env

    instance = build_env("data/envs/deepmimic_g1_env.yaml", "data/engines/mujoco_engine.yaml", 4, "cpu", False)
    instance.reset()
    yield instance
    instance._engine.close()
