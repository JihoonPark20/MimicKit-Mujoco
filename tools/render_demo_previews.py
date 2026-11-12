"""Render official reference trajectories as GIFs, without a learned policy."""

import argparse
import math
import pickle
import sys
from pathlib import Path

import mujoco
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from mimickit_mujoco import bootstrap  # noqa: E402
from mimickit_mujoco.cli import read_yaml  # noqa: E402
from mimickit_mujoco.demos import DEMOS, apply_demo, read_demo  # noqa: E402

bootstrap()
from envs.view_motion_env import ViewMotionEnv  # noqa: E402


def render(name, output):
    preset = read_demo(name)
    with (ROOT / preset["motion_file"]).open("rb") as stream:
        motion = pickle.load(stream)
    duration = (len(motion["frames"]) - 1) / motion["fps"]
    cfg = apply_demo(read_yaml(ROOT / "data/envs/view_motion_g1_env.yaml"), name)
    env = ViewMotionEnv(cfg, read_yaml(ROOT / "data/engines/mujoco_engine.yaml"), 1, "cpu", False)
    renderer = None
    try:
        env.reset()
        # Reset initializes the character; synchronize the actual first motion
        # frame before capturing so the GIF does not flash the default pose.
        env._sync_motion()
        obj = env._engine.objects[0]
        renderer = mujoco.Renderer(obj.model, height=360, width=480)
        data = obj.batch.get_data(0)
        camera = mujoco.MjvCamera()
        camera.distance = 3.1
        camera.azimuth = 135
        camera.elevation = -15
        frames = []
        for step in range(math.ceil(duration * 30) + 1):
            if step:
                env.step(torch.zeros((1, 29)))
            if step % 2:
                continue
            obj.refresh()
            obj.batch.get_data(0, data)
            camera.lookat[:] = [obj.qpos[0, 0], obj.qpos[0, 1], 0.85]
            renderer.update_scene(data, camera=camera)
            frames.append(Image.fromarray(renderer.render().copy()))
        # 15 fps reference animation, with a short pause at the end of the clip.
        frames[0].save(
            output,
            save_all=True,
            append_images=frames[1:],
            loop=0,
            duration=[67] * (len(frames) - 1) + [600],
            optimize=False,
        )
        print(f"{name}: {len(frames)} reference frames -> {output}", flush=True)
    finally:
        if renderer is not None:
            renderer.close()
        env._engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "images/g1_demos")
    args = parser.parse_args()
    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    import os

    os.chdir(ROOT)
    torch.set_num_threads(2)
    for name in DEMOS:
        render(name, out / f"{name}.gif")


if __name__ == "__main__":
    main()
