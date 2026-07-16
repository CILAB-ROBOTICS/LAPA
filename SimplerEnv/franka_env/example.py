import argparse
from pathlib import Path

import gymnasium as gym
import mediapy as media
# import simpler_env  # noqa: F401 - imports and registers SimplerEnv environments
from ManiSkill2_real2sim.mani_skill2_real2sim.utils.sapien_utils import look_at

from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
)


DEFAULT_VIDEO_PATH = Path(__file__).resolve().parent / "videos/example_rollout.mp4"
VIDEO_FPS = 5
CAMERA_NAME = "base_camera"
TABLE_HEIGHT = 0.882


def parse_args():
    parser = argparse.ArgumentParser(description="Run a Panda robot in SimplerEnv.")
    parser.add_argument("--width", type=int, default=640, help="Camera image width.")
    parser.add_argument("--height", type=int, default=480, help="Camera image height.")
    parser.add_argument(
        "--camera-pos",
        type=float,
        nargs=3,
        default=[-1.2, 0.8, 1.6],
        metavar=("X", "Y", "Z"),
        help="World-space camera position.",
    )
    parser.add_argument(
        "--camera-look-at",
        type=float,
        nargs=3,
        default=[0.0, 0.0, TABLE_HEIGHT],
        metavar=("X", "Y", "Z"),
        help="World-space point that the camera looks at.",
    )
    parser.add_argument(
        "--fov",
        type=float,
        default=1.0,
        help="Vertical field of view in radians. Smaller values zoom in.",
    )
    parser.add_argument(
        "--video-path",
        type=Path,
        default=DEFAULT_VIDEO_PATH,
        help="Path where the rollout video is saved.",
    )
    return parser.parse_args()


def make_panda_env(args):
    """Create a Panda environment for the coke-can grasping task."""
    camera_pose = look_at(args.camera_pos, args.camera_look_at)

    return gym.make(
        "GraspSingleAppleInScene-v0",
        robot="panda",
        control_mode="pd_ee_delta_pose",
        obs_mode="rgbd",
        scene_name="dummy_tabletop",
        scene_offset=[0.0, -0.21, 0.0],
        scene_table_height=TABLE_HEIGHT,
        camera_cfgs={
            CAMERA_NAME: {
                "p": camera_pose.p,
                "q": camera_pose.q,
                "width": args.width,
                "height": args.height,
                "fov": args.fov,
            }
        },
    )


def main():
    args = parse_args()
    env = make_panda_env(args)
    base_env = env.unwrapped

    obs, reset_info = env.reset(
        options={
            "robot_init_options": {
                "init_xy": [0.0, 0.0],
                "init_height": TABLE_HEIGHT,
            },
            "obj_init_options": {
                "init_xy": [-0.2, 0.0],
                "init_z": TABLE_HEIGHT,
            },
        }
    )

    print("Reset info:", reset_info)
    print("Instruction:", base_env.get_language_instruction())
    print("Robot:", base_env.robot_uid)
    print("Control mode:", base_env.control_mode)
    print("Action space:", env.action_space)
    print(
        f"Camera: {args.width}x{args.height}, position={args.camera_pos}, "
        f"look_at={args.camera_look_at}, fov={args.fov}"
    )

    frames = [
        get_image_from_maniskill2_obs_dict(
            base_env, obs, camera_name=CAMERA_NAME
        )
    ]

    terminated = False
    truncated = False
    info = {}
    while not (terminated or truncated):
        # action[:3]: end-effector delta xyz
        # action[3:6]: end-effector delta rotation in axis-angle form
        # action[6]: Panda gripper target position
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(
            get_image_from_maniskill2_obs_dict(
                base_env, obs, camera_name=CAMERA_NAME
            )
        )

    print("Episode stats:", info.get("episode_stats", {}))

    args.video_path.parent.mkdir(parents=True, exist_ok=True)
    media.write_video(args.video_path, frames, fps=VIDEO_FPS)
    print(f"Saved video: {args.video_path.resolve()}")
    env.close()


if __name__ == "__main__":
    main()


'''
# available environment
GraspSingleOpenedCokeCanInScene-v0
GraspSingleCokeCanInScene-v0
GraspSinglePepsiCanInScene-v0
GraspSingleOpenedPepsiCanInScene-v0
GraspSingle7upCanInScene-v0
GraspSingleOpened7upCanInScene-v0
GraspSingleSpriteCanInScene-v0
GraspSingleOpenedSpriteCanInScene-v0
GraspSingleFantaCanInScene-v0
GraspSingleOpenedFantaCanInScene-v0
GraspSingleRedBullCanInScene-v0
GraspSingleOpenedRedBullCanInScene-v0
GraspSingleBluePlasticBottleInScene-v0
GraspSingleAppleInScene-v0
GraspSingleOrangeInScene-v0
GraspSingleSpongeInScene-v0
GraspSingleBridgeSpoonInScene-v0
GraspSingleRandomObjectInScene-v0
'''