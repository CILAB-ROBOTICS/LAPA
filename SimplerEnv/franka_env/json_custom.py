"""Export Panda expert rollouts to the raw JSON format for fine-tuning.

This creates a JSON array that can be passed to:

    python ../data/finetune_preprocess.py \
        --input_path ../data/panda_expert_raw.json \
        --output_filename ../data/panda_finetune.jsonl \
        --csv_filename ../data/panda_action_bins.csv
"""

import argparse
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import mediapy as media
import numpy as np
import tensorflow as tf


SCRIPT_DIR = Path(__file__).resolve().parent
SIMPLER_ENV_DIR = SCRIPT_DIR.parent
REPO_DIR = SIMPLER_ENV_DIR.parent
DEFAULT_MODEL_PATH = SIMPLER_ENV_DIR / "panda_expert/checkpoints/panda_ppo.weights.h5"
DEFAULT_OUTPUT_JSON = REPO_DIR / "panda_expert/data/panda_expert_raw.json"
DEFAULT_IMAGE_DIR = REPO_DIR / "panda_expert/data/panda_expert_images"
DEFAULT_EXPERT_SCRIPT = SCRIPT_DIR / "panda_expert.py"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Load a Panda expert checkpoint and export rollouts to JSON."
    )
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--image-dir", type=Path, default=DEFAULT_IMAGE_DIR)
    parser.add_argument("--expert-script", type=Path, default=DEFAULT_EXPERT_SCRIPT)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--instruction", type=str, default="pick coke can")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[1.2, 0.8, 1.6])
    parser.add_argument("--camera-look-at", type=float, nargs=3, default=[0.0, 0.0, 0.882])
    parser.add_argument("--fov", type=float, default=1.0)
    parser.add_argument(
        "--keep-failures",
        action="store_true",
        help="Keep failed episodes too. By default, only successful episodes are exported.",
    )
    parser.add_argument(        # 사용 X, gripper action should be binary (0 or 1)
        "--continuous-gripper",
        action="store_true",
        help="Save the raw continuous gripper action instead of thresholding it to 0/1.",
    )
    return parser.parse_args()


def load_expert_module(expert_script):
    spec = importlib.util.spec_from_file_location("panda_expert_policy", expert_script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_expert_args(args):
    return SimpleNamespace(
        model_path=args.model_path,
        seed=args.seed,
        width=args.width,
        height=args.height,
        camera_pos=args.camera_pos,
        camera_look_at=args.camera_look_at,
        fov=args.fov,
    )


def to_uint8_image(image):
    image = np.asarray(image)
    if image.dtype == np.uint8:
        return image
    if image.max() <= 1.0:
        image = image * 255.0
    return np.clip(image, 0, 255).astype(np.uint8)


def action_to_raw_list(action, continuous_gripper):
    raw_action = np.asarray(action, dtype=np.float32).tolist()
    if not continuous_gripper:
        raw_action[6] = int(raw_action[6] > 0.0)
    return [float(x) for x in raw_action[:6]] + [raw_action[6]]


# def make_sample(sample_id, image_path, instruction, raw_action):
#     return {
#         "id": sample_id,
#         "image": str(image_path),
#         "conversations": [
#             {"value": f"<image>\n{instruction}"},
#             {"raw_actions": raw_action},
#         ],
#     }

# LAPA fintuning data format 참고
def make_sample(sample_id, image_path, instruction, raw_action):
    return {
        "id": sample_id,
        "image": str(image_path),
        "conversations": [
            {
                "from": "human",       # 수정
                "value": f"<image>\nWhat action should the robot take to `{instruction}`" # 수정
            },
            {
                "from": "gpt",         # 수정
                "raw_actions": raw_action
            }
        ],
    }


def main():
    args = parse_args()
    expert = load_expert_module(args.expert_script)

    env_args = make_expert_args(args)
    env = expert.make_env(env_args)
    model = expert.ActorCritic(env.action_space.shape[0])
    model(tf.zeros((1, env.observation_space.shape[0]), dtype=tf.float32))
    model.load_weights(args.model_path)

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.image_dir.mkdir(parents=True, exist_ok=True)

    samples = []
    successful_episodes = 0
    attempted_episodes = 0

    for episode in range(args.episodes):
        attempted_episodes += 1
        observation, _ = env.reset(seed=args.seed + episode)
        episode_samples = []
        done = False
        info = {"success": False}

        for step in range(args.max_steps):
            action, _, _ = expert.choose_action(model, observation, deterministic=True)

            sample_id = f"panda_ep{episode:04d}_step{step:04d}"
            image_path = args.image_dir / f"{sample_id}.png"
            media.write_image(image_path, to_uint8_image(env.last_frame))
            episode_samples.append(
                make_sample(
                    sample_id=sample_id,
                    image_path=image_path,
                    instruction=args.instruction,
                    raw_action=action_to_raw_list(action, args.continuous_gripper),
                )
            )

            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if done:
                break

        success = bool(info.get("success", False))
        if success:
            successful_episodes += 1
        if success or args.keep_failures:
            samples.extend(episode_samples)

        print(
            f"episode={episode + 1}/{args.episodes} "
            f"steps={len(episode_samples)} success={success} "
            f"exported_samples={len(samples)}"
        )

    env.close()
    with open(args.output_json, "w") as outfile:
        json.dump(samples, outfile, indent=2)

    print(f"Saved {len(samples)} samples to {args.output_json.resolve()}")
    print(f"Images saved under {args.image_dir.resolve()}")
    print(
        "Successful episodes: "
        f"{successful_episodes}/{attempted_episodes} "
        f"({successful_episodes / max(attempted_episodes, 1):.1%})"
    )


if __name__ == "__main__":
    main()