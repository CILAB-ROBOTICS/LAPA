"""Train and test a PPO policy for Panda coke-can grasping.

Examples:
    python "example_panda copy.py" train --total-steps 200000
    python "example_panda copy.py" test --episodes 10
"""

import argparse
from pathlib import Path

import gymnasium as gym
import mediapy as media
import numpy as np
import simpler_env  # noqa: F401 - registers SimplerEnv environments
import tensorflow as tf
from mani_skill2_real2sim.utils.sapien_utils import look_at, vectorize_pose

from simpler_env.utils.env.observation_utils import (
    get_image_from_maniskill2_obs_dict,
)


CAMERA_NAME = "base_camera"
TABLE_HEIGHT = 0.882
MODEL_PATH = Path("checkpoints/panda_ppo.weights.h5")
VIDEO_DIR = Path("videos/panda_ppo")
LOG_TWO_PI = tf.constant(np.log(2.0 * np.pi), dtype=tf.float32)
GAUSSIAN_ENTROPY_CONSTANT = tf.constant(
    0.5 * np.log(2.0 * np.pi * np.e), dtype=tf.float32
)


def parse_args():
    parser = argparse.ArgumentParser(description="Train or test PPO with Panda.")
    parser.add_argument("mode", choices=["train", "test"])
    parser.add_argument("--model-path", type=Path, default=MODEL_PATH)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-steps", type=int, default=200_000)
    parser.add_argument("--rollout-steps", type=int, default=2048)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--clip-ratio", type=float, default=0.2)
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--camera-pos", type=float, nargs=3, default=[1.2, 0.8, 1.6])
    parser.add_argument(
        "--camera-look-at",
        type=float,
        nargs=3,
        default=[0.0, 0.0, TABLE_HEIGHT],
    )
    parser.add_argument("--fov", type=float, default=1.0)
    parser.add_argument(
        "--wandb", action="store_true", help="Log metrics to Weights & Biases."
    )
    parser.add_argument("--wandb-project", type=str, default="panda-ppo")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-name", type=str, default=None)
    parser.add_argument("--wandb-group", type=str, default=None)
    parser.add_argument("--wandb-dir", type=Path, default=None)
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline", "disabled"],
        default=None,
        help="Optional wandb mode override.",
    )
    parser.add_argument(
        "--wandb-log-videos",
        action="store_true",
        help="Upload evaluation videos to wandb during test mode.",
    )
    return parser.parse_args()


def wandb_config_from_args(args):
    config = vars(args).copy()
    for key, value in config.items():
        if isinstance(value, Path):
            config[key] = str(value)
    return config


def init_wandb(args):
    if not args.wandb:
        return None

    try:
        import wandb
    except ImportError as exc:
        raise ImportError(
            "wandb is not installed. Install it or run without --wandb."
        ) from exc

    return wandb.init(
        project=args.wandb_project,
        entity=args.wandb_entity,
        name=args.wandb_name,
        group=args.wandb_group,
        dir=str(args.wandb_dir) if args.wandb_dir is not None else None,
        mode=args.wandb_mode,
        config=wandb_config_from_args(args),
    )


def make_raw_env(args):
    camera_pose = look_at(args.camera_pos, args.camera_look_at)
    return gym.make(
        "GraspSingleOpenedCokeCanInScene-v0",
        robot="panda",
        control_mode="pd_ee_delta_pose",
        obs_mode="rgbd",
        scene_name="dummy_tabletop",
        scene_offset=[0.0, -0.21, 0.0],
        scene_table_height=TABLE_HEIGHT,
        success_from_episode_stats=False,
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


class PandaRLWrapper(gym.Wrapper):
    """Expose compact robot state and a shaped reward suitable for PPO."""

    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(36,), dtype=np.float32
        )
        self.last_frame = None
        self.previous_distance = 0.0

    def _state(self):
        qpos = self.base_env.agent.robot.get_qpos()
        qvel = self.base_env.agent.robot.get_qvel() * 0.1
        tcp_pose = vectorize_pose(self.base_env.tcp.pose)
        obj_pose = vectorize_pose(self.base_env.obj_pose)
        tcp_to_obj = self.base_env.obj_pose.p - self.base_env.tcp.pose.p
        obj_height = np.array(
            [self.base_env.obj.pose.p[2] - self.base_env.obj_height_after_settle]
        )
        return np.concatenate(
            [qpos, qvel, tcp_pose, obj_pose, tcp_to_obj, obj_height]
        ).astype(np.float32)

    def _save_frame(self, obs):
        self.last_frame = get_image_from_maniskill2_obs_dict(
            self.base_env, obs, camera_name=CAMERA_NAME
        )

    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        options.setdefault(
            "robot_init_options",
            {"init_xy": [0.0, 0.0], "init_height": TABLE_HEIGHT},
        )
        options.setdefault(
            "obj_init_options",
            {
                "init_xy": np.random.uniform([-0.32, -0.18], [-0.12, 0.18]),
                "init_z": TABLE_HEIGHT,
            },
        )
        obs, info = self.env.reset(options=options, **kwargs)
        self._save_frame(obs)
        self.previous_distance = np.linalg.norm(
            self.base_env.obj_pose.p - self.base_env.tcp.pose.p
        )
        return self._state(), info

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        self._save_frame(obs)

        distance = np.linalg.norm(self.base_env.obj_pose.p - self.base_env.tcp.pose.p)
        progress = self.previous_distance - distance
        self.previous_distance = distance

        reward = 10.0 * progress + 0.1 * np.exp(-10.0 * distance)
        reward -= 0.005 * float(np.square(action).sum())
        reward += 1.0 * float(info["is_grasped"])
        reward += 5.0 * float(info["lifted_object_significantly"])
        reward += 10.0 * float(info["success"])
        info["shaped_reward"] = reward
        return self._state(), float(reward), terminated, truncated, info


def make_env(args):
    return PandaRLWrapper(make_raw_env(args))


class ActorCritic(tf.keras.Model):
    def __init__(self, action_dim):
        super().__init__()
        self.shared = tf.keras.Sequential(
            [
                tf.keras.layers.Dense(256, activation="tanh"),
                tf.keras.layers.Dense(256, activation="tanh"),
            ]
        )
        self.mean = tf.keras.layers.Dense(action_dim, activation="tanh")
        self.value = tf.keras.layers.Dense(1)
        self.log_std = self.add_weight(
            shape=(action_dim,),
            initializer=tf.keras.initializers.Constant(-0.5),
            trainable=True,
            name="log_std",
            dtype=tf.float32,
        )

    def call(self, observations):
        features = self.shared(observations)
        return self.mean(features), tf.squeeze(self.value(features), axis=-1)


def gaussian_log_prob(actions, means, log_std):
    variance = tf.exp(2.0 * log_std)
    log_prob = -0.5 * (
        tf.square(actions - means) / variance + 2.0 * log_std + LOG_TWO_PI
    )
    return tf.reduce_sum(log_prob, axis=-1)


def choose_action(model, observation, deterministic=False):
    obs_tensor = tf.convert_to_tensor(observation[None], dtype=tf.float32)
    mean, value = model(obs_tensor)
    if deterministic:
        action = mean[0]
    else:
        action = mean[0] + tf.exp(model.log_std) * tf.random.normal(mean[0].shape)
    clipped_action = tf.clip_by_value(action, -1.0, 1.0)
    log_prob = gaussian_log_prob(clipped_action[None], mean, model.log_std)[0]
    return clipped_action.numpy(), float(log_prob), float(value[0])


def compute_gae(rewards, values, dones, last_value, gamma, gae_lambda):
    advantages = np.zeros_like(rewards, dtype=np.float32)
    gae = 0.0
    for step in reversed(range(len(rewards))):
        next_value = last_value if step == len(rewards) - 1 else values[step + 1]
        nonterminal = 1.0 - dones[step]
        delta = rewards[step] + gamma * next_value * nonterminal - values[step]
        gae = delta + gamma * gae_lambda * nonterminal * gae
        advantages[step] = gae
    return advantages, advantages + values


def update_policy(model, optimizer, data, args):
    observations, actions, old_log_probs, advantages, returns = data
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
    size = len(observations)

    policy_losses = []
    value_losses = []
    for _ in range(args.epochs):
        indices = np.random.permutation(size)
        for start in range(0, size, args.batch_size):
            batch = indices[start : start + args.batch_size]
            obs_b = tf.convert_to_tensor(observations[batch], dtype=tf.float32)
            act_b = tf.convert_to_tensor(actions[batch], dtype=tf.float32)
            old_log_b = tf.convert_to_tensor(old_log_probs[batch], dtype=tf.float32)
            adv_b = tf.convert_to_tensor(advantages[batch], dtype=tf.float32)
            ret_b = tf.convert_to_tensor(returns[batch], dtype=tf.float32)

            with tf.GradientTape() as tape:
                means, values = model(obs_b)
                log_probs = gaussian_log_prob(act_b, means, model.log_std)
                ratio = tf.exp(log_probs - old_log_b)
                clipped_ratio = tf.clip_by_value(
                    ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                )
                policy_loss = -tf.reduce_mean(
                    tf.minimum(ratio * adv_b, clipped_ratio * adv_b)
                )
                value_loss = tf.reduce_mean(tf.square(ret_b - values))
                entropy = tf.reduce_sum(model.log_std + GAUSSIAN_ENTROPY_CONSTANT)
                loss = (
                    policy_loss
                    + tf.constant(0.5, dtype=tf.float32) * value_loss
                    - tf.constant(0.001, dtype=tf.float32) * entropy
                )

            gradients = tape.gradient(loss, model.trainable_variables)
            gradients, _ = tf.clip_by_global_norm(gradients, 0.5)
            optimizer.apply_gradients(zip(gradients, model.trainable_variables))
            policy_losses.append(float(policy_loss))
            value_losses.append(float(value_loss))

    return np.mean(policy_losses), np.mean(value_losses)


def train(args):
    wandb_run = init_wandb(args)
    np.random.seed(args.seed)
    tf.random.set_seed(args.seed)
    env = make_env(args)
    model = ActorCritic(env.action_space.shape[0])
    model(tf.zeros((1, env.observation_space.shape[0])))
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    observation, _ = env.reset(seed=args.seed)
    episode_return = 0.0
    episode_count = 0
    completed_steps = 0

    while completed_steps < args.total_steps:
        observations = []
        actions = []
        rewards = []
        dones = []
        values = []
        log_probs = []

        for _ in range(min(args.rollout_steps, args.total_steps - completed_steps)):
            action, log_prob, value = choose_action(model, observation)
            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            observations.append(observation)
            actions.append(action)
            rewards.append(reward)
            dones.append(float(done))
            values.append(value)
            log_probs.append(log_prob)
            observation = next_observation
            episode_return += reward
            completed_steps += 1

            if done:
                episode_count += 1
                print(
                    f"episode={episode_count} steps={completed_steps} "
                    f"return={episode_return:.2f} success={info['success']}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/episode_return": episode_return,
                            "train/episode_success": float(info["success"]),
                            "train/episode": episode_count,
                        },
                        step=completed_steps,
                    )
                observation, _ = env.reset()
                episode_return = 0.0

        _, _, last_value = choose_action(model, observation, deterministic=True)
        advantages, returns = compute_gae(
            np.asarray(rewards, dtype=np.float32),
            np.asarray(values, dtype=np.float32),
            np.asarray(dones, dtype=np.float32),
            last_value,
            args.gamma,
            args.gae_lambda,
        )
        policy_loss, value_loss = update_policy(
            model,
            optimizer,
            (
                np.asarray(observations, dtype=np.float32),
                np.asarray(actions, dtype=np.float32),
                np.asarray(log_probs, dtype=np.float32),
                advantages,
                returns,
            ),
            args,
        )
        args.model_path.parent.mkdir(parents=True, exist_ok=True)
        model.save_weights(args.model_path)
        print(
            f"update steps={completed_steps}/{args.total_steps} "
            f"policy_loss={policy_loss:.4f} value_loss={value_loss:.4f}"
        )
        if wandb_run is not None:
            wandb_run.log(
                {
                    "train/policy_loss": policy_loss,
                    "train/value_loss": value_loss,
                    "train/reward_mean": float(np.mean(rewards)),
                    "train/reward_std": float(np.std(rewards)),
                    "train/return_mean": float(np.mean(returns)),
                    "train/advantage_mean": float(np.mean(advantages)),
                    "train/completed_steps": completed_steps,
                    "train/learning_rate": args.learning_rate,
                    "train/log_std_mean": float(tf.reduce_mean(model.log_std)),
                },
                step=completed_steps,
            )

    env.close()
    print(f"Saved PPO weights: {args.model_path.resolve()}")
    if wandb_run is not None:
        wandb_run.finish()


def test(args):
    wandb_run = init_wandb(args)
    env = make_env(args)
    model = ActorCritic(env.action_space.shape[0])
    model(tf.zeros((1, env.observation_space.shape[0])))
    model.load_weights(args.model_path)

    successes = 0
    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    for episode in range(args.episodes):
        observation, _ = env.reset(seed=args.seed + episode)
        frames = [env.last_frame]
        total_reward = 0.0
        done = False
        info = {}

        while not done:
            action, _, _ = choose_action(model, observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(action)
            frames.append(env.last_frame)
            total_reward += reward
            done = terminated or truncated

        success = bool(info["success"])
        successes += int(success)
        video_path = VIDEO_DIR / f"episode_{episode:03d}_success_{int(success)}.mp4"
        media.write_video(video_path, frames, fps=5)
        print(
            f"test episode={episode + 1} return={total_reward:.2f} "
            f"success={success} video={video_path}"
        )
        if wandb_run is not None:
            log_data = {
                "test/episode_return": total_reward,
                "test/episode_success": float(success),
                "test/episode": episode + 1,
            }
            if args.wandb_log_videos:
                import wandb

                log_data["test/video"] = wandb.Video(str(video_path), fps=5)
            wandb_run.log(log_data, step=episode + 1)

    env.close()
    success_rate = successes / args.episodes
    print(f"Success rate: {successes}/{args.episodes} = {success_rate:.1%}")
    if wandb_run is not None:
        wandb_run.log(
            {"test/success_rate": success_rate, "test/successes": successes},
            step=args.episodes,
        )
        wandb_run.finish()


def main():
    args = parse_args()
    if args.mode == "train":
        train(args)
    else:
        test(args)


if __name__ == "__main__":
    main()
