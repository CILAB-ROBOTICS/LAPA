"""Train and test a PPO policy for Panda coke-can grasping.

Examples:
    python "example_panda copy.py" train --total-steps 200000
    python "example_panda copy.py" test --episodes 10
"""

import time
import argparse
from pathlib import Path
from collections import deque
from tqdm import tqdm

import gymnasium as gym
import mediapy as media
import numpy as np
# import simpler_env  # noqa: F401 - registers SimplerEnv environments
import tensorflow as tf
from ManiSkill2_real2sim.mani_skill2_real2sim.utils.sapien_utils import look_at, vectorize_pose

from scipy.spatial.transform import Rotation as R

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

# Forced Final Phase 설정 (e,g., 0.8이면 전체 스텝의 80% 이후부터 난이도 1.0 고정, None이면 비활성)
FORCED_FINAL_PHASE = 0.8 


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
        max_episode_steps=250,  # env에서 최대 step 수
    )


class PandaRLWrapper(gym.Wrapper):
    """Expose compact robot state and a shaped reward suitable for PPO."""

    def __init__(self, env):
        super().__init__(env)
        self.base_env = env.unwrapped
        
        # Phase 정보를 State로 전달하기 위해 shape 36 -> 37로 확장
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(37,), dtype=np.float32
        )
        self.last_frame = None
        
        # Parameters for Rewards & Phase
        self.margin = 0.15
        # Phase 전환 임계값
        self.hover_target_threshold = 0.08
        self.grasp_target_threshold = 0.04
        self.lift_target_margin = 0.12
        
        self.phase = 0
        
        self.min_dist = np.inf      # 신기록 갱신용
        self.max_lift_z = -np.inf
        self.init_can_z = 0.0

    def _state(self):
        qpos = self.base_env.agent.robot.get_qpos()
        qvel = self.base_env.agent.robot.get_qvel() * 0.1
        tcp_pose = vectorize_pose(self.base_env.tcp.pose)
        obj_pose = vectorize_pose(self.base_env.obj_pose)
        tcp_to_obj = self.base_env.obj_pose.p - self.base_env.tcp.pose.p
        obj_height = np.array(
            [self.base_env.obj.pose.p[2] - self.base_env.obj_height_after_settle]
        )
        
        # 현재 Phase 정보 제공
        phase_array = np.array([self.phase], dtype=np.float32)
        
        return np.concatenate(
            [qpos, qvel, tcp_pose, obj_pose, tcp_to_obj, obj_height, phase_array]
        ).astype(np.float32)

    def _save_frame(self, obs):
        self.last_frame = get_image_from_maniskill2_obs_dict(
            self.base_env, obs, camera_name=CAMERA_NAME
        )

    def _get_robot_state_dict(self):
        eef_pos = self.base_env.tcp.pose.p.tolist()
        quat_sapien = self.base_env.tcp.pose.q
        quat_scipy = [quat_sapien[1], quat_sapien[2], quat_sapien[3], quat_sapien[0]]
        eef_euler = R.from_quat(quat_scipy).as_euler('xyz').tolist()
        
        qpos = self.base_env.agent.robot.get_qpos()
        gripper_state = float(np.mean(qpos[-2:]))
        
        return {
            "eef_pos": [float(x) for x in eef_pos],
            "eef_euler": [float(x) for x in eef_euler],
            "gripper_state": gripper_state
        }
    
    def reset(self, **kwargs):
        options = dict(kwargs.pop("options", {}) or {})
        
        # curriculum_factor: 1.0은 가장 먼 시작 위치
        curriculum_factor = options.pop("curriculum_factor", 1.0)
        
        options.setdefault("robot_init_options", {"init_xy": [0.0, 0.0], "init_height": TABLE_HEIGHT})
        options.setdefault("obj_init_options", {"init_z": TABLE_HEIGHT})

        obs, info = self.env.reset(options=options, **kwargs)
        
        # Reverse Curriculum: P-controller로 초기 위치 조절
        if curriculum_factor < 1.0:
            tcp_pos = self.base_env.tcp.pose.p
            can_pos = self.base_env.obj_pose.p
            
            pos_orig = tcp_pos.copy()
            pos_easy = can_pos.copy()
            pos_easy[2] += self.margin + 0.02
            
            target_pos = (1.0 - curriculum_factor) * pos_easy + curriculum_factor * pos_orig
            
            for _ in range(50):
                current_tcp = self.base_env.tcp.pose.p
                error = target_pos - current_tcp
                if np.linalg.norm(error) < 0.005:
                    break
                
                action = np.zeros(self.action_space.shape[0], dtype=np.float32)
                action[:3] = np.clip(error * 10.0, -1.0, 1.0) 
                action[-1] = 1.0  
                self.base_env.step(action)
            
            # dummy step 통해 obs 획득
            dummy_action = np.zeros(self.action_space.shape[0], dtype=np.float32)
            dummy_action[-1] = 1.0
            obs, _, _, _, info = self.env.step(dummy_action)
            
        self._save_frame(obs)
        
        tcp_pos = self.base_env.tcp.pose.p
        can_pos = self.base_env.obj_pose.p
        self.init_can_z = can_pos[2]
        
        self.phase = 0
        self.max_lift_z = self.init_can_z
        
        target_pos = can_pos.copy()
        target_pos[2] += self.margin
        # 에피소드 시작 시 최고 기록(min_dist) 초기화
        self.min_dist = np.linalg.norm(tcp_pos - target_pos) 
        
        info["robot_state"] = self._get_robot_state_dict()
        return self._state(), info

    def step(self, action):
        # 그리퍼 액션 Discretization (0 or 1 -> -1.0 or 1.0)
        discrete_action = np.copy(action)
        discrete_action[-1] = 1.0 if action[-1] == 1.0 else -1.0
        
        obs, _, terminated, truncated, info = self.env.step(discrete_action)
        self._save_frame(obs)

        tcp_pos = self.base_env.tcp.pose.p
        can_pos = self.base_env.obj_pose.p
        
        reward = 0.0
        is_success = False
        
        # 0. 충돌 / 실패 처리
        if can_pos[2] < self.init_can_z - 0.04:
            terminated = True
            info["success"] = False
            info["robot_state"] = self._get_robot_state_dict()
            # 음수 패널티가 없으므로 지금까지 모은 + 보상이 Total Reward
            return self._state(), float(reward), terminated, truncated, info

        # 1. Phase 기반 Reward
        if self.phase == 0:
            target_pos = can_pos.copy()
            target_pos[2] += self.margin
            dist = np.linalg.norm(tcp_pos - target_pos)
            
            # 거리 기록 갱신하면 Reward
            if dist < self.min_dist:
                reward += (self.min_dist - dist) * 2.0
                self.min_dist = dist  
                
            if dist <= self.hover_target_threshold:
                self.phase = 1
                reward += 0.5  # 전환 보너스
                
                # Phase 1 진입 시 그리퍼가 열려있다면 reward (discrete_action[-1]은 1.0 또는 -1.0)
                if discrete_action[-1] > 0:
                    reward += 1.0
                    
                target_pos_grasp = can_pos.copy()
                
                # Phase 1 타겟 (캔 상단 지점)
                target_pos_grasp[2] += 0.08  
                
                self.min_dist = np.linalg.norm(tcp_pos - target_pos_grasp)
                
        elif self.phase == 1:
            target_pos = can_pos.copy()
            
            # Phase 1 타겟 (캔 상단 지점)
            target_pos[2] += 0.08 
            
            dist = np.linalg.norm(tcp_pos - target_pos)
            
            if dist < self.min_dist:
                reward += (self.min_dist - dist) * 2.0
                self.min_dist = dist
                
            if dist <= self.grasp_target_threshold:
                self.phase = 2
                reward += 0.5  
                
                # Phase 2 진입 시 그리퍼가 닫혀있다면 reward
                if discrete_action[-1] < 0:
                    reward += 2.0
                    
                self.max_lift_z = can_pos[2]
                
        elif self.phase == 2:
            # 들어올린 높이 기록 갱신하면 Reward
            if can_pos[2] > self.max_lift_z:
                reward += (can_pos[2] - self.max_lift_z) * 10.0
                self.max_lift_z = can_pos[2]
                
            if can_pos[2] > self.init_can_z + self.lift_target_margin:
                reward += 1.0  
                is_success = True
                terminated = True

        info["success"] = is_success
        info["robot_state"] = self._get_robot_state_dict()
        
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
        # gripper action 분리
        self.cont_dim = action_dim - 1
        self.mean = tf.keras.layers.Dense(self.cont_dim, activation="tanh")
        
        # gripper action categorical logits (0: 닫기, 1: 열기)
        self.gripper_logits = tf.keras.layers.Dense(2)
        
        self.value = tf.keras.layers.Dense(1)
        self.log_std = self.add_weight(
            shape=(self.cont_dim,),
            initializer=tf.keras.initializers.Constant(-0.5),
            trainable=True,
            name="log_std",
            dtype=tf.float32,
        )

    def call(self, observations):
        features = self.shared(observations)
        # mean, gripper_logits, value 리턴
        return self.mean(features), self.gripper_logits(features), tf.squeeze(self.value(features), axis=-1)


def gaussian_log_prob(actions, means, log_std):
    variance = tf.exp(2.0 * log_std)
    log_prob = -0.5 * (
        tf.square(actions - means) / variance + 2.0 * log_std + LOG_TWO_PI
    )
    return tf.reduce_sum(log_prob, axis=-1)


def choose_action(model, observation, deterministic=False):
    obs_tensor = tf.convert_to_tensor(observation[None], dtype=tf.float32)

    mean, gripper_logits, value = model(obs_tensor)
    
    if deterministic:
        cont_action = mean[0]
        gripper_action = tf.argmax(gripper_logits[0], output_type=tf.int32)
    else:
        cont_action = mean[0] + tf.exp(model.log_std) * tf.random.normal(mean[0].shape)
        gripper_action = tf.random.categorical(gripper_logits, 1, dtype=tf.int32)[0, 0]
        
    clipped_cont_action = tf.clip_by_value(cont_action, -1.0, 1.0)
    
    cont_log_prob = gaussian_log_prob(clipped_cont_action[None], mean, model.log_std)[0]
    
    gripper_probs = tf.nn.softmax(gripper_logits[0])
    gripper_log_prob = tf.math.log(gripper_probs[gripper_action] + 1e-8)
    
    total_log_prob = cont_log_prob + gripper_log_prob
    
    final_action = np.concatenate([clipped_cont_action.numpy(), [float(gripper_action.numpy())]])
    
    return final_action, float(total_log_prob), float(value[0])


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
                means, gripper_logits, values = model(obs_b)
                
                cont_act_b = act_b[:, :-1]
                gripper_act_b = tf.cast(act_b[:, -1], tf.int32)
                
                cont_log_probs = gaussian_log_prob(cont_act_b, means, model.log_std)
                gripper_loss_calc = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gripper_act_b, logits=gripper_logits)
                gripper_log_probs = -gripper_loss_calc
                
                log_probs = cont_log_probs + gripper_log_probs
                
                ratio = tf.exp(log_probs - old_log_b)
                clipped_ratio = tf.clip_by_value(
                    ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio
                )
                policy_loss = -tf.reduce_mean(
                    tf.minimum(ratio * adv_b, clipped_ratio * adv_b)
                )
                value_loss = tf.reduce_mean(tf.square(ret_b - values))
                
                entropy_cont = tf.reduce_sum(model.log_std + GAUSSIAN_ENTROPY_CONSTANT)
                gripper_probs_dist = tf.nn.softmax(gripper_logits)
                entropy_gripper = tf.reduce_mean(-tf.reduce_sum(gripper_probs_dist * tf.math.log(gripper_probs_dist + 1e-8), axis=-1))
                entropy = entropy_cont + entropy_gripper
                
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

    episode_return = 0.0
    episode_count = 0
    completed_steps = 0
    
    curriculum_factor = 0.0
    success_history = deque(maxlen=100)  # 100개의 에피소드 성공 기록 관리
    observation, _ = env.reset(seed=args.seed, options={"curriculum_factor": curriculum_factor})
    
    # 에피소드 시작 시간 기록
    episode_start_time = time.time()

    # tqdm 바 생성 (총 스텝 수 기준)
    pbar = tqdm(total=args.total_steps, desc="Training Steps", unit="step")

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
            pbar.update(1)  # tqdm 바 1스텝 업데이트

            if done:
                episode_count += 1
                episode_duration = time.time() - episode_start_time # 소요 시간 계산
                
                success_history.append(float(info["success"]))
                
                if len(success_history) > 0:
                    recent_success_rate = sum(success_history) / len(success_history)
                else:
                    recent_success_rate = 0.0
                
                # 조건 달성시 난이도 상승 (5%)
                if len(success_history) == 100 and recent_success_rate >= 0.5:
                    curriculum_factor = min(1.0, curriculum_factor + 0.05)
                    success_history.clear()
                
                # Forced Final Phase 적용
                progress = completed_steps / args.total_steps
                if FORCED_FINAL_PHASE is not None and progress >= FORCED_FINAL_PHASE:
                    curriculum_factor = 1.0
                
                tqdm.write(
                    f"episode={episode_count} steps={completed_steps} "
                    f"return={episode_return:.2f} success={info['success']} "
                    f"time={episode_duration:.1f}s curriculum={curriculum_factor:.2f} "
                    f"success_rate={recent_success_rate:.2f}"
                )
                if wandb_run is not None:
                    wandb_run.log(
                        {
                            "train/episode_return": episode_return,
                            "train/episode_success": float(info["success"]),
                            "train/episode": episode_count,
                            "train/episode_duration_sec": episode_duration, # WandB에 시간 기록
                            "train/curriculum_factor": curriculum_factor, # WandB에 curriculum_factor 기록
                            "train/recent_success_rate": recent_success_rate, # WandB에 성공률 기록
                        },
                        step=completed_steps,
                    )
                observation, _ = env.reset(options={"curriculum_factor": curriculum_factor})
                episode_return = 0.0
                episode_start_time = time.time() # 다음 에피소드 시간 측정 시작

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
        
        if wandb_run is not None:
            import wandb
            wandb.save(str(args.model_path), base_path=str(args.model_path.parent))
        
        tqdm.write(
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

    pbar.close()
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
    
    for episode in tqdm(range(args.episodes), desc="Testing Episodes", unit="ep"):
        episode_start_time = time.time() # 시간 측정 시작
        
        # 테스트 시 curriculum_factor 1.0으로 고정
        observation, _ = env.reset(seed=args.seed + episode, options={"curriculum_factor": 1.0})
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

        episode_duration = time.time() - episode_start_time # 소요 시간 계산
        success = bool(info["success"])
        successes += int(success)
        video_path = VIDEO_DIR / f"episode_{episode:03d}_success_{int(success)}.mp4"
        media.write_video(video_path, frames, fps=5)
        
        tqdm.write(
            f"test episode={episode + 1} return={total_reward:.2f} "
            f"success={success} time={episode_duration:.1f}s video={video_path}"
        )
        
        if wandb_run is not None:
            log_data = {
                "test/episode_return": total_reward,
                "test/episode_success": float(success),
                "test/episode": episode + 1,
                "test/episode_duration_sec": episode_duration, # 테스트 소요 시간 기록
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