import os
import gym
import numpy as np
import gym_super_mario_bros
from gym import Wrapper
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecVideoRecorder, DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback, EvalCallback
from torch.utils.tensorboard import SummaryWriter
import torch

# ===== Custom Wrappers =====
class MarioIdleDeathWrapper(Wrapper):
    def __init__(self, env, max_idle_steps=60):
        super().__init__(env)
        self.max_idle_steps = max_idle_steps
        self.idle_counter = 0
        self.last_x_pos = 0
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self, **kwargs):
        self.idle_counter = 0
        self.episode_reward = 0
        self.episode_length = 0
        obs = self.env.reset(**kwargs)
        self.last_x_pos = self._get_x_pos()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        current_x = self._get_x_pos()

        self.episode_reward += reward
        self.episode_length += 1

        if current_x == self.last_x_pos:
            self.idle_counter += 1
        else:
            self.idle_counter = 0
            self.last_x_pos = current_x

        if self.idle_counter >= self.max_idle_steps:
            done = True
            info["idle_death"] = True
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length
            }

        return obs, reward, done, info

    def _get_x_pos(self):
        return getattr(self.env.unwrapped, "x_position", 0)

# ===== Custom Callback =====
class CustomTensorboardCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.idle_deaths = 0
        self.episode_count = 0

    def _on_step(self):
        infos = self.locals.get("infos", [])
        for info in infos:
            if "episode" in info:
                self.episode_rewards.append(info["episode"]["r"])
                self.episode_lengths.append(info["episode"]["l"])
                self.logger.record("custom/episode_reward", info["episode"]["r"])
                self.logger.record("custom/episode_length", info["episode"]["l"])
            if info.get("idle_death", False):
                self.idle_deaths += 1
        return True

    def _on_rollout_end(self):
        self.episode_count += 1
        self.logger.record("custom/idle_deaths", self.idle_deaths)
        self.logger.record("custom/idle_deaths_per_episode", self.idle_deaths / max(self.episode_count, 1))
        self.idle_deaths = 0

    def _on_training_end(self):
        if self.episode_rewards:
            self.logger.record("custom/avg_episode_reward", np.mean(self.episode_rewards))
            self.logger.record("custom/avg_episode_length", np.mean(self.episode_lengths))

# ===== Environment Factory =====
def make_env():
    def _init():
        env = gym_super_mario_bros.make("SuperMarioBros-v0")
        env = JoypadSpace(env, COMPLEX_MOVEMENT)
        env = Monitor(env)
        env = MarioIdleDeathWrapper(env)
        return env
    return _init

if __name__ == '__main__':

  # ===== Setup Parallel Environments =====
  num_envs = 4
  log_dir = "./logs/"
  video_folder = "./videos/"

  os.makedirs(log_dir, exist_ok=True)
  os.makedirs(video_folder, exist_ok=True)

  vec_env = SubprocVecEnv([make_env() for _ in range(num_envs)])
  vec_env = VecVideoRecorder(
      vec_env,
      video_folder,
      record_video_trigger=lambda step: step % 50000 == 0,
      video_length=3600,
      name_prefix="mario-train"
  )

  # ===== Model Setup =====
  model = PPO(
      "CnnPolicy",
      vec_env,
      verbose=1,
      tensorboard_log=log_dir,
      device="cuda",
      n_steps=1024
  )

  # ===== Evaluation Environment and Callback =====
  def make_eval_env():
      env = gym_super_mario_bros.make("SuperMarioBros-v0")
      env = JoypadSpace(env, COMPLEX_MOVEMENT)
      env = Monitor(env)
      env = MarioIdleDeathWrapper(env)
      vec_env = DummyVecEnv([lambda: env])
      vec_env = VecTransposeImage(vec_env)
      return vec_env

  eval_env = make_eval_env()

  eval_callback = EvalCallback(
      eval_env,
      best_model_save_path="./checkpoints/",
      log_path="./eval_logs/",
      eval_freq=100_000,
      n_eval_episodes=5,
      deterministic=True,
      render=False
  )

  # ===== Training with Callback =====
  callback = CustomTensorboardCallback()

  model.learn(total_timesteps=1_000_000, callback=[callback, eval_callback])
  model.save("ppo_mario_checkpoint_1m")

  callback.episode_rewards = []
  callback.episode_lengths = []
  callback.idle_deaths = 0
  callback.episode_count = 0

  model.learn(total_timesteps=1_000_000, callback=[callback, eval_callback])
  model.save("ppo_mario_2m_final")

  # ===== Final Evaluation =====
  mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
  print(f"Mean reward after training: {mean_reward}")