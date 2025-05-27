import os
import gym
import numpy as np
from gym import Wrapper
from stable_baselines3 import PPO
from gym_super_mario_bros.actions import COMPLEX_MOVEMENT
from nes_py.wrappers import JoypadSpace
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.callbacks import BaseCallback

# ===== Custom Wrappers =====

class MarioIdleDeathWrapper(Wrapper):
    def __init__(self, env, max_idle_steps=60):
        super().__init__(env)
        self.max_idle_steps = max_idle_steps
        self.idle_counter = 0
        self.last_x_pos = 0

    def reset(self, **kwargs):
        self.idle_counter = 0
        obs = self.env.reset(**kwargs)
        self.last_x_pos = self._get_x_pos()
        return obs

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        current_x = self._get_x_pos()

        if current_x == self.last_x_pos:
            self.idle_counter += 1
        else:
            self.idle_counter = 0
            self.last_x_pos = current_x

        if self.idle_counter >= self.max_idle_steps:
            done = True
            info["idle_death"] = True

        return obs, reward, done, info

    def _get_x_pos(self):
        return self.env.unwrapped._env.env.env.env.x_position

# ===== Custom Callback for TensorBoard =====

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

# ===== Environment Setup =====

video_folder = "./videos/"
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, COMPLEX_MOVEMENT)
env = Monitor(env)
env = MarioIdleDeathWrapper(env)
env = DummyVecEnv([lambda: env])
env = VecVideoRecorder(
    env, video_folder,
    record_video_trigger=lambda step: step % 200_000 == 0,
    video_length=3600,
    name_prefix="mario-train"
)

# ===== Training =====

model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir, device="cuda")
model.learn(total_timesteps=1_000_000, callback=CustomTensorboardCallback())
model.save("ppo_mario_checkpoint_1m")

model.learn(total_timesteps=1_000_000, callback=CustomTensorboardCallback())
model.save("ppo_mario_2m_final")

# ===== Evaluation =====

eval_env = gym_super_mario_bros.make("SuperMarioBros-v0")
eval_env = JoypadSpace(eval_env, COMPLEX_MOVEMENT)
eval_env = Monitor(eval_env)
eval_env = DummyVecEnv([lambda: eval_env])

mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=5)
print(f"Mean reward after training: {mean_reward}")