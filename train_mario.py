import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.evaluation import evaluate_policy
import os

# Create environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)
env = Monitor(env)
env = DummyVecEnv([lambda: env])  # Vectorize for SB3

# Create log folder
log_dir = "./logs/"
os.makedirs(log_dir, exist_ok=True)

# Train model
model = PPO("CnnPolicy", env, verbose=1, tensorboard_log=log_dir)
model.learn(total_timesteps=100_000)

# Save model
model.save("ppo_mario")

# Evaluate
mean_reward, _ = evaluate_policy(model, env, n_eval_episodes=5)
print(f"Mean reward after training: {mean_reward}")