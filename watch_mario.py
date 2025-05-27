from stable_baselines3 import PPO
import gym_super_mario_bros
import time
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
from nes_py.wrappers import JoypadSpace

# Load trained model and environment
env = gym_super_mario_bros.make("SuperMarioBros-v0")
env = JoypadSpace(env, SIMPLE_MOVEMENT)

model = PPO.load("ppo_mario_checkpoint_1m")

obs = env.reset()
done = False
while True:
    action, _ = model.predict(obs.copy())
    action = int(action)  
    obs, reward, done, info = env.step(action)
    env.render()
    time.sleep(0.01)
    if done:
        obs = env.reset()
