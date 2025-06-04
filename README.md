# Super Mario Bros AI

A custom reinforcement learning agent trained to play Super Mario Bros using Stable Baselines3 and a custom convolutional policy. This project utilizes parallel environments, video recording, and custom callbacks to train and monitor performance.

## 🕹 Overview

This repository implements a Proximal Policy Optimization (PPO) agent that learns to play `SuperMarioBros-v0` from the `gym-super-mario-bros` environment. The agent is trained using a custom CNN policy with orthogonal initialization, an idle-death wrapper to discourage stalling, and TensorBoard logging.

## 📁 Structure

- `train_mario.py` – Main training script with parallel environment setup, video recorder, TensorBoard logging, and checkpoint saving.
- `custom_policy.py` – Defines `CustomCnnPolicy`, a subclass of `CnnPolicy` with orthogonal initialization.
- `watch_mario.py` – Loads and renders a trained agent in real-time using PPO.
- `run_mario.py` – Simple script to observe random gameplay from the environment.

## 🚀 Features

- **Parallel training** with `SubprocVecEnv` (8 environments).
- **Video recording** at intervals via `VecVideoRecorder`.
- **Evaluation callback** for periodic model evaluation and checkpointing.
- **Idle death detection** to terminate unproductive episodes.
- **TensorBoard logging** for custom metrics (reward, length, idle deaths).

## 📦 Requirements

Install dependencies with:

```bash
pip install -r requirements.txt
