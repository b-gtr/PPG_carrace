# ppo.py
import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

# sb3-contrib provides RecurrentPPO and CnnLstmPolicy
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy

# For learning rate scheduling
from stable_baselines3.common.utils import get_linear_fn


def train_cnn_policy(total_timesteps=200_000, n_envs=8):
    """
    Train a standard PPO with a CNN policy on CarRacing-v3 using:
      - Multiple parallel environments
      - A linear learning rate schedule
      - Tuned hyperparameters
    """
    print("Training PPO with CnnPolicy on CarRacing-v3...")

    # Create multiple parallel environments for faster sampling
    env = make_vec_env(
        env_id="CarRacing-v3",
        n_envs=n_envs,
        env_kwargs={"continuous": True},
        vec_env_cls=SubprocVecEnv,
    )

    # Linear schedule from 3e-4 down to 1e-5 over total_timesteps
    initial_lr = 3e-4
    final_lr = 1e-5
    lr_schedule = get_linear_fn(initial_value=initial_lr, final_value=final_lr, max_progress=1.0)

    # Create the PPO model with a built-in CNN policy and improved hyperparameters
    model_cnn = PPO(
        policy="CnnPolicy",
        env=env,
        learning_rate=lr_schedule,
        n_steps=1024,       # More steps per rollout helps with training stability
        batch_size=64,      # Reasonable batch size for CNN-based policy
        ent_coef=0.01,      # Encourage exploration
        gamma=0.99,         # Discount factor
        gae_lambda=0.95,    # GAE parameter
        clip_range=0.2,     # PPO clipping
        verbose=1,
        tensorboard_log="./ppo_cnn_tensorboard/"  # Optional: for TensorBoard logs
    )

    # Train the model
    model_cnn.learn(total_timesteps=total_timesteps)

    # Save the model
    model_cnn.save("ppo_cnn_car_racing")
    print("Finished training PPO CNN. Model saved as ppo_cnn_car_racing.zip")


def train_lstm_policy(total_timesteps=200_000, n_envs=8):
    """
    Train a RecurrentPPO (sb3-contrib) with a CNN + LSTM policy on CarRacing-v3 using:
      - Multiple parallel environments
      - A linear learning rate schedule
      - Tuned hyperparameters
    """
    print("Training RecurrentPPO with CnnLstmPolicy on CarRacing-v3...")

    # Create multiple parallel environments for faster sampling
    env = make_vec_env(
        env_id="CarRacing-v3",
        n_envs=n_envs,
        env_kwargs={"continuous": True},
        vec_env_cls=SubprocVecEnv,
    )

    # Linear schedule from 3e-4 down to 1e-5 over total_timesteps
    initial_lr = 3e-4
    final_lr = 1e-5
    lr_schedule = get_linear_fn(initial_value=initial_lr, final_value=final_lr, max_progress=1.0)

    # Create the RecurrentPPO model with a CNN + LSTM policy and improved hyperparameters
    model_lstm = RecurrentPPO(
        policy=CnnLstmPolicy,
        env=env,
        learning_rate=lr_schedule,
        n_steps=1024,
        batch_size=64,
        ent_coef=0.01,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        verbose=1,
        tensorboard_log="./recurrent_ppo_tensorboard/"
    )

    # Train the model
    model_lstm.learn(total_timesteps=total_timesteps)

    # Save the model
    model_lstm.save("ppo_lstm_car_racing")
    print("Finished training Recurrent PPO LSTM. Model saved as ppo_lstm_car_racing.zip")


if __name__ == "__main__":
    # Example: train each policy for 200k timesteps using 8 parallel environments
    train_cnn_policy(total_timesteps=200_000, n_envs=8)
    train_lstm_policy(total_timesteps=200_000, n_envs=8)
