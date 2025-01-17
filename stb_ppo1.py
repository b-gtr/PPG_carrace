import gymnasium as gym
import torch
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy

# For optional logging
# from stable_baselines3.common.logger import configure

def make_env():
    """
    Helper function to create the CarRacing environment.
    Wrap it in a DummyVecEnv for vectorized training.
    """
    return gym.make("CarRacing-v3", continuous=True)

def train_cnn_lstm_policy(total_timesteps=200_000):
    """
    Train a RecurrentPPO model with a CNN + LSTM policy on CarRacing-v3.
    The combination is obs -> CNN -> LSTM -> action.
    """
    print("Training RecurrentPPO with CnnLstmPolicy on CarRacing-v3...")

    # Create environment and wrap it
    env = DummyVecEnv([make_env])
    # Transpose images from (H, W, C) to (C, H, W) for PyTorch
    env = VecTransposeImage(env)

    # Optional: set up logging directory
    # logger = configure("./logs/", ["stdout", "csv", "tensorboard"])

    # Hyperparameters based on common usage/examples
    model_lstm = RecurrentPPO(
        policy=CnnLstmPolicy,
        env=env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=128,            # how many steps to run per environment per update
        batch_size=64,         # mini-batch size (must be a divisor of n_steps * n_envs)
        n_epochs=10,           # number of optimization epochs per update
        gamma=0.99,            # discount factor
        gae_lambda=0.95,       # GAE parameter
        clip_range=0.2,        
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(
            # If you want to configure the CNN or LSTM size, do so here:
            # e.g. features_extractor_kwargs={"features_dim": 256},
            lstm_hidden_size=256,
            n_lstm_layers=1
        ),
        # tensorboard_log="./tensorboard/", # enable if you want tensorboard logs
    )

    # Train the model
    model_lstm.learn(total_timesteps=total_timesteps)

    # Save the model
    model_lstm.save("ppo_lstm_car_racing")
    print("Finished training. Model saved as ppo_lstm_car_racing.zip")

if __name__ == "__main__":
    # Example: train for 200k timesteps
    train_cnn_lstm_policy(total_timesteps=200_000)
