import gymnasium as gym
import torch

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# sb3-contrib provides RecurrentPPO and CnnLstmPolicy
from sb3_contrib import RecurrentPPO
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy


def make_env():
    """
    Helper function to create the CarRacing environment.
    We will wrap it in a DummyVecEnv for vectorized training.
    """
    return gym.make("CarRacing-v3", continuous=True)


def train_cnn_policy(total_timesteps=50_000):
    """
    Train a standard PPO with a CNN policy on CarRacing-v2.
    """
    print("Training PPO with CnnPolicy on CarRacing-v2...")
    env = DummyVecEnv([make_env])
    
    # Create the PPO model with a built-in CNN policy
    model_cnn = PPO(
        policy="CnnPolicy",
        env=env,
        verbose=1,
        # You can fine-tune hyperparameters (e.g., learning_rate, n_steps, etc.)
        # Here we keep them to default for demonstration.
    )
    
    # Train the model
    model_cnn.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model_cnn.save("ppo_cnn_car_racing")
    print("Finished training. Model saved as ppo_cnn_car_racing.zip")


def train_lstm_policy(total_timesteps=50_000):
    """
    Train a Recurrent PPO (sb3-contrib) with a CNN + LSTM policy on CarRacing-v2.
    """
    print("Training RecurrentPPO with CnnLstmPolicy on CarRacing-v2...")
    env = DummyVecEnv([make_env])

    # Create the RecurrentPPO model with a CNN + LSTM policy
    model_lstm = RecurrentPPO(
        policy=CnnLstmPolicy,
        env=env,
        verbose=1,
        # Adjust hyperparameters as needed, e.g.:
        # learning_rate=3e-4, n_steps=128, batch_size=64, etc.
    )
    
    # Train the model
    model_lstm.learn(total_timesteps=total_timesteps)
    
    # Save the model
    model_lstm.save("ppo_lstm_car_racing")
    print("Finished training. Model saved as ppo_lstm_car_racing.zip")


if __name__ == "__main__":
    # Example: train each policy for 50k timesteps
    train_cnn_policy(total_timesteps=50_000)
    train_lstm_policy(total_timesteps=50_000)
