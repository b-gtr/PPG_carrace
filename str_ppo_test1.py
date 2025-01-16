# test_ppo.py

import gymnasium as gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

def test_cnn_policy(model_path="ppo_cnn_car_racing.zip", num_episodes=5):
    """
    Test a CNN-based PPO policy on CarRacing-v3 in human-render mode.
    """
    # Load the saved CNN model
    model = PPO.load(model_path)
    
    # Create a single environment with human rendering
    env = gym.make("CarRacing-v3", continuous=True, render_mode="human")
    
    for episode in range(num_episodes):
        # Reset the environment
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        
        while not done:
            # Predict the next action (inference)
            action, _states = model.predict(obs, deterministic=True)
            
            # Step in the environment
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            # Render the environment (human mode)
            env.render()
        
        print(f"[CNN] Episode {episode + 1}, Reward: {total_reward:.2f}")
    
    env.close()


def test_lstm_policy(model_path="ppo_lstm_car_racing.zip", num_episodes=5):
    """
    Test a RecurrentPPO (CNN + LSTM) policy on CarRacing-v3 in human-render mode.
    """
    # Load the saved LSTM-based model
    model = RecurrentPPO.load(model_path)
    
    # Create a single environment with human rendering
    env = gym.make("CarRacing-v3", continuous=True, render_mode="human")
    
    for episode in range(num_episodes):
        obs, info = env.reset()
        done = False
        total_reward = 0.0
        
        # Recurrent policies require a hidden state (LSTM state)
        lstm_states = None
        episode_start = True
        
        while not done:
            # Predict the next action and LSTM states
            action, lstm_states = model.predict(
                obs,
                state=lstm_states,
                episode_start=episode_start,
                deterministic=True
            )
            
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            
            env.render()
            episode_start = False
        
        print(f"[LSTM] Episode {episode + 1}, Reward: {total_reward:.2f}")
    
    env.close()


if __name__ == "__main__":
    # Test/Validate the CNN policy for 5 episodes
    test_cnn_policy(model_path="ppo_cnn_car_racing.zip", num_episodes=5)
    
    # Test/Validate the LSTM policy for 5 episodes
    test_lstm_policy(model_path="ppo_lstm_car_racing.zip", num_episodes=5)
