import gym
import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

def make_env(render_mode=None):
    """
    Create a CarRacing-v2 environment.
    
    :param render_mode: (str or None) If "human", the environment
                        will attempt to render to the screen.
    :return: (gym.Env) The CarRacing-v2 environment instance.
    """
    # For CarRacing-v2 from gym, "render_mode" can be "human", "rgb_array", or None.
    # If you want interactive training, set it to "human".
    # If you want pure training with no GUI, set to None (default).
    env = gym.make("CarRacing-v2", render_mode=render_mode)
    
    # Optionally, you can also apply a Monitor wrapper to record episode stats:
    env = Monitor(env)
    
    return env

# Create a vectorized environment to allow Stable Baselines3 to handle observations properly
num_envs = 1
env_fns = [lambda: make_env(render_mode=None) for _ in range(num_envs)]
vec_env = DummyVecEnv(env_fns)

# For image-based observations, we need to transpose [H, W, C] to [C, H, W].
vec_env = VecTransposeImage(vec_env)

# Define some PPO hyperparameters
ppo_hyperparams = {
    "n_steps": 1024,               # Number of steps to run per rollout
    "batch_size": 64,              # Minibatch size
    "n_epochs": 10,                # Number of optimization epochs per rollout
    "gamma": 0.99,                 # Discount factor
    "learning_rate": 3e-4,         # Adam learning rate
    "clip_range": 0.2,             # Clipping range for PPO
    "ent_coef": 0.01,              # Entropy coefficient
    "vf_coef": 0.5,                # Value function coefficient
    "gae_lambda": 0.95,            # GAE lambda
    "max_grad_norm": 0.5,          # Maximum norm for gradient clipping
    "verbose": 1,                  # Verbosity mode: 0 = no output, 1 = info, 2 = debug
}

# Create the PPO model using the CNN policy
model = PPO(
    policy="CnnPolicy",
    env=vec_env,
    **ppo_hyperparams
)

# Train the model
total_timesteps = 200_000
model.learn(total_timesteps=total_timesteps)

# Save the trained model
model.save("ppo_car_racing_cnn")

# --- EVALUATION / TESTING ---

# Create a fresh environment for testing
test_env = make_env(render_mode="human")

# Reset the environment
obs, _ = test_env.reset()

# Run for a certain number of steps (you can increase or reduce this)
num_steps = 1000
for _ in range(num_steps):
    # Get an action from the trained model
    action, _states = model.predict(obs, deterministic=True)
    # Step the environment
    next_obs, reward, done, truncated, info = test_env.step(action)
    
    # Render the environment (since render_mode="human", this will display on screen)
    test_env.render()
    
    # If the episode is over or truncated, reset
    if done or truncated:
        next_obs, _ = test_env.reset()
        
    obs = next_obs

test_env.close()
