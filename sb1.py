import gymnasium as gym
from shimmy import GymV26CompatibilityV0

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor


def make_env():
    """
    Create a CarRacing-v3 environment *through* GymV26CompatibilityV0
    so SB3 sees the old (gym <= 0.21) API.
    """
    # Pass the environment *ID string* directly, rather than an instantiated env.
    old_env = GymV26CompatibilityV0("CarRacing-v3", render_mode=None)
    env = Monitor(old_env)
    return env


if __name__ == "__main__":
    # Create a DummyVecEnv with 1 copy of CarRacing-v3
    env_fns = [make_env for _ in range(1)]
    vec_env = DummyVecEnv(env_fns)
    
    # For image observations, use VecTransposeImage to switch [H, W, C] -> [C, H, W]
    vec_env = VecTransposeImage(vec_env)

    # Define some PPO hyperparameters
    ppo_hyperparams = dict(
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        learning_rate=3e-4,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        gae_lambda=0.95,
        max_grad_norm=0.5,
        verbose=1,
    )

    # Instantiate the PPO model with a CNN policy
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        **ppo_hyperparams
    )

    # Train the model
    total_timesteps = 50_000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("ppo_car_racing_v3")

    # --- EVALUATION / TESTING ---
    # Create a test environment in the same way
    test_env = GymV26CompatibilityV0("CarRacing-v3", render_mode="human")
    test_env = Monitor(test_env)

    obs, _ = test_env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        if done or truncated:
            next_obs, _ = test_env.reset()
        obs = next_obs

    test_env.close()
