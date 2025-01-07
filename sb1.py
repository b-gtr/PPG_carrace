import gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

def make_env(render_mode=None):
    """
    Create a CarRacing-v2 environment.
    """
    env = gym.make("CarRacing-v2", render_mode=render_mode)
    # Wrap in Monitor so that episodes are tracked/logged
    env = Monitor(env)
    return env

if __name__ == "__main__":
    # Number of parallel environments you want to run
    num_envs = 1

    # === CRITICAL: pass in callables (lambdas) that create an environment ===
    env_fns = [lambda: make_env(render_mode=None) for _ in range(num_envs)]
    vec_env = DummyVecEnv(env_fns)
    # Transpose images to [C, H, W], as required by SB3's CnnPolicy
    vec_env = VecTransposeImage(vec_env)

    # Some PPO hyperparameters
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
        verbose=1
    )

    # Create the PPO model with a CNN policy
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        **ppo_hyperparams
    )

    # Train the agent
    total_timesteps = 200_000
    model.learn(total_timesteps=total_timesteps)

    # Save the model
    model.save("ppo_car_racing_cnn")

    # --- EVALUATION / TESTING ---
    test_env = make_env(render_mode="human")
    obs, _ = test_env.reset()

    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, info = test_env.step(action)
        test_env.render()
        if done or truncated:
            next_obs, _ = test_env.reset()
        obs = next_obs

    test_env.close()
