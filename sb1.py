import gymnasium as gym
from shimmy import GymV26CompatibilityV0

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecTransposeImage
from stable_baselines3.common.monitor import Monitor

def make_env():
    # Create CarRacing-v2 environment from Gymnasium
    new_env = gym.make("CarRacing-v2", render_mode=None)
    # Convert to old-style gym.Env
    old_env = GymV26CompatibilityV0(new_env)
    # Wrap with SB3 Monitor
    monitored_env = Monitor(old_env)
    return monitored_env

if __name__ == "__main__":
    # Create vectorized environment using callables
    env_fns = [make_env for _ in range(1)]
    vec_env = DummyVecEnv(env_fns)
    # Transpose images so SB3 can handle them
    vec_env = VecTransposeImage(vec_env)

    # Define PPO with CnnPolicy
    model = PPO(
        "CnnPolicy",
        vec_env,
        n_steps=1024,
        batch_size=64,
        n_epochs=10,
        learning_rate=3e-4,
        verbose=1
    )

    # Train
    model.learn(50_000)

    # Save
    model.save("ppo_car_racing_v2")

    # Test in a real-time environment
    test_new_env = gym.make("CarRacing-v2", render_mode="human")
    test_old_env = GymV26CompatibilityV0(test_new_env)
    test_monitored_env = Monitor(test_old_env)

    obs, _ = test_monitored_env.reset()
    for _ in range(1000):
        action, _ = model.predict(obs, deterministic=True)
        next_obs, reward, done, truncated, info = test_monitored_env.step(action)
        test_monitored_env.render()
        if done or truncated:
            next_obs, _ = test_monitored_env.reset()
        obs = next_obs

    test_monitored_env.close()
