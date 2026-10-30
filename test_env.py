import time
import numpy as np
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.wrappers import make_mario_env


def test_single_env():
    print("=== Testing Single Environment Wrapper ===")
    env = make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=True)

    obs, info = env.reset()
    print(f"Observation shape: {obs.shape} (dtype: {obs.dtype})")
    print(f"Action space: {env.action_space}")
    assert obs.shape == (1, 84, 84), f"Expected (1, 84, 84), got {obs.shape}"

    start_time = time.time()
    steps = 500
    for s in range(steps):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - start_time
    fps = steps / elapsed
    print(f"Single env simulated {steps} steps in {elapsed:.2f}s (~{fps:.1f} FPS)")
    env.close()


def test_vectorized_env():
    print("\n=== Testing DummyVecEnv + VecFrameStack(4) ===")
    def _make():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=True)

    vec_env = DummyVecEnv([_make for _ in range(2)])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order='first')

    obs = vec_env.reset()
    print(f"VecEnv Observation shape: {obs.shape}")
    assert obs.shape == (2, 4, 84, 84), f"Expected (2, 4, 84, 84), got {obs.shape}"

    actions = np.array([vec_env.action_space.sample() for _ in range(2)])
    obs, rewards, dones, infos = vec_env.step(actions)
    print(f"Step succeeded! Rewards: {rewards}, Dones: {dones}")
    vec_env.close()
    print("All environment tests passed successfully!\n")


if __name__ == "__main__":
    test_single_env()
    test_vectorized_env()
