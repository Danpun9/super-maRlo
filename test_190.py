from stable_baselines3 import PPO
from envs.wrappers import make_mario_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack

env = VecFrameStack(DummyVecEnv([lambda: make_mario_env(custom_reward=False)]), 4, 'first')
model = PPO.load('models/mario_run_1/mario_ppo_1900000_steps.zip')

print("=== Testing 1.9M Model with deterministic=False ===")
for ep in range(5):
    obs = env.reset()
    done = False
    max_x = 0
    flag_captured = False
    while not done:
        action, _ = model.predict(obs, deterministic=False)
        obs, rew, done_arr, infos = env.step(action)
        done = done_arr[0]
        x = infos[0].get('x_pos', 0)
        if x > max_x:
            max_x = x
        if infos[0].get('flag_get', False):
            flag_captured = True
            break
    print(f"Episode {ep+1}: Max X: {max_x} | Flag Cleared: {flag_captured}")
env.close()
