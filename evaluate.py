import os
import argparse
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.wrappers import make_mario_env


def evaluate(model_path, num_episodes=5, record=True, video_path="evaluation.mp4"):
    print(f"Loading trained model from {model_path}...")
    model = PPO.load(model_path)

    # Raw rendering tracker for high-res video recording
    def make_eval_env():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=False)

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    video_writer = None
    if record:
        # Original Mario NES resolution is 256x240 (W x H)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        video_writer = cv2.VideoWriter(video_path, fourcc, 30.0, (256, 240))
        print(f"Recording evaluation video to {video_path}...")

    total_rewards = []
    max_x_list = []
    cleared_count = 0

    try:
        for ep in range(num_episodes):
            obs = vec_env.reset()
            done = False
            ep_reward = 0.0
            max_x = 0
            flag_captured = False

            while not done:
                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done_flags, infos = vec_env.step(action)
                done = done_flags[0]
                info = infos[0]

                ep_reward += reward[0]
                x_pos = info.get("x_pos", 0)
                if x_pos > max_x:
                    max_x = x_pos

                if info.get("flag_get", False):
                    flag_captured = True

                if record and video_writer is not None:
                    # Access raw NES rendering frame
                    raw_frame = getattr(vec_env.envs[0].unwrapped, "screen", None)
                    if raw_frame is None:
                        raw_frame = vec_env.envs[0].render(mode="rgb_array")
                    if raw_frame is not None:
                        # Convert RGB to BGR for OpenCV
                        bgr_frame = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                        video_writer.write(bgr_frame)

            total_rewards.append(ep_reward)
            max_x_list.append(max_x)
            if flag_captured:
                cleared_count += 1

            status = "CLEARED! 🏁" if flag_captured else "FAILED"
            print(f"Episode {ep + 1}/{num_episodes} - Status: {status} | Max X: {max_x} | Total Reward: {ep_reward:.1f}")

    finally:
        if video_writer is not None:
            video_writer.release()
            print(f"Video saved successfully to {video_path}")
        vec_env.close()

    print("\n=== Evaluation Summary ===")
    print(f"Clear Rate: {cleared_count}/{num_episodes} ({cleared_count/num_episodes*100:.1f}%)")
    print(f"Average Max X: {np.mean(max_x_list):.1f} / 3266 (Flag is around x=3160)")
    print(f"Average Reward: {np.mean(total_rewards):.1f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Mario PPO Model")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip model file")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to evaluate")
    parser.add_argument("--record", action="store_true", help="Record gameplay video")
    parser.add_argument("--output", type=str, default="mario_eval.mp4", help="Video output path")
    args = parser.parse_args()

    evaluate(args.model, num_episodes=args.episodes, record=args.record, video_path=args.output)
