import os
import argparse
from typing import Callable

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, BaseCallback
from envs.wrappers import make_mario_env


def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.
    :param initial_value: Initial learning rate.
    :return: schedule that computes current learning rate depending on remaining progress.
    """
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func


def make_env(env_id="SuperMarioBros-1-1-v0", custom_reward=True):
    def _init():
        return make_mario_env(env_id=env_id, custom_reward=custom_reward)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Train Mario 1-1 with PPO")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--total_steps", type=int, default=2_000_000, help="Total training timesteps (default: 2M)")
    parser.add_argument("--lr", type=float, default=1e-4, help="Initial learning rate (default: 1e-4)")
    parser.add_argument("--n_steps", type=int, default=512, help="Steps per env per rollout (default: 512)")
    parser.add_argument("--batch_size", type=int, default=64, help="Minibatch size (default: 64)")
    parser.add_argument("--ent_coef", type=float, default=0.01, help="Entropy coefficient for exploration (default: 0.01)")
    parser.add_argument("--exp_name", type=str, default="ppo_mario_1_1", help="Experiment name")
    parser.add_argument("--save_freq", type=int, default=50_000, help="Checkpoint save frequency in steps")
    args = parser.parse_args()

    models_dir = os.path.join("models", args.exp_name)
    logs_dir = os.path.join("logs", args.exp_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"==================================================")
    print(f" Starting PPO Training: {args.exp_name}")
    print(f" Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Parallel Envs: {args.num_envs}")
    print(f" Total Steps: {args.total_steps:,}")
    print(f" Models Dir: {models_dir}")
    print(f" TensorBoard Logs: {logs_dir}")
    print(f"==================================================")

    # 1. Create vectorized environments with multiprocessing
    env_fns = [make_env("SuperMarioBros-1-1-v0", custom_reward=True) for _ in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    # 2. Stack 4 frames (channels-first for PyTorch: (num_envs, 4, 84, 84))
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    # 3. Setup Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=models_dir,
        name_prefix="mario_ppo",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 4. Initialize PPO agent
    model = PPO(
        policy="CnnPolicy",
        env=vec_env,
        learning_rate=linear_schedule(args.lr),
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=args.ent_coef,
        vf_coef=0.5,
        max_grad_norm=0.5,
        verbose=1,
        tensorboard_log=logs_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 5. Start learning
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=checkpoint_callback,
            progress_bar=True,
        )
    finally:
        # Save final model
        final_model_path = os.path.join(models_dir, "mario_ppo_final.zip")
        model.save(final_model_path)
        print(f"\nFinal model saved to {final_model_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
