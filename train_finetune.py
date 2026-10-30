import os
import argparse
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback
from envs.wrappers import make_mario_env


def make_env(env_id="SuperMarioBros-1-1-v0", custom_reward=True):
    def _init():
        return make_mario_env(env_id=env_id, custom_reward=custom_reward)
    return _init


def main():
    parser = argparse.ArgumentParser(description="Precision Fine-Tuning for 1-1 Flagpole Clear")
    parser.add_argument("--base_model", type=str, default="models/mario_run_1/mario_ppo_1900000_steps.zip", help="Base 1.9M model")
    parser.add_argument("--num_envs", type=int, default=8, help="Number of parallel environments (default: 8)")
    parser.add_argument("--total_steps", type=int, default=150_000, help="Total steps (default: 150K, ~3-4 mins)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Low learning rate to preserve early-game skills (default: 2e-5)")
    parser.add_argument("--ent_coef", type=float, default=0.005, help="Low entropy to prevent ruining delicate jumps (default: 0.005)")
    parser.add_argument("--exp_name", type=str, default="mario_finetune_2", help="Experiment name")
    parser.add_argument("--save_freq", type=int, default=15_000, help="Checkpoint frequency")
    args = parser.parse_args()

    models_dir = os.path.join("models", args.exp_name)
    logs_dir = os.path.join("logs", args.exp_name)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)

    print(f"==================================================")
    print(f" Precision PPO Fine-Tuning: {args.exp_name}")
    print(f" Base Model: {args.base_model}")
    print(f" Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    if torch.cuda.is_available():
        print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Parallel Envs: {args.num_envs}")
    print(f" Steps: {args.total_steps:,} (~3-4 min)")
    print(f" LR: {args.lr} (Knowledge-preserving)")
    print(f" Entropy: {args.ent_coef} (Precision-preserving)")
    print(f" Models Dir: {models_dir}")
    print(f"==================================================")

    # 1. Parallel envs with precision anti-freeze wrappers
    env_fns = [make_env("SuperMarioBros-1-1-v0", custom_reward=True) for _ in range(args.num_envs)]
    vec_env = SubprocVecEnv(env_fns)
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    # 2. Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=max(args.save_freq // args.num_envs, 1),
        save_path=models_dir,
        name_prefix="mario_flag",
        save_replay_buffer=False,
        save_vecnormalize=False,
    )

    # 3. Load 1.9M checkpoint
    print(f"\nLoading weights from: {args.base_model}...")
    model = PPO.load(
        args.base_model,
        env=vec_env,
        learning_rate=args.lr,
        ent_coef=args.ent_coef,
        tensorboard_log=logs_dir,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )

    # 4. Run fine-tuning
    try:
        model.learn(
            total_timesteps=args.total_steps,
            callback=checkpoint_callback,
            progress_bar=True,
            reset_num_timesteps=False,
        )
    finally:
        final_path = os.path.join(models_dir, "mario_clear_final.zip")
        model.save(final_path)
        print(f"\nFinal clear model saved to: {final_path}")
        vec_env.close()


if __name__ == "__main__":
    main()
