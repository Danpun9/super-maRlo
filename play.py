import os
import glob
import time
import numpy as np
import pygame
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.wrappers import make_mario_env


def get_latest_checkpoint(models_dir="models/mario_run_1"):
    checkpoints = glob.glob(os.path.join(models_dir, "*.zip"))
    if not checkpoints:
        return None
    # Filter out final if needed or sort by modification time
    checkpoints.sort(key=os.path.getmtime)
    return checkpoints[-1]


def play(model_path=None, scale=3, fps=30):
    if model_path is None:
        model_path = get_latest_checkpoint()
        if model_path is None:
            print("No saved checkpoint found in models/mario_run_1!")
            return

    print(f"Loading checkpoint: {model_path}")
    model = PPO.load(model_path)

    # Base NES resolution is 256 x 240
    base_w, base_h = 256, 240
    win_w, win_h = base_w * scale, base_h * scale

    # Initialize Pygame
    pygame.init()
    screen = pygame.display.set_mode((win_w, win_h))
    pygame.display.set_caption(f"Super Mario Bros RL Agent - {os.path.basename(model_path)}")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Arial", 18)

    # Evaluation Environment
    def make_eval_env():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=False)

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    running = True
    episode = 0

    try:
        while running:
            episode += 1
            obs = vec_env.reset()
            done = False
            total_reward = 0.0
            step_count = 0
            max_x = 0

            print(f"\n--- Starting Episode {episode} ---")

            while not done and running:
                # Handle window events (closing, ESC key)
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False
                        break
                    elif event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE or event.key == pygame.K_q:
                            running = False
                            break

                if not running:
                    break

                action, _states = model.predict(obs, deterministic=True)
                obs, reward, done_flags, infos = vec_env.step(action)
                done = done_flags[0]
                info = infos[0]

                total_reward += reward[0]
                step_count += 1
                x_pos = info.get("x_pos", 0)
                if x_pos > max_x:
                    max_x = x_pos
                score = info.get("score", 0)
                life = info.get("life", 0)

                # Emulator raw frame: shape is (240, 256, 3) in RGB
                raw_frame = getattr(vec_env.envs[0].unwrapped, "screen", None)
                if raw_frame is None:
                    raw_frame = vec_env.envs[0].render(mode="rgb_array")
                if raw_frame is not None:
                    # Pygame expects (width, height, 3) with axes transposed (1, 0, 2)
                    surf = pygame.surfarray.make_surface(np.transpose(raw_frame, (1, 0, 2)))
                    # Scale to window size
                    scaled_surf = pygame.transform.scale(surf, (win_w, win_h))
                    screen.blit(scaled_surf, (0, 0))

                    # Overlay status bar
                    overlay_text = f"Ep: {episode} | Step: {step_count} | X: {x_pos} (Max: {max_x}) | Score: {score} | Life: {life}"
                    text_surface = font.render(overlay_text, True, (0, 255, 0))
                    # Background rect for text readability
                    bg_rect = pygame.Rect(8, 8, text_surface.get_width() + 8, text_surface.get_height() + 4)
                    pygame.draw.rect(screen, (0, 0, 0), bg_rect)
                    screen.blit(text_surface, (12, 10))

                    pygame.display.flip()

                clock.tick(fps)

            status = "CLEARED! 🏁" if info.get("flag_get", False) else "FAILED"
            print(f"Episode {episode} finished! Status: {status} | Max X: {max_x} | Reward: {total_reward:.1f}")
            if running:
                time.sleep(1.0)

    finally:
        pygame.quit()
        vec_env.close()
        print("Closed gameplay viewer.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default=None, help="Path to checkpoint model (default: latest)")
    parser.add_argument("--scale", type=int, default=3, help="Window display scale (default: 3)")
    parser.add_argument("--fps", type=int, default=30, help="Target playback FPS (default: 30)")
    args = parser.parse_args()

    play(model_path=args.model, scale=args.scale, fps=args.fps)
