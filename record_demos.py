import os
import cv2
import numpy as np
from PIL import Image
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.wrappers import make_mario_env

os.makedirs("assets", exist_ok=True)


def convert_mp4_to_gif(mp4_path, gif_path, max_frames=None, target_fps=15, scale=0.75):
    cap = cv2.VideoCapture(mp4_path)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_step = max(1, int(round(src_fps / target_fps)))

    pil_frames = []
    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % frame_step == 0:
            h, w = frame.shape[:2]
            new_w, new_h = int(w * scale), int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_AREA)
            rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(rgb)
            pil_img = pil_img.quantize(colors=128, method=Image.Quantize.MEDIANCUT)
            pil_frames.append(pil_img)
            if max_frames and len(pil_frames) >= max_frames:
                break
        frame_idx += 1
    cap.release()

    if pil_frames:
        duration = int(1000 / target_fps)
        pil_frames[0].save(
            gif_path,
            save_all=True,
            append_images=pil_frames[1:],
            optimize=True,
            duration=duration,
            loop=0
        )
        print(f"Saved GIF: {gif_path} ({len(pil_frames)} frames, {os.path.getsize(gif_path)/1024:.1f} KB)")


def record_untrained(output_mp4="assets/demo_untrained.mp4"):
    print("\n--- Recording Untrained (Random) Agent ---")
    def make_eval_env():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=False)

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4, fourcc, 30.0, (256, 240))

    obs = vec_env.reset()
    done = False
    step_count = 0
    max_x = 0

    while not done and step_count < 300:
        action = [vec_env.action_space.sample()]
        obs, reward, done_flags, infos = vec_env.step(action)
        done = done_flags[0]
        step_count += 1
        x = infos[0].get("x_pos", 0)
        if x > max_x:
            max_x = x

        raw_frame = getattr(vec_env.envs[0].unwrapped, "screen", None)
        if raw_frame is not None:
            bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)

    out.release()
    vec_env.close()
    print(f"Untrained demo saved: {output_mp4} (Steps: {step_count}, Max X: {max_x})")
    convert_mp4_to_gif(output_mp4, "assets/demo_untrained.gif", target_fps=15)


def record_baseline_intermediate(model_path="models/mario_run_1/mario_ppo_500000_steps.zip",
                                 output_mp4="assets/demo_baseline_500k.mp4"):
    print(f"\n--- Recording Intermediate Trained Model ({model_path}) ---")
    model = PPO.load(model_path)

    def make_eval_env():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=False)

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4, fourcc, 30.0, (256, 240))

    obs = vec_env.reset()
    done = False
    step_count = 0
    max_x = 0

    while not done and step_count < 1000:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_flags, infos = vec_env.step(action)
        done = done_flags[0]
        step_count += 1
        x = infos[0].get("x_pos", 0)
        if x > max_x:
            max_x = x

        raw_frame = getattr(vec_env.envs[0].unwrapped, "screen", None)
        if raw_frame is not None:
            bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)

    out.release()
    vec_env.close()
    print(f"Baseline demo saved: {output_mp4} (Steps: {step_count}, Max X: {max_x})")
    convert_mp4_to_gif(output_mp4, "assets/demo_baseline_500k.gif", target_fps=15)


def record_staircase_freeze(model_path="models/mario_run_1/mario_ppo_1900000_steps.zip",
                            output_mp4="assets/demo_staircase_freeze.mp4"):
    print(f"\n--- Recording Staircase Freeze Model (Deterministic 1.9M) ---")
    model = PPO.load(model_path)

    def make_eval_env():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=False)

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(output_mp4, fourcc, 30.0, (256, 240))

    obs = vec_env.reset()
    done = False
    step_count = 0
    max_x = 0
    freeze_frames = 0

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done_flags, infos = vec_env.step(action)
        done = done_flags[0]
        step_count += 1
        x = infos[0].get("x_pos", 0)
        if x > max_x:
            max_x = x
            freeze_frames = 0
        else:
            freeze_frames += 1

        raw_frame = getattr(vec_env.envs[0].unwrapped, "screen", None)
        if raw_frame is not None:
            bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
            out.write(bgr)

        # Stop after 90 frames of freezing at stairs
        if max_x >= 2950 and freeze_frames > 90:
            print("Captured staircase freezing demonstration.")
            break

    out.release()
    vec_env.close()
    print(f"Staircase freeze demo saved: {output_mp4} (Steps: {step_count}, Max X: {max_x})")
    convert_mp4_to_gif(output_mp4, "assets/demo_staircase_freeze.gif", target_fps=15)


def record_clear(model_path="models/mario_run_1/mario_ppo_1900000_steps.zip",
                 output_mp4="assets/demo_stage_clear.mp4"):
    print(f"\n--- Recording Full Stage 1-1 Clear Model ---")
    model = PPO.load(model_path)

    def make_eval_env():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=False)

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    for attempt in range(1, 11):
        print(f"Clear Recording Attempt {attempt}...")
        frames = []
        obs = vec_env.reset()
        done = False
        step_count = 0
        max_x = 0
        flag_cleared = False

        while not done and step_count < 1500:
            action, _ = model.predict(obs, deterministic=False)
            obs, reward, done_flags, infos = vec_env.step(action)
            done = done_flags[0]
            step_count += 1
            info = infos[0]
            x = info.get("x_pos", 0)
            if x > max_x:
                max_x = x
            if info.get("flag_get", False):
                flag_cleared = True

            raw_frame = getattr(vec_env.envs[0].unwrapped, "screen", None)
            if raw_frame is not None:
                bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)
                frames.append(bgr)

            if flag_cleared and step_count > 50:
                for _ in range(45):
                    obs, _, _, _ = vec_env.step([action[0]])
                    rf = getattr(vec_env.envs[0].unwrapped, "screen", None)
                    if rf is not None:
                        frames.append(cv2.cvtColor(rf, cv2.COLOR_RGB2BGR))
                break

        print(f"Attempt {attempt} finished: Max X = {max_x}, Flag Cleared = {flag_cleared}")
        if flag_cleared:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            out = cv2.VideoWriter(output_mp4, fourcc, 30.0, (256, 240))
            for f in frames:
                out.write(f)
            out.release()
            print(f"SUCCESS! Stage clear video saved: {output_mp4} ({len(frames)} frames)")
            convert_mp4_to_gif(output_mp4, "assets/demo_stage_clear.gif", target_fps=15)
            break

    vec_env.close()


if __name__ == "__main__":
    record_untrained()
    record_baseline_intermediate()
    record_staircase_freeze()
    record_clear()
    print("\nAll demo recordings completed successfully!")
