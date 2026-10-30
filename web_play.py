import os
import glob
import time
import io
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
import cv2
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from envs.wrappers import make_mario_env

# Shared global state for web streaming
latest_jpeg = None
status_info = {
    "episode": 0,
    "step": 0,
    "x_pos": 0,
    "max_x": 0,
    "score": 0,
    "life": 0,
    "model": "None",
    "status": "Starting..."
}
lock = threading.Lock()


def get_latest_checkpoint():
    # Check finetune_2 first, then finetune_1, then run_1
    for d in ["models/mario_finetune_2", "models/mario_finetune_1", "models/mario_run_1"]:
        checkpoints = glob.glob(os.path.join(d, "*.zip"))
        if checkpoints:
            checkpoints.sort(key=os.path.getmtime)
            return checkpoints[-1]
    return None


class StreamingHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        global latest_jpeg, status_info

        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-type", "text/html; charset=utf-8")
            self.end_headers()
            html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Super Mario Bros - Realtime RL Stream</title>
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <style>
        body {{
            background-color: #121212;
            color: #eeeeee;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin: 0;
            padding: 20px;
        }}
        h1 {{ margin-bottom: 8px; color: #ff4d4d; }}
        .stream-container {{
            border: 4px solid #333;
            border-radius: 8px;
            box-shadow: 0 8px 24px rgba(0,0,0,0.6);
            overflow: hidden;
            background: #000;
        }}
        img {{
            display: block;
            image-rendering: pixelated;
            width: 768px;
            height: 720px;
        }}
        .stats-card {{
            margin-top: 16px;
            padding: 12px 24px;
            background: #1e1e1e;
            border-radius: 6px;
            display: flex;
            gap: 24px;
            font-size: 16px;
            border: 1px solid #333;
        }}
        .stat-item span {{ color: #4ade80; font-weight: bold; }}
    </style>
</head>
<body>
    <h1>Super Mario Bros 1-1 Live Agent</h1>
    <div class="stream-container">
        <img src="/stream" alt="Live Mario Stream" />
    </div>
    <div class="stats-card">
        <div class="stat-item">Model: <span>{os.path.basename(status_info['model'])}</span></div>
        <div class="stat-item">Status: <span>LIVE 🔴</span></div>
    </div>
</body>
</html>"""
            self.wfile.write(html.encode("utf-8"))

        elif self.path == "/stream":
            self.send_response(200)
            self.send_header("Age", "0")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=FRAME")
            self.end_headers()

            try:
                while True:
                    with lock:
                        frame = latest_jpeg
                    if frame is not None:
                        self.wfile.write(b"--FRAME\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame)))
                        self.end_headers()
                        self.wfile.write(frame)
                        self.wfile.write(b"\r\n")
                    time.sleep(0.033)  # ~30 FPS
            except (ConnectionResetError, BrokenPipeError):
                pass
        else:
            self.send_error(404)

    def log_message(self, format, *args):
        # Suppress noisy HTTP request logging
        return


def run_server(port=5000):
    server = HTTPServer(("0.0.0.0", port), StreamingHandler)
    print(f"\n=======================================================")
    print(f" Web Stream Server started!")
    print(f" Open your Windows browser at: http://localhost:{port}")
    print(f"=======================================================\n")
    server.serve_forever()


def run_agent_loop(model_path=None, fps=30, deterministic=False):
    global latest_jpeg, status_info

    if model_path is None:
        model_path = get_latest_checkpoint()
        if model_path is None:
            print("No checkpoint found!")
            return

    status_info["model"] = model_path
    print(f"Loading model: {model_path}")
    model = PPO.load(model_path)

    def make_eval_env():
        return make_mario_env(env_id="SuperMarioBros-1-1-v0", custom_reward=False)

    vec_env = DummyVecEnv([make_eval_env])
    vec_env = VecFrameStack(vec_env, n_stack=4, channels_order="first")

    delay = 1.0 / fps
    episode = 0

    while True:
        episode += 1
        obs = vec_env.reset()
        done = False
        step_count = 0
        max_x = 0
        status_info["episode"] = episode

        while not done:
            start_time = time.time()

            action, _states = model.predict(obs, deterministic=deterministic)
            obs, reward, done_flags, infos = vec_env.step(action)
            done = done_flags[0]
            info = infos[0]

            step_count += 1
            x_pos = info.get("x_pos", 0)
            if x_pos > max_x:
                max_x = x_pos

            status_info["step"] = step_count
            status_info["x_pos"] = x_pos
            status_info["max_x"] = max_x
            status_info["score"] = info.get("score", 0)
            status_info["life"] = info.get("life", 0)

            # Get raw NES frame (240, 256, 3)
            raw_frame = getattr(vec_env.envs[0].unwrapped, "screen", None)
            if raw_frame is not None:
                # Convert RGB to BGR for OpenCV
                bgr = cv2.cvtColor(raw_frame, cv2.COLOR_RGB2BGR)

                # Resize for high clarity (768 x 720)
                bgr = cv2.resize(bgr, (768, 720), interpolation=cv2.INTER_NEAREST)

                # Add OSD info
                osd_text = f"Ep: {episode} | Step: {step_count} | X: {x_pos} (Max: {max_x}) | Score: {status_info['score']} | Life: {status_info['life']}"
                cv2.putText(bgr, osd_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4)
                cv2.putText(bgr, osd_text, (15, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Encode to JPEG
                _, jpeg_buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                with lock:
                    latest_jpeg = jpeg_buf.tobytes()

            elapsed = time.time() - start_time
            if elapsed < delay:
                time.sleep(delay - elapsed)

        status_info["status"] = "CLEARED! 🏁" if info.get("flag_get", False) else "FAILED"
        print(f"Episode {episode} finished! Max X: {max_x}")
        time.sleep(1.0)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="models/mario_run_1/mario_ppo_1900000_steps.zip", help="Checkpoint path (default: 1.9M clear model)")
    parser.add_argument("--port", type=int, default=5000, help="Web port (default: 5000)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS (default: 30)")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic action selection instead of stochastic")
    args = parser.parse_args()

    # Start HTTP streaming server in background thread
    server_thread = threading.Thread(target=run_server, args=(args.port,), daemon=True)
    server_thread.start()

    # Run agent loop in main thread
    run_agent_loop(model_path=args.model, fps=args.fps, deterministic=args.deterministic)
