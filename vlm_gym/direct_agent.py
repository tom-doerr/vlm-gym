"""Direct VLM agent — single token output for minimum latency."""

import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image

from vlm_gym.envs import make_env, render_to_pil, GameDisplay, save_video


def pil_to_data_uri(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    b64 = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/jpeg;base64,{b64}"


class DirectAgent:
    """Single-token VLM agent using raw API calls."""

    def __init__(self, model, api_base, api_key="none"):
        self.model = model
        self.api_base = api_base.rstrip("/")
        if not self.api_base.endswith("/v1"):
            self.api_base += "/v1"
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers["Authorization"] = f"Bearer {api_key}"

    def _build_prompt(self, goal, action_space):
        acts = " ".join(f"{i}={d}" for i, d in action_space.items())
        return (
            f"Game: {goal} Actions: {acts}. "
            f"3 frames shown oldest to newest. "
            f"Reply with ONLY the action number."
        )

    def act(self, frames, goal, action_space):
        """frames: list of 1-3 PIL images (oldest first)."""
        prompt = self._build_prompt(goal, action_space)
        content = []
        labels = ["t-2", "t-1", "now"]
        start = 3 - len(frames)
        for i, f in enumerate(frames):
            label = labels[start + i]
            content.append({"type": "text", "text": f"[{label}]"})
            content.append({"type": "image_url",
                            "image_url": {"url": pil_to_data_uri(f)}})
        content.append({"type": "text", "text": prompt})
        messages = [{"role": "user", "content": content}]
        body = {
            "model": self.model, "messages": messages,
            "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs": {"enable_thinking": False},
        }
        resp = self.session.post(
            f"{self.api_base}/chat/completions",
            json=body, timeout=120,
        )
        resp.raise_for_status()
        tok = resp.json()["choices"][0]["message"]["content"]
        tok = tok.strip()
        valid = list(action_space.keys())
        try:
            action = int(tok)
        except ValueError:
            action = valid[0]
        if action not in valid:
            action = min(valid, key=lambda a: abs(a - action))
        return action, tok

    def run_episode(self, env_name, max_steps=None,
                    save_dir=None, verbose=True, display=True,
                    video_path=None):
        env, config = make_env(env_name)
        if max_steps is None:
            max_steps = config["max_steps"]
        action_space = config["actions"]
        obs, info = env.reset()
        history, total_reward = [], 0.0
        vid_frames = [] if video_path else None
        frame_buf = []  # last 3 frames for velocity estimation
        start = time.time()
        disp = None
        if display:
            disp = GameDisplay(title=f"VLM-Gym: {env_name}")
        if verbose:
            print(f"\n{'='*50}")
            print(f"[direct] {env_name} | {config['goal']}")
            print(f"{'='*50}\n")
        step = 0
        while max_steps == 0 or step < max_steps:
            frame = render_to_pil(env, scale=1.0)
            if vid_frames is not None:
                vid_frames.append(frame.copy())
            frame_buf.append(frame)
            if len(frame_buf) > 3:
                frame_buf.pop(0)
            if disp:
                disp.update(frame)
            t0 = time.time()
            action, tok = self.act(
                frame_buf, config["goal"], action_space
            )
            dt = time.time() - t0
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            history.append({
                "step": step, "action": action,
                "reward": reward, "inference_time": dt,
            })
            if verbose:
                name = action_space.get(action, "?")
                print(f"Step {step:3d} | {name} | "
                      f"r={reward:+.1f} total={total_reward:+.1f}"
                      f" | {dt:.2f}s | tok={tok}")
            if term or trunc:
                break
            step += 1
        elapsed = time.time() - start
        if verbose:
            print(f"\nDone: {len(history)} steps, "
                  f"reward={total_reward:+.1f}, {elapsed:.1f}s")
        if save_dir:
            p = Path(save_dir)
            p.mkdir(parents=True, exist_ok=True)
            f = p / f"{env_name}_{int(start)}.json"
            f.write_text(json.dumps({
                "env": env_name, "steps": len(history),
                "total_reward": total_reward,
                "elapsed_seconds": elapsed, "history": history,
            }, indent=2, default=str))
        if vid_frames and video_path:
            save_video(vid_frames, video_path)
            if verbose:
                print(f"Video saved: {video_path}")
        if disp:
            disp.close()
        env.close()
        return {"env": env_name, "steps": len(history),
                "total_reward": total_reward}
