"""Direct VLM agent — single token output for minimum latency."""

import base64
import io
import json
import time
from pathlib import Path

import requests
from PIL import Image

from vlm_gym.envs import make_env, render_to_pil, GameDisplay


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

    def _build_prompt(self, goal, action_space, history):
        acts = " ".join(f"{i}={d}" for i, d in action_space.items())
        hist = ""
        if history:
            recent = history[-5:]
            hist = " Last: " + ", ".join(
                f"{h['action']}" for h in recent
            )
        return (
            f"Game: {goal} Actions: {acts}.{hist} "
            f"Reply with ONLY the action number."
        )

    def act(self, frame, goal, action_space, history):
        prompt = self._build_prompt(goal, action_space, history)
        data_uri = pil_to_data_uri(frame)
        messages = [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": data_uri}},
            {"type": "text", "text": prompt},
        ]}]
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
                    save_dir=None, verbose=True, display=True):
        env, config = make_env(env_name)
        max_steps = max_steps or config["max_steps"]
        action_space = config["actions"]
        obs, info = env.reset()
        history, total_reward = [], 0.0
        start = time.time()
        disp = None
        if display:
            disp = GameDisplay(title=f"VLM-Gym: {env_name}")
        if verbose:
            print(f"\n{'='*50}")
            print(f"[direct] {env_name} | {config['goal']}")
            print(f"{'='*50}\n")
        for step in range(max_steps):
            frame = render_to_pil(env)
            if disp:
                disp.update(frame)
            t0 = time.time()
            action, tok = self.act(
                frame, config["goal"], action_space, history
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
        if disp:
            disp.close()
        env.close()
        return {"env": env_name, "steps": len(history),
                "total_reward": total_reward}
