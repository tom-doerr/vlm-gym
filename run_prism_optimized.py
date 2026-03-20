#!/usr/bin/env python3
"""Run CarRacing with PRISM-optimized knowledge."""

import time
import requests
from PIL import Image

from vlm_gym.envs import (
    detect_model, make_env, render_to_pil,
    GameDisplay, save_video,
)
from vlm_gym.direct_agent import pil_to_data_uri

KNOWLEDGE = "On straight road, press gas (action 3)"
API = "http://localhost:8000/v1"
PROMPT = (f"Strategy: {KNOWLEDGE}\n"
          "Actions: 0=nothing 1=left 2=right"
          " 3=gas 4=brake."
          " Reply with ONLY the action number.")


def act(sess, model, frame_uri):
    msg = [{"role": "user", "content": [
        {"type": "image_url",
         "image_url": {"url": frame_uri}},
        {"type": "text", "text": PROMPT}]}]
    body = {"model": model, "messages": msg,
            "max_tokens": 1, "temperature": 0,
            "chat_template_kwargs":
                {"enable_thinking": False}}
    r = sess.post(f"{API}/chat/completions",
                  json=body, timeout=30)
    r.raise_for_status()
    tok = r.json()["choices"][0]["message"]["content"]
    try:
        return int(tok.strip()) % 5
    except ValueError:
        return 0


def main():
    model = detect_model(API)
    print(f"Model: {model}")
    print(f"Knowledge: {KNOWLEDGE}")
    sess = requests.Session()
    sess.headers["Authorization"] = "Bearer none"
    env, config = make_env("carracing")
    actions = config["actions"]
    obs, _ = env.reset()
    disp = GameDisplay(title="PRISM-optimized CarRacing")
    total, frames = 0.0, []
    start = time.time()
    for step in range(200):
        frame = render_to_pil(env, scale=0.5)
        frames.append(frame.copy())
        disp.update(frame)
        uri = pil_to_data_uri(frame)
        action = act(sess, model, uri)
        obs, reward, t, tr, _ = env.step(action)
        total += reward
        name = actions.get(action, "?")
        print(f"Step {step:3d} | {name} | "
              f"r={reward:+.1f} total={total:+.1f}")
        if t or tr:
            break
    elapsed = time.time() - start
    print(f"\nDone: {step+1} steps, "
          f"reward={total:+.1f}, {elapsed:.1f}s")
    save_video(frames,
               "episodes/prism_optimized_carracing.mp4")
    disp.close()
    env.close()


if __name__ == "__main__":
    main()
