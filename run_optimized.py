#!/usr/bin/env python3
"""Run optimized DSPy agent."""

import json
import time
import dspy
import gymnasium as gym

from vlm_gym.envs import (
    detect_model, render_to_pil, GameDisplay, save_video,
)
from optimize import GameAction


def load_instruction(path="optimized_agent.json"):
    with open(path) as f:
        data = json.load(f)
    return data["predict_action"]["signature"]["instructions"]


def main():
    api_base = "http://localhost:8000/v1"
    model = detect_model(api_base)
    print(f"Model: {model}")
    lm = dspy.LM(
        f"openai/{model}",
        api_base=api_base, api_key="none",
        max_tokens=100, temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
    dspy.configure(lm=lm)

    instr = load_instruction()
    print(f"Instruction: {instr[:80]}...")

    # Build signature with optimized instruction
    predict = dspy.Predict(GameAction)
    predict.signature = predict.signature.with_instructions(instr)

    env = gym.make("CartPole-v1", render_mode="rgb_array")
    obs, _ = env.reset()
    disp = GameDisplay(title="Optimized VLM CartPole")
    total, frames = 0.0, []
    start = time.time()
    for step in range(500):
        frame = render_to_pil(env, scale=0.5)
        frames.append(frame.copy())
        disp.update(frame)
        try:
            r = predict(frame=dspy.Image(frame))
            action = int(r.action)
            if action not in [0, 1]:
                action = 0
        except Exception:
            action = 0
        obs, reward, t, tr, _ = env.step(action)
        total += reward
        dt = time.time() - start
        print(f"Step {step:3d} | act={action} | "
              f"r={total:+.0f} | {dt:.1f}s")
        if t or tr:
            break
    elapsed = time.time() - start
    print(f"\nDone: {step+1} steps, reward={total:+.0f}, "
          f"{elapsed:.1f}s")
    save_video(frames, "episodes/optimized_cartpole.mp4")
    print("Video saved: episodes/optimized_cartpole.mp4")
    disp.close()
    env.close()


if __name__ == "__main__":
    main()
