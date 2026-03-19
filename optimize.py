#!/usr/bin/env python3
"""Optimize VLM game agent with MIPROv2."""

import argparse
import gymnasium as gym
import numpy as np
from PIL import Image

import dspy
from dspy.teleprompt import MIPROv2

from vlm_gym.envs import detect_model, render_to_pil


class GameAction(dspy.Signature):
    """Look at the game screenshot and choose the best action."""

    frame: dspy.Image = dspy.InputField(desc="game screenshot")
    action: int = dspy.OutputField(desc="action index")


def cartpole_heuristic(obs):
    """Push toward pole lean."""
    _, _, angle, _ = obs
    return 1 if angle > 0 else 0


def collect_examples(n=50):
    """Collect frames + heuristic actions."""
    env = gym.make("CartPole-v1", render_mode="rgb_array")
    examples = []
    while len(examples) < n:
        obs, _ = env.reset()
        for _ in range(500):
            frame = render_to_pil(env, scale=0.5)
            a = cartpole_heuristic(obs)
            examples.append(dspy.Example(
                frame=frame, action=a,
            ).with_inputs("frame"))
            obs, _, t, tr, _ = env.step(a)
            if t or tr or len(examples) >= n:
                break
    env.close()
    return examples[:n]


def metric(example, prediction, trace=None):
    try:
        return int(prediction.action) == int(example.action)
    except (ValueError, TypeError):
        return False


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--api-base",
                    default="http://localhost:8000/v1")
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--save", default="optimized_agent.json")
    args = p.parse_args()

    model_name = detect_model(args.api_base)
    print(f"Task model: {model_name}")
    task_lm = dspy.LM(
        f"openai/{model_name}",
        api_base=args.api_base,
        api_key="none",
        max_tokens=100,
        temperature=0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
    # Use spark-2 (122B) for prompt generation
    spark2 = "http://192.168.110.2:8000/v1"
    pm_name = detect_model(spark2)
    print(f"Prompt model: {pm_name}")
    prompt_lm = dspy.LM(
        f"openai/{pm_name}",
        api_base=spark2, api_key="none",
        max_tokens=1000, temperature=0.7,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
    dspy.configure(lm=task_lm)

    print(f"Collecting {args.n_train} training examples...")
    trainset = collect_examples(args.n_train)
    print(f"Collected {len(trainset)} examples")

    student = dspy.Predict(GameAction)
    optimizer = MIPROv2(
        metric=metric,
        prompt_model=prompt_lm,
        task_model=task_lm,
        auto="light",
        num_threads=1,
        verbose=True,
    )
    print("Starting MIPROv2 optimization...")
    optimized = optimizer.compile(
        student, trainset=trainset,
    )
    optimized.save(args.save)
    print(f"Saved optimized agent to {args.save}")


if __name__ == "__main__":
    main()
