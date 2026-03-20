#!/usr/bin/env python3
"""Optimize VLM game agent with MIPROv2."""

import argparse
import gymnasium as gym
import numpy as np
from PIL import Image

import dspy
from dspy.teleprompt import MIPROv2, SIMBAT

from vlm_gym.envs import (
    detect_model, render_to_pil, ENV_CONFIGS, make_env,
)


class GameAction(dspy.Signature):
    """Look at the game screenshot and choose the best action."""

    frame: dspy.Image = dspy.InputField(desc="game screenshot")
    action: int = dspy.OutputField(desc="action index")

ROLLOUT_STEPS = 30  # steps per episode during optimization


class GamePlayer(dspy.Module):
    """Runs an episode, returns reward."""

    def __init__(self, env_name):
        self.env_name = env_name
        self.predict = dspy.Predict(GameAction)
        config = ENV_CONFIGS[env_name]
        self.valid_actions = list(config["actions"].keys())

    def _act(self, frame):
        try:
            r = self.predict(frame=dspy.Image(frame))
            a = int(r.action)
            if a not in self.valid_actions:
                a = self.valid_actions[0]
            return a
        except Exception:
            return self.valid_actions[0]

    def forward(self, seed):
        env, _ = make_env(self.env_name)
        env.reset(seed=int(seed))
        total = 0.0
        for _ in range(ROLLOUT_STEPS):
            frame = render_to_pil(env, scale=0.5)
            action = self._act(frame)
            _, reward, t, tr, _ = env.step(action)
            total += reward
            if t or tr:
                break
        env.close()
        return dspy.Prediction(reward=total)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="cartpole",
                    choices=list(ENV_CONFIGS.keys()))
    p.add_argument("--api-base",
                    default="http://localhost:8000/v1")
    p.add_argument("--prompt-api-base", default=None,
                    help="separate endpoint for prompt model")
    p.add_argument("--n-train", type=int, default=50)
    p.add_argument("--save", default=None)
    p.add_argument("--optimizer", default="mipro",
                    choices=["mipro", "simbat"])
    args = p.parse_args()

    model_name = detect_model(args.api_base)
    print(f"Task model: {model_name}")
    task_lm = dspy.LM(
        f"openai/{model_name}",
        api_base=args.api_base,
        api_key="none",
        max_tokens=100,
        temperature=1.0,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
    pm_base = args.prompt_api_base or args.api_base
    pm_name = detect_model(pm_base)
    print(f"Prompt model: {pm_name} @ {pm_base}")
    prompt_lm = dspy.LM(
        f"openai/{pm_name}",
        api_base=pm_base, api_key="none",
        max_tokens=1000, temperature=0.7,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}
        },
    )
    dspy.configure(lm=task_lm)

    save_path = args.save or f"optimized_{args.env}.json"

    # Trainset: episode seeds. Metric: normalized reward.
    trainset = [
        dspy.Example(seed=i, reward=ROLLOUT_STEPS
        ).with_inputs("seed")
        for i in range(args.n_train)
    ]
    def metric(ex, pred, trace=None):
        return float(pred.reward) / ROLLOUT_STEPS

    student = GamePlayer(args.env)
    if args.optimizer == "simbat":
        optimizer = SIMBAT(
            metric=metric,
            prompt_model=prompt_lm,
            bsize=min(4, len(trainset)),
            num_candidates=3,
            max_steps=4,
            max_demos=0,
            num_threads=1,
        )
    else:
        optimizer = MIPROv2(
            metric=metric,
            prompt_model=prompt_lm,
            task_model=task_lm,
            auto="light",
            num_threads=1,
            verbose=True,
        )
    print(f"Starting {args.optimizer} optimization...")
    optimized = optimizer.compile(
        student, trainset=trainset,
    )
    optimized.save(save_path)
    print(f"Saved optimized agent to {save_path}")


if __name__ == "__main__":
    main()
