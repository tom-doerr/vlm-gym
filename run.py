#!/usr/bin/env python3
"""CLI entry point for VLM-Gym."""

import argparse
from vlm_gym.envs import ENV_CONFIGS, detect_model


def main():
    p = argparse.ArgumentParser(description="VLM plays Gym")
    p.add_argument("--env", default="cartpole",
                    choices=list(ENV_CONFIGS.keys()))
    p.add_argument("--model", default=None,
                    help="auto-detected from endpoint if omitted")
    p.add_argument("--api-base",
                    default="http://192.168.110.2:8000/v1")
    p.add_argument("--api-key", default="none")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--max-steps", type=int, default=None,
                    help="maximum steps per episode; use 0 for no cap")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--save-dir", default="episodes")
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--direct", action="store_true",
                    help="single-token mode (no DSPy, min latency)")
    p.add_argument("--no-cot", action="store_true",
                    help="disable DSPy Chain-of-Thought wrapper")
    p.add_argument("--history-steps", type=int, default=10,
                    help="number of prior action/reward steps to include in DSPy history")
    p.add_argument("--no-video", action="store_true",
                    help="disable automatic video recording")
    args = p.parse_args()

    raw_model = args.model or detect_model(args.api_base)
    print(f"Model: {raw_model}")

    if args.direct:
        from vlm_gym.direct_agent import DirectAgent
        agent = DirectAgent(raw_model, args.api_base, args.api_key)
    else:
        from vlm_gym.agent import VLMAgent
        agent = VLMAgent(
            "openai/" + raw_model, args.api_base,
            args.api_key, args.temperature,
            cot=not args.no_cot,
            history_steps=args.history_steps,
        )
    import time as _t
    for ep in range(args.episodes):
        if args.episodes > 1:
            print(f"\n--- Episode {ep+1}/{args.episodes} ---")
        vpath = None
        if not args.no_video:
            from pathlib import Path
            Path(args.save_dir).mkdir(parents=True, exist_ok=True)
            vpath = f"{args.save_dir}/{args.env}_{int(_t.time())}.mp4"
        agent.run_episode(
            args.env, max_steps=args.max_steps,
            save_dir=args.save_dir,
            display=not args.no_display,
            video_path=vpath,
        )


if __name__ == "__main__":
    main()
