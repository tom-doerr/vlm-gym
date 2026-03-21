#!/usr/bin/env python3
"""PRISM-driven game agent — online RL via knowledge pool."""

import argparse
import logging
import time

import dspy
import numpy as np
import requests

from vlm_gym.envs import (
    detect_model, make_env, render_to_pil,
    GameDisplay, save_video, ENV_CONFIGS,
)
from vlm_gym.direct_agent import pil_to_data_uri
from dspy.teleprompt.prism import (
    _Piece, _CreditModel, _sample, _build,
    _GenKnowledge, KnowledgePool, KnowledgePiece,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


class GameAction(dspy.Signature):
    """Play a game. Use the strategy knowledge to decide."""

    frame: dspy.Image = dspy.InputField(desc="game screenshot")
    knowledge: str = dspy.InputField(desc="strategy tips")
    action: int = dspy.OutputField(desc="action index")


INITIAL_KNOWLEDGE = [
    "If the road curves left, steer left",
    "If the road curves right, steer right",
    "Gas on straight sections",
    "Brake before sharp turns",
    "Stay on the dark road surface",
    "Push left if pole leans left, right if right",
]


def play_episode(env_name, predict, pieces, credit,
                 gen, gen_every, max_steps, display, gen_lm):
    env, config = make_env(env_name)
    valid = list(config["actions"].keys())
    obs, _ = env.reset()
    disp = GameDisplay(title=f"PRISM: {env_name}") if display else None
    total, frames, start = 0.0, [], time.time()
    sv_buf, rw_buf = [], []  # selection vectors + raw rewards
    gamma = 0.9

    print(f"\n{'='*50}")
    print(f"PRISM: {env_name} | pool={len(pieces)} pieces")
    print(f"{'='*50}\n")

    for step in range(max_steps):
        frame = render_to_pil(env, scale=1.0)
        frames.append(frame.copy())
        if disp:
            disp.update(frame)

        # PRISM: sample knowledge
        sel = _sample(pieces) if pieces else []
        knowledge = _build(pieces, sel) if sel else "none"

        # Get action from VLM
        try:
            r = predict(
                frame=dspy.Image(frame), knowledge=knowledge)
            action = int(r.action)
            if action not in valid:
                action = valid[0]
        except Exception:
            action = valid[0]

        obs, reward, t, tr, _ = env.step(action)
        total += reward

        # PRISM: buffer selection + reward
        sv = [1.0 if i in set(sel) else 0.0
              for i in range(len(pieces))]
        sv_buf.append(sv)
        rw_buf.append(reward)
        for i in sel:
            pieces[i].n_sel += 1

        # Refit with discounted returns every step
        from prism_logprob import discount_rewards
        dr = discount_rewards(rw_buf, gamma)
        credit.X = sv_buf[:]
        credit.y = dr.tolist()
        credit.update(pieces)

        name = config["actions"].get(action, "?")
        print(f"Step {step:3d} | {name} | "
              f"r={reward:+.1f} total={total:+.1f} | "
              f"knowledge={len(sel)} pieces")

        # PRISM: generate new knowledge
        if gen and (step+1) % gen_every == 0:
            _do_gen(pieces, gen, knowledge, reward, gen_lm)

        if t or tr:
            break

    elapsed = time.time() - start
    print(f"\nDone: {step+1} steps, reward={total:+.1f}, "
          f"{elapsed:.1f}s, pool={len(pieces)} pieces")

    # Print final piece stats
    print("\nKnowledge pool:")
    for p in sorted(pieces, key=lambda p: p.coef, reverse=True):
        print(f"  β={p.coef:+.3f} SE={p.stderr:.3f} "
              f"n={p.n_sel:2d} | {p.content}")

    if frames:
        save_video(frames, f"episodes/prism_{env_name}.mp4")
    if disp:
        disp.close()
    env.close()
    return total


def _do_gen(pieces, gen, knowledge, reward, gen_lm):
    pool = KnowledgePool(items=[
        KnowledgePiece(content=p.content, beta=p.coef,
                       se=p.stderr, n=p.n_sel)
        for p in pieces])
    kw = {"pool": pool,
          "rollout": f"r={reward:.1f} k={knowledge}"}
    try:
        if gen_lm:
            with dspy.context(lm=gen_lm):
                r = gen(**kw)
        else:
            r = gen(**kw)
        new = r.new_knowledge if isinstance(
            r.new_knowledge, list) else []
        ext = {p.content for p in pieces}
        for s in new:
            s = str(s).strip()
            if len(s) > 3 and s not in ext:
                pieces.append(_Piece(s)); ext.add(s)
                print(f"  [+] {s[:60]}")
    except Exception as e:
        log.warning(f"Gen: {e}")


def main():
    p = argparse.ArgumentParser(description="PRISM game agent")
    p.add_argument("--env", default="carracing",
                    choices=list(ENV_CONFIGS.keys()))
    p.add_argument("--api-base",
                    default="http://192.168.110.2:8000/v1")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument("--gen-every", type=int, default=15)
    p.add_argument("--no-display", action="store_true")
    p.add_argument("--gen-api-base", default=None)
    args = p.parse_args()

    model = detect_model(args.api_base)
    print(f"Model: {model}")
    lm = dspy.LM(
        f"openai/{model}",
        api_base=args.api_base, api_key="none",
        max_tokens=100, temperature=0.3,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}},
    )
    dspy.configure(lm=lm)

    gb = args.gen_api_base or args.api_base
    gm = detect_model(gb)
    print(f"Gen model: {gm} @ {gb}")
    gen_lm = dspy.LM(
        f"openai/{gm}", api_base=gb, api_key="none",
        max_tokens=1000, temperature=0.7,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False}})

    predict = dspy.Predict(GameAction)
    pieces = [_Piece(s) for s in INITIAL_KNOWLEDGE]
    credit = _CreditModel()
    gen = dspy.Predict(_GenKnowledge)

    play_episode(
        args.env, predict, pieces, credit, gen,
        args.gen_every, args.max_steps,
        not args.no_display, gen_lm=gen_lm,
    )


if __name__ == "__main__":
    main()
