#!/usr/bin/env python3
"""PRISM + logprob optimization for game agents.

Collect rollouts, then optimize knowledge pieces by
maximizing logprob of good actions."""

import argparse
import json
import logging
import time
import random
import math

import numpy as np
import requests
import dspy

from vlm_gym.envs import (
    detect_model, make_env, render_to_pil, ENV_CONFIGS,
)
from vlm_gym.direct_agent import pil_to_data_uri
from dspy.teleprompt.prism import (
    _Piece, _CreditModel, _sample, _build,
    _GenKnowledge, KnowledgePool, KnowledgePiece,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


INITIAL_KNOWLEDGE = [
    "If the road curves left, steer left (action 1)",
    "If the road curves right, steer right (action 2)",
    "On straight road, press gas (action 3)",
    "Brake before sharp turns (action 4)",
    "Stay on the dark gray road surface",
    "Action 3 is gas — use it to move forward",
]


def collect_rollout(env_name, n_steps=50):
    """Play with heuristic, collect (frame_uri, good_action)."""
    env, config = make_env(env_name)
    valid = list(config["actions"].keys())
    obs, _ = env.reset()
    rollout = []
    for _ in range(n_steps):
        frame = render_to_pil(env, scale=0.5)
        uri = pil_to_data_uri(frame)
        action = 3  # gas mostly
        obs, reward, t, tr, _ = env.step(action)
        rollout.append({"frame_uri": uri, "good_action": 3})
        if t or tr:
            obs, _ = env.reset()
    env.close()
    return rollout


def _build_msg(frame_uri, knowledge):
    prompt = (
        f"Strategy: {knowledge}\n"
        f"Actions: 0=nothing 1=left 2=right 3=gas 4=brake. "
        f"Reply with ONLY the action number."
    )
    return [{"role": "user", "content": [
        {"type": "image_url", "image_url": {"url": frame_uri}},
        {"type": "text", "text": prompt},
    ]}]


def get_action_logprob(session, api_base, model,
                       frame_uri, knowledge, target):
    """Get logprob of target action token."""
    body = {
        "model": model,
        "messages": _build_msg(frame_uri, knowledge),
        "max_tokens": 1, "temperature": 0,
        "logprobs": True, "top_logprobs": 5,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    resp = session.post(
        f"{api_base}/chat/completions",
        json=body, timeout=30)
    resp.raise_for_status()
    choice = resp.json()["choices"][0]
    lps = choice.get("logprobs", {}).get("content", [])
    if not lps:
        return -10.0
    for e in lps[0].get("top_logprobs", []):
        if e["token"].strip() == str(target):
            return e["logprob"]
    return -10.0


def _gen_pieces(pieces, gen, knowledge, reward):
    pool = KnowledgePool(items=[
        KnowledgePiece(content=p.content, beta=p.coef,
                       se=p.stderr, n=p.n_sel)
        for p in pieces])
    try:
        r = gen(pool=pool,
                rollout=f"logprob={reward:.2f} k={knowledge}")
        ext = {p.content for p in pieces}
        for s in getattr(r.new_knowledge, 'items', []):
            s = str(s).strip()
            if s and s not in ext:
                pieces.append(_Piece(s)); ext.add(s)
                print(f"  [+] {s[:60]}")
    except Exception as e:
        log.warning(f"Gen: {e}")


def _opt_step(rollout, pieces, cr, sess, api, model):
    ex = random.choice(rollout)
    sel = _sample(pieces) if pieces else []
    k = _build(pieces, sel) if sel else "none"
    lp = get_action_logprob(
        sess, api, model,
        ex["frame_uri"], k, ex["good_action"])
    sv = [1.0 if j in set(sel) else 0.0
          for j in range(len(pieces))]
    cr.add(sv, lp)
    cr.update(pieces)
    for j in sel:
        pieces[j].n_sel += 1
    return lp, k


def optimize(rollout, pieces, sess, api, model,
             steps=100, gen=None, gen_every=20):
    cr = _CreditModel()
    best_lp, best_k = -math.inf, ""
    for i in range(steps):
        lp, k = _opt_step(
            rollout, pieces, cr, sess, api, model)
        if lp > best_lp:
            best_lp, best_k = lp, k
        if (i+1) % 10 == 0:
            print(f"  step {i+1} lp={lp:.2f} "
                  f"best={best_lp:.2f}")
        if gen and (i+1) % gen_every == 0:
            _gen_pieces(pieces, gen, k, lp)
    return best_k, best_lp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="carracing")
    p.add_argument("--api-base",
                    default="http://localhost:8000/v1")
    p.add_argument("--rollout-steps", type=int, default=30)
    p.add_argument("--opt-steps", type=int, default=100)
    p.add_argument("--gen-every", type=int, default=20)
    args = p.parse_args()

    model = detect_model(args.api_base)
    api = args.api_base.rstrip("/")
    if not api.endswith("/v1"):
        api += "/v1"
    print(f"Model: {model}")
    sess = requests.Session()
    sess.headers["Authorization"] = "Bearer none"

    print(f"Collecting {args.rollout_steps} rollout frames...")
    rollout = collect_rollout(args.env, args.rollout_steps)
    print(f"Collected {len(rollout)} frames")

    pieces = [_Piece(s) for s in INITIAL_KNOWLEDGE]
    lm = dspy.LM(f"openai/{model}",
                  api_base=args.api_base, api_key="none",
                  max_tokens=500, temperature=0.7,
                  extra_body={"chat_template_kwargs":
                              {"enable_thinking": False}})
    dspy.configure(lm=lm)
    gen = dspy.Predict(_GenKnowledge)

    print(f"Optimizing {args.opt_steps} steps...")
    best_k, best_lp = optimize(
        rollout, pieces, sess, api, model,
        args.opt_steps, gen, args.gen_every)

    print(f"\nBest logprob: {best_lp:.3f}")
    print(f"Best knowledge:\n{best_k}")
    print("\nFinal pool:")
    for pc in sorted(pieces, key=lambda x: x.coef,
                     reverse=True):
        print(f"  β={pc.coef:+.3f} SE={pc.stderr:.3f}"
              f" n={pc.n_sel:2d} | {pc.content}")


if __name__ == "__main__":
    main()
