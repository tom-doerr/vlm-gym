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


INITIAL_KNOWLEDGE = []


def discount_rewards(rewards, gamma=0.9):
    """Backward-propagate discounted rewards."""
    disc = np.zeros(len(rewards))
    running = 0.0
    for t in reversed(range(len(rewards))):
        running = rewards[t] + gamma * running
        disc[t] = running
    return disc


def collect_rollout(env_name, n_steps=50, gamma=0.9):
    """Play with heuristic, collect frames + discounted rewards."""
    env, config = make_env(env_name)
    obs, _ = env.reset()
    raw = []
    for _ in range(n_steps):
        frame = render_to_pil(env, scale=0.5)
        uri = pil_to_data_uri(frame)
        valid = list(config["actions"].keys())
        action = random.choice(valid)
        obs, reward, t, tr, _ = env.step(action)
        raw.append({"frame_uri": uri,
                     "action": action, "reward": reward})
        if t or tr:
            obs, _ = env.reset()
    env.close()
    rewards = [r["reward"] for r in raw]
    disc = discount_rewards(rewards, gamma)
    for i, r in enumerate(raw):
        r["disc_reward"] = float(disc[i])
    return raw


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


def _gen_pieces(pieces, gen, knowledge, reward,
                gen_lm=None, env_desc=""):
    pool = KnowledgePool(items=[
        KnowledgePiece(content=p.content, beta=p.coef,
                       se=p.stderr, n=p.n_sel)
        for p in pieces])
    rollout = (f"{env_desc}\n"
               f"Score: logprob={reward:.2f}\n"
               f"Knowledge used: {knowledge or 'none'}")
    kw = {"pool": pool, "rollout": rollout}
    if gen_lm:
        kw["lm"] = gen_lm
    try:
        r = gen(**kw)
        ext = {p.content for p in pieces}
        for s in getattr(r.new_knowledge, 'items', []):
            s = str(s).strip()
            if s and s not in ext:
                pieces.append(_Piece(s)); ext.add(s)
                print(f"  [+] {s[:60]}")
    except Exception as e:
        log.warning(f"Gen: {e}")


def _opt_step(rollout, pieces, cr, sess, api, model):
    # Only use positive-reward frames
    pos = [e for e in rollout if e["disc_reward"] > 0]
    if not pos:
        pos = rollout
    ex = random.choice(pos)
    sel = _sample(pieces) if pieces else []
    k = _build(pieces, sel) if sel else "none"
    lp = get_action_logprob(
        sess, api, model,
        ex["frame_uri"], k, ex["action"])
    # Logprob only — reward filtering already selects good frames
    score = lp
    if pieces:
        sv = [1.0 if j in set(sel) else 0.0
              for j in range(len(pieces))]
        cr.add(sv, score)
        cr.update(pieces)
    for j in sel:
        pieces[j].n_sel += 1
    return score, k


def optimize(rollout, pieces, sess, api, model,
             steps=100, gen=None, gen_every=20,
             gen_lm=None, env_desc=""):
    cr = _CreditModel()
    best_lp, best_k = -math.inf, ""
    for i in range(steps):
        lp, k = _opt_step(
            rollout, pieces, cr, sess, api, model)
        if lp > best_lp:
            best_lp, best_k = lp, k
        if (i+1) % 10 == 0:
            print(f"  step {i+1} lp={lp:.2f} "
                  f"best={best_lp:.2f} pool={len(pieces)}")
        if gen and (i+1) % gen_every == 0:
            _gen_pieces(pieces, gen, k, lp, gen_lm, env_desc)
    return best_k, best_lp


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--env", default="carracing")
    p.add_argument("--api-base",
                    default="http://localhost:8000/v1")
    p.add_argument("--rollout-steps", type=int, default=30)
    p.add_argument("--opt-steps", type=int, default=100)
    p.add_argument("--gen-every", type=int, default=20)
    p.add_argument("--gamma", type=float, default=0.9)
    p.add_argument("--gen-api-base", default=None,
                    help="separate endpoint for knowledge gen")
    args = p.parse_args()

    model = detect_model(args.api_base)
    api = args.api_base.rstrip("/")
    if not api.endswith("/v1"):
        api += "/v1"
    print(f"Model: {model}")
    sess = requests.Session()
    sess.headers["Authorization"] = "Bearer none"

    print(f"Collecting {args.rollout_steps} frames (γ={args.gamma})...")
    rollout = collect_rollout(args.env, args.rollout_steps, args.gamma)
    acts = [r["action"] for r in rollout]
    print(f"Collected {len(rollout)} frames, "
          f"actions: {dict((a, acts.count(a)) for a in set(acts))}")

    pieces = [_Piece(s) for s in INITIAL_KNOWLEDGE]
    dspy.configure(lm=dspy.LM(f"openai/{model}",
                  api_base=args.api_base, api_key="none",
                  max_tokens=100, temperature=0))
    # Use separate LM for knowledge generation
    gen_base = args.gen_api_base or args.api_base
    gen_model = detect_model(gen_base)
    print(f"Gen model: {gen_model} @ {gen_base}")
    gen_lm = dspy.LM(f"openai/{gen_model}",
                  api_base=gen_base, api_key="none",
                  max_tokens=500, temperature=0.7,
                  extra_body={"chat_template_kwargs":
                              {"enable_thinking": False}})
    gen = dspy.Predict(_GenKnowledge)

    print(f"Optimizing {args.opt_steps} steps...")
    config = ENV_CONFIGS[args.env]
    acts = " ".join(f"{k}={v}" for k, v in config["actions"].items())
    env_desc = (f"Game: {config['id']} — {config['goal']}\n"
                f"Actions: {acts}")
    print(f"Env: {env_desc}")

    best_k, best_lp = optimize(
        rollout, pieces, sess, api, model,
        args.opt_steps, gen, args.gen_every, gen_lm, env_desc)

    print(f"\nBest logprob: {best_lp:.3f}")
    print(f"Best knowledge:\n{best_k}")
    print("\nFinal pool:")
    for pc in sorted(pieces, key=lambda x: x.coef,
                     reverse=True):
        print(f"  β={pc.coef:+.3f} SE={pc.stderr:.3f}"
              f" n={pc.n_sel:2d} | {pc.content}")


if __name__ == "__main__":
    main()
