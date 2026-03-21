# VLM-Gym

VLM agents playing OpenAI Gymnasium environments via DSPy.

## Architecture

- `run.py` — CLI entry point, `--direct` for single-token mode
- `vlm_gym/agent.py` — DSPy-based agent (ChatAdapter)
- `vlm_gym/direct_agent.py` — Raw API single-token agent
- `vlm_gym/envs.py` — Env configs, rendering, display, video

## Optimization

- `optimize.py` — MIPROv2/SIMBAT (`--optimizer`, `--env`)
- `prism_play.py` — Online PRISM (Thompson-sampled knowledge)
- `prism_logprob.py` — **Best**: offline PRISM + logprobs
  - β=+1.12 for "press gas" → reward -13 → +121
- `run_prism_optimized.py` — Run with optimized knowledge

## Key Findings

- **Lasso→Ridge fix** was root cause of β=0 in all PRISM runs.
  Lasso (L1) zeroed small effects; Ridge (L2) preserves them.
  PRISM now defaults to `reg="ridge"` (configurable).
- Both reward-based and logprob-based PRISM work with Ridge.
  Reward: β≈+0.09 on CarRacing. Logprob: β≈+0.24.
- Logprob metric more sensitive (detects piece effects even
  when argmax action unchanged). Reward works too.
- `_GenKnowledge` output simplified to `list[str]` (was nested
  Pydantic `NewKnowledge` model that small models couldn't produce)
- `enable_thinking: False` disables Qwen3.5 thinking mode
- DSPy fork from `~/git/dspy` (custom branch) for PRISM
