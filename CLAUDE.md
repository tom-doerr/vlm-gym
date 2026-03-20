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

- Logprob PRISM >> reward-based optimizers for weak models
- 0.8B ignores prompts with reward metric; logprobs work
- `enable_thinking: False` disables Qwen3.5 thinking mode
- DSPy fork from `~/git/dspy` (custom branch) for PRISM
