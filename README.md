# VLM-Gym

Use a Vision Language Model to play OpenAI Gymnasium environments.
Renders game frames, sends them to a VLM via DSPy, and executes
the returned actions in a loop.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Usage

```bash
# Defaults: CartPole on spark-2 SGLang
python run.py

# Different environment
python run.py --env mountaincar

# Custom VLM endpoint
python run.py --api-base http://localhost:8000/v1 --model openai/MyModel

# Multiple episodes
python run.py --episodes 5 --max-steps 50
```

## Environments

- `cartpole` — balance a pole on a cart (2 actions)
- `mountaincar` — drive up a hill (3 actions)
- `acrobot` — swing a robot arm up (3 actions)
