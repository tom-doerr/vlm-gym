"""Environment configurations and helpers for VLM-Gym."""

import gymnasium as gym
import numpy as np
from PIL import Image


ENV_CONFIGS = {
    "cartpole": {
        "id": "CartPole-v1",
        "actions": {0: "push cart LEFT", 1: "push cart RIGHT"},
        "goal": "Keep the pole balanced upright on the cart.",
        "max_steps": 500,
    },
    "mountaincar": {
        "id": "MountainCar-v0",
        "actions": {0: "push LEFT", 1: "do NOTHING", 2: "push RIGHT"},
        "goal": "Drive the car up the right hill to reach the flag.",
        "max_steps": 200,
    },
    "acrobot": {
        "id": "Acrobot-v1",
        "actions": {0: "apply +1 torque", 1: "apply 0 torque", 2: "apply -1 torque"},
        "goal": "Swing the free end of the two-link robot above the base.",
        "max_steps": 500,
    },
}


def make_env(env_name: str) -> tuple[gym.Env, dict]:
    """Create a gymnasium environment with rgb_array rendering."""
    if env_name not in ENV_CONFIGS:
        raise ValueError(
            f"Unknown env '{env_name}'. Choose from: {list(ENV_CONFIGS.keys())}"
        )
    config = ENV_CONFIGS[env_name]
    env = gym.make(config["id"], render_mode="rgb_array")
    return env, config


def render_to_pil(env: gym.Env, scale: float = 0.5) -> Image.Image:
    """Render the env to a scaled PIL Image for the VLM."""
    frame = env.render()
    img = Image.fromarray(frame)
    if scale != 1.0:
        new_size = (int(img.width * scale), int(img.height * scale))
        img = img.resize(new_size, Image.LANCZOS)
    return img
