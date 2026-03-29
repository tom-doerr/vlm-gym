"""VLM agent that plays Gymnasium environments using DSPy."""

import json
import time
from pathlib import Path

import dspy
from PIL import Image
from dspy.utils.exceptions import AdapterParseError

from vlm_gym.envs import make_env, render_to_pil, GameDisplay, save_video as _save_video


class GameAction(dspy.Signature):
    """You are an AI playing a game. Analyze the frame and choose an action."""

    frame: dspy.Image = dspy.InputField(desc="current game frame")
    goal: str = dspy.InputField(desc="the goal of the game")
    action_space: str = dspy.InputField(desc="available actions")
    history: str = dspy.InputField(desc="recent actions and rewards")
    action: int = dspy.OutputField(desc="action index to take")


class VLMAgent:
    """Agent that uses a VLM to play Gymnasium environments."""

    def __init__(
        self,
        model: str,
        api_base: str,
        api_key: str = "none",
        temperature: float = 0.7,
        cot: bool = True,
        history_steps: int = 10,
    ):
        self.lm = dspy.LM(
            model,
            api_base=api_base,
            api_key=api_key,
            temperature=temperature,
            max_tokens=1024,
            extra_body={
                "chat_template_kwargs": {"enable_thinking": False}
            },
        )
        dspy.configure(lm=self.lm)
        self.use_cot = cot
        self.history_steps = max(0, int(history_steps))
        self.predict = (
            dspy.ChainOfThought(GameAction)
            if cot else dspy.Predict(GameAction)
        )

    def _format_history(self, history, action_space):
        if not history or self.history_steps == 0:
            return "none yet"
        lines = []
        for h in history[-self.history_steps:]:
            name = action_space.get(h["action"], "?")
            lines.append(f"  step {h['step']}: {name}, r={h['reward']}")
        return "\n".join(lines)

    def _fallback_action(self, action_space):
        for action, desc in action_space.items():
            if "nothing" in desc.lower():
                return action
        if 0 in action_space:
            return 0
        return next(iter(action_space))

    def act(self, frame, goal, action_space, history):
        """Given a game frame, return an action index."""
        action_desc = "\n".join(
            f"  {i}: {d}" for i, d in action_space.items()
        )
        try:
            result = self.predict(
                frame=dspy.Image(frame),
                goal=goal,
                action_space=action_desc,
                history=self._format_history(history, action_space),
            )
            action = int(result.action)
        except (AdapterParseError, AttributeError, TypeError, ValueError) as e:
            fallback = self._fallback_action(action_space)
            print(f"[warn] DSPy parse failed ({e.__class__.__name__}); "
                  f"falling back to action {fallback} "
                  f"({action_space.get(fallback, '?')})")
            return fallback
        valid = list(action_space.keys())
        if action not in valid:
            action = min(valid, key=lambda a: abs(a - action))
        return action

    def run_episode(self, env_name, max_steps=None, save_dir=None,
                    verbose=True, display=True, video_path=None):
        """Run one episode. Returns result dict."""
        env, config = make_env(env_name)
        if max_steps is None:
            max_steps = config["max_steps"]
        action_space = config["actions"]
        obs, info = env.reset()
        history, total_reward = [], 0.0
        frames = [] if video_path else None
        start = time.time()
        disp = GameDisplay(title=f"VLM-Gym: {env_name}") if display else None
        if verbose:
            print(f"\n{'='*50}")
            print(f"Episode: {env_name} | Goal: {config['goal']}")
            print(f"{'='*50}\n")
        step = 0
        while max_steps == 0 or step < max_steps:
            frame = render_to_pil(env, scale=1.0)
            if frames is not None:
                frames.append(frame.copy())
            if disp:
                disp.update(frame)
            t0 = time.time()
            action = self.act(
                frame, config["goal"], action_space, history
            )
            dt = time.time() - t0
            obs, reward, term, trunc, info = env.step(action)
            total_reward += reward
            history.append({
                "step": step, "action": action,
                "reward": reward,
                "inference_time": dt,
            })
            if verbose:
                name = action_space.get(action, "?")
                print(f"Step {step:3d} | {name} | "
                      f"r={reward:+.1f} total={total_reward:+.1f} "
                      f"| {dt:.1f}s")
            if term or trunc:
                break
            step += 1
        elapsed = time.time() - start
        result = {
            "env": env_name, "steps": len(history),
            "total_reward": total_reward,
            "elapsed_seconds": elapsed, "history": history,
        }
        if verbose:
            print(f"\nDone: {len(history)} steps, "
                  f"reward={total_reward:+.1f}, {elapsed:.1f}s")
        if frames and video_path:
            _save_video(frames, video_path)
            if verbose:
                print(f"Video saved: {video_path}")
        if save_dir:
            self._save_log(result, save_dir, env_name, start)
        if disp:
            disp.close()
        env.close()
        return result

    def _save_log(self, result, save_dir, env_name, start):
        p = Path(save_dir)
        p.mkdir(parents=True, exist_ok=True)
        f = p / f"{env_name}_{int(start)}.json"
        f.write_text(json.dumps(result, indent=2, default=str))
