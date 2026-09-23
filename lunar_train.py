import argparse
import os
from pathlib import Path

# SDL (pygame's backend for the render window) otherwise catches SIGTERM and
# turns it into a window-close event, which gymnasium never reads, so a plain
# `kill` would be ignored. Must be set before pygame starts.
os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "LunarLander-v3"
OUTPUT = Path(__file__).resolve().parent
FINAL_MODEL = OUTPUT / "ppo_lunar_lander_final.zip"


def train(args):
    (OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)

    # PPO hyperparameters for LunarLander-v3 from RL Baselines3 Zoo
    # (rl_zoo3/hyperparams/ppo.yml, v2.9.1).
    n_envs = 16
    env = make_vec_env(
        ENV_ID,
        n_envs=n_envs,
        seed=args.seed,
        vec_env_cls=SubprocVecEnv,
    )

    # save_freq counts vectorized steps (one per call across all envs), so
    # divide by n_envs to checkpoint every 100,000 transitions.
    checkpoints = CheckpointCallback(
        save_freq=100_000 // n_envs,
        save_path=str(OUTPUT / "checkpoints"),
        name_prefix="ppo_lunar_lander",
    )

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=1024,
        batch_size=64,
        n_epochs=4,
        gamma=0.999,
        gae_lambda=0.98,
        ent_coef=0.01,
        seed=args.seed,
        device="cpu",
        verbose=1,
        tensorboard_log=str(OUTPUT / "tensorboard"),
    )

    model.learn(total_timesteps=args.timesteps, callback=checkpoints)
    model.save(str(FINAL_MODEL))
    env.close()
    print(f"Saved model to {FINAL_MODEL}")


def evaluate(args):
    if not args.model.is_file():
        raise SystemExit(
            f"No model at {args.model}. Run 'python train.py train' first."
        )
    model = PPO.load(str(args.model), device="cpu")
    render = not args.no_render
    env = gym.make(ENV_ID, render_mode="human" if render else None)
    solved_at = gym.spec(ENV_ID).reward_threshold

    rewards, lengths, timeouts = [], [], []
    try:
        # Seed only the first reset; later resets continue the same RNG
        # stream, so the whole set of episodes is reproducible.
        obs, _ = env.reset(seed=args.seed)
        for episode in range(1, args.episodes + 1):
            total, steps, terminated, truncated = 0.0, 0, False, False
            while not (terminated or truncated):
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, _ = env.step(action)
                total += float(reward)
                steps += 1
                if render and window_closed():
                    raise WindowClosed
            rewards.append(total)
            lengths.append(steps)
            timeouts.append(truncated)
            outcome = "timed out" if truncated else "ended"
            print(f"Episode {episode:>3}: reward {total:7.1f}  "
                  f"length {steps:4d}  ({outcome})")
            obs, _ = env.reset()
    except WindowClosed:
        print("Window closed; stopping early.")
    except KeyboardInterrupt:
        print("\nInterrupted; stopping early.")
    finally:
        env.close()
    summarize(args, rewards, lengths, timeouts, solved_at)


class WindowClosed(Exception):
    pass


def window_closed():
    # Gymnasium's human renderer pumps pygame events but ignores QUIT, so the
    # window's close button does nothing unless we check for it ourselves.
    import pygame

    return bool(pygame.event.get(pygame.QUIT))


def summarize(args, rewards, lengths, timeouts, solved_at):
    if not rewards:
        print("No episodes completed.")
        return
    n = len(rewards)
    rewards, lengths = np.array(rewards), np.array(lengths)
    print()
    print(f"Summary over {n} episodes ({args.model.name})")
    print(f"  Reward: mean {rewards.mean():.1f} +/- {rewards.std():.1f}  "
          f"min {rewards.min():.1f}  median {np.median(rewards):.1f}  "
          f"max {rewards.max():.1f}")
    print(f"  Length: mean {lengths.mean():.1f}  "
          f"min {lengths.min()}  max {lengths.max()}")
    solved = int((rewards >= solved_at).sum())
    print(f"  Scored {solved_at:.0f} or more (solved): {solved}/{n} "
          f"({100 * solved / n:.0f}%)")
    # Hitting the time limit usually means the lander hovered without landing.
    print(f"  Timed out: {sum(timeouts)}/{n}")


def main():
    parser = argparse.ArgumentParser(description="PPO on LunarLander-v3.")
    modes = parser.add_subparsers(dest="mode", required=True)

    train_parser = modes.add_parser("train", help="train a new model")
    train_parser.add_argument("--timesteps", type=int, default=1_000_000)
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    eval_parser = modes.add_parser(
        "eval", help="load a trained model and watch it play"
    )
    eval_parser.add_argument("--model", type=Path, default=FINAL_MODEL)
    eval_parser.add_argument("--episodes", type=int, default=10)
    eval_parser.add_argument("--seed", type=int, default=42)
    eval_parser.add_argument(
        "--no-render",
        action="store_true",
        help="skip the window (for headless machines such as a RunPod pod)",
    )
    eval_parser.set_defaults(func=evaluate)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
