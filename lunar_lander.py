import argparse
import os
import warnings
from pathlib import Path

# SDL (pygame's backend for the render window) otherwise catches SIGTERM and
# turns it into a window-close event, which gymnasium never reads, so a plain
# `kill` would be ignored. Must be set before pygame starts.
os.environ.setdefault("SDL_NO_SIGNAL_HANDLERS", "1")

import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import SubprocVecEnv

ENV_ID = "LunarLander-v3"
# Everything worth downloading from the VM (checkpoints, TensorBoard logs and
# by default the models) goes under this one directory.
OUTPUT = Path(__file__).resolve().parent / "save"
DEFAULT_MODEL = OUTPUT / "ppo_lunar_lander_final.zip"
EVAL_EPISODES = 10


def model_path(value):
    # SB3 appends .zip when saving a path without a suffix; match that here so
    # existence checks and messages use the real filename.
    path = Path(value)
    return path if path.suffix else path.with_suffix(".zip")


def best_path(model):
    return model.with_name(f"{model.stem}_best.zip")


def train(args):
    best_model = best_path(args.model)
    resume = args.model.is_file()
    args.model.parent.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "checkpoints").mkdir(parents=True, exist_ok=True)

    # When resuming, score the existing best model first so the first
    # periodic evaluation can't replace it with a worse one.
    best_so_far = -np.inf
    if resume and best_model.is_file():
        print(f"Scoring existing best model {best_model}...")
        try:
            best_so_far = score(PPO.load(str(best_model), device="cpu"),
                                args.seed)
        except KeyboardInterrupt:
            raise SystemExit("\nInterrupted before training started.")
        print(f"Existing best: mean reward {best_so_far:.1f}")

    # PPO hyperparameters for LunarLander-v3 from RL Baselines3 Zoo
    # (rl_zoo3/hyperparams/ppo.yml, v2.9.1).
    n_envs = 8
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
        name_prefix=args.model.stem,
    )

    # Scores the policy every 50,000 transitions on a separate env and saves
    # best_model whenever the mean reward beats the best so far. The eval env
    # stays in-process (DummyVecEnv) so Ctrl+C doesn't kill it; SB3 warns
    # about the type mismatch with the training env, which is harmless here.
    warnings.filterwarnings("ignore", "Training and eval env are not")
    eval_env = make_eval_env(args.seed)
    best = EvalCallback(
        eval_env,
        eval_freq=50_000 // n_envs,
        n_eval_episodes=EVAL_EPISODES,
        deterministic=True,
        # EvalCallback's own best_model_save_path always writes
        # best_model.zip, so save under our own name instead.
        callback_on_new_best=SaveModel(best_model),
        log_path=str(OUTPUT / "eval" / args.model.stem),
    )
    best.best_mean_reward = best_so_far

    if resume:
        # Hyperparameters come from the saved model. Stable-Baselines3
        # rebuilds the rollout buffer for this env's n_envs.
        print(f"Resuming training from {args.model}")
        model = PPO.load(
            str(args.model),
            env=env,
            device="cpu",
            tensorboard_log=str(OUTPUT / "tensorboard"),
        )
        model.set_random_seed(args.seed)
    else:
        print(f"Training a new model; it will be saved to {args.model}")
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

    try:
        # Without resetting, --timesteps counts extra steps on top of the
        # loaded model's, and TensorBoard continues the same run.
        model.learn(
            total_timesteps=args.timesteps,
            callback=[checkpoints, best],
            reset_num_timesteps=not resume,
        )
    except KeyboardInterrupt:
        print("\nInterrupted; saving and stopping early.")
        model.save(str(args.model))
        save_if_best(model, best, best_model, args.seed)
    else:
        model.save(str(args.model))
    finally:
        close_quietly(env)
        eval_env.close()

    print(f"Saved latest model to {args.model} "
          f"({model.num_timesteps:,} timesteps in total)")
    if best_model.is_file():
        print(f"Best model is {best_model} "
              f"(mean reward {best.best_mean_reward:.1f} over "
              f"{EVAL_EPISODES} episodes)")


class SaveModel(BaseCallback):
    def __init__(self, path):
        super().__init__()
        self.path = path

    def _on_step(self):
        self.model.save(str(self.path))
        return True


def make_eval_env(seed):
    return make_vec_env(ENV_ID, n_envs=1, seed=seed + 1_000)


def score(model, seed):
    # Uses a fresh env each time: if Ctrl+C landed during a periodic
    # evaluation, the eval env's Box2D world is left locked and can't reset.
    score_env = make_eval_env(seed)
    try:
        rewards, _ = evaluate_policy(
            model,
            score_env,
            n_eval_episodes=EVAL_EPISODES,
            deterministic=True,
            return_episode_rewards=True,
        )
    finally:
        score_env.close()
    return float(np.mean(rewards))


def save_if_best(model, best, best_model, seed):
    # The last periodic evaluation may be up to 50,000 transitions old, so
    # score the model as it is now and keep it if it beats the best so far.
    print(f"Scoring the current model over {EVAL_EPISODES} episodes "
          "(Ctrl+C again to skip)...")
    try:
        mean_reward = score(model, seed)
    except KeyboardInterrupt:
        print("Skipped scoring; keeping the existing best model.")
        return
    print(f"Current model: mean reward {mean_reward:.1f} "
          f"(best so far {best.best_mean_reward:.1f})")
    if mean_reward > best.best_mean_reward:
        model.save(str(best_model))
        best.best_mean_reward = mean_reward


def close_quietly(env):
    # Ctrl+C in a terminal also reaches the SubprocVecEnv workers, which exit
    # on KeyboardInterrupt, so their pipes may already be closed.
    try:
        env.close()
    except (EOFError, BrokenPipeError):
        pass


def evaluate(args):
    if not args.model.is_file():
        raise SystemExit(
            f"No model at {args.model}. "
            f"Run 'python {Path(__file__).name} train --model {args.model}' "
            "first."
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

    train_parser = modes.add_parser(
        "train", help="train a model, resuming from --model if it exists"
    )
    train_parser.add_argument(
        "--model",
        type=model_path,
        default=DEFAULT_MODEL,
        help="where to save the model; loaded and trained further if it "
        "already exists (default: %(default)s)",
    )
    train_parser.add_argument(
        "--timesteps",
        type=int,
        default=1_000_000,
        help="steps to train for; when resuming, added to the model's "
        "existing steps (default: %(default)s)",
    )
    train_parser.add_argument("--seed", type=int, default=42)
    train_parser.set_defaults(func=train)

    eval_parser = modes.add_parser(
        "eval", help="load a trained model and watch it play"
    )
    eval_parser.add_argument(
        "--model",
        type=model_path,
        default=DEFAULT_MODEL,
        help="model to load (default: %(default)s)",
    )
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
