#!/usr/bin/env bash
# Set up a Python environment for vectorized, CPU-based RL on a Runpod Pod.
# Run from a Git checkout located on Runpod's persistent /workspace volume.
# Add environment-specific Python packages to requirements.txt in your repository.
# Set RUNPOD_REQUIREMENTS to use a different requirements file.
set -Eeuo pipefail

fail() {
    printf 'ERROR: %s\n' "$*" >&2
    exit 1
}

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
venv_dir="$script_dir/.venv-runpod"
python_command="${RUNPOD_PYTHON:-python3.12}"
requirements_file="${RUNPOD_REQUIREMENTS:-$script_dir/requirements.txt}"
if [[ "$requirements_file" != /* ]]; then
    requirements_file="$script_dir/$requirements_file"
fi

[[ "$script_dir" == /workspace/* ]] || fail \
    "Clone your repository under /workspace and run this script there. Current path: $script_dir"
cd "$script_dir"

command -v "$python_command" >/dev/null 2>&1 || fail \
    "$python_command was not found. Select a Pod image with Python 3.12 or set RUNPOD_PYTHON to a compatible interpreter."

"$python_command" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || fail \
    "Stable-Baselines3 2.9 requires Python 3.10 or later."

# pip downloads and unpacks packages before installing them. Give both the
# persistent project volume and temporary container disk enough headroom.
free_kib() {
    df -Pk "$1" | awk 'NR == 2 {print $4}'
}

workspace_free="$(free_kib "$script_dir")"
root_free="$(free_kib /)"
[[ "$workspace_free" =~ ^[0-9]+$ && "$root_free" =~ ^[0-9]+$ ]] || fail \
    "Could not measure free disk space. Run: df -h / /workspace"

printf 'Project directory: %s\n' "$script_dir"
printf 'Python: %s\n' "$("$python_command" --version 2>&1)"
df -h / "$script_dir"

(( workspace_free >= 6 * 1024 * 1024 )) || fail \
    "Less than 6 GiB is free on the project volume. Increase the Pod's volume disk to about 20 GB (Pods > Edit Pod), then rerun."
(( root_free >= 512 * 1024 )) || fail \
    "Less than 512 MiB is free on the container disk. Increase the Pod's container disk (Pods > Edit Pod), then rerun."

if [[ -e "$venv_dir" && ! -x "$venv_dir/bin/python" ]]; then
    fail "An incomplete environment exists at $venv_dir. Inspect it, then remove it if disposable and rerun."
fi

if [[ ! -d "$venv_dir" ]]; then
    if ! "$python_command" -m venv "$venv_dir"; then
        if [[ "$(id -u)" == 0 ]] && command -v apt-get >/dev/null 2>&1; then
            printf 'Installing the missing Python venv support...\n'
            python_version="$("$python_command" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
            apt-get update
            apt-get install -y "python${python_version}-venv"
            "$python_command" -m venv "$venv_dir" || fail "Virtual environment creation failed."
        else
            fail "Virtual environment creation failed. Install the venv package for your Python version, then rerun."
        fi
    fi
fi

"$venv_dir/bin/python" -c 'import sys; assert sys.version_info >= (3, 10), sys.version' || fail \
    "The existing $venv_dir uses an incompatible Python. Move it aside and rerun."
"$venv_dir/bin/python" -m pip --version >/dev/null || fail \
    "pip is missing from $venv_dir. Install venv support for this Python, remove the incomplete environment, and rerun."

# Keep large temporary downloads off the usually small container disk.
temp_dir="$(mktemp -d "$script_dir/.runpod-tmp.XXXXXXXX")"
trap 'rm -rf -- "$temp_dir"' EXIT
export TMPDIR="$temp_dir"
export PIP_NO_CACHE_DIR=1
printf '%s\n' 'torch==2.8.0+cpu' > "$temp_dir/cpu-constraints.txt"

printf 'Installing CPU PyTorch 2.8 from the official PyTorch wheel index...\n'
"$venv_dir/bin/python" -m pip install --no-cache-dir \
    --index-url https://download.pytorch.org/whl/cpu 'torch==2.8.0'

printf 'Installing Stable-Baselines3, Gymnasium, and TensorBoard...\n'
"$venv_dir/bin/python" -m pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -c "$temp_dir/cpu-constraints.txt" \
    'stable-baselines3==2.9.0' gymnasium tensorboard

if [[ -f "$requirements_file" ]]; then
    printf 'Installing project dependencies from %s...\n' "$requirements_file"
    "$venv_dir/bin/python" -m pip install --no-cache-dir \
        --extra-index-url https://download.pytorch.org/whl/cpu \
        -c "$temp_dir/cpu-constraints.txt" \
        -r "$requirements_file"
elif [[ -n "${RUNPOD_REQUIREMENTS:-}" ]]; then
    fail "The specified requirements file does not exist: $requirements_file"
else
    printf 'No requirements.txt found; install your environment-specific dependencies before training.\n'
fi

printf 'Testing the vectorized RL installation with a minimal synthetic environment...\n'
"$venv_dir/bin/python" - <<'PY'
import gymnasium as gym
import numpy as np
import stable_baselines3
import torch
from gymnasium import spaces
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv

class ProbeEnv(gym.Env):
    observation_space = spaces.Box(-1.0, 1.0, shape=(1,), dtype=np.float32)
    action_space = spaces.Discrete(2)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        return np.zeros(1, dtype=np.float32), {}

    def step(self, action):
        return np.zeros(1, dtype=np.float32), 0.0, False, False, {}

env = DummyVecEnv([ProbeEnv for _ in range(2)])
observations = env.reset()
observations, rewards, dones, infos = env.step(np.array([0, 1]))
assert observations.shape == (2, 1)
env.close()
assert torch.version.cuda is None, "A GPU PyTorch build replaced the intended CPU build"

print(f"Python packages: torch={torch.__version__}, SB3={stable_baselines3.__version__}, gymnasium={gym.__version__}")
print("Two-environment vectorization check: OK")
PY

printf '\nSetup complete. To train, run:\n'
printf '  cd %q\n' "$script_dir"
printf '  source .venv-runpod/bin/activate\n'
printf '  python -u path/to/your_training_script.py\n'
printf 'For environment-specific packages, add them to requirements.txt and rerun this script.\n'
