#!/usr/bin/env bash
# Set up vectorized CPU-based RL and direct SSH/SFTP on a Runpod Pod.
# Run from a Git checkout located on Runpod's persistent /workspace volume.
# Add environment-specific Python packages to requirements.txt in your repository.
# Set RUNPOD_REQUIREMENTS to use a different requirements file.
# Set RUNPOD_SFTP_PUBLIC_KEY to the contents of your local id_ed25519.pub
# if the Pod does not already have your key; some templates set PUBLIC_KEY.
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

printf 'Preparing the SSH server for direct SSH/SFTP access...\n'
if ! command -v sshd >/dev/null 2>&1 && [[ ! -x /usr/sbin/sshd ]]; then
    if [[ "$(id -u)" == 0 ]] && command -v apt-get >/dev/null 2>&1; then
        apt-get update
        DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server
    else
        fail "OpenSSH server is missing. Install it in the Pod image, then rerun this script."
    fi
fi

# Container images may include sshd but omit its host keys. -A creates only
# missing keys; it does not replace existing host keys on a rerun.
[[ "$(id -u)" == 0 ]] || fail "Starting the SSH server requires root privileges."
ssh-keygen -A
mkdir -p /run/sshd
sshd_path="$(command -v sshd || true)"
sshd_path="${sshd_path:-/usr/sbin/sshd}"
"$sshd_path" -t || fail "SSH server configuration check failed."
service ssh start || fail "SSH server could not start. Check its service logs."

sftp_public_key="${RUNPOD_SFTP_PUBLIC_KEY:-${PUBLIC_KEY:-${SSH_PUBLIC_KEY:-}}}"
if [[ -n "$sftp_public_key" ]]; then
    [[ "$sftp_public_key" != *$'\n'* && "$sftp_public_key" != *$'\r'* ]] || fail \
        "The SFTP public key must contain exactly one public-key line."
    printf '%s\n' "$sftp_public_key" > "$temp_dir/sftp-public-key.pub"
    ssh-keygen -lf "$temp_dir/sftp-public-key.pub" >/dev/null || fail \
        "The SFTP public key is invalid. Use the entire contents of your local id_ed25519.pub, not your private key."
    mkdir -p /root/.ssh
    chmod 700 /root/.ssh
    touch /root/.ssh/authorized_keys
    chmod 600 /root/.ssh/authorized_keys
    if ! grep -Fxq -- "$sftp_public_key" /root/.ssh/authorized_keys; then
        printf '%s\n' "$sftp_public_key" >> /root/.ssh/authorized_keys
    fi
fi
if [[ ! -s /root/.ssh/authorized_keys ]]; then
    printf 'Root has no authorized SSH keys. On your local computer run: cat ~/.ssh/id_ed25519.pub\n'
    printf 'Then rerun this script with RUNPOD_SFTP_PUBLIC_KEY set to that entire public-key line (never the private key).\n'
else
    printf 'Root SSH keys are installed. Your local id_ed25519.pub must match one of them for direct SSH/SFTP.\n'
fi
if [[ -n "${RUNPOD_PUBLIC_IP:-}" && -n "${RUNPOD_TCP_PORT_22:-}" ]]; then
    printf 'Direct SFTP from your local computer: sftp -P %s -i ~/.ssh/id_ed25519 root@%s\n' \
        "$RUNPOD_TCP_PORT_22" "$RUNPOD_PUBLIC_IP"
else
    printf 'To use SFTP, make sure TCP port 22 is exposed in your Pod settings and use the current address shown under SSH over exposed TCP.\n'
fi

printf '\nSetup complete. To train, run:\n'
printf '  cd %q\n' "$script_dir"
printf '  source .venv-runpod/bin/activate\n'
printf '  python -u path/to/your_training_script.py\n'
printf 'For environment-specific packages, add them to requirements.txt and rerun this script.\n'
