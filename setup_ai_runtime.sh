#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AI_ROOT="${SPRITE_VIDEO_LAB_AI_ROOT:-$ROOT_DIR/work/models}"
VENV_DIR="${SPRITE_VIDEO_LAB_VENV_DIR:-$AI_ROOT/venv}"
MODEL_CACHE="${SPRITE_VIDEO_LAB_AI_MODEL_CACHE:-$AI_ROOT/huggingface}"
TORCH_INDEX_URL="${SPRITE_VIDEO_LAB_TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu128}"
BOOTSTRAP_PYTHON="${SPRITE_VIDEO_LAB_BOOTSTRAP_PYTHON:-python3}"

mkdir -p "$AI_ROOT" "$MODEL_CACHE"

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
  "$BOOTSTRAP_PYTHON" -m venv "$VENV_DIR"
fi

PYTHON_EXE="$VENV_DIR/bin/python"
"$PYTHON_EXE" -m pip install --upgrade pip
"$PYTHON_EXE" -m pip install -r "$ROOT_DIR/requirements.txt"
"$PYTHON_EXE" -m pip install --upgrade --index-url "$TORCH_INDEX_URL" torch torchvision
"$PYTHON_EXE" -m pip install -r "$ROOT_DIR/requirements-ai.txt"

export SPRITE_VIDEO_LAB_AI_MODEL_CACHE="$MODEL_CACHE"
export HF_HOME="$MODEL_CACHE"
export HUGGINGFACE_HUB_CACHE="$MODEL_CACHE/hub"
export TRANSFORMERS_CACHE="$MODEL_CACHE/transformers"
export HF_MODULES_CACHE="$MODEL_CACHE/modules"
export HF_XET_CACHE="$MODEL_CACHE/xet"

"$PYTHON_EXE" - <<'PY'
import torch

cuda_available = torch.cuda.is_available()
print(
    {
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuda_available": cuda_available,
        "device": torch.cuda.get_device_name(0) if cuda_available else "",
    }
)
if not cuda_available:
    raise SystemExit(
        "CUDA is not available; check the NVIDIA driver and PyTorch CUDA wheel"
    )
PY

printf 'AI runtime ready: %s\n' "$VENV_DIR"
printf 'Model cache: %s\n' "$MODEL_CACHE"
