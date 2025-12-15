#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

source .venv/bin/activate || true

# Single GPU QLoRA training (stable and memory-efficient)
CUDA_VISIBLE_DEVICES=0 python -m policy.sft_train \
    --config configs/train_sft.yaml \
    --out_dir outputs/sft_lora
