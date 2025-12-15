#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_DIR"

source .venv/bin/activate || true

# Use single GPU for stable QLoRA training
export CUDA_VISIBLE_DEVICES=0

python -m policy.dpo_train --config configs/train_dpo.yaml --out_dir outputs/dpo_lora


