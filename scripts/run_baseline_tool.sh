#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

source .venv/bin/activate || true

# Suppress verbose logging for cleaner output
export PYTHONUNBUFFERED=1
export VLLM_LOGGING_LEVEL=WARNING
export TOKENIZERS_PARALLELISM=false

python -m eval.run_eval_tool --limit 0
