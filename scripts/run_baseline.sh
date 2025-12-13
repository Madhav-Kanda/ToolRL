#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
export PYTHONUNBUFFERED=1
python -m eval.run_eval --limit 5


