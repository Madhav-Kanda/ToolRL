#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
python -m envs.tooluse.trajectory --cfg configs/tool.yaml --out_train data/sft/train.jsonl --out_val data/sft/val.jsonl --max_examples 1000
echo "Tool-use SFT trajectories generated."


