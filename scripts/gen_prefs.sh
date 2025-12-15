#!/usr/bin/env bash
set -euo pipefail
source .venv/bin/activate || true
python -m policy.gen_prefs --sft_train data/sft/train.jsonl --out_pairs data/prefs/pairs.jsonl
echo "Preference pairs written to data/prefs/pairs.jsonl"


