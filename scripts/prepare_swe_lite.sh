#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$REPO_DIR/.venv/bin/activate" || true
python - <<'PY'
import os, json
from datasets import load_dataset
os.makedirs("data/swebench", exist_ok=True)
try:
    # Try local file first
    from datasets import load_dataset as ld
    ds = ld("json", data_files={"dev":"data/swebench/lite.jsonl"}, split="dev")
except Exception:
    # Fallback to HF dev+test
    ds_dev = load_dataset("princeton-nlp/SWE-bench_Lite", split="dev")
    ds_test = load_dataset("princeton-nlp/SWE-bench_Lite", split="test")
    with open("data/swebench/lite_normalized.jsonl","w",encoding="utf-8") as f:
        for rec in ds_dev:
            f.write(json.dumps(rec)+"\n")
        for rec in ds_test:
            f.write(json.dumps(rec)+"\n")
    print("Wrote data/swebench/lite_normalized.jsonl")
    raise SystemExit(0)
# If local file was used, just mirror to normalized
with open("data/swebench/lite_normalized.jsonl","w",encoding="utf-8") as f:
    for rec in ds:
        f.write(json.dumps(rec)+"\n")
print("Wrote data/swebench/lite_normalized.jsonl")
PY
echo "Prepared SWE-bench Lite normalized specs."