#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_DIR}"

source .venv/bin/activate || true

OUT_JSONL="data/tooluse/apibank.jsonl"
mkdir -p "$(dirname "${OUT_JSONL}")"

echo "[prepare_apibank] Downloading API-Bank from Hugging Face (liminghao1630/API-Bank)..."
python -m envs.tooluse.normalize_apibank --out "${OUT_JSONL}"

sed -i 's#^dataset_jsonl: .*#dataset_jsonl: data/tooluse/apibank.jsonl#' configs/tool.yaml || true
sed -i 's#^subset_limit: .*#subset_limit: 0#' configs/tool.yaml || true
echo "[prepare_apibank] Prepared API-Bank at ${OUT_JSONL} and pointed configs/tool.yaml to it."
