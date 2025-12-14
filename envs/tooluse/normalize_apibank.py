"""
Download and normalize API-Bank dataset from Hugging Face (liminghao1630/API-Bank).
Downloads individual JSON files and processes them with their varying schemas.
"""
import argparse
import json
from pathlib import Path
from typing import Dict, Any, List
import re

from huggingface_hub import hf_hub_download, list_repo_files


REPO_ID = "liminghao1630/API-Bank"

# Files we want to process (test data for evaluation)
TEST_FILES = [
    "test-data/level-1-api.json",
    "test-data/level-1-response.json",
    "test-data/level-2-api.json",
    "test-data/level-2-response.json",
    "test-data/level-3.json",
]

# Training files (optional, for SFT)
TRAIN_FILES = [
    "training-data/lv1-train.json",
    "training-data/lv2-train.json",
    "training-data/lv3-train.json",
]


def parse_tool_calls(api_call_str: str) -> List[Dict[str, Any]]:
    """
    Parse the API call string from API-Bank.
    Example format: "ToolSearcher(keywords='xxx')" or chained calls.
    """
    calls = []
    if not api_call_str or str(api_call_str).strip() == "":
        return calls
    
    api_call_str = str(api_call_str)
    # Match patterns like: FunctionName(args...)
    pattern = r"(\w+)\(([^)]*)\)"
    matches = re.findall(pattern, api_call_str)
    
    for func_name, args_str in matches:
        args = {}
        if args_str.strip():
            # Parse key='value' or key="value" pairs
            arg_pattern = r"(\w+)\s*=\s*['\"]?([^'\",$)]+)['\"]?"
            arg_matches = re.findall(arg_pattern, args_str)
            for k, v in arg_matches:
                args[k] = v.strip()
        calls.append({"name": func_name, "args": args, "output": None})
    
    return calls


def normalize_record(rec: Dict[str, Any], idx: int, source_file: str) -> Dict[str, Any]:
    """Normalize a single API-Bank record based on its source file schema."""
    rid = rec.get("id") or f"apibank_{idx}"
    
    # Different files have different schemas:
    # - level-X-api.json: instruction, input, expected_output, file, id
    # - level-X-response.json: instruction, input, output
    # - level-3.json: has different format
    # - training files: instruction, input, output
    
    # Get the question/instruction
    instruction = rec.get("instruction") or ""
    input_text = rec.get("input") or ""
    question = f"{instruction}\n{input_text}".strip() if input_text else instruction
    
    # Get the expected output/answer
    final_answer = (
        rec.get("expected_output") or 
        rec.get("output") or 
        rec.get("response") or 
        rec.get("answer") or 
        ""
    )
    
    # Parse tool calls from the output if it looks like an API call
    calls = []
    if final_answer and "(" in str(final_answer) and ")" in str(final_answer):
        calls = parse_tool_calls(str(final_answer))
    
    # Determine level from source file
    level = "1"
    if "level-2" in source_file or "lv2" in source_file:
        level = "2"
    elif "level-3" in source_file or "lv3" in source_file:
        level = "3"
    
    return {
        "id": f"{source_file.replace('/', '_')}_{rid}",
        "question": question,
        "tools": calls,
        "final_answer": str(final_answer),
        "level": level,
    }


def download_and_process_file(filename: str, cache_dir: Path) -> List[Dict[str, Any]]:
    """Download a single file from the repo and parse it."""
    try:
        local_path = hf_hub_download(
            repo_id=REPO_ID,
            filename=filename,
            repo_type="dataset",
            cache_dir=str(cache_dir),
        )
        
        with open(local_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        records = []
        if isinstance(data, list):
            for idx, rec in enumerate(data):
                records.append(normalize_record(rec, idx, filename))
        elif isinstance(data, dict):
            # Some files might be a dict with a data key
            items = data.get("data") or data.get("examples") or [data]
            if isinstance(items, list):
                for idx, rec in enumerate(items):
                    records.append(normalize_record(rec, idx, filename))
            else:
                records.append(normalize_record(data, 0, filename))
        
        return records
    except Exception as e:
        print(f"  Warning: Could not process {filename}: {e}")
        return []


def main(out_path: str, include_train: bool = False):
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    cache_dir = out.parent / "hf_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    
    print("[normalize_apibank] Downloading API-Bank files from Hugging Face...")
    
    files_to_process = TEST_FILES.copy()
    if include_train:
        files_to_process.extend(TRAIN_FILES)
    
    all_records = []
    for filename in files_to_process:
        print(f"  Processing {filename}...")
        records = download_and_process_file(filename, cache_dir)
        all_records.extend(records)
        print(f"    -> {len(records)} records")
    
    # Filter out records with empty questions
    valid_records = [r for r in all_records if r["question"].strip()]
    
    with out.open("w", encoding="utf-8") as w:
        for rec in valid_records:
            w.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print(f"[normalize_apibank] Wrote {len(valid_records)} records to {out}")
    
    # Print breakdown by level
    by_level = {}
    for r in valid_records:
        lv = r.get("level", "?")
        by_level[lv] = by_level.get(lv, 0) + 1
    print(f"  Breakdown by level: {by_level}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/tooluse/apibank.jsonl", help="Output JSONL path")
    ap.add_argument("--include-train", action="store_true", help="Also include training data")
    args = ap.parse_args()
    main(args.out, args.include_train)
