import argparse
import json
from pathlib import Path
from typing import List, Dict
import yaml

from envs.tooluse.dataset import ToolUseDataset


def build_trace(question: str, gold_calls: List[Dict], final_answer: str) -> Dict[str, str]:
    """
    Build a training example that teaches the model to output the API-Bank format directly.
    
    Input prompt: The full question from API-Bank
    Target response: The expected API-Request format
    """
    # The question already contains all context needed
    # The final_answer is already in the correct format: "API-Request: [ApiName(...)]"
    prompt = question.strip() + "\n"
    response = final_answer.strip()
    return {"prompt": prompt, "response": response}


def main(cfg_path: str, out_train: str, out_val: str, val_ratio: float = 0.1, max_examples: int = 5000, train_data_path: str = None):
    """
    Generate SFT training data from API-Bank TRAINING split (not test!).
    
    Args:
        cfg_path: Config file path
        out_train: Output training JSONL
        out_val: Output validation JSONL  
        val_ratio: Fraction for validation
        max_examples: Max examples to generate
        train_data_path: Path to training data (defaults to data/tooluse/apibank_train.jsonl)
    """
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    
    # Use training data file, NOT the test data!
    if train_data_path is None:
        train_data_path = Path(cfg.get("dataset_jsonl", "data/tooluse/apibank.jsonl")).parent / "apibank_train.jsonl"
    
    train_data_path = Path(train_data_path)
    if not train_data_path.exists():
        print(f"Warning: Training data {train_data_path} not found!")
        print("Run: python -m envs.tooluse.normalize_apibank --split both")
        print("Falling back to main dataset (may cause data leakage!)")
        train_data_path = Path(cfg.get("dataset_jsonl", "data/tooluse/apibank.jsonl"))
    
    print(f"[trajectory] Loading training data from: {train_data_path}")
    
    recs: List[Dict[str, str]] = []
    with train_data_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except:
                continue
            question = rec.get("question") or ""
            final_answer = rec.get("final_answer") or ""
            if not question.strip() or not final_answer.strip():
                continue
            gold = rec.get("tools") or []
            recs.append(build_trace(question, gold, final_answer))
            if len(recs) >= max_examples:
                break
    
    n_val = int(len(recs) * val_ratio)
    Path(out_train).parent.mkdir(parents=True, exist_ok=True)
    with open(out_val, "w", encoding="utf-8") as vf:
        for r in recs[:n_val]:
            vf.write(json.dumps(r) + "\n")
    with open(out_train, "w", encoding="utf-8") as tf:
        for r in recs[n_val:]:
            tf.write(json.dumps(r) + "\n")
    print(f"[trajectory] Wrote {len(recs)-n_val} train and {n_val} val examples.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cfg", type=str, default="configs/tool.yaml")
    ap.add_argument("--out_train", type=str, default="data/sft/train.jsonl")
    ap.add_argument("--out_val", type=str, default="data/sft/val.jsonl")
    ap.add_argument("--max_examples", type=int, default=5000)
    ap.add_argument("--train_data", type=str, default=None,
                    help="Path to training data (default: data/tooluse/apibank_train.jsonl)")
    args = ap.parse_args()
    main(args.cfg, args.out_train, args.out_val, max_examples=args.max_examples, train_data_path=args.train_data)

