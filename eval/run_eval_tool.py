import argparse
import os
import logging

# Suppress verbose logging before imports
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
logging.getLogger("vllm").setLevel(logging.WARNING)
logging.getLogger("transformers").setLevel(logging.WARNING)

from envs.tooluse.harness import run_subset


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0, help="0 means no limit (entire dataset)")
    args = ap.parse_args()
    results = run_subset(limit=args.limit)
    ok_count = sum(int(ok) for _, ok, _ in results)
    print("=== Results ===")
    for tid, ok, out in results:
        print(f"{tid}: {'PASS' if ok else 'FAIL'}")
        if not ok:
            print(str(out)[:200])
        print("-" * 40)
    print(f"Summary: {ok_count}/{len(results)} passed")


if __name__ == "__main__":
    main()

