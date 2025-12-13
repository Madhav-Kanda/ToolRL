import argparse
from pathlib import Path
from envs.swebench.harness import run_subset
from eval.metrics import parse_pytest_summary


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=5)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    results = run_subset(limit=args.limit, verbose=not args.quiet)
    print("=== Results ===")
    ok_count = 0
    for task_id, ok, out in results:
        ok_count += int(ok)
        print(f"{task_id}: {'PASS' if ok else 'FAIL'}")
        print(parse_pytest_summary(out))
        print("-" * 40)
    print(f"Summary: {ok_count}/{len(results)} passed")


if __name__ == "__main__":
    main()