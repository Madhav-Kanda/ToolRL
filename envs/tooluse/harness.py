import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional
import yaml
from tqdm import tqdm

from agent.policy_wrapper import Policy
from envs.tooluse.dataset import ToolUseDataset, ToolTask
from envs.tooluse.tools import ToolRegistry, ToolResult
from envs.tooluse.validators import validate_answer


@dataclass
class Step:
    tool_name: str
    args: Dict[str, Any]
    result: ToolResult


def run_episode(task: ToolTask, cfg: Dict[str, Any], policy: Policy | None = None, registry: ToolRegistry | None = None, verbose: bool = False) -> Tuple[List[Step], bool, str]:
    """
    Simplified evaluation for API-Bank:
    - Feed the question directly to the model
    - Compare model output with expected API-Request format
    """
    pol = policy or Policy()
    metrics = cfg.get("metrics", {}) or {}
    use_regex = bool(metrics.get("regex_match", False))
    
    # API-Bank: just feed the question and get the API request
    prompt = task.question.strip() + "\n"
    
    if verbose:
        print(f"    [Eval] Generating response...")
    
    # Get raw model output (not parsed as JSON tool call)
    raw_output = pol.generate_raw(prompt)
    
    # Extract the API-Request from the output
    # Look for pattern: API-Request: [ApiName(...)]
    api_match = re.search(r'API-Request:\s*\[.*?\]', raw_output, re.DOTALL)
    if api_match:
        answer = api_match.group(0).strip()
    else:
        # Fallback: use the whole output
        answer = raw_output.strip()
    
    if verbose:
        print(f"    [Eval] Output: {answer[:100]}...")
        print(f"    [Eval] Expected: {task.final_answer[:100]}...")
    
    # Compare with expected answer
    ok, _ = validate_answer(answer, task.final_answer, use_regex=use_regex)
    
    return [], ok, answer


def run_subset(limit: int = 50, verbose: bool = True) -> List[Tuple[str, bool, str]]:
    with open("configs/tool.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    ds = ToolUseDataset()
    
    # Get total count for progress bar
    tasks = list(ds)
    if limit and limit > 0:
        tasks = tasks[:limit]
    total = len(tasks)
    
    print(f"\n{'='*60}", flush=True)
    print(f"[Eval] Loading model...", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    # Initialize once to avoid reloading model for each task
    policy = Policy()
    registry = ToolRegistry(cfg)
    
    print(f"\n{'='*60}", flush=True)
    print(f"[Eval] Starting evaluation on {total} tasks", flush=True)
    print(f"{'='*60}\n", flush=True)
    sys.stdout.flush()
    
    results = []
    passed = 0
    
    pbar = tqdm(tasks, desc="Evaluating", unit="task", file=sys.stdout, dynamic_ncols=True)
    for task in pbar:
        hist, ok, out = run_episode(task, cfg, policy=policy, registry=registry, verbose=False)
        results.append((task.task_id, ok, out))
        if ok:
            passed += 1
        # Update progress bar with running accuracy
        acc = 100 * passed / len(results)
        pbar.set_postfix({"pass": passed, "fail": len(results) - passed, "acc": f"{acc:.1f}%"}, refresh=True)
    
    print(f"\n{'='*60}", flush=True)
    print(f"[Eval] Completed: {passed}/{total} passed ({100*passed/total:.1f}%)", flush=True)
    print(f"{'='*60}\n", flush=True)
    
    return results
