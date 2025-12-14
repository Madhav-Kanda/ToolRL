import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any, Tuple
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


def build_prompt(tools_doc: str, question: str, history: List[Step]) -> str:
    hist = []
    for s in history[-5:]:
        hist.append(f"<CALL name='{s.tool_name}' args='{json.dumps(s.args, ensure_ascii=False)}'>")
        hist.append(f"<RESULT ok={s.result.ok}>{s.result.content[:200]}</RESULT>")
    hist_str = "\n".join(hist)
    return (
        f"You are a tool-using assistant. Use only JSON tool calls until you can return the final answer.\n"
        f"Tools:\n{tools_doc}\n"
        f"Rules:\n- Output only a single JSON object per turn: {{\"tool_name\": str, \"args\": dict}}.\n"
        f"- When you are done, output {{\"tool_name\": \"final_answer\", \"args\": {{\"answer\": str}}}}.\n\n"
        f"Question:\n{question}\n\n"
        f"History:\n{hist_str}\n\n"
        f"Your next tool call as JSON:"
    )


def tools_documentation(enabled: List[str]) -> str:
    specs = {
        "calculator": "calculator(expression: str) -> str  # arithmetic",
        "retriever": "retriever(query: str) -> json[list[{{id,title,text}}]]  # local corpus search",
        "calendar": "calendar(date: 'YYYY-MM-DD', op: 'weekday') -> str",
        "weather": "weather(city: str) -> str  # mocked",
        "sqldb": "sqldb(query: str) -> json[list[rows]]  # sqlite over local db",
        "final_answer": "final_answer(answer: str) -> end",
    }
    return "\n".join(f"- {n}: {specs[n]}" for n in enabled + ["final_answer"])


def run_episode(task: ToolTask, cfg: Dict[str, Any], policy: Policy | None = None, registry: ToolRegistry | None = None, verbose: bool = False) -> Tuple[List[Step], bool, str]:
    reg = registry or ToolRegistry(cfg)
    pol = policy or Policy()
    max_calls = int(cfg.get("max_calls", 5))
    metrics = cfg.get("metrics", {}) or {}
    use_regex = bool(metrics.get("regex_match", False))
    hist: List[Step] = []
    tools_doc = tools_documentation(list(reg.enabled))
    
    for step_i in range(max_calls):
        prompt = build_prompt(tools_doc, task.question, hist)
        if verbose:
            print(f"    [Step {step_i+1}/{max_calls}] Generating...")
        call = pol.decide(prompt)
        name = call.get("tool_name")
        args = call.get("args") or {}
        if verbose:
            print(f"    [Step {step_i+1}/{max_calls}] Tool: {name}, Args: {str(args)[:80]}")
        if name == "final_answer":
            answer = str(args.get("answer", ""))
            ok, _ = validate_answer(answer, task.final_answer, use_regex=use_regex)
            return hist, ok, answer
        res = reg.call(name, args)
        hist.append(Step(tool_name=name, args=args, result=res))
    
    # If no final, default to last content
    final = hist[-1].result.content if hist else ""
    ok, _ = validate_answer(final, task.final_answer, use_regex=use_regex)
    return hist, ok, final


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
