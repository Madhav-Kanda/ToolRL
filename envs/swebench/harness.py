from pathlib import Path
from typing import Tuple, List
from .dataset import SWELiteDataset, TaskSpec
from .sandbox import clone_or_update, prepare_episode_workspace, load_swe_cfg
from agent.loop import AgentLoop
from agent import tools


def run_episode(task: TaskSpec, verbose: bool = True) -> Tuple[List, bool, str]:
    cfg = load_swe_cfg()
    cache_dir = Path(cfg.get("cache_dir", "data/swebench/repos_cache"))
    work_dir = Path(cfg.get("work_dir", "data/swebench/episodes"))
    clone_timeout = int(cfg.get("clone_timeout_s", 300))
    test_timeout = int(cfg.get("test_timeout_s", 180))

    if verbose:
        print(f"[Task {task.task_id}] Clone {task.repo}@{task.base_commit} ...", flush=True)
    repo_path = clone_or_update(task.repo, task.base_commit, cache_dir, timeout_s=clone_timeout)
    if verbose:
        print(f"[Task {task.task_id}] Prepare workspace ...", flush=True)
    ep = prepare_episode_workspace(repo_path, work_dir, task.task_id)
    tools.ensure_git_repo(ep)
    if verbose:
        print(f"[Task {task.task_id}] Run agent ...", flush=True)
    agent = AgentLoop(ep, task.issue_text, verbose=verbose)
    history = agent.run()
    if verbose:
        print(f"[Task {task.task_id}] Run tests ...", flush=True)
    final = tools.run_tests(ep, timeout_s=test_timeout, cmd=task.test_cmd)
    return history, final.ok, final.content


def run_subset(limit: int = 10, verbose: bool = True):
    ds = SWELiteDataset()
    results = []
    for i, task in enumerate(ds):
        if i >= limit:
            break
        try:
            if verbose:
                print(f"[{i+1}/{limit}] Start task_id={task.task_id}", flush=True)
            hist, ok, out = run_episode(task, verbose=verbose)
        except Exception as e:
            ok, out = False, f"ERROR: {e}"
        results.append((task.task_id, ok, out))
    return results


