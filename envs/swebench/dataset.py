import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional
import yaml


@dataclass
class TaskSpec:
    task_id: str
    repo: str               # git URL or org/repo
    base_commit: str        # commit to checkout (buggy)
    test_cmd: str           # command to run tests
    issue_text: str         # description/issue
    gold_patch: Optional[str] = None  # optional unified diff


def _normalize_record(rec: dict) -> Optional[TaskSpec]:
    """
    Normalize diverse SWE-bench Lite formats into TaskSpec.
    Expected fields (best-effort):
      - 'instance_id' or 'task_id'
      - 'repo' or 'repo_url' or 'repo_name'
      - 'base_commit' or 'version' or 'bug_commit'
      - 'test_cmd' (else default)
      - 'problem_statement' or 'title'+'body' or 'issue' as issue_text
      - optional 'patch'/'gold_patch' unified diff
    """
    task_id = rec.get("instance_id") or rec.get("task_id") or rec.get("id")
    if not task_id:
        return None
    repo = rec.get("repo") or rec.get("repo_url") or rec.get("repo_name")
    if not repo:
        return None
    base_commit = rec.get("base_commit") or rec.get("version") or rec.get("bug_commit")
    if not base_commit:
        return None
    issue_text = rec.get("problem_statement") or rec.get("issue") or (
        (rec.get("title") or "") + "\n" + (rec.get("body") or "")
    )
    test_cmd = rec.get("test_cmd") or "pytest -q"
    gold_patch = rec.get("gold_patch") or rec.get("patch")
    return TaskSpec(
        task_id=str(task_id),
        repo=str(repo),
        base_commit=str(base_commit),
        test_cmd=str(test_cmd),
        issue_text=str(issue_text),
        gold_patch=gold_patch if isinstance(gold_patch, str) else None,
    )


class SWELiteDataset:
    def __init__(self, swe_cfg_path: str = "configs/swe.yaml"):
        with open(swe_cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.path = Path(cfg["dataset_jsonl"])
        self.subset_limit = int(cfg.get("subset_limit", 50))
        self.default_test_cmd = cfg.get("default_test_cmd", "pytest -q")
        self.shuffle = bool(cfg.get("shuffle", False))
        self.seed = int(cfg.get("seed", 42))

    def __iter__(self) -> Iterator[TaskSpec]:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        specs = []
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                spec = _normalize_record(rec)
                if not spec:
                    continue
                if not spec.test_cmd:
                    spec.test_cmd = self.default_test_cmd
                specs.append(spec)
        if self.shuffle:
            import random
            random.Random(self.seed).shuffle(specs)
        limit = self.subset_limit or len(specs)
        for spec in specs[:limit]:
            yield spec

    def load_all(self) -> List[TaskSpec]:
        return list(iter(self))


