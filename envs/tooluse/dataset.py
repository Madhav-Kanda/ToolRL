import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Dict, Any
import yaml


@dataclass
class ToolCall:
    name: str
    args: Dict[str, Any]
    output: Optional[str] = None


@dataclass
class ToolTask:
    task_id: str
    question: str
    gold_calls: List[ToolCall]
    final_answer: str


class ToolUseDataset:
    def __init__(self, cfg_path: str = "configs/tool.yaml"):
        with open(cfg_path, "r") as f:
            cfg = yaml.safe_load(f)
        self.path = Path(cfg["dataset_jsonl"])
        self.subset_limit = int(cfg.get("subset_limit", 200))

    def __iter__(self) -> Iterator[ToolTask]:
        if not self.path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self.path}")
        count = 0
        with self.path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                task_id = str(rec.get("id") or rec.get("task_id") or count)
                question = rec.get("question") or rec.get("prompt") or ""
                final_answer = rec.get("final_answer") or rec.get("answer") or ""
                calls_raw = rec.get("tools") or []
                calls: List[ToolCall] = []
                for c in calls_raw:
                    name = c.get("name")
                    args = c.get("args") or {}
                    out = c.get("output")
                    if name:
                        calls.append(ToolCall(name=name, args=args, output=out))
                yield ToolTask(task_id=task_id, question=question, gold_calls=calls, final_answer=final_answer)
                count += 1
                if self.subset_limit and count >= self.subset_limit:
                    break


