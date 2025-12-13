from pathlib import Path
from typing import List
import yaml
from .schema import ToolCall, Step
from .policy_wrapper import Policy
from .prompts import build_prompt
from . import tools
from .memory import RecentMemory


class AgentLoop:
    def __init__(self, repo_root: Path, issue_text: str, env_cfg_path: str = "configs/env.yaml", verbose: bool = False):
        self.repo_root = repo_root
        self.issue = issue_text
        with open(env_cfg_path, "r") as f:
            self.cfg = yaml.safe_load(f)
        self.policy = Policy()
        self.history: List[Step] = []
        self.mem = RecentMemory()
        self.verbose = verbose
        if self.verbose:
            print(f"[Agent] Initialized for repo={self.repo_root} | max_steps={self.cfg.get('max_steps', 6)}", flush=True)

    def format_context(self) -> str:
        files = "\n".join(
            str(p.relative_to(self.repo_root))
            for p in sorted(self.repo_root.rglob("*.py"))
            if ".git" not in str(p)
        )
        tail = self.mem.tail()
        return f"Files:\n{files}\n\nLast:\n{tail}"

    def step(self) -> Step:
        prompt = build_prompt(self.issue, self.format_context())
        call_dict = self.policy.decide(prompt)
        call = ToolCall(tool_name=call_dict["tool_name"], args=call_dict["args"])
        res = self.dispatch(call)
        if self.verbose:
            preview = (res.content or "")[:120].replace("\n", " ")
            print(f"[Agent] {call.tool_name} -> ok={res.ok} | {preview}", flush=True)
        self.mem.add(res.content[-600:])
        step = Step(call=call, result=res)
        self.history.append(step)
        return step

    def dispatch(self, call: ToolCall):
        tn = call.tool_name
        args = call.args
        per_step_to = int(self.cfg.get("per_step_timeout_s", 60))
        if tn == "search":
            return tools.search(self.repo_root, pattern=args.get("pattern", ""))
        if tn == "read_file":
            return tools.read_file(
                self.repo_root,
                path=args.get("path", ""),
                start_line=int(args.get("start_line", 1)),
                end_line=int(args.get("end_line", 400)),
            )
        if tn == "apply_diff":
            return tools.apply_diff(
                self.repo_root,
                patch=args.get("patch", ""),
                max_chars=int(self.cfg.get("max_diff_chars", 20000)),
            )
        if tn == "git_diff":
            return tools.git_diff(self.repo_root)
        if tn == "run_tests":
            return tools.run_tests(
                self.repo_root,
                timeout_s=int(args.get("timeout_s", per_step_to)),
                cmd=args.get("cmd", "pytest -q"),
            )
        if tn == "revert":
            return tools.revert(self.repo_root)
        return tools.run_tests(self.repo_root, timeout_s=per_step_to)

    def run(self):
        if self.verbose:
            print("[Agent] Starting episode loop...", flush=True)
        for _ in range(int(self.cfg.get("max_steps", 6))):
            st = self.step()
            if st.result.ok and "passed" in st.result.content.lower():
                if self.verbose:
                    print("[Agent] Detected tests passed. Stopping.", flush=True)
                break
        return self.history


