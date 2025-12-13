import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Tuple
from .schema import ToolResult


def _run(cmd, cwd: Path, timeout: int = 120) -> Tuple[int, str]:
    proc = subprocess.Popen(
        cmd, cwd=str(cwd), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        return 124, "TIMEOUT"
    return proc.returncode, out


def ensure_git_repo(root: Path):
    if not (root / ".git").exists():
        _run(["git", "init"], root)
        _run(["git", "add", "-A"], root)
        _run(["git", "commit", "-m", "init"], root)


def search(root: Path, pattern: str) -> ToolResult:
    if shutil.which("rg"):
        code, out = _run(["rg", "-n", "-S", pattern], root)
        return ToolResult(ok=(code in (0, 1)), content=out, meta={"tool": "rg"})
    code, out = _run(["grep", "-RIn", pattern, "."], root)
    return ToolResult(ok=(code in (0, 1)), content=out, meta={"tool": "grep"})


def read_file(root: Path, path: str, start_line: int = 1, end_line: int = 300) -> ToolResult:
    p = (root / path).resolve()
    if not p.exists() or not p.is_file():
        return ToolResult(ok=False, content=f"File not found: {path}", meta={})
    try:
        lines = p.read_text(encoding="utf-8", errors="ignore").splitlines()
    except Exception as e:
        return ToolResult(ok=False, content=f"Read error: {e}", meta={})
    s, e = max(1, start_line), min(len(lines), end_line)
    snippet = "\n".join(lines[s - 1 : e])
    return ToolResult(ok=True, content=snippet, meta={"path": path, "start": s, "end": e})


def git_diff(root: Path) -> ToolResult:
    code, out = _run(["git", "diff"], root)
    return ToolResult(ok=(code == 0), content=out, meta={})


def apply_diff(root: Path, patch: str, max_chars: int = 20000) -> ToolResult:
    ensure_git_repo(root)
    if len(patch) > max_chars:
        return ToolResult(ok=False, content="Patch too large.", meta={"applied": False})
    fd, patch_path = tempfile.mkstemp(prefix="patch_", suffix=".diff")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(patch)
        code, out = _run(["git", "apply", "--reject", patch_path], root)
        if code != 0:
            return ToolResult(ok=False, content=out, meta={"applied": False})
        _run(["git", "add", "-A"], root)
        _run(["git", "commit", "-m", "agent patch"], root)
        return ToolResult(ok=True, content="Patch applied and committed.", meta={"applied": True})
    finally:
        try:
            os.remove(patch_path)
        except Exception:
            pass


def revert(root: Path) -> ToolResult:
    code, out = _run(["git", "reset", "--hard", "HEAD~1"], root)
    return ToolResult(ok=(code == 0), content=out, meta={})


def run_tests(root: Path, timeout_s: int = 120, cmd: str = "pytest -q") -> ToolResult:
    code, out = _run(cmd.split(), root, timeout=timeout_s)
    ok = (code == 0)
    return ToolResult(ok=ok, content=out, meta={"exit_code": code})


