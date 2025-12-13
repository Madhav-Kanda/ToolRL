from pathlib import Path
from ..agent import tools


def check_success(repo: Path, test_cmd: str = "pytest -q", timeout_s: int = 180) -> bool:
    res = tools.run_tests(repo, timeout_s=timeout_s, cmd=test_cmd)
    return res.ok


