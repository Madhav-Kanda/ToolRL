import shutil
import subprocess
from pathlib import Path
from typing import Tuple
import yaml


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


def _normalize_repo_url(repo: str) -> str:
    r = repo.strip()
    if r.startswith(("http://", "https://", "git@")):
        return r
    if r.startswith("github.com/"):
        return "https://" + r
    # owner/repo → https://github.com/owner/repo.git
    if "/" in r and not r.endswith(".git"):
        return f"https://github.com/{r}.git"
    return r


def clone_or_update(repo: str, commit: str, cache_dir: Path, timeout_s: int = 300) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    url = _normalize_repo_url(repo)
    repo_name = url.split("/")[-1].replace(".git", "")
    dst = cache_dir / repo_name
    attempts = 3
    last_err = ""
    for i in range(attempts):
        # Clean destination each attempt
        if dst.exists():
            if dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst, ignore_errors=True)
        # Partial clone to reduce bandwidth
        code, out = _run(["git", "clone", "--filter=blob:none", "--no-checkout", url, repo_name], cache_dir, timeout=timeout_s)
        if code != 0 or not dst.exists():
            last_err = f"clone: {out}"
            continue
        # Try fetching the specific commit shallowly first
        code, out = _run(["git", "fetch", "--depth", "1", "origin", commit], dst, timeout=timeout_s)
        if code != 0:
            # Fallback to full fetch
            code, out = _run(["git", "fetch", "--all", "--tags"], dst, timeout=timeout_s)
            if code != 0:
                last_err = f"fetch: {out}"
                continue
        code, out = _run(["git", "checkout", commit], dst, timeout=timeout_s)
        if code != 0:
            last_err = f"checkout: {out}"
            continue
        # Success
        return dst
    raise RuntimeError(f"Clone failed for {repo} ({url}): {last_err}")
    return dst

def prepare_episode_workspace(repo_path: Path, work_root: Path, task_id: str) -> Path:
    ep = work_root / task_id
    if ep.exists():
        shutil.rmtree(ep)
    shutil.copytree(repo_path, ep)
    return ep


def load_swe_cfg(path: str = "configs/swe.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


