from __future__ import annotations

import os
import shutil
import subprocess
from typing import Optional
from pathlib import Path


EGOHUB_MODEL_HOME = Path(
    os.getenv("EGOMODEL_HOME", str(Path.home() / ".cache/egohub/models"))
).expanduser()


def _run(cmd: list[str], cwd: Optional[Path] = None) -> tuple[int, str, str]:
    proc = subprocess.run(
        cmd, cwd=str(cwd) if cwd else None, capture_output=True, text=True
    )
    return proc.returncode, proc.stdout, proc.stderr


def ensure_repo(folder: str, git_url: str) -> Path:
    """Clone a repo into EGOMODEL_HOME if missing and return its path."""
    EGOHUB_MODEL_HOME.mkdir(parents=True, exist_ok=True)
    dest = EGOHUB_MODEL_HOME / folder
    if dest.exists() and (dest / ".git").exists():
        return dest
    if dest.exists():
        shutil.rmtree(dest)
    code, out, err = _run(["git", "clone", "--depth", "1", git_url, str(dest)])
    if code != 0:
        raise RuntimeError(f"git clone failed for {git_url}: {err}")
    return dest


def have_colmap() -> bool:
    code, out, err = _run(["colmap", "--version"])  # type: ignore
    return code == 0


def ensure_colmap() -> None:
    if have_colmap():
        return
    # Best effort install on macOS via brew
    if shutil.which("brew"):
        _run(["brew", "install", "colmap"])  # ignore errors; user may not want brew
    # Recheck silently
    _run(["colmap", "--version"])  # may still fail; caller can handle
