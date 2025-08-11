from __future__ import annotations

import os
from pathlib import Path
from typing import Optional


class CheckpointResolver:
    """
    Resolve external repo paths and model checkpoints in a consistent way.

    Priority order for a given key (e.g., "SAM2"):
      1) Explicit function argument
      2) Environment variable (e.g., SAM2_HOME / SAM2_CHECKPOINT)
      3) Default under EGOMODEL_HOME (~/.cache/egohub/models)
    """

    def __init__(self, model_home_env: str = "EGOMODEL_HOME") -> None:
        default_home = Path.home() / ".cache" / "egohub" / "models"
        self.model_home = Path(os.getenv(model_home_env, default_home)).expanduser()
        self.model_home.mkdir(parents=True, exist_ok=True)

    def resolve_repo(self, key: str, explicit: Optional[str] = None) -> Optional[Path]:
        if explicit:
            p = Path(explicit).expanduser()
            return p if p.exists() else None
        env = os.getenv(f"{key.upper()}_HOME")
        if env:
            p = Path(env).expanduser()
            return p if p.exists() else None
        # Default location under model home
        candidate = self.model_home / key.lower()
        return candidate if candidate.exists() else None

    def resolve_checkpoint(
        self, key: str, filename: Optional[str] = None, explicit: Optional[str] = None
    ) -> Optional[Path]:
        if explicit:
            p = Path(explicit).expanduser()
            return p if p.exists() else None
        env = os.getenv(f"{key.upper()}_CHECKPOINT")
        if env:
            p = Path(env).expanduser()
            return p if p.exists() else None
        if filename:
            candidate = self.model_home / key.lower() / filename
            return candidate if candidate.exists() else None
        return None
