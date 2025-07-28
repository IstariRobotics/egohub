from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import h5py
import numpy as np


class BaseBackend(ABC):
    @abstractmethod
    def run(self, traj_group: h5py.Group, **kwargs: Any) -> dict:
        """
        Runs the backend-specific logic.
        """
        raise NotImplementedError

    @abstractmethod
    def get_embeddings(
        self, frames: list[np.ndarray], **kwargs: Any
    ) -> np.ndarray | None:
        """
        Generates embeddings for a list of video frames.
        """
        raise NotImplementedError


def get_backend(backend_name: str) -> type[BaseBackend]:
    """Retrieries a backend class by its name."""
    from egohub import backends

    backend_cls = getattr(backends, backend_name, None)
    if backend_cls is None:
        raise ValueError(f"Unknown backend: {backend_name}")
    if not isinstance(backend_cls, type) or not issubclass(backend_cls, BaseBackend):
        raise TypeError(f"{backend_name} is not a valid Backend class.")
    return backend_cls