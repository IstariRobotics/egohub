from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import h5py


class BaseBackend(ABC):
    @abstractmethod
    def run(self, traj_group: h5py.Group, **kwargs: Any) -> dict:
        """
        Runs the backend-specific logic.

        This method should be implemented by all subclasses to perform
        the actual model loading and inference.
        """
        raise NotImplementedError


def get_backend(backend_name: str) -> type[BaseBackend]:
    """Retrieves a backend class by its name."""
    from egohub import backends

    backend_cls = getattr(backends, backend_name, None)
    if backend_cls is None:
        raise ValueError(f"Unknown backend: {backend_name}")
    if not issubclass(backend_cls, BaseBackend):
        raise TypeError(f"{backend_name} is not a valid Backend class.")
    return backend_cls
