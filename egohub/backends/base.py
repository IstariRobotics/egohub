from __future__ import annotations

from abc import ABC, abstractmethod


class BaseBackend(ABC):
    @abstractmethod
    def run(self, **kwargs):
        """
        Runs the backend-specific logic.

        This method should be implemented by all subclasses to perform
        the actual model loading and inference.
        """
        raise NotImplementedError
