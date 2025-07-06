from __future__ import annotations

from abc import ABC, abstractmethod

import h5py

from egohub.backends.base import BaseBackend


class BaseTask(ABC):
    def __init__(self, output_group_name: str):
        self.output_group_name = output_group_name

    @abstractmethod
    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs):
        """
        Runs the task using a specified backend.

        Args:
            traj_group: The HDF5 group for the trajectory to process.
            backend: The backend instance to use for inference.
            **kwargs: Backend-specific arguments.
        """
        raise NotImplementedError
