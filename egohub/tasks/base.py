from __future__ import annotations

import logging
from abc import ABC, abstractmethod

import h5py

if False:
    from egohub.backends.base import BaseBackend


logger = logging.getLogger(__name__)


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


def get_task(task_name: str) -> type[BaseTask]:
    """Retrieves a task class by its name."""
    from egohub import tasks

    task_cls = getattr(tasks, task_name, None)
    if task_cls is None:
        raise ValueError(f"Unknown task: {task_name}")
    if not issubclass(task_cls, BaseTask):
        raise TypeError(f"{task_name} is not a valid Task class.")
    return task_cls
