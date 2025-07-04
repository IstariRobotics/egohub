"""Base class for data processing tools."""
from __future__ import annotations

from abc import ABC, abstractmethod
import h5py


class BaseTool(ABC):
    """Abstract base class for all data processing tools.

    A tool is a callable class that performs a specific processing task on a
    single trajectory group within an HDF5 file.
    """

    @abstractmethod
    def __call__(self, traj_group: h5py.Group) -> None:
        """Process a single trajectory group in-place.

        Args:
            traj_group: The HDF5 group for a single trajectory.
                        (e.g., f['trajectory_0000'])
        """
        raise NotImplementedError 