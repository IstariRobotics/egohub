from __future__ import annotations

import logging
from typing import Dict

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class ObjectPoseEstimationTask(BaseTask):
    def __init__(self, output_group_name: str = "objects"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs) -> None:
        logger.info(
            "Running ObjectPoseEstimationTask with backend: %s",
            backend.__class__.__name__,
        )
        results: Dict[str, Dict] = backend.run(traj_group, **kwargs)
        if not results:
            logger.warning("Backend returned no results. Skipping HDF5 write.")
            return

        objects = results.get("objects")
        if not objects:
            logger.warning("No objects found in backend results.")
            return

        # Overwrite the output group if it exists
        if self.output_group_name in traj_group:
            del traj_group[self.output_group_name]
        output_group = traj_group.create_group(self.output_group_name)

        for label, obj in objects.items():
            obj_group = output_group.create_group(label)
            center = np.asarray(obj.get("center", np.zeros((0, 3), dtype=np.float32)))
            dimensions = np.asarray(
                obj.get("dimensions", np.zeros((3,), dtype=np.float32))
            )
            rotation = np.asarray(
                obj.get("rotation", np.zeros((0, 3), dtype=np.float32))
            )
            frame_indices = np.asarray(
                obj.get("frame_indices", np.zeros((center.shape[0],), dtype=np.uint64))
            )

            obj_group.create_dataset("center", data=center)
            obj_group.create_dataset("dimensions", data=dimensions)
            obj_group.create_dataset("rotation", data=rotation)
            obj_group.create_dataset("frame_indices", data=frame_indices)

            # Optional: if backend provided full 4x4 pose matrices, persist them
            pose_mats = obj.get("pose_matrices")
            if pose_mats is not None:
                pose_mats = np.asarray(pose_mats, dtype=np.float32)
                obj_group.create_dataset("pose_matrices", data=pose_mats)

        logger.info("Saved %d object tracks to '%s'", len(objects), output_group.name)
