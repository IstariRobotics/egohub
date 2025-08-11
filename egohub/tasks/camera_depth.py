from __future__ import annotations

import logging
from typing import Dict

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class CameraDepthTask(BaseTask):
    """
    Estimates camera intrinsics and per-frame camera poses and writes them to
    `cameras/ego_camera/{intrinsics, pose_in_world, pose_indices}`. Depth is
    optional and may be added by the backend.
    """

    # Not used for group naming; this task writes to `cameras/` directly
    output_group = "cameras/"

    def __init__(self, output_group_name: str = "cameras"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs) -> None:
        logger.info(
            "Running CameraDepthTask with backend: %s", backend.__class__.__name__
        )

        results: Dict[str, np.ndarray] = backend.run(traj_group, **kwargs)
        if not results:
            logger.warning("Backend returned no results. Skipping HDF5 write.")
            return

        cameras_group = traj_group.get("cameras")
        if not isinstance(cameras_group, h5py.Group):
            logger.error("No 'cameras' group found in trajectory.")
            return

        ego_group = cameras_group.get("ego_camera")
        if not isinstance(ego_group, h5py.Group):
            ego_group = cameras_group.create_group("ego_camera")

        # Write intrinsics
        intrinsics = results.get("intrinsics")
        if intrinsics is not None:
            if "intrinsics" in ego_group:
                del ego_group["intrinsics"]
            ego_group.create_dataset("intrinsics", data=intrinsics.astype(np.float32))

        # Write poses
        poses = results.get("poses")
        pose_indices = results.get("pose_indices")
        if poses is not None and pose_indices is not None:
            if "pose_in_world" in ego_group:
                del ego_group["pose_in_world"]
            if "pose_indices" in ego_group:
                del ego_group["pose_indices"]
            ego_group.create_dataset("pose_in_world", data=poses.astype(np.float32))
            ego_group.create_dataset(
                "pose_indices", data=pose_indices.astype(np.uint64)
            )

        logger.info("Wrote camera intrinsics and poses to cameras/ego_camera")
