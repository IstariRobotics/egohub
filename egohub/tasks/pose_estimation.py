from __future__ import annotations

import logging

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class PoseEstimationTask(BaseTask):
    def __init__(self, output_group_name: str = "skeleton"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs):
        """
        Runs pose estimation using the specified backend and saves the results.

        Args:
            traj_group: The HDF5 group for the trajectory to process.
            backend: The backend instance to use for inference.
        """
        logger.info(
            f"Running PoseEstimationTask with backend: {backend.__class__.__name__}"
        )
        results = backend.run(traj_group, **kwargs)

        if not results:
            logger.warning("Backend returned no results. Skipping HDF5 write.")
            return

        keypoints = results.get("keypoints")
        confidences = results.get("confidences")
        frame_indices = results.get("frame_indices")

        if keypoints is None or confidences is None:
            logger.error("Backend results missing 'keypoints' or 'confidences'.")
            return

        # Overwrite the output group if it exists
        if self.output_group_name in traj_group:
            del traj_group[self.output_group_name]
        output_group = traj_group.create_group(self.output_group_name)

        output_group.create_dataset(
            "positions", data=np.array(keypoints, dtype=np.float32)
        )
        output_group.create_dataset(
            "confidences", data=np.array(confidences, dtype=np.float32)
        )

        # Frame indices might not be returned by all backends
        if frame_indices is not None:
            output_group.create_dataset(
                "frame_indices", data=np.array(frame_indices, dtype=np.int64)
            )
        else:
            # If not provided, assume it's one pose per frame, aligned with positions
            num_frames = len(keypoints)
            output_group.create_dataset(
                "frame_indices", data=np.arange(num_frames, dtype=np.int64)
            )

        logger.info(
            f"Saved {len(keypoints)} poses to HDF5 group: '{output_group.name}'"
        )
