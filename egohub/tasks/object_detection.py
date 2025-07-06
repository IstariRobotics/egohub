from __future__ import annotations

import logging

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class ObjectDetectionTask(BaseTask):
    def __init__(self, output_group_name: str = "objects"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs):
        """
        Runs object detection using the specified backend and saves the results.

        Args:
            traj_group: The HDF5 group for the trajectory to process.
            backend: The backend instance to use for inference.
        """
        logger.info(
            f"Running ObjectDetectionTask with backend: {backend.__class__.__name__}"
        )
        results = backend.run(traj_group, **kwargs)

        if not results:
            logger.warning("Backend returned no results. Skipping HDF5 write.")
            return

        detections = results.get("detections")
        num_frames = results.get("num_frames")

        if detections is None or num_frames is None:
            logger.error("Backend results missing 'detections' or 'num_frames'.")
            return

        # Overwrite the output group if it exists
        if self.output_group_name in traj_group:
            del traj_group[self.output_group_name]
        output_group = traj_group.create_group(self.output_group_name)

        logger.info(
            f"Writing {len(detections)} detected object classes to HDF5 group "
            f"'{self.output_group_name}'."
        )
        for label, dets in detections.items():
            obj_group = output_group.create_group(label)

            bboxes_2d = np.zeros((num_frames, 4), dtype=np.float32)
            scores = np.zeros(num_frames, dtype=np.float32)

            for frame_idx, bbox, score in dets:
                bboxes_2d[frame_idx] = bbox
                scores[frame_idx] = score

            obj_group.create_dataset("bboxes_2d", data=bboxes_2d)
            obj_group.create_dataset("scores", data=scores)
