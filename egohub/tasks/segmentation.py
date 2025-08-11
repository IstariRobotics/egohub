from __future__ import annotations

import logging
from typing import Dict

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class SegmentationTask(BaseTask):
    """
    Runs an image segmentation backend and writes per-frame 2D bounding boxes
    to the HDF5 file under `objects/{label}/bboxes_2d` with corresponding
    `frame_indices`.

    This task focuses on lightweight outputs needed by downstream tracking and
    pose modules. Binary masks can be added later if needed for reconstruction.
    """

    # Hint CLI to use the objects/ group
    output_group = "objects/"

    def __init__(self, output_group_name: str = "objects"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs) -> None:
        logger.info(
            "Running SegmentationTask with backend: %s", backend.__class__.__name__
        )

        results: Dict[str, Dict] = backend.run(traj_group, **kwargs)
        if not results:
            logger.warning("Backend returned no results. Skipping HDF5 write.")
            return

        objects = results.get("objects")
        if not objects:
            logger.warning("No objects found in backend results.")
            return

        # Ensure parent output group exists
        if self.output_group_name not in traj_group:
            traj_group.create_group(self.output_group_name)
        output_root = traj_group[self.output_group_name]

        for label, obj in objects.items():
            # Overwrite existing label group if present
            if label in output_root:
                del output_root[label]
            label_group = output_root.create_group(label)

            bboxes_2d = np.asarray(
                obj.get("bboxes_2d", np.zeros((0, 4), dtype=np.float32)),
                dtype=np.float32,
            )
            frame_indices = np.asarray(
                obj.get("frame_indices", np.zeros((0,), dtype=np.uint64)),
                dtype=np.uint64,
            )

            label_group.create_dataset("bboxes_2d", data=bboxes_2d)
            label_group.create_dataset("frame_indices", data=frame_indices)

        logger.info("Wrote segmentation outputs for %d object(s)", len(objects))
