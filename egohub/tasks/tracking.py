from __future__ import annotations

import logging
from typing import Dict

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logger = logging.getLogger(__name__)


class TrackingTask(BaseTask):
    """
    Consumes prior segmentation outputs and runs a tracking backend to produce
    `track_ids` and `frame_indices` under `objects/{label}`.
    """

    output_group = "objects/"

    def __init__(self, output_group_name: str = "objects"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs) -> None:
        logger.info("Running TrackingTask with backend: %s", backend.__class__.__name__)

        results: Dict[str, Dict] = backend.run(traj_group, **kwargs)
        if not results:
            logger.warning("Backend returned no results. Skipping HDF5 write.")
            return

        objects = results.get("objects")
        if not objects:
            logger.warning("No objects found in backend results.")
            return

        if self.output_group_name not in traj_group:
            traj_group.create_group(self.output_group_name)
        output_root = traj_group[self.output_group_name]

        for label, obj in objects.items():
            if label not in output_root:
                label_group = output_root.create_group(label)
            else:
                label_group = output_root[label]

            track_ids = np.asarray(
                obj.get("track_ids", np.zeros((0,), dtype=np.int32)), dtype=np.int32
            )
            frame_indices = np.asarray(
                obj.get("frame_indices", np.zeros((0,), dtype=np.uint64)),
                dtype=np.uint64,
            )

            # Overwrite datasets if they exist
            for name in ("track_ids", "track_frame_indices"):
                if name in label_group:
                    del label_group[name]

            label_group.create_dataset("track_ids", data=track_ids)
            label_group.create_dataset("track_frame_indices", data=frame_indices)

        logger.info("Wrote tracking outputs for %d object(s)", len(objects))
