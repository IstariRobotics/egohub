from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.utils.auto_setup import ensure_repo
from egohub.utils.checkpoints import CheckpointResolver

logger = logging.getLogger(__name__)


class SpaTrackerBackend(BaseBackend):
    """
    Wrapper for SpaTrackerV2 offline tracking.

    MVP: consume existing `objects/{label}/bboxes_2d` and produce contiguous
    `track_ids` per frame. For now, a simple constant-ID passthrough is used as
    placeholder; integrate real SpaTrackerV2 next.

    Expected kwargs:
      - object_label: str
    """

    def __init__(self, spatracker_repo: Optional[str] = None, **_: Any) -> None:
        resolver = CheckpointResolver()
        self.repo = resolver.resolve_repo("spatrackerv2", explicit=spatracker_repo)
        if self.repo is None:
            self.repo = ensure_repo(
                "spatrackerv2", "https://github.com/henry123-boy/SpaTrackerV2.git"
            )

    def run(self, traj_group: h5py.Group, **kwargs: Any) -> Dict[str, Any]:
        object_label: str = kwargs.get("object_label", "object")

        objects_group = traj_group.get("objects")
        if (
            not isinstance(objects_group, h5py.Group)
            or object_label not in objects_group
        ):
            logger.warning("No segmentation outputs found for label '%s'", object_label)
            return {}

        label_group = objects_group[object_label]
        if "bboxes_2d" not in label_group or "frame_indices" not in label_group:
            logger.warning("Missing bboxes_2d/frame_indices for '%s'", object_label)
            return {}

        frame_indices = np.asarray(label_group["frame_indices"])  # (N,)
        # Placeholder: assign a single track id for all frames
        track_ids = np.zeros_like(frame_indices, dtype=np.int32)

        return {
            "objects": {
                object_label: {
                    "track_ids": track_ids,
                    "frame_indices": frame_indices,
                }
            }
        }
