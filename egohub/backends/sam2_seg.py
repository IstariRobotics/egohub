from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.utils.auto_setup import ensure_repo
from egohub.utils.checkpoints import CheckpointResolver
from egohub.utils.video_utils import hdf5_to_cv2_video

logger = logging.getLogger(__name__)


class Sam2SegmentationBackend(BaseBackend):
    """
    Minimal SAM2 segmentation wrapper.

    For now, this backend supports a bounding-box prompt per frame and returns
    per-frame bounding boxes (possibly refined by a mask in the future).

    Expected kwargs:
      - object_label: str
      - prompt_bbox: Optional[Tuple[int,int,int,int]] (x, y, w, h) to seed masks
      - max_frames: Optional[int] to limit processing
    """

    def __init__(
        self,
        sam2_repo: Optional[str] = None,
        sam2_weights: Optional[str] = None,
        **_: Any,
    ) -> None:
        resolver = CheckpointResolver()
        # Hardcode default: auto-clone SAM2 into cache on first use
        self.repo_path = resolver.resolve_repo("sam2", explicit=sam2_repo)
        if self.repo_path is None:
            self.repo_path = ensure_repo(
                "sam2", "https://github.com/facebookresearch/sam2.git"
            )
        self.sam2_weights = resolver.resolve_checkpoint("sam2", explicit=sam2_weights)

    def run(self, traj_group: h5py.Group, **kwargs: Any) -> Dict[str, Any]:
        object_label: str = kwargs.get("object_label", "object")
        prompt_bbox: Optional[Tuple[int, int, int, int]] = kwargs.get("prompt_bbox")
        max_frames: Optional[int] = kwargs.get("max_frames")

        # Allow passing bbox as a string like "x,y,w,h" via CLI backend-args
        if (
            prompt_bbox is None
            and "prompt_bbox" in kwargs
            and isinstance(kwargs["prompt_bbox"], str)
        ):
            try:
                raw = kwargs["prompt_bbox"].replace(" ", ",")
                parts = [int(p) for p in raw.split(",") if p]
                if len(parts) == 4:
                    prompt_bbox = (parts[0], parts[1], parts[2], parts[3])
            except Exception:
                logger.warning(
                    "Failed to parse prompt_bbox from string; using fallback."
                )

        frames = list(hdf5_to_cv2_video(traj_group["cameras"]["ego_camera"]["rgb"]))
        if max_frames is not None:
            frames = frames[: int(max_frames)]

        bboxes: List[Tuple[int, int, int, int]] = []
        frame_indices: List[int] = []

        for idx, frame in enumerate(frames):
            if prompt_bbox is not None:
                x, y, w, h = prompt_bbox
            else:
                # Fallback: full-frame bbox as placeholder
                h_img, w_img = frame.shape[:2]
                x, y, w, h = 0, 0, int(w_img), int(h_img)

            bboxes.append((x, y, w, h))
            frame_indices.append(idx)

        return {
            "objects": {
                object_label: {
                    "bboxes_2d": np.array(bboxes, dtype=np.float32),
                    "frame_indices": np.array(frame_indices, dtype=np.uint64),
                }
            }
        }
