from __future__ import annotations

import logging
from typing import Any, Dict, Optional

import h5py
import numpy as np

from egohub.backends.base import BaseBackend
from egohub.utils.auto_setup import ensure_colmap, ensure_repo
from egohub.utils.checkpoints import CheckpointResolver

logger = logging.getLogger(__name__)


class VGGTCameraDepthBackend(BaseBackend):
    """
    Scaffold for VGGT-based camera/depth estimation.

    MVP: return identity poses and a simple focal length intrinsics for smoke
    testing. Replace with real VGGT/COLMAP outputs in the next step.

    Depth policy: if the HDF5 already contains `cameras/ego_camera/depth`, we do
    NOT estimate depth; we use the provided depth. Only if depth is absent will
    this backend (later) compute/attach a depth map.
    """

    def __init__(
        self,
        vggt_repo: Optional[str] = None,
        vggt_weights: Optional[str] = None,
        **_: Any,
    ) -> None:
        resolver = CheckpointResolver()
        self.repo = resolver.resolve_repo("vggt", explicit=vggt_repo)
        if self.repo is None:
            self.repo = ensure_repo(
                "vggt", "https://github.com/facebookresearch/vggt.git"
            )
        self.weights = resolver.resolve_checkpoint("vggt", explicit=vggt_weights)
        # Ensure COLMAP availability for fallback
        ensure_colmap()

    def run(self, traj_group: h5py.Group, **kwargs: Any) -> Dict[str, np.ndarray]:
        # Determine number of frames from RGB stream
        cameras_group = traj_group.get("cameras")
        if (
            not isinstance(cameras_group, h5py.Group)
            or "ego_camera" not in cameras_group
        ):
            logger.error("No cameras/ego_camera found")
            return {}
        ego = cameras_group["ego_camera"]
        rgb_group = ego.get("rgb")
        if not isinstance(rgb_group, h5py.Group) or "frame_sizes" not in rgb_group:
            logger.error("No RGB frames found to infer pose count")
            return {}

        num_frames = int(len(rgb_group["frame_sizes"]))
        # Simple intrinsics: fx=fy=1000, cx,cy at 640x480 center for smoke
        width, height = 640, 480
        intrinsics = np.array(
            [
                [1000.0, 0.0, width / 2.0],
                [0.0, 1000.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )

        poses = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
        pose_indices = np.arange(num_frames, dtype=np.uint64)

        results: Dict[str, np.ndarray] = {
            "intrinsics": intrinsics,
            "poses": poses,
            "pose_indices": pose_indices,
        }

        # Respect existing depth if present
        if "depth" in ego:
            logger.info("Depth found in HDF5; skipping depth estimation.")
        else:
            logger.info("No depth in HDF5; backend may estimate depth in a later step.")
            # Placeholder: omit depth for MVP. Later: attach estimated depth here.

        return results
