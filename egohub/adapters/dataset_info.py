from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass(frozen=True)
class DatasetInfo:
    """Dataset-specific metadata required by exporters/visualisers."""

    # --- Camera ---
    camera_intrinsics: np.ndarray  # shape (3,3)
    view_coordinates: str = "RDF"
    frame_rate: float = 30.0

    # --- Skeleton ---
    joint_names: List[str] = field(default_factory=list)
    joint_hierarchy: Dict[str, str] = field(default_factory=dict)
    joint_remap: Dict[str, str] = field(default_factory=dict)

    # --- Optional camera distortion ---
    camera_distortion: Optional[Dict[str, Any]] = None

    # --- Depth / scale ---
    depth_scale: float = 1.0

    # --- Object detection palette ---
    object_categories: List[str] = field(default_factory=list)
    object_palette: Optional[np.ndarray] = None  # shape (N,3) uint8

    # --- Hand model parameters ---
    mano_betas_left: Optional[np.ndarray] = None  # (10,)
    mano_betas_right: Optional[np.ndarray] = None

    # --- Modalities present ---
    modalities: Dict[str, bool] = field(
        default_factory=lambda: {
            "rgb": True,
            "depth": False,
            "pointcloud": False,
            "imu": False,
        }
    )

    def get_mano_betas(self, side: str) -> Optional[np.ndarray]:
        side = side.lower()
        if side == "left":
            return self.mano_betas_left
        if side == "right":
            return self.mano_betas_right
        raise ValueError(f"Side must be 'left' or 'right', got {side}")
