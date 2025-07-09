"""
Advanced camera parameter system for egocentric data processing.

This module provides comprehensive camera parameter classes with automatic
transformation matrix computation, support for different camera conventions,
and distortion models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from jaxtyping import Float

# from einops import rearrange


@dataclass
class Distortion:
    """Camera distortion parameters."""

    k1: float = 0.0
    k2: float = 0.0
    p1: float = 0.0
    p2: float = 0.0
    k3: float = 0.0
    k4: Optional[float] = None
    k5: Optional[float] = None
    k6: Optional[float] = None


@dataclass
class Extrinsics:
    """Represents the camera extrinsics (pose) in the world frame."""

    # These fields are mutually exclusive and are used to initialize
    # the object from different representations. The properties below
    # ensure that they are consistent.
    world_r_cam: Optional[Float[np.ndarray, "3 3"]] = None  # noqa: F722
    world_t_cam: Optional[Float[np.ndarray, "3"]] = None  # noqa: F722
    cam_r_world: Optional[Float[np.ndarray, "3 3"]] = None  # noqa: F722
    cam_t_world: Optional[Float[np.ndarray, "3"]] = None  # noqa: F722

    def __post_init__(self):
        # Validate that exactly one representation is provided
        r_provided = self.world_r_cam is not None and self.world_t_cam is not None
        r_inv_provided = self.cam_r_world is not None and self.cam_t_world is not None

        if not (r_provided ^ r_inv_provided):
            raise ValueError(
                "Must provide exactly one of (world_r_cam, world_t_cam) or "
                "(cam_r_world, cam_t_world)."
            )

        # Ensure consistency
        if r_provided:
            assert self.world_r_cam is not None and self.world_t_cam is not None
            self.cam_r_world = np.transpose(self.world_r_cam)
            self.cam_t_world = -np.transpose(self.world_r_cam) @ self.world_t_cam
        else:  # r_inv_provided
            assert self.cam_r_world is not None and self.cam_t_world is not None
            self.world_r_cam = np.transpose(self.cam_r_world)
            self.world_t_cam = -np.transpose(self.cam_r_world) @ self.cam_t_world

    @property
    def world_h_cam(self) -> np.ndarray:
        """Homogeneous transformation from camera to world coordinates."""
        h = np.eye(4)
        h[:3, :3] = self.world_r_cam
        h[:3, 3] = self.world_t_cam
        return h

    @property
    def cam_h_world(self) -> np.ndarray:
        """Homogeneous transformation from world to camera coordinates."""
        h = np.eye(4)
        h[:3, :3] = self.cam_r_world
        h[:3, 3] = self.cam_t_world
        return h


@dataclass
class CameraIntrinsics:
    """Camera intrinsic parameters."""

    fx: float
    fy: float
    cx: float
    cy: float
    height: Optional[int] = None
    width: Optional[int] = None

    @property
    def k(self) -> np.ndarray:
        """The 3x3 camera matrix."""
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]])


@dataclass
class CameraParameters:
    """A container for all camera parameters."""

    intrinsics: CameraIntrinsics
    extrinsics: Extrinsics
    distortion: Optional[Distortion] = None

    @classmethod
    def from_egodex(cls, data: dict, frame_idx: int = 0) -> "CameraParameters":
        """Factory method to create CameraParameters from EgoDex HDF5 data."""

        intrinsics_data = data["intrinsics"][:]
        intrinsics = CameraIntrinsics(
            fx=intrinsics_data[0, 0],
            fy=intrinsics_data[1, 1],
            cx=intrinsics_data[0, 2],
            cy=intrinsics_data[1, 2],
            width=data["color_size"][1],
            height=data["color_size"][0],
        )

        extrinsics_data = data["extrinsics"][frame_idx]
        extrinsics = Extrinsics(
            world_r_cam=extrinsics_data[:3, :3], world_t_cam=extrinsics_data[:3, 3]
        )

        distortion_data = data.get("distortion")
        distortion = (
            Distortion(*distortion_data[:].tolist()) if distortion_data else None
        )

        return cls(intrinsics=intrinsics, extrinsics=extrinsics, distortion=distortion)
