"""
Advanced camera parameter system for egocentric data processing.

This module provides comprehensive camera parameter classes with automatic
transformation matrix computation, support for different camera conventions,
and distortion models.
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from einops import rearrange
from jaxtyping import Float
from numpy import ndarray


@dataclass
class Distortion:
    """Brown Conrady distortion model."""

    k1: float
    k2: float
    p1: float
    p2: float
    k3: float
    k4: float | None = None
    k5: float | None = None
    k6: float | None = None


@dataclass
class Extrinsics:
    """Camera extrinsics with automatic transformation matrix computation.

    Rotation and translation can be provided for both world-to-camera
    and camera-to-world transformations. The system automatically computes
    the inverse transformation.
    """

    world_r_cam: Float[ndarray, "3 3"] | None = None  # noqa: F722
    world_t_cam: Float[ndarray, "3"] | None = None  # noqa: F722
    cam_r_world: Float[ndarray, "3 3"] | None = None  # noqa: F722
    cam_t_world: Float[ndarray, "3"] | None = None  # noqa: F722

    # The projection matrix and transformation matrices will be computed in post-init
    world_t_cam_matrix: Float[ndarray, "4 4"] = field(init=False)  # noqa: F722, N815
    cam_t_world_matrix: Float[ndarray, "4 4"] = field(init=False)  # noqa: F722, N815

    def __post_init__(self) -> None:
        self.compute_transformation_matrices()

    def compute_transformation_matrices(self) -> None:
        """Compute transformation matrices from rotation and translation."""
        # If world-to-camera is provided, compute the transformation matrix and
        # its inverse
        if self.world_r_cam is not None and self.world_t_cam is not None:
            self.world_t_cam_matrix = self.compose_transformation_matrix(
                self.world_r_cam, self.world_t_cam
            )
            self.cam_t_world_matrix = np.linalg.inv(self.world_t_cam_matrix)
            # Extract camera-to-world rotation and translation from the inverse matrix
            self.cam_r_world, self.cam_t_world = self.decompose_transformation_matrix(
                self.cam_t_world_matrix
            )
        # If camera-to-world is provided, compute the transformation matrix and
        # its inverse
        elif self.cam_r_world is not None and self.cam_t_world is not None:
            self.cam_t_world_matrix = self.compose_transformation_matrix(
                self.cam_r_world, self.cam_t_world
            )
            self.world_t_cam_matrix = np.linalg.inv(self.cam_t_world_matrix)
            # Extract world-to-camera rotation and translation from the inverse matrix
            self.world_r_cam, self.world_t_cam = self.decompose_transformation_matrix(
                self.world_t_cam_matrix
            )
        else:
            raise ValueError(
                "Either world-to-camera or camera-to-world rotation and translation "
                "must be provided."
            )

    def compose_transformation_matrix(
        self, r: Float[ndarray, "3 3"], t: Float[ndarray, "3"]  # noqa: F722, N803
    ) -> Float[ndarray, "4 4"]:  # noqa: F722
        """Compose a 4x4 transformation matrix from rotation and translation."""
        rt: Float[ndarray, "3 4"] = np.hstack(
            [r, rearrange(t, "c -> c 1")]
        )  # noqa: F722, N806
        t_matrix: Float[ndarray, "4 4"] = np.vstack(
            [rt, np.array([0, 0, 0, 1])]
        )  # noqa: F722, N806
        return t_matrix

    def decompose_transformation_matrix(
        self, t_matrix: Float[ndarray, "4 4"]  # noqa: F722, N803
    ) -> tuple[Float[ndarray, "3 3"], Float[ndarray, "3"]]:  # noqa: F722
        """Decompose a 4x4 transformation matrix into rotation and translation."""
        r: Float[ndarray, "3 3"] = t_matrix[:3, :3]  # noqa: F722, N806
        t: Float[ndarray, "3"] = t_matrix[:3, 3]  # noqa: F722, N806
        return r, t


@dataclass
class Intrinsics:
    """Camera intrinsics with automatic K-matrix computation."""

    camera_conventions: Literal["RDF", "RUB"]
    """RDF(OpenCV): X Right - Y Down - Z Front | RUB (OpenGL): X Right- Y Up - Z Back"""
    fl_x: float
    fl_y: float
    cx: float
    cy: float
    height: int | None = None
    width: int | None = None
    k_matrix: Float[ndarray, "3 3"] = field(init=False)  # noqa: F722

    def __post_init__(self):
        self.compute_k_matrix()
        if self.height is None or self.width is None:
            self.height = int(2 * self.cy)
            self.width = int(2 * self.cx)

    def compute_k_matrix(self):
        """Compute the camera matrix using the focal length and principal point."""
        self.k_matrix = np.array(
            [
                [self.fl_x, 0, self.cx],
                [0, self.fl_y, self.cy],
                [0, 0, 1],
            ]
        )

    def __repr__(self):
        return (
            f"Intrinsics(camera_conventions={self.camera_conventions}, "
            f"fl_x={self.fl_x}, fl_y={self.fl_y}, cx={self.cx}, cy={self.cy}, "
            f"height={self.height}, width={self.width})"
        )


@dataclass
class PinholeParameters:
    """Complete pinhole camera parameters with automatic projection matrix
    computation."""

    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)  # noqa: F722
    distortion: Distortion | None = None

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        """Compute the projection matrix using K-matrix and world_T_cam."""
        self.projection_matrix = (
            self.intrinsics.k_matrix @ self.extrinsics.cam_t_world_matrix[:3, :]
        )


@dataclass
class Fisheye62Parameters:
    """Fisheye camera parameters with 6 radial (k) parameters and 2 tangential (p)
    distortion parameters."""

    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)  # noqa: F722
    distortion: Distortion | None = None

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        """Compute the projection matrix using K-matrix and world_T_cam."""
        self.projection_matrix = (
            self.intrinsics.k_matrix @ self.extrinsics.cam_t_world_matrix[:3, :]
        )


def to_homogeneous(
    points: Float[np.ndarray, "num_points _"],  # noqa: F722
) -> Float[np.ndarray, "num_points _"]:  # noqa: F722
    """Convert a set of 3D points to homogeneous coordinates.

    Args:
        points: A numpy array containing the 3D coordinates of the points.

    Returns:
        A numpy array containing the homogeneous coordinates of the points.
    """
    ones_column: Float[ndarray, "num_points 1"] = np.ones(  # noqa: F722
        (points.shape[0], 1), dtype=points.dtype
    )
    return np.hstack([points, ones_column])


def from_homogeneous(
    points_hom: Float[np.ndarray, "num_points _"],  # noqa: F722
) -> Float[np.ndarray, "num_points _"]:  # noqa: F722
    """Convert a set of 3D points from homogeneous coordinates to Euclidean
    coordinates.

    Args:
        points_hom: A numpy array containing the homogeneous coordinates of the
            points.

    Returns:
        A numpy array containing the Euclidean coordinates of the points.
    """
    return points_hom[:, :-1] / points_hom[:, -1:]


def rescale_intri(
    camera_intrinsics: Intrinsics, *, target_width: int, target_height: int
) -> Intrinsics:
    """Rescale camera intrinsics to a new resolution.

    Args:
        camera_intrinsics: The original camera intrinsics.
        target_width: The target width.
        target_height: The target height.

    Returns:
        A new Intrinsics object with rescaled parameters.
    """
    if camera_intrinsics.width is None or camera_intrinsics.height is None:
        raise ValueError("Camera intrinsics must have width and height set.")

    scale_x = target_width / camera_intrinsics.width
    scale_y = target_height / camera_intrinsics.height

    return Intrinsics(
        camera_conventions=camera_intrinsics.camera_conventions,
        fl_x=camera_intrinsics.fl_x * scale_x,
        fl_y=camera_intrinsics.fl_y * scale_y,
        cx=camera_intrinsics.cx * scale_x,
        cy=camera_intrinsics.cy * scale_y,
        width=target_width,
        height=target_height,
    )


def perspective_projection(
    points_3d: Float[np.ndarray, "num_points 3"],
    k: Float[np.ndarray, "3 3"],  # noqa: F722, N803
) -> Float[np.ndarray, "num_points 2"]:  # noqa: F722
    """Project 3D points using perspective projection.

    Args:
        points_3d: A numpy array containing the 3D coordinates of the points.
        k: The camera intrinsic matrix.

    Returns:
        A numpy array containing the 2D projected coordinates of the points.
    """
    # Convert to homogeneous coordinates
    points_hom = to_homogeneous(points_3d)
    # Project using the camera matrix
    projected_hom = (k @ points_hom.T).T
    # Convert back to Euclidean coordinates
    return from_homogeneous(projected_hom)


def arctan_projection(
    points_3d: Float[np.ndarray, "num_points 3"],
    k: Float[np.ndarray, "3 3"],  # noqa: F722, N803
) -> Float[np.ndarray, "num_points 2"]:  # noqa: F722
    """Project 3D points using arctan projection (for fisheye cameras).

    Args:
        points_3d: A numpy array containing the 3D coordinates of the points.
        k: The camera intrinsic matrix.

    Returns:
        A numpy array containing the 2D projected coordinates of the points.
    """
    # Convert to homogeneous coordinates
    points_hom = to_homogeneous(points_3d)
    # Project using the camera matrix
    projected_hom = (k @ points_hom.T).T
    # Convert back to Euclidean coordinates
    return from_homogeneous(projected_hom)


def apply_radial_tangential_distortion(
    dist_coeffs: Distortion, points2d: Float[np.ndarray, "num_points 2"]  # noqa: F722
) -> Float[np.ndarray, "num_points 2"]:  # noqa: F722
    """Apply radial and tangential distortion to 2D points.

    Args:
        dist_coeffs: The distortion coefficients.
        points2d: A numpy array containing the 2D coordinates of the points.

    Returns:
        A numpy array containing the distorted 2D coordinates of the points.
    """
    # Extract distortion coefficients
    k1 = dist_coeffs.k1
    k2 = dist_coeffs.k2
    p1 = dist_coeffs.p1
    p2 = dist_coeffs.p2
    k3 = dist_coeffs.k3

    # Normalize coordinates
    x = points2d[:, 0]
    y = points2d[:, 1]

    # Compute radial distance
    r2 = x * x + y * y
    r4 = r2 * r2
    r6 = r4 * r2

    # Apply radial distortion
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6

    # Apply tangential distortion
    tangential_x = 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
    tangential_y = p1 * (r2 + 2 * y * y) + 2 * p2 * x * y

    # Apply distortion
    x_distorted = x * radial + tangential_x
    y_distorted = y * radial + tangential_y

    return np.column_stack([x_distorted, y_distorted])


def fisheye_projection(
    points_3d_world: Float[ndarray, "num_points 3"],
    camera: Fisheye62Parameters,  # noqa: F722
) -> Float[ndarray, "num_points 2"]:  # noqa: F722
    """Project 3D world points using fisheye camera model.

    Args:
        points_3d_world: A numpy array containing the 3D world coordinates of the
            points.
        camera: The fisheye camera parameters.

    Returns:
        A numpy array containing the 2D projected coordinates of the points.
    """
    # Transform points to camera coordinates
    points_cam = (
        camera.extrinsics.cam_t_world_matrix @ to_homogeneous(points_3d_world).T
    )
    points_cam = points_cam[:3, :].T

    # Project using arctan projection
    return arctan_projection(points_cam, camera.intrinsics.k_matrix)
