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
from jaxtyping import Bool, Float
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
    world_R_cam: Float[ndarray, "3 3"] | None = None
    world_t_cam: Float[ndarray, "3"] | None = None
    cam_R_world: Float[ndarray, "3 3"] | None = None
    cam_t_world: Float[ndarray, "3"] | None = None
    
    # The projection matrix and transformation matrices will be computed in post-init
    world_T_cam: Float[ndarray, "4 4"] = field(init=False)
    cam_T_world: Float[ndarray, "4 4"] = field(init=False)

    def __post_init__(self) -> None:
        self.compute_transformation_matrices()

    def compute_transformation_matrices(self) -> None:
        """Compute transformation matrices from rotation and translation."""
        # If world-to-camera is provided, compute the transformation matrix and its inverse
        if self.world_R_cam is not None and self.world_t_cam is not None:
            self.world_T_cam = self.compose_transformation_matrix(
                self.world_R_cam, self.world_t_cam
            )
            self.cam_T_world = np.linalg.inv(self.world_T_cam)
            # Extract camera-to-world rotation and translation from the inverse matrix
            self.cam_R_world, self.cam_t_world = self.decompose_transformation_matrix(self.cam_T_world)
        # If camera-to-world is provided, compute the transformation matrix and its inverse
        elif self.cam_R_world is not None and self.cam_t_world is not None:
            self.cam_T_world = self.compose_transformation_matrix(
                self.cam_R_world, self.cam_t_world
            )
            self.world_T_cam = np.linalg.inv(self.cam_T_world)
            # Extract world-to-camera rotation and translation from the inverse matrix
            self.world_R_cam, self.world_t_cam = self.decompose_transformation_matrix(self.world_T_cam)
        else:
            raise ValueError("Either world-to-camera or camera-to-world rotation and translation must be provided.")

    def compose_transformation_matrix(self, R: Float[ndarray, "3 3"], t: Float[ndarray, "3"]) -> Float[ndarray, "4 4"]:
        """Compose a 4x4 transformation matrix from rotation and translation."""
        Rt: Float[ndarray, "3 4"] = np.hstack([R, rearrange(t, "c -> c 1")])
        T: Float[ndarray, "4 4"] = np.vstack([Rt, np.array([0, 0, 0, 1])])
        return T

    def decompose_transformation_matrix(
        self, T: Float[ndarray, "4 4"]
    ) -> tuple[Float[ndarray, "3 3"], Float[ndarray, "3"]]:
        """Decompose a 4x4 transformation matrix into rotation and translation."""
        R: Float[ndarray, "3 3"] = T[:3, :3]
        t: Float[ndarray, "3"] = T[:3, 3]
        return R, t


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
    k_matrix: Float[ndarray, "3 3"] = field(init=False)

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
    """Complete pinhole camera parameters with automatic projection matrix computation."""
    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)
    distortion: Distortion | None = None

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        """Compute the projection matrix using K-matrix and world_T_cam."""
        self.projection_matrix = self.intrinsics.k_matrix @ self.extrinsics.cam_T_world[:3, :]


@dataclass
class Fisheye62Parameters:
    """Fisheye camera parameters with 6 radial (k) parameters and 2 tangential (p) distortion parameters."""
    name: str
    extrinsics: Extrinsics
    intrinsics: Intrinsics
    projection_matrix: Float[ndarray, "3 4"] = field(init=False)
    distortion: Distortion | None = None

    def __post_init__(self) -> None:
        self.compute_projection_matrix()

    def compute_projection_matrix(self) -> None:
        """Compute the projection matrix using K-matrix and world_T_cam."""
        self.projection_matrix = self.intrinsics.k_matrix @ self.extrinsics.cam_T_world[:3, :]


def to_homogeneous(
    points: Float[np.ndarray, "num_points _"],
) -> Float[np.ndarray, "num_points _"]:
    """Convert a set of 3D points to homogeneous coordinates.

    Args:
        points: A numpy array containing the 3D coordinates of the points.

    Returns:
        A numpy array containing the homogeneous coordinates of the points.
    """
    ones_column: Float[ndarray, "num_points 1"] = np.ones((points.shape[0], 1), dtype=points.dtype)
    return np.hstack([points, ones_column])


def from_homogeneous(
    points_hom: Float[np.ndarray, "num_points _"],
) -> Float[np.ndarray, "num_points _"]:
    """Convert a set of 3D points from homogeneous coordinates to Euclidean coordinates.

    Args:
        points_hom: A numpy array containing the homogeneous coordinates of the points.

    Returns:
        A numpy array containing the 3D coordinates of the points.
    """
    points = points_hom / points_hom[:, 3:]
    return points[:, :3]


def rescale_intri(camera_intrinsics: Intrinsics, *, target_width: int, target_height: int) -> Intrinsics:
    """Rescale camera intrinsics for a different image size.

    Args:
        camera_intrinsics: The camera intrinsics to rescale.
        target_width: Target image width.
        target_height: Target image height.

    Returns:
        The rescaled camera intrinsics.
    """
    assert camera_intrinsics.height is not None, "Set Camera Height, currently None"
    assert camera_intrinsics.width is not None, "Set Camera Width, currently None"
    
    x_scale: float = target_width / camera_intrinsics.width
    y_scale: float = target_height / camera_intrinsics.height

    new_fl_x: float = camera_intrinsics.fl_x * x_scale
    new_fl_y: float = camera_intrinsics.fl_y * y_scale

    rescaled_intri = Intrinsics(
        camera_conventions=camera_intrinsics.camera_conventions,
        fl_x=new_fl_x,
        fl_y=new_fl_y,
        cx=camera_intrinsics.cx * x_scale,
        cy=camera_intrinsics.cy * y_scale,
        height=target_height,
        width=target_width,
    )
    return rescaled_intri


def perspective_projection(
    points_3d: Float[np.ndarray, "num_points 3"], K: Float[np.ndarray, "3 3"]
) -> Float[np.ndarray, "num_points 2"]:
    """Project 3D points using perspective projection.

    Args:
        points_3d: 3D points in camera coordinates.
        K: Camera intrinsic matrix.

    Returns:
        2D projected points.
    """
    # Convert to homogeneous coordinates
    points_hom = to_homogeneous(points_3d)
    
    # Project using camera matrix
    projected_hom = (K @ points_hom.T).T
    
    # Convert back to Euclidean coordinates
    projected_2d = from_homogeneous(projected_hom)
    
    return projected_2d


def arctan_projection(
    points_3d: Float[np.ndarray, "num_points 3"], K: Float[np.ndarray, "3 3"]
) -> Float[np.ndarray, "num_points 2"]:
    """Project 3D points using arctan projection (for fisheye cameras).

    Args:
        points_3d: 3D points in camera coordinates.
        K: Camera intrinsic matrix.

    Returns:
        2D projected points.
    """
    # Extract focal lengths and principal point
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]
    
    # Compute angles
    r = np.sqrt(points_3d[:, 0]**2 + points_3d[:, 1]**2)
    theta = np.arctan2(r, points_3d[:, 2])
    
    # Project using arctan model
    u = fx * theta * points_3d[:, 0] / r + cx
    v = fy * theta * points_3d[:, 1] / r + cy
    
    return np.column_stack([u, v])


def apply_radial_tangential_distortion(
    dist_coeffs: Distortion, points2d: Float[np.ndarray, "num_points 2"]
) -> Float[np.ndarray, "num_points 2"]:
    """Apply radial and tangential distortion to 2D points.

    Args:
        dist_coeffs: Distortion coefficients.
        points2d: 2D points to distort.

    Returns:
        Distorted 2D points.
    """
    x, y = points2d[:, 0], points2d[:, 1]
    
    # Convert to normalized coordinates
    x_norm = x
    y_norm = y
    
    # Compute radius
    r2 = x_norm**2 + y_norm**2
    r4 = r2**2
    r6 = r2**3
    
    # Apply radial distortion
    k1, k2, k3 = dist_coeffs.k1, dist_coeffs.k2, dist_coeffs.k3
    radial = 1 + k1 * r2 + k2 * r4 + k3 * r6
    
    # Apply tangential distortion
    p1, p2 = dist_coeffs.p1, dist_coeffs.p2
    tangential_x = 2 * p1 * x_norm * y_norm + p2 * (r2 + 2 * x_norm**2)
    tangential_y = p1 * (r2 + 2 * y_norm**2) + 2 * p2 * x_norm * y_norm
    
    # Apply distortion
    x_dist = x_norm * radial + tangential_x
    y_dist = y_norm * radial + tangential_y
    
    return np.column_stack([x_dist, y_dist])


def fisheye_projection(
    points_3d_world: Float[ndarray, "num_points 3"], camera: Fisheye62Parameters
) -> Float[ndarray, "num_points 2"]:
    """Project 3D world points using fisheye camera model.

    Args:
        points_3d_world: 3D points in world coordinates.
        camera: Fisheye camera parameters.

    Returns:
        2D projected points.
    """
    # Transform world points to camera coordinates
    points_hom = to_homogeneous(points_3d_world)
    points_cam = (camera.extrinsics.world_T_cam @ points_hom.T).T
    points_cam = from_homogeneous(points_cam)
    
    # Project using fisheye model
    projected = arctan_projection(points_cam, camera.intrinsics.k_matrix)
    
    # Apply distortion if available
    if camera.distortion is not None:
        projected = apply_radial_tangential_distortion(camera.distortion, projected)
    
    return projected 