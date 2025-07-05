"""
Coordinate frame transformation utilities.

This module provides functions to transform poses and points between different
coordinate systems, particularly from the ARKit coordinate system used by EgoDex
to our canonical Z-up world coordinate system.
"""

import numpy as np
from scipy.spatial.transform import Rotation


def arkit_to_canonical_transform() -> np.ndarray:
    """
    Get the transformation matrix from ARKit's Y-up to our canonical Z-up system.

    ARKit (Source): Right-handed, +Y is up, +X is right, +Z is toward the user.
    Canonical (Target): Right-handed, +Z is up, +Y is forward, +X is right.

    This transformation corresponds to a +90 degree rotation around the X-axis.
    """
    transform = Rotation.from_euler("x", 90, degrees=True).as_matrix()

    # Make it a 4x4 homogeneous matrix
    homogeneous_transform = np.eye(4)
    homogeneous_transform[:3, :3] = transform

    return homogeneous_transform.astype(np.float32)


def transform_pose(pose: np.ndarray, transform_matrix: np.ndarray) -> np.ndarray:
    """
    Apply a coordinate transformation to a 4x4 pose matrix.

    Args:
        pose: 4x4 pose matrix in the source coordinate system.
        transform_matrix: 4x4 transformation matrix from source to target system.

    Returns:
        np.ndarray: 4x4 pose matrix in the target coordinate system.
    """
    if pose.shape != (4, 4):
        raise ValueError(f"Expected 4x4 pose matrix, got shape {pose.shape}")
    if transform_matrix.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 transform matrix, got shape {transform_matrix.shape}"
        )

    return transform_matrix @ pose


def transform_poses_batch(
    poses: np.ndarray, transform_matrix: np.ndarray
) -> np.ndarray:
    """
    Apply coordinate transformation to a batch of pose matrices.

    Args:
        poses: Array of shape (N, 4, 4) containing N pose matrices.
        transform_matrix: 4x4 transformation matrix from source to target system.

    Returns:
        np.ndarray: Array of shape (N, 4, 4) with transformed poses.
    """
    if len(poses.shape) != 3 or poses.shape[1:] != (4, 4):
        raise ValueError(f"Expected poses of shape (N, 4, 4), got {poses.shape}")
    if transform_matrix.shape != (4, 4):
        raise ValueError(
            f"Expected 4x4 transform matrix, got shape {transform_matrix.shape}"
        )

    # Apply transformation to each pose in the batch
    transformed_poses = np.zeros_like(poses)
    for i in range(poses.shape[0]):
        transformed_poses[i] = transform_matrix @ poses[i]

    return transformed_poses


def arkit_to_canonical_pose(arkit_pose: np.ndarray) -> np.ndarray:
    """
    Convert a single pose from ARKit coordinate system to canonical Z-up system.

    Args:
        arkit_pose: 4x4 pose matrix in ARKit coordinate system.

    Returns:
        np.ndarray: 4x4 pose matrix in canonical Z-up coordinate system.
    """
    transform = arkit_to_canonical_transform()
    return transform_pose(arkit_pose, transform)


def arkit_to_canonical_poses(poses: np.ndarray) -> np.ndarray:
    """
    Converts a batch of poses from ARKit's coordinate system to the canonical system.

    Args:
        poses: A numpy array of shape (N, 4, 4) or (N, 68, 3) representing poses.

    Returns:
        A numpy array with the same shape, with poses in the canonical coordinate
        system.
    """
    transform_matrix = arkit_to_canonical_transform()

    # Handle batched 4x4 matrices
    if poses.ndim == 3 and poses.shape[1:] == (4, 4):
        # This is equivalent to: transformed_pose = transform_matrix @ original_pose
        return transform_matrix @ poses

    # Handle batched 3D points (N, K, 3)
    elif poses.ndim == 3 and poses.shape[-1] == 3:
        original_shape = poses.shape
        points = poses.reshape(-1, 3)  # Flatten to (N*K, 3)
        # Apply only the rotation part of the transform
        transformed_points = points @ transform_matrix[:3, :3].T
        return transformed_points.reshape(original_shape)

    # Handle single 4x4 matrix
    elif poses.ndim == 2 and poses.shape == (4, 4):
        return transform_matrix @ poses

    else:
        raise ValueError(f"Unsupported pose shape: {poses.shape}")
