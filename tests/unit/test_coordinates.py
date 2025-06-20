"""
Unit tests for coordinate transformation functions.

Tests the conversion from ARKit coordinate system to our canonical Z-up system.
"""

import numpy as np
import pytest

from egohub.transforms.coordinates import (
    arkit_to_canonical_transform,
    transform_pose,
    transform_poses_batch,
    arkit_to_canonical_pose,
    arkit_to_canonical_poses,
)


class TestCoordinateTransforms:
    """Test coordinate transformation functions."""
    
    def test_arkit_to_canonical_transform_shape(self):
        """Test that the transformation matrix has the correct shape."""
        transform = arkit_to_canonical_transform()
        assert transform.shape == (4, 4)
        assert transform.dtype == np.float32
    
    def test_arkit_to_canonical_transform_properties(self):
        """Test that the transformation matrix has expected properties."""
        transform = arkit_to_canonical_transform()
        
        # Should be a valid homogeneous transformation matrix
        assert np.allclose(transform[3, :], [0, 0, 0, 1])
        
        # Should be orthogonal (rotation part)
        rotation_part = transform[:3, :3]
        should_be_identity = rotation_part @ rotation_part.T
        assert np.allclose(should_be_identity, np.eye(3), atol=1e-6)
    
    def test_transform_pose_identity(self):
        """Test transforming with identity matrix."""
        identity_pose = np.eye(4, dtype=np.float32)
        identity_transform = np.eye(4, dtype=np.float32)
        
        result = transform_pose(identity_pose, identity_transform)
        assert np.allclose(result, identity_pose)
    
    def test_transform_pose_invalid_shapes(self):
        """Test that transform_pose raises errors for invalid shapes."""
        valid_pose = np.eye(4)
        invalid_pose = np.eye(3)
        valid_transform = np.eye(4)
        invalid_transform = np.eye(3)
        
        with pytest.raises(ValueError, match="Expected 4x4 pose matrix"):
            transform_pose(invalid_pose, valid_transform)
            
        with pytest.raises(ValueError, match="Expected 4x4 transform matrix"):
            transform_pose(valid_pose, invalid_transform)
    
    def test_transform_poses_batch_shape(self):
        """Test batch transformation with correct shapes."""
        n_poses = 5
        poses = np.tile(np.eye(4), (n_poses, 1, 1)).astype(np.float32)
        transform_matrix = np.eye(4, dtype=np.float32)
        
        result = transform_poses_batch(poses, transform_matrix)
        assert result.shape == (n_poses, 4, 4)
    
    def test_transform_poses_batch_invalid_shapes(self):
        """Test that batch transform raises errors for invalid shapes."""
        invalid_poses = np.eye(4)  # Missing batch dimension
        valid_transform = np.eye(4)
        
        with pytest.raises(ValueError, match="Expected poses of shape"):
            transform_poses_batch(invalid_poses, valid_transform)
    
    def test_arkit_to_canonical_coordinate_mapping(self):
        """Test that ARKit coordinates map correctly to canonical coordinates."""
        # Create a test pose at origin with standard orientation
        arkit_pose = np.eye(4, dtype=np.float32)
        
        canonical_pose = arkit_to_canonical_pose(arkit_pose)
        
        # The transformation should maintain the origin
        assert np.allclose(canonical_pose[3, :3], [0, 0, 0])
        assert canonical_pose[3, 3] == 1.0
        
        # Test specific coordinate mappings
        # ARKit Y-up should become canonical Z-up
        arkit_y_axis = np.array([0, 1, 0, 0])  # Y direction in ARKit
        canonical_result = canonical_pose @ arkit_y_axis
        # Should map to Z direction in canonical system
        expected_canonical_z = np.array([0, 0, 1, 0])
        assert np.allclose(canonical_result, expected_canonical_z)
    
    def test_arkit_to_canonical_poses_batch(self):
        """Test batch conversion from ARKit to canonical coordinates."""
        n_poses = 3
        arkit_poses = np.tile(np.eye(4), (n_poses, 1, 1)).astype(np.float32)
        
        canonical_poses = arkit_to_canonical_poses(arkit_poses)
        
        assert canonical_poses.shape == (n_poses, 4, 4)
        
        # Each pose should be the same since we used identity matrices
        for i in range(n_poses):
            for j in range(n_poses):
                assert np.allclose(canonical_poses[i], canonical_poses[j])
    
    def test_coordinate_system_handedness(self):
        """Test that the coordinate system remains right-handed after transformation."""
        transform = arkit_to_canonical_transform()
        
        # Extract rotation matrix
        rotation = transform[:3, :3]
        
        # Determinant should be +1 for right-handed system
        det = np.linalg.det(rotation)
        assert np.isclose(det, 1.0, atol=1e-6) 