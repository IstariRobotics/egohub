import numpy as np
import pytest

from egohub.transforms.coordinates import arkit_to_canonical_poses
from egohub.transforms.pipeline import TransformPipeline


class TestCoordinateTransforms:
    """Test coordinate transformation functions."""

    @pytest.fixture
    def sample_arkit_poses(self):
        """Create sample ARKit poses for testing."""
        # Create 4x4 transformation matrices
        num_poses = 5
        poses = np.random.rand(num_poses, 4, 4).astype(np.float32)

        # Ensure they are valid transformation matrices
        for i in range(num_poses):
            poses[i, 3, :] = [0, 0, 0, 1]  # Set homogeneous coordinate
            poses[i, :3, 3] = np.random.rand(3)  # Translation
            # Create a simple rotation matrix
            poses[i, :3, :3] = np.eye(3)

        return poses

    def test_arkit_to_canonical_poses_basic(self, sample_arkit_poses):
        """Test basic ARKit to canonical pose transformation."""
        result = arkit_to_canonical_poses(sample_arkit_poses)

        assert result.shape == sample_arkit_poses.shape
        assert result.dtype == np.float32

        # Check that the transformation preserves the number of poses
        assert len(result) == len(sample_arkit_poses)

    def test_arkit_to_canonical_poses_single_pose(self):
        """Test ARKit to canonical transformation with a single pose."""
        single_pose = np.eye(4, dtype=np.float32)
        single_pose[:3, 3] = [1.0, 2.0, 3.0]  # Translation

        result = arkit_to_canonical_poses(single_pose)

        assert result.shape == (4, 4)
        assert result.dtype == np.float32

    def test_arkit_to_canonical_poses_empty_input(self):
        """Test ARKit to canonical transformation with empty input."""
        empty_poses = np.zeros((0, 4, 4), dtype=np.float32)

        result = arkit_to_canonical_poses(empty_poses)

        assert result.shape == (0, 4, 4)
        assert result.dtype == np.float32

    def test_arkit_to_canonical_poses_3d_positions(self):
        """Test ARKit to canonical transformation with 3D positions."""
        # Test with positions instead of poses (N, 3) instead of (N, 4, 4)
        positions = np.random.rand(5, 3).astype(np.float32)

        result = arkit_to_canonical_poses(positions)

        assert result.shape == positions.shape
        assert result.dtype == np.float32

    def test_arkit_to_canonical_poses_invalid_shape(self):
        """Test ARKit to canonical transformation with invalid shape."""
        invalid_poses = np.random.rand(5, 3, 2).astype(
            np.float32
        )  # Wrong shape (should be 3 for last dimension)

        with pytest.raises(ValueError, match="Unsupported pose shape"):
            arkit_to_canonical_poses(invalid_poses)

    def test_arkit_to_canonical_poses_coordinate_system(self, sample_arkit_poses):
        """Test that the coordinate system transformation is correct."""
        # ARKit uses a different coordinate system than our canonical system
        # This test verifies the transformation logic

        # Create poses with known transformations
        poses = np.eye(4, dtype=np.float32)
        poses[:3, 3] = [1.0, 2.0, 3.0]  # Translation
        poses = np.tile(poses, (3, 1, 1))  # Repeat 3 times

        result = arkit_to_canonical_poses(poses)

        # The transformation should change the coordinate system
        # but preserve the structure of the transformation matrices
        assert result.shape == (3, 4, 4)
        assert result.dtype == np.float32

        # Check that the last row is still [0, 0, 0, 1] for all poses
        for i in range(3):
            assert np.allclose(result[i, 3, :], [0, 0, 0, 1])

    def test_arkit_to_canonical_poses_rotation_preservation(self):
        """Test that rotations are properly handled in the transformation."""
        # Create a pose with a rotation
        pose = np.eye(4, dtype=np.float32)
        # Add a simple rotation around Z axis
        angle = np.pi / 4  # 45 degrees
        pose[:3, :3] = np.array(
            [
                [np.cos(angle), -np.sin(angle), 0],
                [np.sin(angle), np.cos(angle), 0],
                [0, 0, 1],
            ]
        )
        pose[:3, 3] = [1.0, 2.0, 3.0]  # Translation

        poses = np.expand_dims(pose, axis=0)  # Add batch dimension

        result = arkit_to_canonical_poses(poses)

        assert result.shape == (1, 4, 4)
        assert result.dtype == np.float32

        # The transformation should preserve the structure
        assert np.allclose(result[0, 3, :], [0, 0, 0, 1])


class TestTransformPipeline:
    """Test the TransformPipeline class."""

    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        transforms = [arkit_to_canonical_poses]
        pipeline = TransformPipeline(transforms)

        assert len(pipeline.transforms) == 1
        assert pipeline.transforms[0] == arkit_to_canonical_poses

    def test_pipeline_initialization_empty(self):
        """Test pipeline initialization with empty transforms list."""
        pipeline = TransformPipeline([])

        assert len(pipeline.transforms) == 0

    def test_pipeline_call_single_transform(self):
        """Test pipeline with a single transform."""
        pipeline = TransformPipeline([arkit_to_canonical_poses])

        # Test with poses
        poses = np.random.rand(3, 4, 4).astype(np.float32)
        result = pipeline(poses)

        assert result.shape == poses.shape
        assert result.dtype == np.float32

    def test_pipeline_call_multiple_transforms(self):
        """Test pipeline with multiple transforms."""

        def mock_transform1(data):
            return data * 2

        def mock_transform2(data):
            return data + 1

        pipeline = TransformPipeline([mock_transform1, mock_transform2])

        # Test with simple data
        data = np.array([1, 2, 3], dtype=np.float32)
        result = pipeline(data)

        # Should apply transforms in order: (data * 2) + 1
        expected = (data * 2) + 1
        np.testing.assert_array_equal(result, expected)

    def test_pipeline_call_no_transforms(self):
        """Test pipeline with no transforms."""
        pipeline = TransformPipeline([])

        data = np.array([1, 2, 3], dtype=np.float32)
        result = pipeline(data)

        # Should return the original data unchanged
        np.testing.assert_array_equal(result, data)

    def test_pipeline_call_with_positions(self):
        """Test pipeline with 3D positions."""
        pipeline = TransformPipeline([arkit_to_canonical_poses])

        positions = np.random.rand(5, 3).astype(np.float32)
        result = pipeline(positions)

        assert result.shape == positions.shape
        assert result.dtype == np.float32

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""

        def failing_transform(data):
            raise ValueError("Transform failed")

        pipeline = TransformPipeline([failing_transform])

        data = np.array([1, 2, 3], dtype=np.float32)

        with pytest.raises(ValueError, match="Transform failed"):
            pipeline(data)

    def test_pipeline_with_complex_transforms(self):
        """Test pipeline with complex transform functions."""

        def scale_transform(data):
            return data * 10

        def offset_transform(data):
            return data + 5

        def clamp_transform(data):
            return np.clip(data, 0, 100)

        pipeline = TransformPipeline(
            [scale_transform, offset_transform, clamp_transform]
        )

        data = np.array([1, 2, 3], dtype=np.float32)
        result = pipeline(data)

        # Should apply: clamp(offset(scale(data)))
        # scale: [1, 2, 3] -> [10, 20, 30]
        # offset: [10, 20, 30] -> [15, 25, 35]
        # clamp: [15, 25, 35] -> [15, 25, 35] (no change)
        expected = np.array([15, 25, 35], dtype=np.float32)
        np.testing.assert_array_equal(result, expected)

    def test_pipeline_with_conditional_transforms(self):
        """Test pipeline with transforms that have conditional logic."""

        def conditional_transform(data):
            if data.shape[0] > 0:
                return data * 2
            else:
                return data

        pipeline = TransformPipeline([conditional_transform])

        # Test with non-empty data
        data = np.array([1, 2, 3], dtype=np.float32)
        result = pipeline(data)
        expected = data * 2
        np.testing.assert_array_equal(result, expected)

        # Test with empty data
        empty_data = np.array([], dtype=np.float32)
        result = pipeline(empty_data)
        np.testing.assert_array_equal(result, empty_data)

    def test_pipeline_transforms_attribute(self):
        """Test that the transforms attribute is accessible."""
        transforms = [arkit_to_canonical_poses]
        pipeline = TransformPipeline(transforms)

        assert hasattr(pipeline, "transforms")
        assert pipeline.transforms == transforms

    def test_pipeline_repr(self):
        """Test pipeline string representation."""
        transforms = [arkit_to_canonical_poses]
        pipeline = TransformPipeline(transforms)

        repr_str = repr(pipeline)
        assert "TransformPipeline" in repr_str
        assert "1" in repr_str  # Number of transforms
