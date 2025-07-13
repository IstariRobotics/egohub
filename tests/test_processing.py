from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from egohub.constants import CANONICAL_SKELETON_JOINTS
from egohub.processing.skeleton import SkeletonProcessor, remap_skeleton
from egohub.processing.synchronization import generate_indices


class TestSkeletonProcessor:
    """Test the SkeletonProcessor class."""

    @pytest.fixture
    def mock_hdf5_file(self):
        """Create a mock HDF5 file for testing."""
        mock_file = Mock()

        # Mock transforms group
        transforms_group = Mock()
        transforms_group.__class__ = h5py.Group  # type: ignore[attr-defined]
        transforms_group.keys.return_value = [
            "hip",
            "spine1",
            "camera",
        ]  # Excluding camera

        # Mock datasets for transforms
        hip_dataset = Mock()
        hip_dataset.__class__ = h5py.Dataset
        hip_dataset.shape = (10, 4, 4)
        # Return 4x4 transform matrices, the method will extract [:3, 3] (translation)
        # When indexed as joint_transform[:, :3, 3], it should return (10, 3)
        hip_dataset.__getitem__ = Mock(
            return_value=np.random.rand(10, 3).astype(np.float32)
        )

        spine1_dataset = Mock()
        spine1_dataset.__class__ = h5py.Dataset
        spine1_dataset.shape = (10, 4, 4)
        # Return 4x4 transform matrices, the method will extract [:3, 3] (translation)
        # When indexed as joint_transform[:, :3, 3], it should return (10, 3)
        spine1_dataset.__getitem__ = Mock(
            return_value=np.random.rand(10, 3).astype(np.float32)
        )

        def transforms_get_side_effect(key):
            if key == "hip":
                return hip_dataset
            elif key == "spine1":
                return spine1_dataset
            else:
                return None

        transforms_group.get.side_effect = transforms_get_side_effect

        # Mock confidences group
        confidences_group = Mock()
        confidences_group.__class__ = h5py.Group  # type: ignore[attr-defined]

        # Mock datasets for confidences
        hip_conf_dataset = Mock()
        hip_conf_dataset.__class__ = h5py.Dataset
        hip_conf_dataset.shape = (10,)
        hip_conf_dataset.__getitem__ = Mock(
            return_value=np.random.rand(10).astype(np.float32)
        )

        spine1_conf_dataset = Mock()
        spine1_conf_dataset.__class__ = h5py.Dataset
        spine1_conf_dataset.shape = (10,)
        spine1_conf_dataset.__getitem__ = Mock(
            return_value=np.random.rand(10).astype(np.float32)
        )

        def confidences_get_side_effect(key):
            if key == "hip":
                return hip_conf_dataset
            elif key == "spine1":
                return spine1_conf_dataset
            else:
                return None

        confidences_group.get.side_effect = confidences_get_side_effect

        def get_side_effect(key):
            if key == "transforms":
                return transforms_group
            elif key == "confidences":
                return confidences_group
            else:
                return None

        mock_file.get.side_effect = get_side_effect

        return mock_file

    def test_get_skeleton_data_valid(self, mock_hdf5_file):
        """Test get_skeleton_data with valid data."""
        processor = SkeletonProcessor()
        result = processor.get_skeleton_data(mock_hdf5_file, 10)

        assert result is not None
        joint_names, positions, confidences = result
        assert len(joint_names) == 2  # hip, spine1 (excluding camera)
        assert positions.shape == (10, 2, 3)
        assert confidences.shape == (10, 2)

    def test_get_skeleton_data_missing_groups(self):
        """Test get_skeleton_data when required groups are missing."""
        mock_file = Mock()
        mock_file.get.return_value = None

        processor = SkeletonProcessor()
        result = processor.get_skeleton_data(mock_file, 10)

        assert result is None

    def test_get_skeleton_data_no_joints(self):
        """Test get_skeleton_data when no joints are found."""
        mock_file = Mock()
        transforms_group = Mock()
        transforms_group.keys.return_value = ["camera"]  # Only camera, no joints
        mock_file.get.return_value = transforms_group

        processor = SkeletonProcessor()
        result = processor.get_skeleton_data(mock_file, 10)

        assert result is None


class TestRemapSkeleton:
    """Test the remap_skeleton function."""

    @pytest.fixture
    def sample_skeleton_data(self):
        """Create sample skeleton data for testing."""
        num_frames = 5
        num_source_joints = 3

        source_positions = np.random.rand(num_frames, num_source_joints, 3).astype(
            np.float32
        )
        source_confidences = np.random.rand(num_frames, num_source_joints).astype(
            np.float32
        )
        source_joint_names = ["hip", "left_shoulder", "left_elbow"]

        return source_positions, source_confidences, source_joint_names

    def test_remap_skeleton_with_joint_map(self, sample_skeleton_data):
        """Test remap_skeleton with a joint mapping."""
        source_positions, source_confidences, source_joint_names = sample_skeleton_data

        joint_map = {
            "hip": "pelvis",
            "left_shoulder": "left_shoulder",
            "left_elbow": "left_elbow"
        }

        canonical_positions, canonical_confidences = remap_skeleton(
            source_positions, source_confidences, source_joint_names, joint_map
        )

        assert canonical_positions.shape == (5, len(CANONICAL_SKELETON_JOINTS), 3)
        assert canonical_confidences.shape == (5, len(CANONICAL_SKELETON_JOINTS))

        # Check that mapped joints have data
        pelvis_idx = CANONICAL_SKELETON_JOINTS.index("pelvis")
        shoulder_idx = CANONICAL_SKELETON_JOINTS.index("left_shoulder")
        elbow_idx = CANONICAL_SKELETON_JOINTS.index("left_elbow")

        assert np.all(~np.isnan(canonical_positions[:, pelvis_idx, :]))
        assert np.all(~np.isnan(canonical_positions[:, shoulder_idx, :]))
        assert np.all(~np.isnan(canonical_positions[:, elbow_idx, :]))

        # Check that unmapped joints have NaN positions and zero confidences
        unmapped_idx = CANONICAL_SKELETON_JOINTS.index("right_shoulder")
        assert np.all(np.isnan(canonical_positions[:, unmapped_idx, :]))
        assert np.all(canonical_confidences[:, unmapped_idx] == 0)

    def test_remap_skeleton_without_joint_map(self, sample_skeleton_data):
        """Test remap_skeleton without a joint mapping."""
        source_positions, source_confidences, source_joint_names = sample_skeleton_data

        canonical_positions, canonical_confidences = remap_skeleton(
            source_positions, source_confidences, source_joint_names
        )

        assert canonical_positions.shape == (5, len(CANONICAL_SKELETON_JOINTS), 3)
        assert canonical_confidences.shape == (5, len(CANONICAL_SKELETON_JOINTS))

        # Check that matching joints have data
        shoulder_idx = CANONICAL_SKELETON_JOINTS.index("left_shoulder")
        elbow_idx = CANONICAL_SKELETON_JOINTS.index("left_elbow")

        assert not np.isnan(canonical_positions[:, shoulder_idx, :]).all()
        assert not np.isnan(canonical_positions[:, elbow_idx, :]).all()

    def test_remap_skeleton_empty_input(self):
        """Test remap_skeleton with empty input."""
        source_positions = np.zeros((0, 0, 3), dtype=np.float32)
        source_confidences = np.zeros((0, 0), dtype=np.float32)
        source_joint_names = []

        canonical_positions, canonical_confidences = remap_skeleton(
            source_positions, source_confidences, source_joint_names
        )

        assert canonical_positions.shape == (0, len(CANONICAL_SKELETON_JOINTS), 3)
        assert canonical_confidences.shape == (0, len(CANONICAL_SKELETON_JOINTS))

    def test_remap_skeleton_partial_mapping(self, sample_skeleton_data):
        """Test remap_skeleton with partial joint mapping."""
        source_positions, source_confidences, source_joint_names = sample_skeleton_data

        # Only map some joints
        joint_map = {
            "hip": "pelvis",
            "left_shoulder": "left_shoulder",
            # "left_elbow" not mapped
        }

        canonical_positions, canonical_confidences = remap_skeleton(
            source_positions, source_confidences, source_joint_names, joint_map
        )

        # Check that mapped joints have data
        pelvis_idx = CANONICAL_SKELETON_JOINTS.index("pelvis")
        shoulder_idx = CANONICAL_SKELETON_JOINTS.index("left_shoulder")

        assert not np.isnan(canonical_positions[:, pelvis_idx, :]).all()
        assert not np.isnan(canonical_positions[:, shoulder_idx, :]).all()

        # Check that unmapped joints (including left_elbow) have NaN positions
        elbow_idx = CANONICAL_SKELETON_JOINTS.index("left_elbow")
        assert np.isnan(canonical_positions[:, elbow_idx, :]).all()


class TestSynchronization:
    """Test the synchronization module."""

    def test_generate_indices_basic(self):
        """Test generate_indices with basic input."""
        master_timestamps = np.array([0, 10, 20, 30, 40], dtype=np.uint64)
        stream_timestamps = np.array([5, 15, 25, 35], dtype=np.uint64)

        indices = generate_indices(master_timestamps, stream_timestamps)

        expected = np.array([0, 1, 2, 3], dtype=np.uint64)  # Closest indices
        np.testing.assert_array_equal(indices, expected)

    def test_generate_indices_exact_matches(self):
        """Test generate_indices with exact timestamp matches."""
        master_timestamps = np.array([0, 10, 20, 30, 40], dtype=np.uint64)
        stream_timestamps = np.array([0, 10, 20, 30], dtype=np.uint64)

        indices = generate_indices(master_timestamps, stream_timestamps)

        expected = np.array([0, 1, 2, 3], dtype=np.uint64)
        np.testing.assert_array_equal(indices, expected)

    def test_generate_indices_empty_master(self):
        """Test generate_indices with empty master timestamps."""
        master_timestamps = np.array([], dtype=np.uint64)
        stream_timestamps = np.array([5, 15], dtype=np.uint64)

        with pytest.raises(ValueError):
            generate_indices(master_timestamps, stream_timestamps)

    def test_generate_indices_empty_stream(self):
        """Test generate_indices with empty stream timestamps."""
        master_timestamps = np.array([0, 10, 20], dtype=np.uint64)
        stream_timestamps = np.array([], dtype=np.uint64)

        indices = generate_indices(master_timestamps, stream_timestamps)

        assert len(indices) == 0

    def test_generate_indices_2d_input(self):
        """Test generate_indices with 2D input (should raise error)."""
        master_timestamps = np.array([[0, 1], [10, 11]], dtype=np.uint64)
        stream_timestamps = np.array([5, 15], dtype=np.uint64)

        with pytest.raises(ValueError):
            generate_indices(master_timestamps, stream_timestamps)

    def test_generate_indices_large_numbers(self):
        """Test generate_indices with large timestamp values."""
        master_timestamps = np.array([1000000, 2000000, 3000000], dtype=np.uint64)
        stream_timestamps = np.array([1500000, 2500000], dtype=np.uint64)

        indices = generate_indices(master_timestamps, stream_timestamps)

        expected = np.array([0, 1], dtype=np.uint64)
        np.testing.assert_array_equal(indices, expected)

    def test_generate_indices_out_of_bounds(self):
        """Test generate_indices with stream timestamps outside master range."""
        master_timestamps = np.array([10, 20, 30], dtype=np.uint64)
        stream_timestamps = np.array([5, 15, 25, 35], dtype=np.uint64)

        indices = generate_indices(master_timestamps, stream_timestamps)

        # Should find closest indices (0, 0, 1, 2)
        expected = np.array([0, 0, 1, 2], dtype=np.uint64)
        np.testing.assert_array_equal(indices, expected)
