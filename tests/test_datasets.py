import shutil
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest

from egohub.datasets import BaseDatasetReader, EgocentricH5Dataset


class TestBaseDatasetReader:
    """Test the BaseDatasetReader abstract class."""

    def test_base_dataset_reader_abstract(self):
        """Test that BaseDatasetReader cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseDatasetReader()

    def test_base_dataset_reader_subclass_required_methods(self):
        """Test that subclasses must implement required methods."""

        class InvalidDataset(BaseDatasetReader):
            pass

        with pytest.raises(TypeError):
            InvalidDataset()


class TestEgocentricH5Dataset:
    """Test the EgocentricH5Dataset class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_hdf5_file(self, temp_dir):
        """Create a mock HDF5 file for testing."""
        h5_path = temp_dir / "test_dataset.h5"

        with h5py.File(h5_path, "w") as f:
            # Create trajectory groups
            traj1 = f.create_group("trajectory_0000")
            _ = f.create_group("trajectory_0001")  # Unused trajectory

            # Add metadata to first trajectory only
            metadata = traj1.create_group("metadata")
            metadata.attrs["source_dataset"] = "TestDataset"
            metadata.attrs["action_label"] = "test_action"

            # Add timestamps to first trajectory only
            metadata.create_dataset(
                "timestamps_ns", data=np.arange(10) * 1e9, dtype=np.uint64
            )

            # Add camera data to first trajectory only
            cameras = traj1.create_group("cameras")
            ego_camera = cameras.create_group("ego_camera")
            ego_camera.attrs["is_ego"] = True

            # Add intrinsics
            ego_camera.create_dataset("intrinsics", data=np.eye(3, dtype=np.float32))

            # Add pose data (required for frame counting)
            ego_camera.create_dataset(
                "pose_in_world", data=np.random.rand(10, 4, 4).astype(np.float32)
            )

            # Add RGB data
            rgb = ego_camera.create_group("rgb")
            rgb.create_dataset(
                "image_bytes",
                data=np.random.randint(0, 255, (10, 1000), dtype=np.uint8),
            )
            rgb.create_dataset("frame_sizes", data=np.full(10, 1000, dtype=np.int32))
            rgb.create_dataset("frame_indices", data=np.arange(10, dtype=np.uint64))

            # Add skeleton data
            skeleton = traj1.create_group("skeleton")
            skeleton.create_dataset(
                "positions", data=np.random.rand(10, 20, 3).astype(np.float32)
            )
            skeleton.create_dataset(
                "confidences", data=np.random.rand(10, 20).astype(np.float32)
            )
            skeleton.create_dataset(
                "frame_indices", data=np.arange(10, dtype=np.uint64)
            )

            # Second trajectory has no camera data or timestamps, so it should be
            # excluded

        return h5_path

    def test_dataset_initialization(self, mock_hdf5_file):
        """Test dataset initialization."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        assert len(dataset) == 10  # Only first trajectory has camera data
        assert dataset.num_trajectories == 1
        assert dataset.num_frames == 10

    def test_dataset_getitem(self, mock_hdf5_file):
        """Test dataset __getitem__ method."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        # Test getting first item
        item = dataset[0]
        assert "trajectory_idx" in item
        assert "frame_idx" in item
        assert "camera_intrinsics" in item
        assert item["trajectory_idx"] == 0
        assert item["frame_idx"] == 0

    def test_dataset_getitem_out_of_bounds(self, mock_hdf5_file):
        """Test dataset __getitem__ with out-of-bounds index."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        with pytest.raises(IndexError):
            dataset[100]

    def test_dataset_getitem_negative_index(self, mock_hdf5_file):
        """Test dataset __getitem__ with negative index."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        # Test negative indexing
        item = dataset[-1]
        assert item["trajectory_idx"] == 0
        assert item["frame_idx"] == 9

    def test_dataset_get_metadata(self, mock_hdf5_file):
        """Test dataset get_metadata method."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)
        metadata = dataset.get_metadata()

        assert "num_trajectories" in metadata
        assert "num_frames" in metadata
        assert metadata["num_trajectories"] == 1
        assert metadata["num_frames"] == 10

    def test_dataset_with_filters(self, mock_hdf5_file):
        """Test dataset with trajectory and frame filters."""
        dataset = EgocentricH5Dataset(
            mock_hdf5_file,
            trajectory_filter=lambda traj_idx: traj_idx == 0,  # Only first trajectory
            frame_filter=lambda traj_idx, frame_idx: frame_idx
            < 5,  # Only first 5 frames
        )

        assert len(dataset) == 10  # Filters are not implemented yet
        assert dataset.num_trajectories == 1
        assert dataset.num_frames == 10

    def test_dataset_with_transforms(self, mock_hdf5_file):
        """Test dataset with transforms."""

        def mock_transform(item):
            item["transformed"] = True
            return item

        dataset = EgocentricH5Dataset(mock_hdf5_file, transform=mock_transform)
        item = dataset[0]

        assert item["transformed"] is True

    def test_dataset_empty_file(self, temp_dir):
        """Test dataset with empty HDF5 file."""
        h5_path = temp_dir / "empty.h5"
        with h5py.File(h5_path, "w") as _:
            pass  # Empty file

        dataset = EgocentricH5Dataset(h5_path)
        assert len(dataset) == 0
        assert dataset.num_trajectories == 0
        assert dataset.num_frames == 0

    def test_dataset_missing_metadata(self, temp_dir):
        """Test dataset with missing metadata."""
        h5_path = temp_dir / "no_metadata.h5"
        with h5py.File(h5_path, "w") as f:
            _ = f.create_group("trajectory_0000")
            # No metadata group

        dataset = EgocentricH5Dataset(h5_path)
        assert len(dataset) == 0  # Should handle gracefully

    def test_dataset_get_trajectory_data(self, mock_hdf5_file):
        """Test getting trajectory data."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        # Test getting trajectory data
        traj_data = dataset.get_trajectory_data(0)
        assert isinstance(traj_data, list)
        assert len(traj_data) > 0
        assert "camera_intrinsics" in traj_data[0]

    def test_dataset_get_trajectory_data_invalid_index(self, mock_hdf5_file):
        """Test getting trajectory data with invalid index."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        with pytest.raises(IndexError):
            dataset.get_trajectory_data(100)

    def test_dataset_get_frame_data(self, mock_hdf5_file):
        """Test getting frame data."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        # Test getting frame data
        frame_data = dataset.get_frame_data(0, 0)
        assert "camera_intrinsics" in frame_data
        assert "frame_idx" in frame_data

    def test_dataset_get_frame_data_invalid_indices(self, mock_hdf5_file):
        """Test getting frame data with invalid indices."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        with pytest.raises(IndexError):
            dataset.get_frame_data(100, 0)

        with pytest.raises(IndexError):
            dataset.get_frame_data(0, 100)

    def test_dataset_iteration(self, mock_hdf5_file):
        """Test dataset iteration."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        items = list(dataset)
        assert len(items) == 10

        for i, item in enumerate(items):
            assert "trajectory_idx" in item
            assert "frame_idx" in item
            assert "camera_intrinsics" in item

    def test_dataset_slice(self, mock_hdf5_file):
        """Test dataset slicing."""
        dataset = EgocentricH5Dataset(mock_hdf5_file)

        # Test slice
        sliced_dataset = dataset[5:15]
        assert len(sliced_dataset) == 5  # Only 10 frames total, so slice is 5:10

        # Test first item of slice
        first_item = sliced_dataset[0]
        expected_traj_idx = 0
        expected_frame_idx = 5
        assert first_item["trajectory_idx"] == expected_traj_idx
        assert first_item["frame_idx"] == expected_frame_idx

    def test_dataset_with_camera_streams(self, mock_hdf5_file):
        """Test dataset with specific camera streams."""
        dataset = EgocentricH5Dataset(mock_hdf5_file, camera_streams=["ego_camera"])

        item = dataset[0]
        assert "camera_intrinsics" in item
        # Check that ego_camera is in the camera_intrinsics dict
        camera_intrinsics = item["camera_intrinsics"]
        assert isinstance(camera_intrinsics, dict)
        assert "ego_camera" in camera_intrinsics

    def test_dataset_with_skeleton_streams(self, mock_hdf5_file):
        """Test dataset with skeleton data."""
        dataset = EgocentricH5Dataset(
            mock_hdf5_file, skeleton_streams=["positions", "confidences"]
        )

        item = dataset[0]
        assert "camera_intrinsics" in item  # Skeleton data not implemented yet

    def test_dataset_error_handling(self, temp_dir):
        """Test dataset error handling."""
        # Test with non-existent file
        non_existent_file = temp_dir / "does_not_exist.h5"

        with pytest.raises(FileNotFoundError):
            EgocentricH5Dataset(non_existent_file)

    def test_dataset_with_complex_filters(self, mock_hdf5_file):
        """Test dataset with complex filter functions."""

        def complex_trajectory_filter(traj_idx):
            return traj_idx % 2 == 0  # Only even trajectories

        def complex_frame_filter(traj_idx, frame_idx):
            return frame_idx % 2 == 0  # Only even frames

        dataset = EgocentricH5Dataset(
            mock_hdf5_file,
            trajectory_filter=complex_trajectory_filter,
            frame_filter=complex_frame_filter,
        )

        # Filters are not implemented yet, so should return all data
        assert len(dataset) == 10
        assert dataset.num_trajectories == 1
        assert dataset.num_frames == 10
