import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import h5py
import numpy as np
import pytest

from egohub.adapters.base import BaseAdapter
from egohub.adapters.egodex.egodex import (
    EGODEX_TO_CANONICAL_SKELETON_MAP,
    EgoDexAdapter,
)
from egohub.constants import CANONICAL_SKELETON_JOINTS
from egohub.constants.egodex_skeleton import (
    EGODEX_SKELETON_HIERARCHY,
    EGODEX_SKELETON_JOINTS,
)


class TestBaseAdapter:
    """Test the BaseAdapter abstract class."""

    def test_base_adapter_abstract_methods(self):
        """Test that BaseAdapter cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseAdapter(Path("test"), Path("test"))

    def test_base_adapter_subclass_required_methods(self):
        """Test that subclasses must implement required methods."""

        class InvalidAdapter(BaseAdapter):
            name = "test"

        with pytest.raises(TypeError):
            InvalidAdapter(Path("test"), Path("test"))


class TestEgoDexAdapter:
    """Test the EgoDexAdapter class."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def mock_sequence(self, temp_dir):
        """Create mock HDF5 and MP4 files for a test sequence."""
        sequence_name = "test_sequence"
        hdf5_path = temp_dir / f"{sequence_name}.hdf5"
        mp4_path = temp_dir / f"{sequence_name}.mp4"

        with h5py.File(hdf5_path, "w") as f:
            # Add metadata
            f.attrs["llm_description"] = "Test action"

            # Add camera transforms
            transforms_group = f.create_group("transforms")
            transforms_group.create_dataset(
                "camera", data=np.random.rand(10, 4, 4).astype(np.float32)
            )

            # Add camera intrinsics
            camera_group = f.create_group("camera")
            camera_group.create_dataset(
                "intrinsic",
                data=np.array(
                    [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                ),
            )

            # Add hand transforms
            transforms_group.create_dataset(
                "LeftHand", data=np.random.rand(10, 4, 4).astype(np.float32)
            )
            transforms_group.create_dataset(
                "RightHand", data=np.random.rand(10, 4, 4).astype(np.float32)
            )

            # Add skeleton transforms and confidences
            transforms_group.create_dataset(
                "hip", data=np.random.rand(10, 4, 4).astype(np.float32)
            )
            transforms_group.create_dataset(
                "spine1", data=np.random.rand(10, 4, 4).astype(np.float32)
            )

            confidences_group = f.create_group("confidences")
            confidences_group.create_dataset(
                "hip", data=np.random.rand(10).astype(np.float32)
            )
            confidences_group.create_dataset(
                "spine1", data=np.random.rand(10).astype(np.float32)
            )

        mp4_path.touch()  # Create an empty MP4 file
        return {"name": sequence_name, "hdf5_path": hdf5_path, "mp4_path": mp4_path}

    def test_egodex_adapter_properties(self):
        """Test EgoDexAdapter properties."""
        adapter = EgoDexAdapter(Path("test"), Path("test"))

        assert adapter.source_joint_names == EGODEX_SKELETON_JOINTS
        assert adapter.source_skeleton_hierarchy == EGODEX_SKELETON_HIERARCHY
        assert adapter.name == "egodex"

    def test_discover_sequences_no_files(self, temp_dir):
        """Test discover_sequences when no files exist."""
        adapter = EgoDexAdapter(temp_dir, Path("test"))
        sequences = adapter.discover_sequences()
        assert sequences == []

    @patch.object(Path, "glob")
    def test_discover_sequences_with_files(self, mock_glob, temp_dir, mock_sequence):
        """Test discover_sequences with valid files."""
        mock_glob.return_value = [mock_sequence["hdf5_path"]]
        adapter = EgoDexAdapter(temp_dir, Path("test"))
        sequences = adapter.discover_sequences()

        assert len(sequences) == 1
        seq = sequences[0]
        assert seq["hdf5_path"] == mock_sequence["hdf5_path"]
        assert seq["mp4_path"] == mock_sequence["mp4_path"]
        assert seq["sequence_name"] == mock_sequence["name"]

    @patch.object(Path, "glob")
    def test_discover_sequences_missing_mp4(self, mock_glob, temp_dir, mock_sequence):
        """Test discover_sequences when MP4 file is missing."""
        mock_glob.return_value = [mock_sequence["hdf5_path"]]
        mock_sequence["mp4_path"].unlink()
        adapter = EgoDexAdapter(temp_dir, Path("test"))
        sequences = adapter.discover_sequences()
        assert sequences == []

    def test_process_metadata(self, temp_dir, mock_sequence):
        """Test _process_metadata method."""
        adapter = EgoDexAdapter(temp_dir, Path("test"))

        with h5py.File(mock_sequence["hdf5_path"], "r") as f_in:
            traj_group = MagicMock()
            metadata_group = MagicMock()
            traj_group.create_group.return_value = metadata_group

            found_streams, timestamps = adapter._process_metadata(
                {"sequence_name": "test"}, f_in, traj_group
            )

            assert "metadata/action_label" in found_streams
            assert "metadata/timestamps_ns" in found_streams
            assert len(timestamps) == 10  # Based on camera transforms

    def test_process_camera_data(self, temp_dir, mock_sequence):
        """Test _process_camera_data method."""
        adapter = EgoDexAdapter(temp_dir, mock_sequence["hdf5_path"])

        with h5py.File(mock_sequence["hdf5_path"], "r") as f_in:
            traj_group = MagicMock()
            cameras_group = MagicMock()
            ego_camera_group = MagicMock()
            traj_group.create_group.return_value = cameras_group
            cameras_group.create_group.return_value = ego_camera_group

            master_timestamps = np.arange(10) * (1e9 / 30.0)

            found_streams = adapter._process_camera_data(
                {"mp4_path": mock_sequence["hdf5_path"]},
                f_in,
                traj_group,
                master_timestamps,
            )

            assert "cameras/ego_camera/intrinsics" in found_streams
            assert "cameras/ego_camera/pose_in_world" in found_streams

    @patch("cv2.VideoCapture")
    def test_process_rgb_data(self, mock_cv2, temp_dir, mock_sequence):
        """Test _process_rgb_data method."""
        # Mock OpenCV video capture
        mock_cap = Mock()
        mock_cv2.return_value = mock_cap
        mock_cap.isOpened.return_value = True

        # Mock frame reading
        mock_frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        mock_cap.read.side_effect = [(True, mock_frame), (False, None)]

        adapter = EgoDexAdapter(temp_dir, mock_sequence["hdf5_path"])
        ego_camera_group = MagicMock()
        rgb_group = MagicMock()
        ego_camera_group.create_group.return_value = rgb_group

        master_timestamps = np.arange(10) * (1e9 / 30.0)

        found_streams = adapter._process_rgb_data(
            {"mp4_path": mock_sequence["hdf5_path"]},
            ego_camera_group,
            master_timestamps,
        )

        assert "cameras/ego_camera/rgb/image_bytes" in found_streams
        mock_cap.release.assert_called_once()

    def test_process_hand_data(self, temp_dir, mock_sequence):
        """Test _process_hand_data method."""
        adapter = EgoDexAdapter(temp_dir, mock_sequence["hdf5_path"])

        with h5py.File(mock_sequence["hdf5_path"], "r") as f_in:
            traj_group = MagicMock()
            hands_group = MagicMock()
            traj_group.create_group.return_value = hands_group

            master_timestamps = np.arange(10) * (1e9 / 30.0)

            found_streams = adapter._process_hand_data(
                f_in, traj_group, master_timestamps
            )

            assert "hands/left/pose_in_world" in found_streams
            assert "hands/right/pose_in_world" in found_streams

    def test_process_skeleton_data(self, temp_dir, mock_sequence):
        """Test _process_skeleton_data method."""
        adapter = EgoDexAdapter(temp_dir, mock_sequence["hdf5_path"])

        with h5py.File(mock_sequence["hdf5_path"], "r") as f_in:
            traj_group = MagicMock()
            skeleton_group = MagicMock()
            traj_group.create_group.return_value = skeleton_group

            master_timestamps = np.arange(10) * (1e9 / 30.0)

            found_streams = adapter._process_skeleton_data(
                f_in, traj_group, master_timestamps
            )

            assert "skeleton/positions" in found_streams
            assert "skeleton/confidences" in found_streams

    def test_egodex_to_canonical_skeleton_map(self):
        """Test the EGODEX_TO_CANONICAL_SKELETON_MAP constant."""
        # Test that all keys are valid EgoDex joint names
        for source_joint in EGODEX_TO_CANONICAL_SKELETON_MAP.keys():
            assert source_joint in EGODEX_SKELETON_JOINTS

        # Test that all values are valid canonical joint names
        for canonical_joint in EGODEX_TO_CANONICAL_SKELETON_MAP.values():
            assert canonical_joint in CANONICAL_SKELETON_JOINTS

    @patch("egohub.adapters.egodex.EgoDexAdapter.process_sequence")
    def test_adapter_run_method(self, mock_process_sequence, temp_dir, mock_sequence):
        """Test the main run method of the adapter."""
        output_file = temp_dir / "output.hdf5"
        adapter = EgoDexAdapter(raw_dir=temp_dir, output_file=output_file)

        # Test with a single sequence
        with patch.object(
            EgoDexAdapter,
            "discover_sequences",
            return_value=[
                {
                    "hdf5_path": mock_sequence["hdf5_path"],
                    "mp4_path": mock_sequence["mp4_path"],
                    "sequence_name": mock_sequence["name"],
                }
            ],
        ):
            adapter.run(num_sequences=1)
            mock_process_sequence.assert_called_once()
            assert output_file.exists()
            with h5py.File(output_file, "r") as f_out:
                assert "trajectory_0000" in f_out

        # Test with no sequences found
        mock_process_sequence.reset_mock()
        with patch.object(EgoDexAdapter, "discover_sequences", return_value=[]):
            adapter.run()
            mock_process_sequence.assert_not_called()
