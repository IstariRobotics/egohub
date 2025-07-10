from unittest.mock import Mock

import h5py
import numpy as np
import pytest

from egohub.tasks.base import BaseTask
from egohub.tasks.object_detection import ObjectDetectionTask
from egohub.tasks.pose_estimation import PoseEstimationTask


class TestBaseTask:
    """Test the BaseTask abstract class."""

    def test_base_task_abstract(self):
        """Test that BaseTask cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseTask()

    def test_base_task_subclass_required_methods(self):
        """Test that subclasses must implement required methods."""

        class InvalidTask(BaseTask):
            pass

        with pytest.raises(TypeError):
            InvalidTask()

    def test_valid_task_subclass(self):
        """Test a valid task subclass."""

        class ValidTask(BaseTask):
            def run(self, traj_group: h5py.Group, backend, **kwargs):
                return {"result": "success"}

        task = ValidTask("test_output")
        result = task.run(Mock(), Mock())
        assert result == {"result": "success"}


class TestObjectDetectionTask:
    """Test the ObjectDetectionTask class."""

    @pytest.fixture
    def mock_traj_group(self):
        """Create a mock trajectory group for testing."""
        mock_group = Mock()

        # Mock camera data
        mock_cameras = Mock()
        mock_ego_camera = Mock()
        mock_rgb = Mock()

        # Mock RGB image data
        mock_image_dataset = Mock()
        mock_image_dataset.shape = (10, 1000)  # 10 frames, 1000 bytes each
        mock_image_dataset.__getitem__ = Mock(
            return_value=np.random.randint(0, 255, 1000, dtype=np.uint8)
        )

        mock_frame_sizes = Mock()
        mock_frame_sizes.__getitem__ = Mock(return_value=1000)

        mock_rgb.__getitem__ = Mock(
            side_effect=lambda x: {
                "image_bytes": mock_image_dataset,
                "frame_sizes": mock_frame_sizes,
            }[x]
        )

        mock_ego_camera.__getitem__ = Mock(return_value=mock_rgb)
        mock_cameras.__getitem__ = Mock(return_value=mock_ego_camera)
        mock_group.__getitem__ = Mock(
            side_effect=lambda x: {"cameras": mock_cameras}[x]
        )
        mock_group.__contains__ = Mock(
            side_effect=lambda item: item in ["cameras", "objects"]
        )
        mock_group.__delitem__ = Mock()

        return mock_group

    def test_object_detection_task_initialization(self):
        """Test ObjectDetectionTask initialization."""
        task = ObjectDetectionTask()

        assert task.output_group == "objects/"
        assert hasattr(task, "run")

    def test_object_detection_task_run(self, mock_traj_group):
        """Test ObjectDetectionTask run method."""
        task = ObjectDetectionTask()

        # Mock backend
        mock_backend = Mock()
        mock_backend.run.return_value = {
            "detections": {
                "person": [(0, [0, 0, 100, 100], 0.9), (1, [200, 200, 300, 300], 0.8)],
                "car": [(2, [50, 50, 150, 150], 0.7)],
            },
            "num_frames": 10,
        }

        result = task.run(mock_traj_group, mock_backend)

        assert result is None
        mock_backend.run.assert_called_once()

    def test_object_detection_task_with_kwargs(self, mock_traj_group):
        """Test ObjectDetectionTask with additional kwargs."""
        task = ObjectDetectionTask()

        mock_backend = Mock()
        mock_backend.run.return_value = {
            "detections": {"person": [(0, [0, 0, 100, 100], 0.9)]},
            "num_frames": 5,
        }

        result = task.run(mock_traj_group, mock_backend, confidence_threshold=0.7)

        assert result is None
        # Check that kwargs were passed to backend
        mock_backend.run.assert_called_once()
        call_args = mock_backend.run.call_args
        assert "confidence_threshold" in call_args[1]
        assert call_args[1]["confidence_threshold"] == 0.7

    def test_object_detection_task_error_handling(self, mock_traj_group):
        """Test ObjectDetectionTask error handling."""
        task = ObjectDetectionTask()

        # Mock backend that raises an error
        mock_backend = Mock()
        mock_backend.run.side_effect = RuntimeError("Backend failed")

        with pytest.raises(RuntimeError, match="Backend failed"):
            task.run(mock_traj_group, mock_backend)

    def test_object_detection_task_no_camera_data(self, mock_traj_group):
        """Test ObjectDetectionTask with no camera data."""
        task = ObjectDetectionTask()

        # Mock backend that raises KeyError when trying to access camera data
        mock_backend = Mock()
        mock_backend.run.side_effect = KeyError("cameras")

        with pytest.raises(KeyError, match="cameras"):
            task.run(mock_traj_group, mock_backend)


class TestPoseEstimationTask:
    """Test the PoseEstimationTask class."""

    @pytest.fixture
    def mock_traj_group(self):
        """Create a mock trajectory group for testing."""
        mock_group = Mock()

        # Mock camera data
        mock_cameras = Mock()
        mock_ego_camera = Mock()
        mock_rgb = Mock()

        # Mock RGB image data
        mock_image_dataset = Mock()
        mock_image_dataset.shape = (10, 1000)  # 10 frames, 1000 bytes each
        mock_image_dataset.__getitem__ = Mock(
            return_value=np.random.randint(0, 255, 1000, dtype=np.uint8)
        )

        mock_frame_sizes = Mock()
        mock_frame_sizes.__getitem__ = Mock(return_value=1000)

        mock_rgb.__getitem__ = Mock(
            side_effect=lambda x: {
                "image_bytes": mock_image_dataset,
                "frame_sizes": mock_frame_sizes,
            }[x]
        )

        mock_ego_camera.__getitem__ = Mock(return_value=mock_rgb)
        mock_cameras.__getitem__ = Mock(return_value=mock_ego_camera)
        mock_group.__getitem__ = Mock(
            side_effect=lambda x: {"cameras": mock_cameras}[x]
        )
        mock_group.__contains__ = Mock(
            side_effect=lambda item: item in ["cameras", "skeleton"]
        )
        mock_group.__delitem__ = Mock()

        return mock_group

    def test_pose_estimation_task_initialization(self):
        """Test PoseEstimationTask initialization."""
        task = PoseEstimationTask()

        assert task.output_group == "skeleton/"
        assert hasattr(task, "run")

    def test_pose_estimation_task_run(self, mock_traj_group):
        """Test PoseEstimationTask run method."""
        task = PoseEstimationTask()

        # Mock backend
        mock_backend = Mock()
        mock_backend.run.return_value = {
            "keypoints": np.random.rand(10, 17, 3),  # 10 frames, 17 keypoints, 3D
            "confidences": np.random.rand(10, 17),  # 10 frames, 17 confidences
        }

        result = task.run(mock_traj_group, mock_backend)

        assert result is None
        mock_backend.run.assert_called_once()

    def test_pose_estimation_task_with_kwargs(self, mock_traj_group):
        """Test PoseEstimationTask with additional kwargs."""
        task = PoseEstimationTask()

        mock_backend = Mock()
        mock_backend.run.return_value = {
            "keypoints": np.random.rand(5, 17, 3),
            "confidences": np.random.rand(5, 17),
        }

        result = task.run(mock_traj_group, mock_backend, confidence_threshold=0.6)

        assert result is None
        # Check that kwargs were passed to backend
        mock_backend.run.assert_called_once()
        call_args = mock_backend.run.call_args
        assert "confidence_threshold" in call_args[1]
        assert call_args[1]["confidence_threshold"] == 0.6

    def test_pose_estimation_task_error_handling(self, mock_traj_group):
        """Test PoseEstimationTask error handling."""
        task = PoseEstimationTask()

        # Mock backend that raises an error
        mock_backend = Mock()
        mock_backend.run.side_effect = RuntimeError("Backend failed")

        with pytest.raises(RuntimeError, match="Backend failed"):
            task.run(mock_traj_group, mock_backend)

    def test_pose_estimation_task_no_camera_data(self, mock_traj_group):
        """Test PoseEstimationTask with no camera data."""
        task = PoseEstimationTask()

        # Mock backend that raises KeyError when trying to access camera data
        mock_backend = Mock()
        mock_backend.run.side_effect = KeyError("cameras")

        with pytest.raises(KeyError, match="cameras"):
            task.run(mock_traj_group, mock_backend)
