from unittest.mock import Mock, patch

import h5py
import numpy as np
import pytest

from egohub.backends.base import BaseBackend, get_backend


class TestBaseBackend:
    """Test the BaseBackend abstract class."""

    def test_base_backend_abstract(self):
        """Test that BaseBackend cannot be instantiated directly."""
        with pytest.raises(TypeError):
            BaseBackend()

    def test_base_backend_subclass_required_methods(self):
        """Test that subclasses must implement the run method."""

        class InvalidBackend(BaseBackend):
            pass

        with pytest.raises(TypeError):
            InvalidBackend()

    def test_valid_backend_subclass(self):
        """Test a valid backend subclass."""

        class ValidBackend(BaseBackend):
            def run(self, traj_group: h5py.Group, **kwargs):
                return {"result": "success"}

        backend = ValidBackend()
        result = backend.run(Mock())
        assert result == {"result": "success"}


class TestGetBackend:
    """Test the get_backend function."""

    def test_get_backend_success(self):
        """Test get_backend with a valid backend name."""
        from egohub.backends.hf import HuggingFaceBackend

        result = get_backend("HuggingFaceBackend")
        assert result == HuggingFaceBackend

    @patch("egohub.backends")
    def test_get_backend_not_found(self, mock_backends):
        """Test get_backend with a non-existent backend name."""
        mock_backends.TestBackend = None

        with pytest.raises(ValueError, match="Unknown backend: TestBackend"):
            get_backend("TestBackend")

    @patch("egohub.backends")
    def test_get_backend_invalid_type(self, mock_backends):
        """Test get_backend with a name that exists but is not a Backend class."""
        mock_backends.TestBackend = "not_a_class"

        with pytest.raises(TypeError, match="TestBackend is not a valid Backend class"):
            get_backend("TestBackend")

    @patch("egohub.backends")
    def test_get_backend_not_subclass(self, mock_backends):
        """Test get_backend with a class that is not a BaseBackend subclass."""

        class NotABackend:
            pass

        mock_backends.TestBackend = NotABackend

        with pytest.raises(TypeError, match="TestBackend is not a valid Backend class"):
            get_backend("TestBackend")


class TestBackendIntegration:
    """Test backend integration scenarios."""

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

        return mock_group

    def test_backend_with_camera_data(self, mock_traj_group):
        """Test a backend that processes camera data."""

        class CameraBackend(BaseBackend):
            def run(self, traj_group: h5py.Group, **kwargs):
                # Access camera data
                cameras = traj_group["cameras"]
                ego_camera = cameras["ego_camera"]
                rgb = ego_camera["rgb"]

                # Process some frames
                num_frames = rgb["image_bytes"].shape[0]
                results = []

                for i in range(min(num_frames, 3)):  # Process first 3 frames
                    frame_size = rgb["frame_sizes"][i]
                    frame_data = rgb["image_bytes"][i][:frame_size]
                    results.append(len(frame_data))

                return {"processed_frames": results}

        backend = CameraBackend()
        result = backend.run(mock_traj_group)

        assert "processed_frames" in result
        assert len(result["processed_frames"]) == 3
        assert all(size == 1000 for size in result["processed_frames"])

    def test_backend_with_kwargs(self, mock_traj_group):
        """Test a backend that uses kwargs."""

        class KwargsBackend(BaseBackend):
            def run(self, traj_group: h5py.Group, **kwargs):
                model_name = kwargs.get("model_name", "default_model")
                device = kwargs.get("device", "cpu")
                threshold = kwargs.get("threshold", 0.5)

                return {
                    "model_name": model_name,
                    "device": device,
                    "threshold": threshold,
                    "processed": True,
                }

        backend = KwargsBackend()

        # Test with kwargs
        result = backend.run(
            mock_traj_group, model_name="test_model", device="cuda", threshold=0.8
        )
        assert result["model_name"] == "test_model"
        assert result["device"] == "cuda"
        assert result["threshold"] == 0.8

        # Test with default kwargs
        result = backend.run(mock_traj_group)
        assert result["model_name"] == "default_model"
        assert result["device"] == "cpu"
        assert result["threshold"] == 0.5

    def test_backend_error_handling(self, mock_traj_group):
        """Test backend error handling."""

        class ErrorBackend(BaseBackend):
            def run(self, traj_group: h5py.Group, **kwargs):
                raise RuntimeError("Backend processing failed")

        backend = ErrorBackend()

        with pytest.raises(RuntimeError, match="Backend processing failed"):
            backend.run(mock_traj_group)

    def test_backend_empty_result(self, mock_traj_group):
        """Test a backend that returns empty results."""

        class EmptyBackend(BaseBackend):
            def run(self, traj_group: h5py.Group, **kwargs):
                return {}

        backend = EmptyBackend()
        result = backend.run(mock_traj_group)

        assert result == {}


class TestBackendDiscovery:
    """Test backend discovery functionality."""

    @patch("egohub.backends")
    def test_backend_discovery(self, mock_backends):
        """Test that backends can be discovered dynamically."""

        # Mock some backend classes
        class TestBackend1(BaseBackend):
            def run(self, traj_group: h5py.Group, **kwargs):
                return {"backend": "test1"}

        class TestBackend2(BaseBackend):
            def run(self, traj_group: h5py.Group, **kwargs):
                return {"backend": "test2"}

        # Mock the getattr function to return our test backends
        mock_backends.TestBackend1 = TestBackend1
        mock_backends.TestBackend2 = TestBackend2

        # Test that we can get the backends
        backend1 = get_backend("TestBackend1")
        backend2 = get_backend("TestBackend2")

        assert backend1 == TestBackend1
        assert backend2 == TestBackend2

        # Test that we get the expected results
        instance1 = backend1()
        instance2 = backend2()

        result1 = instance1.run(Mock())
        result2 = instance2.run(Mock())

        assert result1 == {"backend": "test1"}
        assert result2 == {"backend": "test2"}
