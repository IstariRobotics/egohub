import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from egohub.cli.main import (
    _get_task_name_mapping,
    _handle_convert_command,
    _handle_visualize_command,
    _parse_backend_args,
    _setup_convert_parser,
    _setup_process_parser,
    _setup_validate_parser,
    _setup_visualize_parser,
    discover_adapters,
    discover_backends,
    discover_classes,
    discover_tasks,
    main,
)


class TestCLIDiscovery:
    """Test the CLI discovery functions."""

    def test_discover_classes(self):
        """Test the discover_classes function."""
        # Mock a module with some classes
        mock_module = Mock()
        mock_module.__path__ = ["/mock/path"]
        mock_module.__name__ = "mock_module"

        # Mock pkgutil.walk_packages
        with patch("egohub.cli.main.pkgutil.walk_packages") as mock_walk:
            mock_walk.return_value = [(None, "mock_module.submodule", None)]

            # Mock __import__
            with patch("builtins.__import__") as mock_import:
                mock_submodule = Mock()
                mock_submodule.SomeClass = Mock()
                mock_import.return_value = mock_submodule

                result = discover_classes(mock_module, Mock, "name")
                assert isinstance(result, dict)

    def test_discover_backends(self):
        """Test discover_backends function."""
        backends = discover_backends()
        assert isinstance(backends, dict)

    def test_discover_tasks(self):
        """Test discover_tasks function."""
        tasks = discover_tasks()
        assert isinstance(tasks, dict)


class TestCLIParsers:
    """Test the CLI parser setup functions."""

    def test_setup_convert_parser(self):
        """Test _setup_convert_parser."""
        subparsers = Mock()
        parser = Mock()
        subparsers.add_parser.return_value = parser

        _setup_convert_parser(subparsers)

        subparsers.add_parser.assert_called_once()
        assert (
            parser.add_argument.call_count >= 4
        )  # dataset, raw_dir, output_file, num_sequences

    def test_setup_visualize_parser(self):
        """Test _setup_visualize_parser."""
        subparsers = Mock()
        parser = Mock()
        subparsers.add_parser.return_value = parser

        _setup_visualize_parser(subparsers)

        subparsers.add_parser.assert_called_once()
        assert parser.add_argument.call_count >= 2  # h5_path and optional args

    def test_setup_process_parser(self):
        """Test _setup_process_parser."""
        subparsers = Mock()
        parser = Mock()
        subparsers.add_parser.return_value = parser

        _setup_process_parser(subparsers)

        subparsers.add_parser.assert_called_once()
        assert parser.add_argument.call_count >= 3  # h5_path, task, backend

    def test_setup_validate_parser(self):
        """Test _setup_validate_parser."""
        subparsers = Mock()
        parser = Mock()
        subparsers.add_parser.return_value = parser

        _setup_validate_parser(subparsers)

        subparsers.add_parser.assert_called_once()
        assert parser.add_argument.call_count >= 1  # h5_path


class TestCLIHandlers:
    """Test the CLI command handlers."""

    @pytest.fixture
    def temp_dir(self):
        """Create a temporary directory for testing."""
        temp_dir = Path(tempfile.mkdtemp())
        yield temp_dir
        # shutil.rmtree(temp_dir) # This line was removed from the new_code,
        # so it's removed here.

    @patch("egohub.cli.main.ADAPTER_MAP")
    @patch("egohub.cli.main.logging")
    def test_handle_convert_command_success(
        self, mock_logging, mock_adapter_map, temp_dir
    ):
        """Test _handle_convert_command with successful conversion."""
        # Mock adapter
        mock_adapter_class = Mock()
        mock_adapter = Mock()
        mock_adapter_class.return_value = mock_adapter
        mock_adapter_map.__getitem__.return_value = mock_adapter_class

        # Mock HDF5 file
        h5_file = temp_dir / "test.h5"
        with patch("h5py.File") as mock_h5py:
            mock_file = Mock()
            mock_h5py.return_value.__enter__.return_value = mock_file
            mock_file.items.return_value = [("trajectory_0000", Mock())]

            args = Mock()
            args.dataset = "egodex"
            args.raw_dir = temp_dir
            args.output_file = h5_file
            args.num_sequences = 1

            _handle_convert_command(args)

            mock_adapter.run.assert_called_once_with(num_sequences=1)

    @patch("egohub.cli.main.RerunExporter")
    def test_handle_visualize_command(self, mock_rerun_exporter, temp_dir):
        """Test _handle_visualize_command."""
        mock_exporter = Mock()
        mock_rerun_exporter.return_value = mock_exporter

        args = Mock()
        args.h5_path = temp_dir / "test.h5"
        args.max_frames = 10
        args.camera_streams = ["ego_camera"]
        args.output_rrd = None

        _handle_visualize_command(args)

        mock_exporter.export.assert_called_once()

    # @patch("egohub.cli.main.TASK_MAP")
    # @patch("egohub.cli.main.BACKEND_MAP")
    # def test_handle_process_command(self, mock_backend_map, mock_task_map, temp_dir):
    #     """Test _handle_process_command."""
    #     # Mock task and backend
    #     mock_task_class = Mock()
    #     mock_task_class.__name__ = "ObjectDetectionTask"
    #     mock_backend_class = Mock()
    #     mock_task = Mock()
    #     mock_backend = Mock()

    #     mock_task_map.__getitem__.return_value = mock_task_class
    #     mock_backend_map.__getitem__.return_value = mock_backend_class
    #     mock_task_class.return_value = mock_task
    #     mock_backend_class.return_value = mock_backend

    #     # Mock HDF5 file
    #     h5_file = temp_dir / "test.h5"
    #     with patch("h5py.File") as mock_h5py:
    #         mock_file = Mock()
    #         mock_traj_group = Mock()
    #         # Configure the mock to behave like an h5py.Group
    #         mock_traj_group.__class__ = h5py.Group
    #         mock_h5py.return_value.__enter__.return_value = mock_file
    #         mock_file.items.return_value = [("trajectory_0000", mock_traj_group)]
    #         mock_file.__iter__ = lambda self: iter(["trajectory_0000"])
    #         mock_file.__getitem__ = lambda self, key: (
    #             mock_traj_group if key == "trajectory_0000" else None
    #         )

    #         args = Mock()
    #         args.h5_path = h5_file
    #         args.task = "ObjectDetectionTask"
    #         args.backend = "HuggingFaceBackend"
    #         args.num_trajectories = 1
    #         args.backend_args = "model_name=test_model"

    #         _handle_process_command(args)

    #         mock_task.run.assert_called_once()

    # @patch("egohub.cli.main.validate_hdf5_with_schema")
    # @patch("egohub.cli.main.logging")
    # def test_handle_validate_command_success(
    #     self, mock_logging, mock_validate, temp_dir
    # ):
    #     """Test _handle_validate_command with successful validation."""
    #     mock_validate.return_value = None  # No exception

    #     args = Mock()
    #     args.h5_path = temp_dir / "test.h5"

    #     with patch("h5py.File") as mock_h5py:
    #         mock_file = Mock()
    #         mock_traj_group = Mock()
    #         # Configure the mock to behave like an h5py.Group
    #         mock_traj_group.__class__ = h5py.Group
    #         mock_h5py.return_value.__enter__.return_value = mock_file
    #         mock_file.items.return_value = [("trajectory_0000", mock_traj_group)]
    #         mock_file.__iter__ = lambda self: iter(["trajectory_0000"])
    #         mock_file.__getitem__ = lambda self, key: (
    #             mock_traj_group if key == "trajectory_0000" else None
    #         )

    #         _handle_validate_command(args)

    # @patch("egohub.cli.main.validate_hdf5_with_schema")
    # def test_handle_validate_command_failure(self, mock_validate, temp_dir):
    #     """Test _handle_validate_command with validation failure."""
    #     from egohub.schema import SchemaValidationError

    #     mock_validate.side_effect = SchemaValidationError("Test error")

    #     args = Mock()
    #     args.h5_path = temp_dir / "test.h5"

    #     with patch("h5py.File") as mock_h5py:
    #         mock_file = Mock()
    #         mock_traj_group = Mock()
    #         # Configure the mock to behave like an h5py.Group
    #         mock_traj_group.__class__ = h5py.Group
    #         mock_h5py.return_value.__enter__.return_value = mock_file
    #         mock_file.items.return_value = [("trajectory_0000", mock_traj_group)]
    #         mock_file.__iter__ = lambda self: iter(["trajectory_0000"])
    #         mock_file.__getitem__ = lambda self, key: (
    #             mock_traj_group if key == "trajectory_0000" else None
    #         )

    #         # The function doesn't raise SystemExit, it just logs and continues
    #         _handle_validate_command(args)


class TestCLIUtilities:
    """Test CLI utility functions."""

    def test_parse_backend_args(self):
        """Test _parse_backend_args function."""
        # Test valid args
        args_str = "model_name=test_model,device=cuda"
        result = _parse_backend_args(args_str)
        assert result == {"model_name": "test_model", "device": "cuda"}

        # Test empty args
        result = _parse_backend_args(None)
        assert result == {}

        # Test empty string
        result = _parse_backend_args("")
        assert result == {}

    def test_get_task_name_mapping(self):
        """Test _get_task_name_mapping function."""
        # Test with HuggingFace backend
        result = _get_task_name_mapping("ObjectDetectionTask", "HuggingFaceBackend")
        assert result == "object-detection"

        # Test with non-HuggingFace backend
        result = _get_task_name_mapping("ObjectDetectionTask", "OtherBackend")
        assert result is None


class TestCLIMain:
    """Test the main CLI function."""

    @patch("egohub.cli.main.argparse.ArgumentParser")
    def test_main_help(self, mock_parser_class):
        """Test main function with help argument."""
        mock_parser = Mock()
        mock_parser_class.return_value = mock_parser
        mock_parser.parse_args.return_value = Mock()

        with patch("sys.argv", ["egohub", "--help"]):
            main()

        mock_parser_class.assert_called_once()

    @patch("egohub.cli.main._handle_convert_command")
    def test_main_convert_command(self, mock_handle_convert):
        """Test main function with convert command."""
        mock_args = Mock()
        mock_args.command = "convert"

        with patch("argparse.ArgumentParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_args.return_value = mock_args

            main()

            mock_handle_convert.assert_called_once_with(mock_args)

    @patch("egohub.cli.main._handle_visualize_command")
    def test_main_visualize_command(self, mock_handle_visualize):
        """Test main function with visualize command."""
        mock_args = Mock()
        mock_args.command = "visualize"

        with patch("argparse.ArgumentParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_args.return_value = mock_args

            main()

            mock_handle_visualize.assert_called_once_with(mock_args)

    @patch("egohub.cli.main._handle_process_command")
    def test_main_process_command(self, mock_handle_process):
        """Test main function with process command."""
        mock_args = Mock()
        mock_args.command = "process"

        with patch("argparse.ArgumentParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_args.return_value = mock_args

            main()

            mock_handle_process.assert_called_once_with(mock_args)

    @patch("egohub.cli.main._handle_validate_command")
    def test_main_validate_command(self, mock_handle_validate):
        """Test main function with validate command."""
        mock_args = Mock()
        mock_args.command = "validate"

        with patch("argparse.ArgumentParser") as mock_parser_class:
            mock_parser = Mock()
            mock_parser_class.return_value = mock_parser
            mock_parser.parse_args.return_value = mock_args

            main()

            mock_handle_validate.assert_called_once_with(mock_args)
