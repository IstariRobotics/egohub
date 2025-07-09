from __future__ import annotations

import argparse
import inspect
import logging
import pkgutil
from pathlib import Path

import h5py

import egohub.adapters
import egohub.backends
import egohub.tasks
from egohub.adapters.base import BaseAdapter
from egohub.backends.base import BaseBackend
from egohub.exporters.rerun_exporter import RerunExporter
from egohub.schema import SchemaValidationError, Trajectory, validate_hdf5_with_schema
from egohub.tasks.base import BaseTask


def discover_classes(module, base_class, name_attr=None):
    """Dynamically discovers and loads classes from a module."""
    class_map = {}
    module_path = module.__path__  # type: ignore
    module_name_prefix = module.__name__ + "."  # type: ignore

    for _, module_name, _ in pkgutil.walk_packages(
        module_path, prefix=module_name_prefix
    ):
        try:
            module_obj = __import__(module_name, fromlist="dummy")
            for _, obj in inspect.getmembers(module_obj):
                if (
                    inspect.isclass(obj)
                    and issubclass(obj, base_class)
                    and obj is not base_class
                ):
                    if name_attr and hasattr(obj, name_attr):
                        class_map[getattr(obj, name_attr)] = obj
                    else:
                        class_map[obj.__name__] = obj
        except ImportError as e:
            logging.warning(f"Could not import {module_name}: {e}")

    return class_map


def discover_adapters() -> dict[str, type[BaseAdapter]]:
    """Dynamically discovers and loads adapter classes from the
    'egohub.adapters' module."""
    return discover_classes(egohub.adapters, BaseAdapter, "name")


def discover_backends() -> dict[str, type[BaseBackend]]:
    """Dynamically discovers and loads backend classes from the
    'egohub.backends' module."""
    return discover_classes(egohub.backends, BaseBackend)


def discover_tasks() -> dict[str, type[BaseTask]]:
    """Dynamically discovers and loads task classes from the
    'egohub.tasks' module."""
    return discover_classes(egohub.tasks, BaseTask)


# Registry maps for dynamically discovered classes
ADAPTER_MAP = discover_adapters()
BACKEND_MAP = discover_backends()
TASK_MAP = discover_tasks()


def _setup_convert_parser(subparsers):
    """Set up the convert command parser."""
    parser_convert = subparsers.add_parser(
        "convert", help="Converts a raw dataset to the canonical HDF5 format."
    )
    parser_convert.add_argument(
        "dataset",
        choices=ADAPTER_MAP.keys(),
        help="The name of the dataset to process.",
    )
    parser_convert.add_argument(
        "--raw-dir",
        required=True,
        type=Path,
        help="The path to the raw dataset directory.",
    )
    parser_convert.add_argument(
        "--output-file",
        required=True,
        type=Path,
        help="The path to the output HDF5 file.",
    )
    parser_convert.add_argument(
        "--num-sequences", type=int, help="Optional number of sequences to process."
    )


def _setup_visualize_parser(subparsers):
    """Set up the visualize command parser."""
    parser_visualize = subparsers.add_parser(
        "visualize", help="Visualize an HDF5 file using the Rerun viewer."
    )
    parser_visualize.add_argument(
        "h5_path", type=Path, help="The path to the canonical HDF5 file."
    )
    parser_visualize.add_argument(
        "--max-frames", type=int, help="Optional maximum number of frames to visualize."
    )
    parser_visualize.add_argument(
        "--camera-streams",
        nargs="+",
        help=(
            "Optional list of camera streams to visualize "
            "(e.g., ego_camera_left ego_camera_right). "
            "Defaults to the first one found."
        ),
    )
    parser_visualize.add_argument(
        "--output-rrd",
        type=Path,
        help="Optional path to save the RRD data to instead of spawning the viewer.",
    )


def _setup_process_parser(subparsers):
    """Set up the process command parser."""
    parser_process = subparsers.add_parser(
        "process", help="Process an HDF5 file with a task and backend."
    )
    parser_process.add_argument(
        "h5_path", type=Path, help="The path to the canonical HDF5 file to process."
    )
    parser_process.add_argument(
        "--task",
        required=True,
        choices=TASK_MAP.keys(),
        help="The task to run (e.g., ObjectDetectionTask, PoseEstimationTask).",
    )
    parser_process.add_argument(
        "--backend",
        required=True,
        choices=BACKEND_MAP.keys(),
        help="The backend to use (e.g., HuggingFaceBackend).",
    )
    parser_process.add_argument(
        "--num-trajectories",
        type=int,
        help="Optional number of trajectories to process (default: all).",
    )
    parser_process.add_argument(
        "--backend-args",
        type=str,
        help=(
            "Optional backend arguments as key=value pairs "
            "(e.g., 'model_name=facebook/detr-resnet-50')."
        ),
    )


def _setup_validate_parser(subparsers):
    """Set up the validate command parser."""
    parser_validate = subparsers.add_parser(
        "validate", help="Validates an HDF5 file against the canonical schema."
    )
    parser_validate.add_argument(
        "h5_path", type=Path, help="The path to the canonical HDF5 file to validate."
    )


def _handle_convert_command(args):
    """Handle the convert command."""
    try:
        adapter_class = ADAPTER_MAP[args.dataset]
        adapter = adapter_class(args.raw_dir, args.output_file)
        adapter.run(num_sequences=args.num_sequences)

        # --- Automatic Validation Step ---
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Conversion complete. Validating output file: {args.output_file}")
        
        with h5py.File(args.output_file, "r") as f:
            for traj_name, traj_group in f.items():
                if isinstance(traj_group, h5py.Group) and traj_name.startswith(
                    "trajectory_"
                ):
                    logging.info(f"--- Validating trajectory: {traj_name} ---")
                    validate_hdf5_with_schema(
                        traj_group, Trajectory, path=traj_name, strict=True
                    )
        logging.info(
            "Validation successful: The HDF5 file is compliant with the "
            "canonical schema."
        )
    except SchemaValidationError as e:
        logging.error("Validation Failed: The output HDF5 file is not compliant.")
        logging.error(f"Error details: {e}")
        # Optionally, clean up the invalid file
        # args.output_file.unlink()
        exit(1)  # Exit with an error code
    except FileNotFoundError:
        logging.error(f"Error: Output file not found after conversion: {args.output_file}")
        logging.error("This likely means the adapter failed to find any sequences to process or could not write the output file.")
        exit(1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during conversion: {e}")
        import traceback
        traceback.print_exc()
        exit(1)


def _handle_visualize_command(args):
    """Handle the visualize command."""
    exporter = RerunExporter(max_frames=args.max_frames)
    exporter.export(args.h5_path, output_path=args.output_rrd)


def _parse_backend_args(backend_args_str):
    """Parse backend arguments from string format."""
    backend_kwargs = {}
    if backend_args_str:
        for arg in backend_args_str.split(","):
            if "=" in arg:
                key, value = arg.split("=", 1)
                backend_kwargs[key.strip()] = value.strip()
    return backend_kwargs


def _get_task_name_mapping(task, backend):
    """Get task name mapping for HuggingFace backend."""
    if backend == "HuggingFaceBackend":
        task_name_mapping = {
            "ObjectDetectionTask": "object-detection",
            "PoseEstimationTask": "pose-estimation",
        }
        return task_name_mapping.get(task)
    return None


def _handle_process_command(args):
    """Handle the process command."""
    logging.basicConfig(level=logging.INFO)

    # Parse backend arguments
    backend_kwargs = _parse_backend_args(args.backend_args)

    # Get task and backend classes
    task_class = TASK_MAP[args.task]
    backend_class = BACKEND_MAP[args.backend]

    # Add task_name to backend kwargs if it's a HuggingFaceBackend
    task_name = _get_task_name_mapping(args.task, args.backend)
    if task_name:
        backend_kwargs["task_name"] = task_name

    # Instantiate task and backend
    output_group_name = task_class.__name__.replace("Task", "").lower()
    task_instance = task_class(output_group_name=output_group_name)
    try:
        backend_instance = backend_class(**backend_kwargs)
    except (TypeError, FileNotFoundError) as e:
        logging.error(f"Failed to instantiate backend '{args.backend}': {e}")
        return

    # Process the file
    logging.info(
        f"Processing file: {args.h5_path} with Task: {args.task} "
        f"and Backend: {args.backend}"
    )

    with h5py.File(args.h5_path, "a") as f:
        # Get trajectory names
        trajectory_names = sorted(
            [name for name in f.keys() if name.startswith("trajectory_")]
        )

        if not trajectory_names:
            logging.error("No trajectories found in the HDF5 file.")
            return

        # Limit number of trajectories if specified
        if args.num_trajectories:
            trajectory_names = trajectory_names[: args.num_trajectories]

        # Process each trajectory
        for traj_name in trajectory_names:
            logging.info(f"Processing trajectory: {traj_name}")
            traj_group = f[traj_name]
            if isinstance(traj_group, h5py.Group):
                task_instance.run(traj_group, backend_instance)
            else:
                logging.warning(f"'{traj_name}' is not a group, skipping.")

    logging.info("Processing complete.")


def _handle_validate_command(args):
    """Handle the validate command."""
    logging.basicConfig(level=logging.INFO)
    logging.info(f"Validating {args.h5_path} against the canonical schema...")
    with h5py.File(args.h5_path, "r") as f:
        for traj_name, traj_group in f.items():
            if isinstance(traj_group, h5py.Group) and traj_name.startswith(
                "trajectory_"
            ):
                logging.info(f"--- Validating trajectory: {traj_name} ---")
                validate_hdf5_with_schema(traj_group, Trajectory, path=traj_name)
    logging.info("Validation complete.")


def main():
    """The main entry point for the egohub command-line interface."""
    parser = argparse.ArgumentParser(
        description="A toolkit for egocentric data processing."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # Set up command parsers
    _setup_convert_parser(subparsers)
    _setup_visualize_parser(subparsers)
    _setup_process_parser(subparsers)
    _setup_validate_parser(subparsers)

    args = parser.parse_args()

    # Handle commands
    if args.command == "convert":
        _handle_convert_command(args)
    elif args.command == "visualize":
        _handle_visualize_command(args)
    elif args.command == "process":
        _handle_process_command(args)
    elif args.command == "validate":
        _handle_validate_command(args)


if __name__ == "__main__":
    main()
