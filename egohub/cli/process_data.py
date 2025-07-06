"""Command-line script to apply processing tools to an HDF5 dataset."""

from __future__ import annotations

import argparse
import importlib
import inspect
import logging
from pathlib import Path

import h5py
from tqdm import tqdm

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Apply a processing task to an HDF5 dataset.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the input HDF5 file.",
    )
    parser.add_argument(
        "--task",
        type=str,
        required=True,
        help="The task to perform (e.g., 'PoseEstimationTask', 'ObjectDetectionTask').",
    )
    parser.add_argument(
        "--backend",
        type=str,
        required=True,
        help=(
            "The backend to use for the task "
            "(e.g., 'HuggingFaceBackend')."
        ),
    )
    parser.add_argument(
        "--backend-args",
        nargs="*",
        help="""Additional keyword arguments to pass to the backend's constructor,
in 'key=value' format. For example: model_name=sapiens-pose-1b""",
    )
    parser.add_argument(
        "--num-trajectories",
        type=int,
        help="Optional: Process only the first N trajectories.",
    )
    return parser.parse_args()


def _find_class(class_name: str, base_class: type, search_module: str) -> type | None:
    """Finds a class by name by scanning a given module path."""
    try:
        module_path = f"egohub.{search_module}"
        module_spec = importlib.util.find_spec(module_path)
        if module_spec is None or module_spec.origin is None:
            return None

        search_dir = Path(module_spec.origin).parent

        for f in search_dir.iterdir():
            if f.is_file() and f.suffix == ".py" and not f.name.startswith("_"):
                module_short_name = f.stem
                try:
                    module = importlib.import_module(
                        f".{module_short_name}", module_path
                    )
                    for name, obj in inspect.getmembers(module, inspect.isclass):
                        if name == class_name and issubclass(obj, base_class):
                            return obj
                except ImportError as e:
                    logger.warning(f"Could not import {module_short_name}: {e}")
    except Exception as e:
        logger.error(f"Error finding class '{class_name}': {e}")
    return None


def _parse_backend_arguments(args: argparse.Namespace) -> dict:
    """Parse backend-specific arguments from command line."""
    backend_kwargs = {}
    if args.backend_args:
        for arg in args.backend_args:
            try:
                key, value = arg.split("=", 1)
                backend_kwargs[key] = value
            except ValueError:
                logger.error(
                    f"Invalid backend argument format: '{arg}'. "
                    "Expected format: 'key=value'"
                )
                return {}
    return backend_kwargs


def _process_trajectories(
    f: h5py.File, task: BaseTask, backend: BaseBackend, num_trajectories: int | None
) -> None:
    """Process trajectories with the given task and backend."""
    trajectory_keys = sorted([key for key in f.keys() if key.startswith("trajectory_")])

    if num_trajectories:
        trajectory_keys = trajectory_keys[:num_trajectories]

    for key in tqdm(trajectory_keys, desc="Processing Trajectories"):
        traj_group = f[key]
        if not isinstance(traj_group, h5py.Group):
            continue
        task.run(traj_group, backend)


def main() -> None:
    """Main entry point for the script."""
    args = get_args()

    if not args.input_file.exists():
        logger.error(f"Input file not found: {args.input_file}")
        return

    # Find and instantiate the Task
    task_class = _find_class(args.task, BaseTask, "tasks")
    if not task_class:
        logger.error(f"Task class '{args.task}' not found in egohub.tasks.")
        return
    task_instance = task_class()

    # Map task class names to task names for backends
    task_name_mapping = {
        "ObjectDetectionTask": "object-detection",
        "PoseEstimationTask": "pose-estimation",
    }
    task_name = task_name_mapping.get(args.task)

    # Find and instantiate the Backend
    backend_class = _find_class(args.backend, BaseBackend, "backends")
    if not backend_class:
        logger.error(f"Backend class '{args.backend}' not found in egohub.backends.")
        return
    backend_kwargs = _parse_backend_arguments(args)

    # Add task_name to backend kwargs if it's a HuggingFaceBackend
    if args.backend == "HuggingFaceBackend" and task_name:
        backend_kwargs["task_name"] = task_name

    try:
        backend_instance = backend_class(**backend_kwargs)
    except (TypeError, FileNotFoundError) as e:
        logger.error(f"Failed to instantiate backend '{args.backend}': {e}")
        return

    # Process the file
    logger.info(
        f"Processing file: {args.input_file} with Task: {args.task} "
        f"and Backend: {args.backend}"
    )
    with h5py.File(args.input_file, "a") as f:
        _process_trajectories(f, task_instance, backend_instance, args.num_trajectories)

    logger.info("Processing complete.")


if __name__ == "__main__":
    main()
