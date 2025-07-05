from __future__ import annotations

import argparse
import inspect
import logging
import pkgutil
from pathlib import Path

import h5py

import egohub.adapters
from egohub.adapters.base import BaseAdapter
from egohub.exporters.rerun import RerunExporter
from egohub.schema import SchemaValidationError, Trajectory, validate_hdf5_with_schema


def discover_adapters() -> dict[str, type[BaseAdapter]]:
    """Dynamically discovers and loads adapter classes from the
    'egohub.adapters' module."""
    adapter_map = {}
    adapter_module = egohub.adapters
    module_path = adapter_module.__path__  # type: ignore
    module_name_prefix = adapter_module.__name__ + "."  # type: ignore

    for _, module_name, _ in pkgutil.walk_packages(
        module_path, prefix=module_name_prefix
    ):
        module = __import__(module_name, fromlist="dummy")
        for _, obj in inspect.getmembers(module):
            if (
                inspect.isclass(obj)
                and issubclass(obj, BaseAdapter)
                and obj is not BaseAdapter
            ):
                if obj.name:
                    adapter_map[obj.name] = obj

    return adapter_map


# A simple registry to map dataset names to their adapter classes.
ADAPTER_MAP = discover_adapters()


def main():
    """The main entry point for the egohub command-line interface."""
    parser = argparse.ArgumentParser(
        description="A toolkit for egocentric data processing."
    )
    subparsers = parser.add_subparsers(
        dest="command", required=True, help="Available commands"
    )

    # --- Convert Command ---
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

    # --- Visualize Command ---
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

    # --- Validate Command ---
    parser_validate = subparsers.add_parser(
        "validate", help="Validates an HDF5 file against the canonical schema."
    )
    parser_validate.add_argument(
        "h5_path", type=Path, help="The path to the canonical HDF5 file to validate."
    )

    args = parser.parse_args()

    if args.command == "convert":
        adapter_class = ADAPTER_MAP[args.dataset]
        adapter = adapter_class(args.raw_dir, args.output_file)
        adapter.run(num_sequences=args.num_sequences)

        # --- Automatic Validation Step ---
        logging.basicConfig(level=logging.INFO)
        logging.info(f"Conversion complete. Validating output file: {args.output_file}")
        try:
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
    elif args.command == "visualize":
        exporter = RerunExporter(max_frames=args.max_frames)
        exporter.export(args.h5_path, output_path=args.output_rrd)
    elif args.command == "validate":
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


if __name__ == "__main__":
    main()
