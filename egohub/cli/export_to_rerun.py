"""Command-line script to export an HDF5 dataset to Rerun for visualization."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from egohub.exporters.rerun import RerunExporter


def get_args() -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(description="Export an HDF5 dataset to Rerun.")
    parser.add_argument(
        "--input-file",
        type=Path,
        required=True,
        help="Path to the input HDF5 file.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Maximum number of frames to export.",
    )
    parser.add_argument(
        "--save-rrd",
        type=Path,
        default=None,
        help=(
            "If specified, save the Rerun data to this path instead of spawning a "
            "viewer."
        ),
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the script."""
    logging.basicConfig(level=logging.INFO)
    args = get_args()

    if not args.input_file.exists():
        logging.error(f"Input file not found: {args.input_file}")
        return

    logging.info(f"Exporting data from: {args.input_file}")

    exporter = RerunExporter(max_frames=args.max_frames)
    exporter.export(args.input_file, output_path=args.save_rrd)


if __name__ == "__main__":
    main()
