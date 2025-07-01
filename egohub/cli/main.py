from __future__ import annotations
from pathlib import Path
import argparse
from egohub.adapters.egodex import EgoDexAdapter
from egohub.datasets import EgocentricH5Dataset
from egohub.exporters.rerun import RerunExporter

# A simple registry to map dataset names to their adapter classes.
ADAPTER_MAP = {
    "egodex": EgoDexAdapter
}

def main():
    """The main entry point for the egohub command-line interface."""
    parser = argparse.ArgumentParser(description="A toolkit for egocentric data processing.")
    subparsers = parser.add_subparsers(dest="command", required=True, help="Available commands")

    # --- Convert Command ---
    parser_convert = subparsers.add_parser("convert", help="Converts a raw dataset to the canonical HDF5 format.")
    parser_convert.add_argument("dataset", choices=ADAPTER_MAP.keys(), help="The name of the dataset to process.")
    parser_convert.add_argument("--raw-dir", required=True, type=Path, help="The path to the raw dataset directory.")
    parser_convert.add_argument("--output-file", required=True, type=Path, help="The path to the output HDF5 file.")
    parser_convert.add_argument("--num-sequences", type=int, help="Optional number of sequences to process.")

    # --- Visualize Command ---
    parser_visualize = subparsers.add_parser("visualize", help="Visualize an HDF5 file using the Rerun viewer.")
    parser_visualize.add_argument("h5_path", type=Path, help="The path to the canonical HDF5 file.")
    parser_visualize.add_argument("--max-frames", type=int, help="Optional maximum number of frames to visualize.")

    args = parser.parse_args()

    if args.command == "convert":
        adapter_class = ADAPTER_MAP[args.dataset]
        adapter = adapter_class(args.raw_dir, args.output_file)
        adapter.run(num_sequences=args.num_sequences)
    elif args.command == "visualize":
        dataset = EgocentricH5Dataset(args.h5_path)
        exporter = RerunExporter(max_frames=args.max_frames)
        exporter.export(dataset)

if __name__ == "__main__":
    main() 