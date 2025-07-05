"""
Base classes for data exporters.
"""

import argparse
import logging
from abc import ABC, abstractmethod
from pathlib import Path

from egohub.datasets import EgocentricH5Dataset


class BaseExporter(ABC):
    """
    Abstract base class for all data exporters.

    Exporters are responsible for consuming data from a canonical HDF5 file
    (via an EgocentricH5Dataset) and exporting it to another format or application,
    such as Rerun, a robotics simulator, or another dataset format.
    """

    def __init__(self):
        pass

    @abstractmethod
    def export(self, dataset: EgocentricH5Dataset, output_path: Path | None, **kwargs):
        """
        Exports data from the dataset to a specific format or application.

        Args:
            dataset: The loaded EgocentricH5Dataset instance.
            output_path: Optional path for the exported output.
            **kwargs: Additional exporter-specific arguments.
        """
        pass

    def get_arg_parser(self) -> argparse.ArgumentParser:
        """
        Gets a command-line argument parser with common arguments.
        Subclasses can extend this to add their own specific arguments.
        """
        parser = argparse.ArgumentParser(description=self.__doc__)
        parser.add_argument("h5_path", help="Path to the input canonical HDF5 file.")
        parser.add_argument(
            "--output_path", help="Optional path for the exported file(s)."
        )
        parser.add_argument(
            "--trajectories", nargs="*", help="Specific trajectory names to process."
        )
        return parser

    def run_from_main(self):
        """
        Main entry point for running the exporter from a script.
        This method handles argument parsing and dataset loading.
        """
        logging.basicConfig(
            level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
        )

        parser = self.get_arg_parser()
        args = parser.parse_args()

        h5_path = Path(args.h5_path)
        if not h5_path.exists():
            logging.error(f"Error: HDF5 file not found at {h5_path}")
            return

        # Pass all parsed args to the export method
        arg_dict = vars(args)

        # Dataset loading is handled here
        dataset = EgocentricH5Dataset(
            h5_path, trajectories=arg_dict.get("trajectories")
        )

        output_path_str = arg_dict.pop("output_path", None)
        output_path = Path(output_path_str) if output_path_str else None

        self.export(dataset, output_path, **arg_dict)
