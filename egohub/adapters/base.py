"""
Base classes for dataset adapters.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import yaml
from tqdm import tqdm


class BaseAdapter(ABC):
    """
    Abstract base class for all dataset adapters.

    Adapters are responsible for converting data from a raw, dataset-specific
    format into our canonical HDF5 format. This class provides a standard
    interface and a common run loop for all adapters.
    """

    name: str = ""

    def __init__(
        self, raw_dir: Path, output_file: Path, config: Optional[Dict[str, Any]] = None
    ):
        self.raw_dir = raw_dir
        self.output_file = output_file
        self.config = config or self._load_config()
        logging.info(f"Starting adapter: {self.__class__.__name__}")

    def _load_config(self) -> dict:
        """Loads the adapter-specific YAML configuration file."""
        if not self.name:
            logging.warning(
                f"Adapter '{self.__class__.__name__}' has no name. "
                "Skipping config load."
            )
            return {}

        # Get the directory of the current file (base.py)
        base_dir = Path(__file__).parent.resolve()

        # Path for config inside adapter's own folder
        adapter_config_path = base_dir / self.name / f"{self.name}.yaml"

        # Path for config in the project's root configs folder
        # Assumes egohub/adapters/base.py -> egohub/configs/
        root_config_path = base_dir.parent / "configs" / f"{self.name}.yaml"

        config_path = None
        if adapter_config_path.exists():
            config_path = adapter_config_path
        elif root_config_path.exists():
            config_path = root_config_path
        else:
            logging.warning(
                f"No config file found for adapter '{self.name}'.\n"
                f"Searched paths:\n"
                f"- {adapter_config_path}\n"
                f"- {root_config_path}"
            )
            return {}

        logging.info(f"Loading configuration from {config_path}...")
        with open(config_path) as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file {config_path}: {e}")
                raise

    @property
    @abstractmethod
    def source_joint_names(self) -> List[str]:
        """Returns the list of joint names for the source skeleton."""
        raise NotImplementedError

    @property
    @abstractmethod
    def source_skeleton_hierarchy(self) -> Dict[str, str]:
        """Returns the kinematic hierarchy for the source skeleton."""
        raise NotImplementedError

    @abstractmethod
    def discover_sequences(self) -> List[Dict[str, Any]]:
        """
        Discovers all processable data sequences in the raw_dir.

        This method should scan the raw data directory and return a list of
        dictionaries, where each dictionary contains the necessary information
        (e.g., file paths, sequence names) to process one sequence.

        Returns:
            A list of sequence information dictionaries.
        """
        pass

    @abstractmethod
    def process_sequence(self, seq_info: Dict[str, Any], traj_group: h5py.Group):
        """
        Processes a single data sequence and writes it to the HDF5 group.

        This is the core method where dataset-specific parsing, transformation,
        and writing logic resides.

        Args:
            seq_info (Dict[str, Any]): A dictionary from the list returned by
                discover_sequences().
            traj_group (h5py.Group): The HDF5 group to write the processed data into.
        """
        pass

    def run(self, num_sequences: Optional[int] = None):
        """
        Main loop to discover, process, and save all sequences.

        This method orchestrates the conversion process:
        1. Calls discover_sequences() to find all data.
        2. Creates/opens the output HDF5 file.
        3. Iterates through sequences, calling process_sequence() for each one.

        Args:
            num_sequences (Optional[int]): If specified, only process the first
                                        `num_sequences` sequences.
        """
        logging.info(f"Input directory: {self.raw_dir}")
        logging.info(f"Output file: {self.output_file}")

        sequences = self.discover_sequences()
        if num_sequences is not None:
            logging.info(
                f"Limiting processing to the first {num_sequences} sequence(s)."
            )
            sequences = sequences[:num_sequences]

        if not sequences:
            logging.warning("No sequences found to process.")
            return

        logging.info("Creating HDF5 file and processing sequences...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_file, "w") as f_out:
            for i, seq_info in enumerate(tqdm(sequences, desc="Processing Sequences")):
                traj_group = f_out.create_group(f"trajectory_{i:04d}")
                self.process_sequence(seq_info, traj_group)

        logging.info("Conversion process completed successfully.")
