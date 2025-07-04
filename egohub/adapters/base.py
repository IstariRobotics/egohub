"""
Base classes for dataset adapters.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import logging
import yaml

import h5py
from tqdm import tqdm


class BaseAdapter(ABC):
    """
    Abstract base class for all dataset adapters.

    Adapters are responsible for converting data from a raw, dataset-specific
    format into our canonical HDF5 format. This class provides a standard
    interface and a common run loop for all adapters.
    """

    name: str = ""

    def __init__(self, raw_dir: Path, output_file: Path):
        self.raw_dir = raw_dir
        self.output_file = output_file
        self.config = self._load_config()
        logging.info(f"Starting adapter: {self.__class__.__name__}")

    def _load_config(self) -> dict:
        """Loads the adapter-specific YAML configuration file."""
        if not self.name:
            logging.warning(f"Adapter '{self.__class__.__name__}' has no name. Skipping config load.")
            return {}

        # Assuming the script is run from the project root
        config_path = Path(f"configs/{self.name}.yaml")
        if not config_path.exists():
            logging.warning(f"No config file found for adapter '{self.name}' at {config_path}.")
            return {}

        logging.info(f"Loading configuration from {config_path}...")
        with open(config_path, 'r') as f:
            try:
                return yaml.safe_load(f)
            except yaml.YAMLError as e:
                logging.error(f"Error parsing YAML file {config_path}: {e}")
                raise

    @abstractmethod
    def discover_sequences(self) -> list[dict]:
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
    def process_sequence(self, seq_info: dict, traj_group: h5py.Group):
        """
        Processes a single data sequence and writes it to the HDF5 group.

        This is the core method where dataset-specific parsing, transformation,
        and writing logic resides.

        Args:
            seq_info (dict): A dictionary from the list returned by discover_sequences().
            traj_group (h5py.Group): The HDF5 group to write the processed data into.
        """
        pass

    def run(self, num_sequences: int | None = None):
        """
        Main loop to discover, process, and save all sequences.

        This method orchestrates the conversion process:
        1. Calls discover_sequences() to find all data.
        2. Creates/opens the output HDF5 file.
        3. Iterates through sequences, calling process_sequence() for each one.

        Args:
            num_sequences (int | None): If specified, only process the first
                                        `num_sequences` sequences.
        """
        logging.info(f"Input directory: {self.raw_dir}")
        logging.info(f"Output file: {self.output_file}")

        sequences = self.discover_sequences()
        if num_sequences is not None:
            logging.info(f"Limiting processing to the first {num_sequences} sequence(s).")
            sequences = sequences[:num_sequences]

        if not sequences:
            logging.warning("No sequences found to process.")
            return

        logging.info("Creating HDF5 file and processing sequences...")
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with h5py.File(self.output_file, 'w') as f_out:
            for i, seq_info in enumerate(tqdm(sequences, desc="Processing Sequences")):
                traj_group = f_out.create_group(f"trajectory_{i:04d}")
                self.process_sequence(seq_info, traj_group)

        logging.info("Conversion process completed successfully.") 