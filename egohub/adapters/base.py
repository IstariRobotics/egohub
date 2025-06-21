"""
Base classes for dataset adapters.
"""
from abc import ABC, abstractmethod
from pathlib import Path
import logging

import h5py
from tqdm import tqdm


class BaseAdapter(ABC):
    """
    Abstract base class for all dataset adapters.

    Each adapter is responsible for converting a raw dataset format
    into the canonical HDF5 schema defined in `egohub.schema`.

    The core workflow is managed by the `run` method, which orchestrates
    the discovery and processing of sequences. Subclasses must implement
    `discover_sequences` and `process_sequence`.
    """

    def __init__(self, raw_dir: Path, output_file: Path):
        self.raw_dir = raw_dir
        self.output_file = output_file

    @abstractmethod
    def discover_sequences(self) -> list[dict]:
        """
        Scans the raw data directory and returns a list of discovered
        sequences to process. Each item in the list should be a dictionary
        containing the necessary information to process that sequence.
        """
        pass

    @abstractmethod
    def process_sequence(self, sequence_info: dict, traj_group: h5py.Group):
        """
        Processes a single sequence and writes its data to the
        provided HDF5 trajectory group.
        """
        pass

    def run(self, num_sequences: int | None = None):
        """
        The main entry point for running the adapter. This method handles
        the boilerplate logic of file creation, sequence discovery, looping,
        and progress tracking.
        """
        logging.info(f"Starting adapter: {self.__class__.__name__}")

        if not self.raw_dir.is_dir():
            logging.error(f"Error: Raw data directory not found at '{self.raw_dir}'")
            return

        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        sequences = self.discover_sequences()
        if not sequences:
            logging.warning("No sequences found. Exiting.")
            return
            
        if num_sequences is not None and num_sequences > 0:
            sequences = sequences[:num_sequences]
            logging.info(f"Limiting processing to the first {len(sequences)} sequence(s).")

        logging.info(f"Input directory: {self.raw_dir}")
        logging.info(f"Output file: {self.output_file}")
        
        with h5py.File(self.output_file, 'w') as f_out:
            logging.info("Creating HDF5 file and processing sequences...")
            
            for i, seq_info in enumerate(tqdm(sequences, desc="Processing Sequences")):
                traj_group_name = f"trajectory_{i:04d}"
                traj_group = f_out.create_group(traj_group_name)
                
                try:
                    self.process_sequence(seq_info, traj_group)
                except Exception as e:
                    logging.error(f"Failed to process sequence {seq_info.get('hdf5_path', 'N/A')}: {e}")
                    # Clean up the partially created group on failure
                    del f_out[traj_group_name]

        logging.info("Conversion process completed successfully.") 