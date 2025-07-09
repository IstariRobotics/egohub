"""
Base classes for dataset adapters.
"""

import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import yaml
from PIL import Image
from tqdm import tqdm

from egohub.constants import (
    CANONICAL_SKELETON_HIERARCHY,
    CANONICAL_SKELETON_JOINTS,
)


class BaseAdapter(ABC):
    """
    Abstract base class for all dataset adapters.

    Adapters are responsible for converting data from a raw, dataset-specific
    format into our canonical HDF5 format. This class provides a standard
    interface and a common run loop for all adapters.
    """

    name: str = ""

    def __init__(
        self, raw_dir: Path, output_path: Path, config: Optional[Dict[str, Any]] = None
    ):
        self.raw_dir = raw_dir
        self.output_path = output_path
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

        # Assuming the script is run from the project root
        config_path = Path(f"configs/{self.name}.yaml")
        if not config_path.exists():
            logging.warning(
                f"No config file found for adapter '{self.name}' at {config_path}."
            )
            return {}

        logging.info(f"Loading configuration from {config_path}...")
        with open(config_path, "r") as f:
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
    def process_sequence(self, seq_info: dict) -> dict:
        """
        Processes a single data sequence and returns it as a dictionary.

        This is the core method where dataset-specific parsing and transformation
        logic resides. The returned dictionary should follow the canonical
        schema structure.

        Args:
            seq_info (dict): A dictionary from the list returned by
                             discover_sequences().

        Returns:
            A dictionary containing the processed trajectory data.
        """
        pass

    def _confidence_scores_to_rgb(self, confidence_scores: np.ndarray) -> np.ndarray:
        """Map confidence scores to a color gradient (blue to green)."""
        colors = np.zeros((confidence_scores.shape[0], 3), dtype=np.uint8)
        # Simple blue-to-green gradient for low-to-high confidence
        colors[:, 1] = (confidence_scores * 255).astype(np.uint8)  # Green channel
        colors[:, 2] = 255 - (confidence_scores * 255).astype(np.uint8)  # Blue channel
        return colors

    def _write_hdf5(self, data: Dict[str, Any], h5_group: h5py.Group):
        """Recursively writes a dictionary to an HDF5 group."""
        for key, value in data.items():
            if isinstance(value, dict):
                next_group = h5_group.create_group(key)
                self._write_hdf5(value, next_group)
            elif isinstance(value, np.ndarray):
                h5_group.create_dataset(key, data=value)
            elif value is not None:
                h5_group.attrs[key] = value

    def _write_rrd(self, traj_data: Dict[str, Any], traj_name: str):
        """Logs a single trajectory dictionary to Rerun."""
        cam_keys = list(traj_data.get("cameras", {}).keys())

        blueprint = self._create_rrd_blueprint(cam_keys)
        rr.send_blueprint(blueprint)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self._rrd_log_static_metadata(traj_data)
        self._rrd_log_static_camera_data(traj_data, cam_keys)
        self._rrd_set_pose_annotation_context()

        master_timestamps_ns = traj_data.get("metadata", {}).get("timestamps_ns")
        if master_timestamps_ns is None:
            logging.error("No 'timestamps_ns' found in trajectory data. Cannot create RRD.")
            return

        num_frames = len(master_timestamps_ns)
        logging.info(f"Logging {num_frames} frames for {traj_name} to RRD...")

        for i in range(num_frames):
            timestamp_ns = master_timestamps_ns[i]
            rr.set_time_sequence("frame", i)
            rr.set_time_nanos("timestamp", timestamp_ns)
            self._rrd_log_temporal_camera_data(traj_data, cam_keys, i)
            self._rrd_log_temporal_skeleton_data(traj_data, i)
            # Add object data logging if/when it's part of the dictionary
            # self._rrd_log_temporal_object_data(traj_data, cam_keys, i)

    def _create_rrd_blueprint(self, camera_names: list[str]) -> rrb.Blueprint:
        """Creates a Rerun blueprint for visualization."""
        spatial2d_views = [
            rrb.Spatial2DView(origin=f"world/cameras/{name}/image", name=f"{name} View")
            for name in camera_names
        ]
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world"),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="llm_description"),
                    rrb.Tabs(*spatial2d_views),
                    row_shares=[1, 10],
                ),
                column_shares=[2, 1],
            ),
            collapse_panels=True,
        )

    def _rrd_log_static_metadata(self, traj_data: dict):
        """Logs static metadata to Rerun."""
        action_label = traj_data.get("metadata", {}).get("action_label")
        if action_label:
            rr.log(
                "llm_description",
                rr.TextDocument(text=str(action_label)),
                static=True,
            )

    def _rrd_log_static_camera_data(self, traj_data: dict, cam_keys: list[str]):
        """Logs static camera data like intrinsics to Rerun."""
        for cam_key in cam_keys:
            cam_dict = traj_data.get("cameras", {}).get(cam_key, {})
            intr = cam_dict.get("intrinsics")
            rgb_dict = cam_dict.get("rgb", {})
            frame_sizes = rgb_dict.get("frame_sizes")
            image_bytes = rgb_dict.get("image_bytes")

            if (
                intr is not None
                and frame_sizes is not None
                and image_bytes is not None
                and len(frame_sizes) > 0
            ):
                img_path = f"world/cameras/{cam_key}/image"
                num_bytes = frame_sizes[0]
                encoded_frame = image_bytes[0, :num_bytes]
                try:
                    image = Image.open(io.BytesIO(encoded_frame))
                    width, height = image.size
                    rr.log(
                        img_path,
                        rr.Pinhole(
                            image_from_camera=intr,
                            width=width,
                            height=height,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        ),
                        static=True,
                    )
                except Exception as e:
                    logging.warning(f"Could not decode image for static log: {e}")

    def _rrd_set_pose_annotation_context(self) -> None:
        """Sets up the annotation context for skeleton visualization."""
        keypoint_connections = []
        for child, parent in CANONICAL_SKELETON_HIERARCHY.items():
            if (
                child in CANONICAL_SKELETON_JOINTS
                and parent in CANONICAL_SKELETON_JOINTS
            ):
                child_id = CANONICAL_SKELETON_JOINTS.index(child)
                parent_id = CANONICAL_SKELETON_JOINTS.index(parent)
                keypoint_connections.append((parent_id, child_id))

        rr.log(
            "world/skeleton",
            rr.AnnotationContext(
                [
                    rr.ClassDescription(
                        info=rr.AnnotationInfo(id=0, label="Canonical Skeleton"),
                        keypoint_annotations=[
                            rr.AnnotationInfo(id=i, label=name)
                            for i, name in enumerate(CANONICAL_SKELETON_JOINTS)
                        ],
                        keypoint_connections=keypoint_connections,
                    )
                ]
            ),
            static=True,
        )

    def _rrd_log_temporal_camera_data(
        self, traj_data: dict, cam_keys: list[str], frame_idx: int
    ):
        """Logs temporal camera data (poses, images) for a frame."""
        for cam_key in cam_keys:
            cam_dict = traj_data.get("cameras", {}).get(cam_key, {})
            pose_in_world = cam_dict.get("pose_in_world")
            pose_indices = cam_dict.get("pose_indices")

            if pose_in_world is not None and pose_indices is not None:
                current_indices = np.where(pose_indices == frame_idx)[0]
                for idx in current_indices:
                    pose = pose_in_world[idx]
                    rr.log(
                        f"world/cameras/{cam_key}",
                        rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
                    )

            rgb_dict = cam_dict.get("rgb", {})
            image_bytes = rgb_dict.get("image_bytes")
            frame_sizes = rgb_dict.get("frame_sizes")
            frame_indices = rgb_dict.get("frame_indices")

            if (
                image_bytes is not None
                and frame_sizes is not None
                and frame_indices is not None
            ):
                current_indices = np.where(frame_indices == frame_idx)[0]
                for idx in current_indices:
                    num_bytes = frame_sizes[idx]
                    encoded_frame = image_bytes[idx, :num_bytes]
                    try:
                        image = Image.open(io.BytesIO(encoded_frame))
                        image_np = np.array(image)
                        rr.log(f"world/cameras/{cam_key}/image", rr.Image(image_np))
                    except Exception as e:
                        logging.warning(f"Could not decode image for frame {idx}: {e}")

    def _rrd_log_temporal_skeleton_data(self, traj_data: dict, frame_idx: int):
        """Logs temporal skeleton data for a frame."""
        skeleton_dict = traj_data.get("skeleton", {})
        positions = skeleton_dict.get("positions")
        confidences = skeleton_dict.get("confidences")
        position_indices = skeleton_dict.get("position_indices")

        if positions is not None and position_indices is not None:
            current_indices = np.where(position_indices == frame_idx)[0]
            for idx in current_indices:
                points = positions[idx]
                point_colors = (
                    self._confidence_scores_to_rgb(confidences[idx])
                    if confidences is not None
                    else None
                )
                rr.log(
                    "world/skeleton",
                    rr.Points3D(points, colors=point_colors, class_ids=0),
                )

    def run(self, num_sequences: int | None = None, output_format: str = "hdf5"):
        """
        Main loop to discover, process, and save all sequences.

        This method orchestrates the conversion process:
        1. Calls discover_sequences() to find all data.
        2. Creates/opens the output file (HDF5 or RRD).
        3. Iterates through sequences, calling process_sequence() for each one.
        4. Writes the data to the file in the specified format.

        Args:
            num_sequences (int | None): If specified, only process the first
                                        `num_sequences` sequences.
            output_format (str): The desired output format ('hdf5' or 'rrd').
        """
        logging.info(f"Input directory: {self.raw_dir}")
        logging.info(f"Output path: {self.output_path}")
        logging.info(f"Output format: {output_format}")

        sequences = self.discover_sequences()
        if num_sequences is not None:
            logging.info(
                f"Limiting processing to the first {num_sequences} sequence(s)."
            )
            sequences = sequences[:num_sequences]

        if not sequences:
            logging.warning("No sequences found to process.")
            return

        self.output_path.parent.mkdir(parents=True, exist_ok=True)

        if output_format == "hdf5":
            with h5py.File(self.output_path, "w") as f_out:
                for i, seq_info in enumerate(
                    tqdm(sequences, desc="Processing Sequences")
                ):
                    traj_data = self.process_sequence(seq_info)
                    traj_group = f_out.create_group(f"trajectory_{i:04d}")
                    self._write_hdf5(traj_data, traj_group)
        elif output_format == "rrd":
            rr.init("egohub_converter", spawn=False)
            rr.save(str(self.output_path))
            for i, seq_info in enumerate(
                tqdm(sequences, desc="Processing Sequences")
            ):
                traj_data = self.process_sequence(seq_info)
                if traj_data:
                    self._write_rrd(traj_data, f"trajectory_{i:04d}")
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

        logging.info("Conversion process completed successfully.")
