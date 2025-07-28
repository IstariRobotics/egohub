import logging
import uuid
from pathlib import Path
from typing import Any, Dict

import h5py
import numpy as np
from PIL import Image

from egohub.adapters.base import BaseAdapter
from egohub.adapters.dataset_info import DatasetInfo
from egohub.processing.synchronization import generate_indices


class HO3DAdapter(BaseAdapter):
    """Adapter for the HO3D dataset."""

    name = "ho3d"

    @property
    def source_joint_names(self) -> list[str]:
        return []

    @property
    def source_skeleton_hierarchy(self) -> dict[str, str]:
        return {}

    def get_camera_intrinsics(self) -> Dict[str, Any]:
        # Intrinsics are sequence-specific for HO3D, so this is a placeholder.
        # The actual intrinsics are loaded in `_process_camera_data`.
        return {"matrix": np.identity(3, dtype=np.float32)}

    def get_dataset_info(self) -> DatasetInfo:
        # This will be implemented later
        return DatasetInfo(camera_intrinsics=self.get_camera_intrinsics()["matrix"])

    def discover_sequences(self) -> list[dict]:
        """
        Discovers all sequences in the raw data directory.
        A sequence is a folder containing 'rgb', 'depth' and 'meta' folders.
        """
        sequences = []
        logging.info(
            "Searching for sequences in '%s' (absolute path: '%s')...",
            self.raw_dir,
            self.raw_dir.resolve(),
        )

        for split in ["train", "evaluation"]:
            split_dir = self.raw_dir / split
            logging.info(f"Checking for split directory: {split_dir}")
            if not split_dir.is_dir():
                logging.warning(f"Split directory not found: {split_dir}")
                continue

            for seq_dir in split_dir.iterdir():
                logging.info(f"Checking for sequence directory: {seq_dir}")
                if seq_dir.is_dir():
                    if (
                        (seq_dir / "rgb").is_dir()
                        and (seq_dir / "depth").is_dir()
                        and (seq_dir / "meta").is_dir()
                    ):
                        logging.info(f"Found sequence: {seq_dir.name}")
                        sequences.append(
                            {
                                "sequence_path": seq_dir,
                                "sequence_name": f"{split}/{seq_dir.name}",
                                "split": split,
                            }
                        )

        logging.info(f"Found {len(sequences)} sequences.")
        return sequences

    def _process_metadata(
        self, seq_info: dict, traj_group: h5py.Group, num_frames: int
    ) -> tuple[set, np.ndarray]:
        """Process metadata and timestamps."""
        found_streams = set()

        # --- Metadata ---
        metadata_group = traj_group.create_group("metadata")
        metadata_group.attrs["uuid"] = str(uuid.uuid4())
        metadata_group.attrs["source_dataset"] = "HO3D"
        metadata_group.attrs["source_identifier"] = seq_info["sequence_name"]
        metadata_group.attrs["action_label"] = "N/A"  # No action labels in HO3D
        found_streams.add("metadata/action_label")

        # --- Timestamps ---
        master_timestamps_ns = np.arange(num_frames) * (1e9 / 30.0)  # Assume 30 FPS
        metadata_group.create_dataset(
            "timestamps_ns", data=master_timestamps_ns.astype(np.uint64)
        )
        found_streams.add("metadata/timestamps_ns")

        return found_streams, master_timestamps_ns

    def _decode_depth_img(self, depth_img_path: Path) -> np.ndarray:
        """Decodes a 16-bit depth image stored in a PNG file."""
        depth_img = Image.open(depth_img_path)
        depth = np.array(depth_img, dtype=np.uint16)
        # The LSB is in the red channel, and the MSB is in the green channel.
        # This is a bit unusual, normally it's the other way around.
        # Let's assume the README means we should interpret the RG values
        # as a 16-bit integer.
        # Some datasets store depth as (G * 256 + R).
        depth_scaled = (
            depth[..., 1].astype(np.uint16) * 256 + depth[..., 0].astype(np.uint16)
        ).astype(np.float32)
        depth_scaled /= 1000.0  # Convert to meters, assuming scale factor of 1000
        return depth_scaled

    def _process_rgb_data(
        self,
        seq_info: dict,
        camera_group: h5py.Group,
        master_timestamps_ns: np.ndarray,
    ) -> set:
        """Process RGB video data."""
        found_streams = set()
        rgb_group = camera_group.create_group("rgb")
        rgb_dir = seq_info["sequence_path"] / "rgb"
        image_files = sorted(list(rgb_dir.glob("*.jpg")))

        if image_files:
            temp_frames = []
            for img_path in image_files:
                with open(img_path, "rb") as f:
                    temp_frames.append(f.read())

            max_frame_size = max(len(f) for f in temp_frames)
            image_dataset = rgb_group.create_dataset(
                "image_bytes",
                (len(temp_frames), max_frame_size),
                dtype=np.uint8,
            )
            for i, frame_bytes in enumerate(temp_frames):
                padded_frame = frame_bytes.ljust(max_frame_size, b"\x00")
                image_dataset[i] = np.frombuffer(padded_frame, dtype=np.uint8)
            rgb_group.create_dataset(
                "frame_sizes",
                data=[len(f) for f in temp_frames],
                dtype=np.int32,
            )
            found_streams.add(
                f"cameras/{camera_group.name.split('/')[-1]}/rgb/image_bytes"
            )

            stream_timestamps_ns = np.arange(len(temp_frames)) * (1e9 / 30.0)
            frame_indices = generate_indices(master_timestamps_ns, stream_timestamps_ns)
            rgb_group.create_dataset("frame_indices", data=frame_indices)
        return found_streams

    def _process_depth_data(
        self,
        seq_info: dict,
        camera_group: h5py.Group,
        master_timestamps_ns: np.ndarray,
    ) -> set:
        """Process depth map data."""
        found_streams = set()
        depth_group = camera_group.create_group("depth")
        depth_dir = seq_info["sequence_path"] / "depth"
        depth_files = sorted(list(depth_dir.glob("*.png")))

        if depth_files:
            depth_maps = [self._decode_depth_img(p) for p in depth_files]
            depth_dataset = depth_group.create_dataset(
                "image_meters",
                (
                    len(depth_maps),
                    depth_maps[0].shape[0],
                    depth_maps[0].shape[1],
                ),
                dtype=np.float32,
            )
            for i, depth_map in enumerate(depth_maps):
                depth_dataset[i] = depth_map
            found_streams.add(
                f"cameras/{camera_group.name.split('/')[-1]}/depth/image_meters"
            )

            stream_timestamps_ns = np.arange(len(depth_maps)) * (1e9 / 30.0)
            frame_indices = generate_indices(master_timestamps_ns, stream_timestamps_ns)
            depth_group.create_dataset("frame_indices", data=frame_indices)
        return found_streams

    def _process_hand_data(self, traj_group: h5py.Group) -> set:
        """Process hand tracking data."""
        found_streams = set()
        hands_group = traj_group.create_group("hands")
        hands_group.create_group("left")
        hands_group.create_group("right")
        found_streams.add("hands/left")
        found_streams.add("hands/right")
        return found_streams

    def _process_skeleton_data(self, traj_group: h5py.Group) -> set:
        """Process skeleton tracking data."""
        found_streams = set()
        traj_group.create_group("skeleton")
        found_streams.add("skeleton")
        return found_streams

    def _process_camera_data(
        self,
        seq_info: dict,
        traj_group: h5py.Group,
        master_timestamps_ns: np.ndarray,
    ) -> set:
        """Process camera data including intrinsics, poses, and RGB."""
        found_streams = set()
        cameras_group = traj_group.create_group("cameras")
        # HO-3D has a single camera per sequence
        camera_group = cameras_group.create_group("default_camera")
        camera_group.attrs["is_ego"] = True  # Assume egocentric

        # --- Static Rotation (180 degrees around Z-axis) ---
        # This is needed because the camera is mounted upside down
        # Quaternion for 180-degree rotation around Z-axis: (0, 0, 1, 0) -> (x, y, z, w)
        rotation_quaternion = np.array([0.0, 0.0, 1.0, 0.0])

        # Create a static transform for the camera pose
        num_frames_len = len(master_timestamps_ns)
        poses = np.zeros((num_frames_len, 7))  # x, y, z, qx, qy, qz, qw
        for i in range(num_frames_len):
            poses[i, 3:7] = rotation_quaternion

        camera_group.create_dataset("poses", data=poses)
        found_streams.add(f"cameras/{camera_group.name.split('/')[-1]}/poses")

        # --- Intrinsics (from metadata) ---
        meta_dir = seq_info["sequence_path"] / "meta"
        meta_files = sorted(list(meta_dir.glob("*.pkl")))
        if meta_files:
            try:
                import pickle

                with open(meta_files[0], "rb") as f:
                    meta_data = pickle.load(f)
                if "camMat" in meta_data and meta_data["camMat"] is not None:
                    intrinsics = meta_data["camMat"]
                    camera_group.create_dataset("intrinsics", data=intrinsics)
                    found_streams.add(
                        f"cameras/{camera_group.name.split('/')[-1]}/intrinsics"
                    )
            except (ImportError, pickle.UnpicklingError, KeyError) as e:
                logging.warning(f"Could not load camera intrinsics: {e}")

        # Process RGB data
        rgb_streams = self._process_rgb_data(
            seq_info, camera_group, master_timestamps_ns
        )
        found_streams.update(rgb_streams)

        # Process Depth data
        depth_streams = self._process_depth_data(
            seq_info, camera_group, master_timestamps_ns
        )
        found_streams.update(depth_streams)

        return found_streams

    def process_sequence(self, seq_info: dict, traj_group: h5py.Group):
        """
        Processes a single sequence and writes its data to the given HDF5 group.
        """
        logging.info(f"--- Processing sequence: {seq_info['sequence_name']} ---")
        found_streams = set()

        # For HO3D, the number of frames is determined by the number of RGB images
        rgb_dir = seq_info["sequence_path"] / "rgb"
        num_frames = len(list(rgb_dir.glob("*.jpg")))

        # Process metadata and get master timestamps
        metadata_streams, master_timestamps_ns = self._process_metadata(
            seq_info, traj_group, num_frames
        )
        found_streams.update(metadata_streams)

        # Process camera data
        camera_streams = self._process_camera_data(
            seq_info, traj_group, master_timestamps_ns
        )
        found_streams.update(camera_streams)

        # Process hand data
        hand_streams = self._process_hand_data(traj_group)
        found_streams.update(hand_streams)

        # Process skeleton data
        skeleton_streams = self._process_skeleton_data(traj_group)
        found_streams.update(skeleton_streams)

        logging.info(
            f"Finished processing sequence. Found streams: "
            f"{sorted(list(found_streams))}"
        )
