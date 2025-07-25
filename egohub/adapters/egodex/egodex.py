import logging
import uuid
from typing import Any, Dict

import cv2
import h5py
import numpy as np

from egohub.adapters.base import BaseAdapter
from egohub.adapters.dataset_info import DatasetInfo
from egohub.constants import EGODEX_SKELETON_HIERARCHY, EGODEX_SKELETON_JOINTS
from egohub.processing.skeleton import remap_skeleton
from egohub.processing.synchronization import generate_indices

# Defines the mapping from the EgoDex source skeleton to the canonical skeleton.
# This is necessary because the joint names and structure differ.
EGODEX_TO_CANONICAL_SKELETON_MAP = {
    # Body
    "hip": "pelvis",
    "leftShoulder": "left_shoulder",
    "leftArm": "left_elbow",
    "leftForearm": "left_wrist",  # EgoDex `leftForearm` is closer to the wrist
    "leftHand": "left_wrist",
    "rightShoulder": "right_shoulder",
    "rightArm": "right_elbow",
    "rightForearm": "right_wrist",  # EgoDex `rightForearm` is closer to the wrist
    "rightHand": "right_wrist",
    # Left Hand Fingers
    "leftIndexFingerMetacarpal": "left_index_finger_mcp",
    "leftIndexFingerKnuckle": "left_index_finger_pip",
    "leftIndexFingerIntermediateBase": "left_index_finger_dip",
    "leftIndexFingerIntermediateTip": "left_index_finger_dip",
    "leftIndexFingerTip": "left_index_finger_tip",
    "leftLittleFingerMetacarpal": "left_pinky_mcp",
    "leftLittleFingerKnuckle": "left_pinky_pip",
    "leftLittleFingerIntermediateBase": "left_pinky_dip",
    "leftLittleFingerIntermediateTip": "left_pinky_dip",
    "leftLittleFingerTip": "left_pinky_tip",
    "leftMiddleFingerMetacarpal": "left_middle_finger_mcp",
    "leftMiddleFingerKnuckle": "left_middle_finger_pip",
    "leftMiddleFingerIntermediateBase": "left_middle_finger_dip",
    "leftMiddleFingerIntermediateTip": "left_middle_finger_dip",
    "leftMiddleFingerTip": "left_middle_finger_tip",
    "leftRingFingerMetacarpal": "left_ring_finger_mcp",
    "leftRingFingerKnuckle": "left_ring_finger_pip",
    "leftRingFingerIntermediateBase": "left_ring_finger_dip",
    "leftRingFingerIntermediateTip": "left_ring_finger_dip",
    "leftRingFingerTip": "left_ring_finger_tip",
    "leftThumbKnuckle": "left_thumb_cmc",
    "leftThumbIntermediateBase": "left_thumb_mcp",
    "leftThumbIntermediateTip": "left_thumb_ip",
    "leftThumbTip": "left_thumb_tip",
    # Right Hand Fingers
    "rightIndexFingerMetacarpal": "right_index_finger_mcp",
    "rightIndexFingerKnuckle": "right_index_finger_pip",
    "rightIndexFingerIntermediateBase": "right_index_finger_dip",
    "rightIndexFingerIntermediateTip": "right_index_finger_dip",
    "rightIndexFingerTip": "right_index_finger_tip",
    "rightLittleFingerMetacarpal": "right_pinky_mcp",
    "rightLittleFingerKnuckle": "right_pinky_pip",
    "rightLittleFingerIntermediateBase": "right_pinky_dip",
    "rightLittleFingerIntermediateTip": "right_pinky_dip",
    "rightLittleFingerTip": "right_pinky_tip",
    "rightMiddleFingerMetacarpal": "right_middle_finger_mcp",
    "rightMiddleFingerKnuckle": "right_middle_finger_pip",
    "rightMiddleFingerIntermediateBase": "right_middle_finger_dip",
    "rightMiddleFingerIntermediateTip": "right_middle_finger_dip",
    "rightMiddleFingerTip": "right_middle_finger_tip",
    "rightRingFingerMetacarpal": "right_ring_finger_mcp",
    "rightRingFingerKnuckle": "right_ring_finger_pip",
    "rightRingFingerIntermediateBase": "right_ring_finger_dip",
    "rightRingFingerIntermediateTip": "right_ring_finger_dip",
    "rightRingFingerTip": "right_ring_finger_tip",
    "rightThumbKnuckle": "right_thumb_cmc",
    "rightThumbIntermediateBase": "right_thumb_mcp",
    "rightThumbIntermediateTip": "right_thumb_ip",
    "rightThumbTip": "right_thumb_tip",
}


class EgoDexAdapter(BaseAdapter):
    """Adapter for the EgoDex dataset."""

    name = "egodex"

    @property
    def source_joint_names(self) -> list[str]:
        return EGODEX_SKELETON_JOINTS

    @property
    def source_skeleton_hierarchy(self) -> dict[str, str]:
        return EGODEX_SKELETON_HIERARCHY

    def get_camera_intrinsics(self) -> Dict[str, Any]:
        if self.config and "default_intrinsics" in self.config:
            return {
                "matrix": np.array(self.config["default_intrinsics"], dtype=np.float32)
            }
        else:
            return {
                "matrix": np.array(
                    [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
                    dtype=np.float32,
                )
            }

    def get_dataset_info(self) -> DatasetInfo:  # noqa: D401
        intr_dict = self.get_camera_intrinsics()
        intr: np.ndarray
        if isinstance(intr_dict, dict):
            intr = intr_dict["matrix"]  # type: ignore[assignment]
        else:
            intr = intr_dict  # type: ignore[assignment]

        from egohub.constants import (
            CANONICAL_SKELETON_HIERARCHY,
            CANONICAL_SKELETON_JOINTS,
        )

        return DatasetInfo(
            camera_intrinsics=intr,
            view_coordinates="RDF",
            frame_rate=30.0,
            joint_names=list(CANONICAL_SKELETON_JOINTS),
            joint_hierarchy=CANONICAL_SKELETON_HIERARCHY,
            joint_remap=EGODEX_TO_CANONICAL_SKELETON_MAP,
            depth_scale=1.0,
        )

    def discover_sequences(self) -> list[dict]:
        """
        Discovers all paired HDF5 and MP4 sequences in the raw data directory.
        """
        sequences = []
        logging.info(
            "Searching for sequences in '%s' (absolute path: '%s')...",
            self.raw_dir,
            self.raw_dir.resolve(),
        )
        hdf5_files = sorted(list(self.raw_dir.glob("**/*.hdf5")))

        for hdf5_path in hdf5_files:
            mp4_path = hdf5_path.with_suffix(".mp4")
            if mp4_path.exists():
                sequences.append(
                    {
                        "hdf5_path": hdf5_path,
                        "mp4_path": mp4_path,
                        "task_name": hdf5_path.parent.name,
                        "sequence_name": hdf5_path.stem,
                    }
                )

        logging.info(f"Found {len(sequences)} sequences.")
        return sequences

    def _load_config(self) -> dict:
        return {}

    def _process_metadata(
        self, seq_info: dict, f_in: h5py.File, traj_group: h5py.Group
    ) -> tuple[set, np.ndarray]:
        """Process metadata and timestamps."""
        found_streams = set()

        # --- Metadata ---
        metadata_group = traj_group.create_group("metadata")
        metadata_group.attrs["uuid"] = str(uuid.uuid4())
        metadata_group.attrs["source_dataset"] = "EgoDex"
        metadata_group.attrs["source_identifier"] = seq_info["sequence_name"]

        action_label_raw = f_in.attrs.get("llm_description", "N/A")
        if isinstance(action_label_raw, bytes):
            action_label = action_label_raw.decode("utf-8", "replace").replace(
                "\x00", ""
            )
        elif isinstance(action_label_raw, str):
            action_label = action_label_raw.replace("\x00", "")
        else:
            action_label = "N/A"

        metadata_group.attrs["action_label"] = action_label
        found_streams.add("metadata/action_label")

        # --- Timestamps ---
        # EgoDex does not provide explicit timestamps, so we generate them.
        # We assume a constant frame rate of 30 FPS for all streams.
        num_frames_source = 0
        camera_transforms = f_in.get("transforms/camera")
        if isinstance(camera_transforms, h5py.Dataset):
            num_frames_source = camera_transforms.shape[0]

        # Master timestamps are the definitive timeline for the trajectory.
        master_timestamps_ns = np.arange(num_frames_source) * (1e9 / 30.0)
        metadata_group.create_dataset(
            "timestamps_ns", data=master_timestamps_ns.astype(np.uint64)
        )
        found_streams.add("metadata/timestamps_ns")

        return found_streams, master_timestamps_ns

    def _process_camera_data(
        self,
        seq_info: dict,
        f_in: h5py.File,
        traj_group: h5py.Group,
        master_timestamps_ns: np.ndarray,
    ) -> set:
        """Process camera data including intrinsics, poses, and RGB."""
        found_streams = set()

        # --- Camera Data ---
        cameras_group = traj_group.create_group("cameras")
        ego_camera_group = cameras_group.create_group("ego_camera")
        ego_camera_group.attrs["is_ego"] = True

        # Process intrinsics
        intrinsics_data = f_in.get("camera/intrinsic")
        if isinstance(intrinsics_data, h5py.Dataset) and intrinsics_data.shape[0] > 0:
            intrinsics = intrinsics_data[:]
        else:
            logging.warning(
                f"No valid intrinsics found in {f_in.filename}. Using default values."
            )
            intrinsics = np.array(
                [[736.6339, 0.0, 960.0], [0.0, 736.6339, 540.0], [0.0, 0.0, 1.0]],
                dtype=np.float32,
            )
        ego_camera_group.create_dataset("intrinsics", data=intrinsics)
        found_streams.add("cameras/ego_camera/intrinsics")

        # Process camera poses
        camera_transforms_data = f_in.get("transforms/camera")
        if isinstance(camera_transforms_data, h5py.Dataset):
            raw_camera_poses = camera_transforms_data[:]

            ego_camera_group.create_dataset("pose_in_world", data=raw_camera_poses)
            found_streams.add("cameras/ego_camera/pose_in_world")

            # Generate and save pose indices
            stream_timestamps_ns = np.arange(raw_camera_poses.shape[0]) * (1e9 / 30.0)
            pose_indices = generate_indices(master_timestamps_ns, stream_timestamps_ns)
            ego_camera_group.create_dataset("pose_indices", data=pose_indices)

        # Process RGB data
        rgb_streams = self._process_rgb_data(
            seq_info, ego_camera_group, master_timestamps_ns
        )
        found_streams.update(rgb_streams)

        return found_streams

    def _process_rgb_data(
        self,
        seq_info: dict,
        ego_camera_group: h5py.Group,
        master_timestamps_ns: np.ndarray,
    ) -> set:
        """Process RGB video data."""
        found_streams = set()

        rgb_group = ego_camera_group.create_group("rgb")
        cap = cv2.VideoCapture(str(seq_info["mp4_path"]))
        if cap.isOpened():
            temp_frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, encoded_image = cv2.imencode(
                    ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 95]
                )
                temp_frames.append(encoded_image.tobytes())
            cap.release()

            if temp_frames:
                max_frame_size = max(len(f) for f in temp_frames)
                image_dataset = rgb_group.create_dataset(
                    "image_bytes",
                    (len(temp_frames), max_frame_size),
                    dtype=np.uint8,
                )
                for i, frame_bytes in enumerate(temp_frames):
                    padded_frame = frame_bytes + b"\x00" * (
                        max_frame_size - len(frame_bytes)
                    )
                    image_dataset[i] = np.frombuffer(padded_frame, dtype=np.uint8)
                rgb_group.create_dataset(
                    "frame_sizes",
                    data=[len(f) for f in temp_frames],
                    dtype=np.int32,
                )
                found_streams.add("cameras/ego_camera/rgb/image_bytes")

                # Generate and save frame indices
                stream_timestamps_ns = np.arange(len(temp_frames)) * (1e9 / 30.0)
                frame_indices = generate_indices(
                    master_timestamps_ns, stream_timestamps_ns
                )
                rgb_group.create_dataset("frame_indices", data=frame_indices)

        return found_streams

    def _process_hand_data(
        self, f_in: h5py.File, traj_group: h5py.Group, master_timestamps_ns: np.ndarray
    ) -> set:
        """Process hand tracking data."""
        found_streams = set()

        # --- Hand Data ---
        hands_group = traj_group.create_group("hands")
        for hand in ["left", "right"]:
            hand_group = hands_group.create_group(hand)
            # AVP uses lowercase first letter (e.g. 'leftHand', 'rightHand')
            source_key = f"transforms/{hand}Hand"
            hand_transforms = f_in.get(source_key)
            if isinstance(hand_transforms, h5py.Dataset):
                raw_hand_poses = hand_transforms[:]

                hand_group.create_dataset("pose_in_world", data=raw_hand_poses)
                found_streams.add(f"hands/{hand}/pose_in_world")

                # Generate and save pose indices
                stream_timestamps_ns = np.arange(raw_hand_poses.shape[0]) * (1e9 / 30.0)
                pose_indices = generate_indices(
                    master_timestamps_ns, stream_timestamps_ns
                )
                hand_group.create_dataset("pose_indices", data=pose_indices)
            else:
                logging.warning(f"No '{source_key}' data found in {f_in.filename}")

        return found_streams

    def _process_skeleton_data(
        self, f_in: h5py.File, traj_group: h5py.Group, master_timestamps_ns: np.ndarray
    ) -> set:
        """Process skeleton tracking data."""
        found_streams = set()

        # --- Skeleton Data ---
        skeleton_group = traj_group.create_group("skeleton")
        transforms_group = f_in.get("transforms")
        confidences_group = f_in.get("confidences")

        if isinstance(transforms_group, h5py.Group) and isinstance(
            confidences_group, h5py.Group
        ):
            # Use the pre-defined list of joints for EgoDex
            joint_names = self.source_joint_names

            if joint_names:
                positions_list, confidences_list = [], []
                for joint_name in joint_names:
                    joint_transform = transforms_group.get(joint_name)
                    if isinstance(joint_transform, h5py.Dataset):
                        # Extract the translation part of the 4x4 matrix
                        positions_list.append(joint_transform[:, :3, 3])
                    else:
                        # If a joint is missing, fill with zeros for now
                        positions_list.append(
                            np.zeros(
                                (master_timestamps_ns.shape[0], 3), dtype=np.float32
                            )
                        )

                    joint_conf = confidences_group.get(joint_name)
                    if isinstance(joint_conf, h5py.Dataset):
                        confidences_list.append(joint_conf[:])
                    else:
                        confidences_list.append(
                            np.zeros(master_timestamps_ns.shape[0], dtype=np.float32)
                        )

                if positions_list and confidences_list:
                    source_positions = np.stack(positions_list, axis=1)
                    all_confidences = np.stack(confidences_list, axis=1)

                    # Remap the skeleton to the canonical joint definition
                    info = self.get_dataset_info()
                    canonical_positions, canonical_confidences = remap_skeleton(
                        source_positions=source_positions,
                        source_confidences=all_confidences,
                        source_joint_names=self.source_joint_names,
                        joint_map=info.joint_remap,
                    )

                    skeleton_group.create_dataset(
                        "positions", data=canonical_positions.astype(np.float32)
                    )
                    found_streams.add("skeleton/positions")
                    skeleton_group.create_dataset(
                        "confidences", data=canonical_confidences.astype(np.float32)
                    )
                    found_streams.add("skeleton/confidences")

                    # Generate and save frame indices
                    stream_timestamps_ns = np.arange(source_positions.shape[0]) * (
                        1e9 / 30.0
                    )
                    frame_indices = generate_indices(
                        master_timestamps_ns, stream_timestamps_ns
                    )
                    skeleton_group.create_dataset("frame_indices", data=frame_indices)

        return found_streams

    def process_sequence(self, seq_info: dict, traj_group: h5py.Group):
        """
        Processes a single sequence and writes its data to the given HDF5 group.

        Args:
            seq_info (dict): A dictionary containing paths and names for the sequence.
            traj_group (h5py.Group): The HDF5 group to write the processed data into.
        """
        logging.info(
            f"--- Processing sequence: {seq_info['task_name']}/"
            f"{seq_info['sequence_name']} ---"
        )
        found_streams = set()

        with h5py.File(seq_info["hdf5_path"], "r") as f_in:
            # Process metadata and get master timestamps
            metadata_streams, master_timestamps_ns = self._process_metadata(
                seq_info, f_in, traj_group
            )
            found_streams.update(metadata_streams)

            # Process camera data
            camera_streams = self._process_camera_data(
                seq_info, f_in, traj_group, master_timestamps_ns
            )
            found_streams.update(camera_streams)

            # Process hand data
            hand_streams = self._process_hand_data(
                f_in, traj_group, master_timestamps_ns
            )
            found_streams.update(hand_streams)

            # Process skeleton data
            skeleton_streams = self._process_skeleton_data(
                f_in, traj_group, master_timestamps_ns
            )
            found_streams.update(skeleton_streams)

        logging.info(
            f"Finished processing sequence. Found streams: "
            f"{sorted(list(found_streams))}"
        )
