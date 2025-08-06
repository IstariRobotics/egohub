import logging
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np

from egohub.adapters.base import BaseAdapter
from egohub.adapters.dataset_info import DatasetInfo


class HOI4DAdapter(BaseAdapter):
    name = "hoi4d"

    def _get_handpose(self, file_path: Path) -> Dict[str, Any]:
        with open(file_path, "rb") as file:
            data = pickle.load(file)
        return {
            "poseCoeff": data["poseCoeff"],
            "beta": data["beta"],
            "trans": data["trans"],
            "kps2D": data["kps2D"],
        }

    def _get_handpose_none(self) -> Dict[str, Any]:
        return {
            "poseCoeff": np.full(48, np.nan, dtype=np.float32),
            "beta": np.full(10, np.nan, dtype=np.float32),
            "trans": np.full(3, np.nan, dtype=np.float32),
            "kps2D": np.full((21, 2), np.nan, dtype=np.float32),
        }

    @property
    def source_joint_names(self) -> List[str]:
        return self.config["mano"]["joint_names"]

    @property
    def source_skeleton_hierarchy(self) -> Dict[str, str]:
        return self.config["mano"]["skeleton_hierarchy"]

    def get_camera_intrinsics(self, camera_file: Path) -> np.ndarray:
        """Load camera intrinsics from .npy file or .txt trajectory file."""
        if camera_file.suffix == ".npy":
            return np.load(camera_file)
        elif camera_file.suffix == ".txt":
            import open3d as o3d

            trajectory = o3d.io.read_pinhole_camera_trajectory(str(camera_file))
            if not trajectory.parameters:
                raise ValueError(f"No camera parameters found in {camera_file}")
            return trajectory.parameters[0].intrinsic.intrinsic_matrix
        else:
            raise ValueError(f"Unsupported camera file format: {camera_file.suffix}")

    def get_dataset_info(self) -> DatasetInfo:
        """
        Loads dataset-wide metadata from the adapter's configuration file.
        """
        return DatasetInfo(
            name=self.name,
            source_fps=self.config["metadata"]["source_fps"],
            target_fps=self.config["metadata"]["target_fps"],
            joint_name_to_source_id={},
            joint_name_to_remap_source_id={},
        )

    def discover_sequences(self) -> List[Dict[str, Any]]:
        """
        Discovers all processable data sequences in the raw_dir.

        The HOI4D dataset structure is:
        HOI4D_release/{subject}/{hand}/{camera}/{noun}/{sequence}/{action}/{task}/align_rgb/image.mp4
        """
        sequences = []
        base_dir = self.raw_dir / "HOI4D_release"
        logging.info(f"Searching for sequences in: {base_dir}")

        if not base_dir.is_dir():
            logging.warning(f"Base directory not found: {base_dir}")
            return []

        # Walk through the complex directory structure
        for root, dirs, _ in os.walk(base_dir):
            if "align_rgb" in dirs:
                align_rgb_dir = Path(root) / "align_rgb"
                video_file = align_rgb_dir / "image.mp4"

                if not video_file.exists():
                    continue

                # Parse path: subject/hand/camera/noun/sequence/action/task
                try:
                    relative_path = Path(root).relative_to(base_dir)
                    path_parts = relative_path.parts

                    # Need at least subject/hand/camera/noun/sequence/action/task
                    if len(path_parts) < 7:
                        logging.debug(
                            f"Skipping {relative_path}: insufficient path depth"
                        )
                        continue

                    subject = path_parts[0]
                    hand = path_parts[1]
                    camera = path_parts[2]
                    noun = path_parts[3]
                    sequence = path_parts[4]
                    action = path_parts[5]
                    task = path_parts[6]

                    # Look for camera intrinsics file
                    camera_npy_file = (
                        self.raw_dir / "camera_params" / subject / "intrin.npy"
                    )
                    camera_txt_file = (
                        self.raw_dir / "camera_params" / f"{relative_path}.txt"
                    )

                    camera_file = None
                    if camera_npy_file.exists():
                        camera_file = camera_npy_file
                    elif camera_txt_file.exists():
                        camera_file = camera_txt_file
                    else:
                        logging.debug(f"No camera file found for {relative_path}")
                        continue

                    # Create sequence metadata
                    sequence_info = {
                        "video_file": video_file,
                        "camera_file": camera_file,
                        "relative_path": relative_path,
                        "subject": subject,
                        "hand": hand,
                        "camera": camera,
                        "noun": noun,
                        "sequence": sequence,
                        "action": action,
                        "task": task,
                        "annotation_file": None,  # No color.json in current data
                    }

                    sequences.append(sequence_info)
                    logging.debug(
                        f"Found sequence: {subject}/{hand}/{camera}/"
                        f"{noun}/{sequence}/{action}/{task}"
                    )

                except Exception as e:
                    logging.warning(f"Error processing path {root}: {e}")
                    continue

        logging.info(f"Discovered {len(sequences)} sequences.")
        return sequences

    def _determine_frame_range(
        self, seq_info: Dict[str, Any], total_frames: int
    ) -> range:
        """Determine the frame range based on available hand pose files."""
        subject = seq_info["subject"]

        # Find directory with hand pose files
        for hand_type in ["handpose_right_hand", "handpose_left_hand"]:
            hand_dir = (
                self.raw_dir
                / "Hand_pose"
                / hand_type
                / subject
                / seq_info["hand"]
                / seq_info["camera"]
                / seq_info["noun"]
                / seq_info["sequence"]
                / seq_info["action"]
                / seq_info["task"]
            )
            if hand_dir.exists():
                pickle_files = list(hand_dir.glob("*.pickle"))
                if pickle_files:
                    frame_numbers = []
                    for f in pickle_files:
                        try:
                            frame_numbers.append(int(f.stem))
                        except ValueError:
                            continue

                    if frame_numbers:
                        start_frame = min(frame_numbers)
                        end_frame = min(max(frame_numbers), total_frames - 1)
                        frame_step = max(1, (end_frame - start_frame) // 100)
                        return range(start_frame, end_frame + 1, frame_step)

        return range(0, min(100, total_frames))

    def _get_hand_poses_for_frame(self, seq_info: Dict[str, Any], frame_idx: int):
        """Get hand poses for a specific frame."""
        subject = seq_info["subject"]

        # Build hand pose file paths
        l_hand_path = (
            self.raw_dir
            / "Hand_pose"
            / "handpose_left_hand"
            / subject
            / seq_info["hand"]
            / seq_info["camera"]
            / seq_info["noun"]
            / seq_info["sequence"]
            / seq_info["action"]
            / seq_info["task"]
            / f"{frame_idx}.pickle"
        )
        r_hand_path = (
            self.raw_dir
            / "Hand_pose"
            / "handpose_right_hand"
            / subject
            / seq_info["hand"]
            / seq_info["camera"]
            / seq_info["noun"]
            / seq_info["sequence"]
            / seq_info["action"]
            / seq_info["task"]
            / f"{frame_idx}.pickle"
        )

        # Try different frame number formats if exact one doesn't exist
        for padding in [0, 5]:  # Try unpadded and 5-digit padded
            if padding > 0:
                frame_str = f"{frame_idx:0{padding}d}.pickle"
            else:
                frame_str = f"{frame_idx}.pickle"

            l_hand_alt = l_hand_path.parent / frame_str
            r_hand_alt = r_hand_path.parent / frame_str

            if l_hand_alt.exists():
                l_hand_path = l_hand_alt
                break
            if r_hand_alt.exists():
                r_hand_path = r_hand_alt
                break

        # Get hand poses
        left_pose = (
            self._get_handpose(l_hand_path)
            if l_hand_path.exists()
            else self._get_handpose_none()
        )
        right_pose = (
            self._get_handpose(r_hand_path)
            if r_hand_path.exists()
            else self._get_handpose_none()
        )

        return left_pose, right_pose

    def _store_sequence_data(
        self,
        traj_group,
        rgb_images,
        camera_poses,
        frame_indices,
        hand_poses_left,
        hand_poses_right,
    ):
        """Store processed sequence data in HDF5 groups."""
        hands_group = traj_group["hands"]
        ego_camera_group = traj_group["cameras/ego_camera"]

        # Store camera data
        ego_camera_group.create_dataset("rgb", data=np.array(rgb_images))
        ego_camera_group.create_dataset("pose_in_world", data=np.array(camera_poses))
        ego_camera_group.create_dataset("pose_indices", data=np.array(frame_indices))

        # Store hand poses
        if hand_poses_left:
            left_hand_group = hands_group.create_group("left")
            for key in hand_poses_left[0]:
                data = np.array([pose[key] for pose in hand_poses_left])
                left_hand_group.create_dataset(key, data=data)

        if hand_poses_right:
            right_hand_group = hands_group.create_group("right")
            for key in hand_poses_right[0]:
                data = np.array([pose[key] for pose in hand_poses_right])
                right_hand_group.create_dataset(key, data=data)

    def process_sequence(self, seq_info: Dict[str, Any], traj_group: Any):
        """Processes a single data sequence and writes it to the HDF5 group."""
        video_file = seq_info["video_file"]
        camera_file = seq_info["camera_file"]
        relative_path = seq_info["relative_path"]

        logging.info(f"Processing sequence: {relative_path}")

        # Load and validate video
        cap = cv2.VideoCapture(str(video_file))
        if not cap.isOpened():
            logging.error(f"Failed to open video file: {video_file}")
            return

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        logging.info(f"Video info: {total_frames} frames at {fps} fps")

        # Load camera intrinsics
        try:
            intrinsics = self.get_camera_intrinsics(camera_file)
        except Exception as e:
            logging.error(f"Failed to load camera intrinsics from {camera_file}: {e}")
            cap.release()
            return

        # Create HDF5 groups and add metadata
        traj_group.create_group("hands")
        camera_group = traj_group.create_group("cameras")
        ego_camera_group = camera_group.create_group("ego_camera")
        ego_camera_group.attrs["is_ego"] = True
        ego_camera_group.create_dataset("intrinsics", data=intrinsics)

        # Add sequence metadata
        for key in ["subject", "hand", "camera", "noun", "sequence", "action", "task"]:
            traj_group.attrs[key] = seq_info[key]

        # Determine frame range and process frames
        frame_range = self._determine_frame_range(seq_info, total_frames)
        logging.info(
            f"Processing frame range: {min(frame_range) if frame_range else 0} "
            f"to {max(frame_range) if frame_range else 0}"
        )

        rgb_images = []
        hand_poses_left = []
        hand_poses_right = []
        frame_indices = []
        camera_poses = []

        for frame_idx in frame_range:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue

            rgb_images.append(frame)
            frame_indices.append(frame_idx)
            camera_poses.append(np.eye(4))  # Identity pose (no trajectory data)

            # Get hand poses for this frame
            left_pose, right_pose = self._get_hand_poses_for_frame(seq_info, frame_idx)
            hand_poses_left.append(left_pose)
            hand_poses_right.append(right_pose)

        cap.release()

        if not rgb_images:
            logging.warning(f"No frames extracted from {video_file}")
            return

        # Store data in HDF5
        self._store_sequence_data(
            traj_group,
            rgb_images,
            camera_poses,
            frame_indices,
            hand_poses_left,
            hand_poses_right,
        )

        logging.info(
            f"Successfully processed {len(rgb_images)} frames from {relative_path}"
        )
