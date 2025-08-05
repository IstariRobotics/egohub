import json
import os
import pickle
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

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
        import open3d as o3d

        trajectory = o3d.io.read_pinhole_camera_trajectory(str(camera_file))
        if not trajectory.parameters:
            raise ValueError(f"No camera parameters found in {camera_file}")
        return trajectory.parameters[0].intrinsic.intrinsic_matrix

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
        """
        sequences = []
        annotations_dir = self.raw_dir / "HOI4D_annotations"
        if not annotations_dir.is_dir():
            return []

        for root, _, files in os.walk(annotations_dir):
            if "color.json" in files:
                relative_path = Path(root).relative_to(annotations_dir)
                video_file = (
                    self.raw_dir
                    / "HOI4D_release"
                    / relative_path
                    / "align_rgb"
                    / "image.mp4"
                )
                camera_file = annotations_dir / relative_path / "3Dseg" / "output.log"

                if video_file.exists() and camera_file.exists():
                    sequences.append(
                        {
                            "annotation_file": Path(root) / "color.json",
                            "video_file": video_file,
                            "camera_file": camera_file,
                            "relative_path": relative_path,
                        }
                    )
        return sequences

    def process_sequence(self, seq_info: Dict[str, Any], traj_group: Any):
        """
        Processes a single data sequence and writes it to the HDF5 group.
        """
        annotation_file = seq_info["annotation_file"]
        video_file = seq_info["video_file"]
        camera_file = seq_info["camera_file"]
        relative_path = seq_info["relative_path"]

        with open(annotation_file) as f:
            annotations = json.load(f)

        cap = cv2.VideoCapture(str(video_file))
        fps = cap.get(cv2.CAP_PROP_FPS)

        # Load camera data
        intrinsics = self.get_camera_intrinsics(camera_file)
        import open3d as o3d

        trajectory = o3d.io.read_pinhole_camera_trajectory(str(camera_file))
        camera_poses = [param.extrinsic for param in trajectory.parameters]

        for i, event in enumerate(annotations["events"]):
            start_frame = int(event["startTime"] * fps)
            end_frame = int(event["endTime"] * fps)

            # Create groups for this event
            event_group = traj_group.create_group(f"event_{i:04d}")
            hands_group = event_group.create_group("hands")
            objects_group = event_group.create_group("objects")
            camera_group = event_group.create_group("cameras")
            ego_camera_group = camera_group.create_group("ego_camera")
            ego_camera_group.attrs["is_ego"] = True
            ego_camera_group.create_dataset("intrinsics", data=intrinsics)

            rgb_images = []
            hand_poses_left = []
            hand_poses_right = []
            object_poses = []
            frame_indices = []
            pose_indices = []

            for frame_idx in range(start_frame, end_frame):
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue

                rgb_images.append(frame)
                frame_indices.append(frame_idx)
                pose_indices.append(frame_idx)

                # Hand poses
                l_hand_path = (
                    self.raw_dir
                    / "Hand_pose"
                    / "handpose_left_hand"
                    / relative_path
                    / f"{frame_idx:05d}.pickle"
                )
                r_hand_path = (
                    self.raw_dir
                    / "Hand_pose"
                    / "handpose_right_hand"
                    / relative_path
                    / f"{frame_idx:05d}.pickle"
                )

                hand_poses_left.append(
                    self._get_handpose(l_hand_path)
                    if l_hand_path.exists()
                    else self._get_handpose_none()
                )
                hand_poses_right.append(
                    self._get_handpose(r_hand_path)
                    if r_hand_path.exists()
                    else self._get_handpose_none()
                )

                # Object poses
                obj_pose_path = (
                    self.raw_dir
                    / "HOI4D_annotations"
                    / relative_path
                    / "objpose"
                    / f"{frame_idx:05d}.json"
                )
                if obj_pose_path.exists():
                    with open(obj_pose_path) as f:
                        object_poses.append(json.load(f))
                else:
                    object_poses.append({})

            ego_camera_group.create_dataset("rgb", data=np.array(rgb_images))
            ego_camera_group.create_dataset(
                "pose_in_world", data=np.array(camera_poses[start_frame:end_frame])
            )
            ego_camera_group.create_dataset("pose_indices", data=np.array(pose_indices))

            # Store hand poses
            left_hand_group = hands_group.create_group("left")
            for key in hand_poses_left[0]:
                data = np.array([pose[key] for pose in hand_poses_left])
                left_hand_group.create_dataset(key, data=data)

            right_hand_group = hands_group.create_group("right")
            for key in hand_poses_right[0]:
                data = np.array([pose[key] for pose in hand_poses_right])
                right_hand_group.create_dataset(key, data=data)

            # Store object poses
            if (
                object_poses
                and "dataList" in object_poses[0]
                and object_poses[0]["dataList"]
            ):
                obj_data = object_poses[0]["dataList"][0]
                obj_label = obj_data["label"]
                obj_group = objects_group.create_group(obj_label)
                obj_group.attrs["label"] = obj_label

                centers = np.array(
                    [
                        [
                            frame_data["dataList"][0]["center"]["x"],
                            frame_data["dataList"][0]["center"]["y"],
                            frame_data["dataList"][0]["center"]["z"],
                        ]
                        for frame_data in object_poses
                        if "dataList" in frame_data and frame_data["dataList"]
                    ]
                )
                dimensions = np.array(
                    [
                        [
                            frame_data["dataList"][0]["dimensions"]["height"],
                            frame_data["dataList"][0]["dimensions"]["length"],
                            frame_data["dataList"][0]["dimensions"]["width"],
                        ]
                        for frame_data in object_poses
                        if "dataList" in frame_data and frame_data["dataList"]
                    ]
                )
                rotations_euler = np.array(
                    [
                        [
                            frame_data["dataList"][0]["rotation"]["x"],
                            frame_data["dataList"][0]["rotation"]["y"],
                            frame_data["dataList"][0]["rotation"]["z"],
                        ]
                        for frame_data in object_poses
                        if "dataList" in frame_data and frame_data["dataList"]
                    ]
                )
                rotations_vec = Rotation.from_euler("XYZ", rotations_euler).as_rotvec()

                obj_group.create_dataset("center", data=centers)
                obj_group.create_dataset("dimensions", data=dimensions)
                obj_group.create_dataset("rotation", data=rotations_vec)
                obj_group.create_dataset("frame_indices", data=np.array(frame_indices))

        cap.release()
