from typing import Any, Dict, List

import h5py
import numpy as np

from egohub.adapters.base import BaseAdapter
from egohub.adapters.dataset_info import DatasetInfo


class DummyMultiViewAdapter(BaseAdapter):
    name = "dummy_multi_view"

    def get_dataset_info(self) -> DatasetInfo:
        return DatasetInfo(
            camera_intrinsics=np.eye(3, dtype=np.float32),
            view_coordinates="RDF",
            frame_rate=30.0,
            depth_scale=0.001,
            modalities={"rgb": True, "depth": True},
        )

    def get_camera_intrinsics(self) -> Dict[str, Any]:
        return {"matrix": np.eye(3, dtype=np.float32)}

    @property
    def source_joint_names(self) -> List[str]:
        return []

    @property
    def source_skeleton_hierarchy(self) -> Dict[str, str]:
        return {}

    def discover_sequences(self) -> List[Dict[str, Any]]:
        return [{"name": "dummy_seq"}]

    def process_sequence(self, seq_info: Dict[str, Any], traj_group: h5py.Group):
        num_frames = 10
        master_timestamps_ns = np.arange(num_frames) * (1e9 / 30.0)

        metadata_group = traj_group.create_group("metadata")
        metadata_group.attrs["source_dataset"] = "dummy"
        metadata_group.create_dataset(
            "timestamps_ns", data=master_timestamps_ns.astype(np.uint64)
        )  # noqa: E501

        cameras_group = traj_group.create_group("cameras")
        for cam_id in ["cam1", "cam2"]:
            cam_group = cameras_group.create_group(cam_id)
            cam_group.attrs["is_ego"] = cam_id == "cam1"
            cam_group.create_dataset("intrinsics", data=np.eye(3, dtype=np.float32))

            # Dummy poses
            poses = np.tile(np.eye(4), (num_frames, 1, 1))
            cam_group.create_dataset("pose_in_world", data=poses.astype(np.float32))
            cam_group.create_dataset(
                "pose_indices", data=np.arange(num_frames, dtype=np.uint64)
            )  # noqa: E501

            # Dummy RGB
            rgb_group = cam_group.create_group("rgb")
            rgb_group.create_dataset(
                "frame_sizes", data=np.full(num_frames, 10, dtype=np.int32)
            )  # noqa: E501
            rgb_group.create_dataset(
                "image_bytes", shape=(num_frames, 10), dtype=np.uint8
            )  # noqa: E501
            rgb_group.create_dataset(
                "frame_indices", data=np.arange(num_frames, dtype=np.uint64)
            )  # noqa: E501

            # Dummy Depth
            depth_group = cam_group.create_group("depth")
            depth_group.create_dataset(
                "frame_sizes", data=np.full(num_frames, 10, dtype=np.int32)
            )  # noqa: E501
            depth_group.create_dataset(
                "depth_bytes", shape=(num_frames, 10), dtype=np.uint8
            )  # noqa: E501
            depth_group.create_dataset(
                "frame_indices", data=np.arange(num_frames, dtype=np.uint64)
            )  # noqa: E501
