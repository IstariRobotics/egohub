"""
PyTorch Dataset classes for loading egocentric data from canonical HDF5 format.

This module provides the main interface for accessing processed egocentric datasets
in a PyTorch-compatible format.

# pyright: reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false
"""

from __future__ import annotations

import io
import json
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from egohub.adapters.dataset_info import DatasetInfo


class BaseDatasetReader(Dataset, ABC):
    """Abstract base class defining the required interface for dataset readers."""

    @abstractmethod
    def __len__(self) -> int:  # noqa: D401
        """Return total number of samples in the dataset."""

    @abstractmethod
    def __getitem__(self, idx: int):  # noqa: D401
        """Return the sample at the given index."""

    @abstractmethod
    def get_metadata(self):  # noqa: D401
        """Return dataset-level metadata useful for diagnostics or training."""


class EgocentricH5Dataset(BaseDatasetReader):
    """
    PyTorch Dataset for loading egocentric data from canonical HDF5 files.

    This dataset provides access to processed egocentric sequences stored in
    our canonical HDF5 format. It efficiently handles multiple trajectories
    and provides a unified interface for accessing frames across all sequences.

    Args:
        h5_path: Path to the HDF5 file containing processed data.
        trajectories: Optional list of trajectory names to include. If None,
            includes all.
        camera_streams: Optional list of camera names to load data from.
                        If None, it will try to load from the first discovered camera.
        transform: Optional transform to apply to loaded data.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        trajectories: Optional[List[str]] = None,
        camera_streams: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        trajectory_filter: Optional[Callable] = None,
        frame_filter: Optional[Callable] = None,
        transforms: Optional[List[Callable]] = None,
        skeleton_streams: Optional[List[str]] = None,
    ):
        self.h5_path = Path(h5_path)
        self.transform = transform
        self.trajectory_filter = trajectory_filter
        self.frame_filter = frame_filter
        self.transforms = transforms or []
        self.skeleton_streams = skeleton_streams

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.camera_streams = self._resolve_camera_streams(camera_streams)

        # Build global frame index
        self.frame_index = self._build_frame_index(trajectories)

        # Load DatasetInfo if present
        with h5py.File(self.h5_path, "r") as f:
            if "dataset_info" in f.attrs:
                info_str = f.attrs["dataset_info"]
                if isinstance(info_str, bytes):
                    info_str = info_str.decode()
                info_dict = json.loads(info_str)
                camera_intrinsics = np.array(
                    info_dict.get(
                        "camera_intrinsics", [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
                    )
                )
                view_coordinates = info_dict.get("view_coordinates", "RDF")
                frame_rate = info_dict.get("frame_rate", 30.0)
                joint_names = info_dict.get("joint_names", [])
                joint_hierarchy = info_dict.get("joint_hierarchy", {})
                joint_remap = info_dict.get("joint_remap", {})
                depth_scale = info_dict.get("depth_scale", 1.0)
                camera_distortion = info_dict.get("camera_distortion")
                object_categories = info_dict.get("object_categories", [])
                object_palette = (
                    np.array(info_dict.get("object_palette"))
                    if info_dict.get("object_palette") is not None
                    else None
                )
                mano_betas_left = (
                    np.array(info_dict.get("mano_betas_left"))
                    if info_dict.get("mano_betas_left") is not None
                    else None
                )
                mano_betas_right = (
                    np.array(info_dict.get("mano_betas_right"))
                    if info_dict.get("mano_betas_right") is not None
                    else None
                )
                modalities = info_dict.get("modalities", {"rgb": True, "depth": False})
                self.dataset_info = DatasetInfo(
                    camera_intrinsics=camera_intrinsics,
                    view_coordinates=view_coordinates,
                    frame_rate=frame_rate,
                    joint_names=joint_names,
                    joint_hierarchy=joint_hierarchy,
                    joint_remap=joint_remap,
                    depth_scale=depth_scale,
                    camera_distortion=camera_distortion,
                    object_categories=object_categories,
                    object_palette=object_palette,
                    mano_betas_left=mano_betas_left,
                    mano_betas_right=mano_betas_right,
                    modalities=modalities,
                )
            else:
                self.dataset_info = None

        # Calculate number of trajectories and frames
        self.num_trajectories = len(set(idx[0] for idx in self.frame_index))
        self.num_frames = len(self.frame_index)

        logging.info(
            f"Loaded EgocentricH5Dataset from {self.h5_path} with "
            f"{len(self.frame_index)} total frames across "
            f"{self.num_trajectories} trajectories"
        )

    def _resolve_camera_streams(
        self, requested_streams: Optional[List[str]]
    ) -> List[str]:
        """Determine which camera streams to use."""
        with h5py.File(self.h5_path, "r") as f:
            # Check the first trajectory for available camera streams
            first_traj_name = next(
                (key for key in f.keys() if key.startswith("trajectory_")), None
            )
            if not first_traj_name:
                logging.warning("No trajectories found in HDF5 file.")
                return []

            cameras_group = f.get(f"{first_traj_name}/cameras")
            if not isinstance(cameras_group, h5py.Group):
                logging.warning(
                    f"No 'cameras' group found in trajectory {first_traj_name}."
                )
                return []

            available_streams = list(cameras_group.keys())
            if not available_streams:
                logging.warning(
                    f"No camera streams found in trajectory {first_traj_name}."
                )
                return []

            if requested_streams:
                valid_streams = [s for s in requested_streams if s in available_streams]
                if len(valid_streams) != len(requested_streams):
                    missing = set(requested_streams) - set(valid_streams)
                    logging.warning(
                        f"Requested camera streams not found: {missing}. "
                        f"Available: {available_streams}"
                    )
                return valid_streams
            else:
                # Default to the first available camera stream
                default_stream = available_streams[0]
                logging.info(
                    f"No camera streams specified, defaulting to '{default_stream}'"
                )
                return [default_stream]

    def _build_frame_index(
        self, trajectories: Optional[List[str]]
    ) -> List[Tuple[str, int]]:
        """
        Build a global frame index mapping dataset indices to (trajectory, frame) pairs.
        """
        frame_index = []

        with h5py.File(self.h5_path, "r") as f:
            all_trajectories = sorted(
                [name for name in f.keys() if name.startswith("trajectory_")]
            )  # type: ignore[attr-defined]

            available_trajectories = (
                trajectories if trajectories is not None else all_trajectories
            )

            for traj_name in available_trajectories:
                if traj_name not in f:  # type: ignore[operator]
                    logging.warning(
                        f"Requested trajectory '{traj_name}' not found, skipping."
                    )
                    continue

                traj_group = f[traj_name]
                if not isinstance(traj_group, h5py.Group):
                    logging.warning(
                        f"'{traj_name}' is not a valid trajectory group, skipping."
                    )
                    continue

                # Determine number of frames. We need at least one camera stream.
                if not self.camera_streams:
                    logging.warning(
                        f"No camera streams configured for {traj_name}, skipping."
                    )
                    continue

                # Use the first configured camera stream to determine frame count
                main_camera = self.camera_streams[0]
                pose_path = f"cameras/{main_camera}/pose_in_world"
                pose_dset = traj_group.get(pose_path)
                timestamps_dset = traj_group.get("metadata/timestamps_ns")

                if isinstance(pose_dset, h5py.Dataset):
                    num_frames = pose_dset.shape[0]
                elif isinstance(timestamps_dset, h5py.Dataset):
                    num_frames = timestamps_dset.shape[0]
                else:
                    logging.warning(
                        f"Cannot determine frame count for {traj_name}, skipping."
                    )
                    continue

                for frame_idx in range(num_frames):
                    frame_index.append((traj_name, frame_idx))

        return frame_index

    def __len__(self) -> int:
        """Return the total number of frames across all trajectories."""
        return len(self.frame_index)

    def _load_camera_data(  # noqa: C901
        self, traj_group: h5py.Group, frame_idx: int, f: h5py.File
    ) -> Dict[str, Dict]:
        """Load camera data including RGB, poses, and intrinsics."""
        rgb_data = {}
        camera_pose_data = {}
        camera_intrinsics_data = {}
        depth_data = {}

        for cam_name in self.camera_streams:
            cam_group_path = f"cameras/{cam_name}"
            cam_group = traj_group.get(cam_group_path)
            if not isinstance(cam_group, h5py.Group):
                continue

            # Load RGB image
            rgb_path = f"{cam_group_path}/rgb"
            rgb_grp = f.get(str(rgb_path))
            if isinstance(rgb_grp, h5py.Group) and "image_bytes" in rgb_grp:
                frame_sizes_dset = rgb_grp.get("frame_sizes")
                image_bytes_dset = rgb_grp.get("image_bytes")
                if isinstance(frame_sizes_dset, h5py.Dataset) and isinstance(
                    image_bytes_dset, h5py.Dataset
                ):
                    num_bytes = frame_sizes_dset[frame_idx]
                    encoded_frame = image_bytes_dset[frame_idx, :num_bytes]
                    img_tensor = torch.from_numpy(
                        np.array(Image.open(io.BytesIO(encoded_frame)))
                    )
                    rgb_data[cam_name] = img_tensor.permute(2, 0, 1)  # HWC to CHW

            # Load camera pose
            pose_dset = cam_group.get("pose_in_world")
            if isinstance(pose_dset, h5py.Dataset):
                pose = pose_dset[frame_idx]
                camera_pose_data[cam_name] = torch.from_numpy(pose).float()

            # Load camera intrinsics (static)
            intrinsics_dset = cam_group.get("intrinsics")
            if isinstance(intrinsics_dset, h5py.Dataset):
                intrinsics = intrinsics_dset[:]
                camera_intrinsics_data[cam_name] = torch.from_numpy(intrinsics).float()

            # Load depth if available
            depth_grp = cam_group.get("depth")
            if isinstance(depth_grp, h5py.Group):
                frame_sizes = depth_grp.get("frame_sizes")
                depth_bytes = depth_grp.get("depth_bytes")
                frame_indices_dset = depth_grp.get("frame_indices")
                if (
                    isinstance(frame_sizes, h5py.Dataset)
                    and isinstance(depth_bytes, h5py.Dataset)
                    and isinstance(frame_indices_dset, h5py.Dataset)
                ):
                    frame_indices = frame_indices_dset[:]
                    current_indices = np.where(frame_indices == frame_idx)[0]
                    for idx in current_indices:
                        num_bytes = frame_sizes[idx]
                        encoded_depth = depth_bytes[idx, :num_bytes]
                        depth_img = Image.open(io.BytesIO(encoded_depth))
                        depth = np.array(depth_img).astype(np.float32)
                        if self.dataset_info:
                            depth *= self.dataset_info.depth_scale
                        depth_data[cam_name] = torch.from_numpy(depth)

        return {
            "rgb": rgb_data,
            "camera_pose": camera_pose_data,
            "camera_intrinsics": camera_intrinsics_data,
            "depth": depth_data,
        }

    def _load_hand_data(
        self, traj_group: h5py.Group, frame_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Load hand tracking data."""
        hand_data = {}
        for hand in ["left", "right"]:
            hand_key = f"hands/{hand}/pose_in_world"
            pose_dset = traj_group.get(hand_key)
            if isinstance(pose_dset, h5py.Dataset):
                pose = pose_dset[frame_idx]
                hand_data[f"{hand}_hand_pose"] = torch.from_numpy(pose).float()
        return hand_data

    def _load_skeleton_data(
        self, traj_group: h5py.Group, frame_idx: int
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """Load skeleton tracking data."""
        skeleton_data = {}
        positions_dset = traj_group.get("skeleton/positions")
        confidences_dset = traj_group.get("skeleton/confidences")

        if isinstance(positions_dset, h5py.Dataset) and isinstance(
            confidences_dset, h5py.Dataset
        ):
            positions = positions_dset[frame_idx]
            confidences = confidences_dset[frame_idx]
            skeleton_data["skeleton_positions"] = torch.from_numpy(positions).float()
            skeleton_data["skeleton_confidences"] = torch.from_numpy(
                confidences
            ).float()

            skeleton_group = traj_group.get("skeleton")
            if (
                isinstance(skeleton_group, h5py.Group)
                and "joint_names" in skeleton_group.attrs
            ):
                skeleton_data["skeleton_joint_names"] = skeleton_group.attrs[
                    "joint_names"
                ]

        return skeleton_data

    def _load_object_data(
        self, traj_group: h5py.Group, frame_idx: int, f: h5py.File, traj_name: str
    ) -> Dict[str, Dict]:
        """Load object detection data."""
        objects_data = {}
        objects_group = traj_group.get("objects")
        if not isinstance(objects_group, h5py.Group):
            return objects_data

        for label, obj_grp in objects_group.items():
            if isinstance(obj_grp, h5py.Group):
                for dset_name, dset in obj_grp.items():
                    if isinstance(dset, h5py.Dataset):
                        group_name = dset_name.rsplit("_", 1)[0]
                        if group_name not in objects_data:
                            objects_data[group_name] = {}
                        objects_data[group_name][label] = torch.from_numpy(
                            dset[frame_idx]
                        )

        if self.dataset_info:
            objects_data["categories"] = self.dataset_info.object_categories
            objects_data["palette"] = self.dataset_info.object_palette

        return objects_data

    def __getitem__(
        self, idx: Union[int, slice]
    ) -> Dict[str, Union[str, int, torch.Tensor, Dict]]:
        """
        Load a single frame or slice of frames from the dataset.
        """
        if isinstance(idx, slice):
            # Handle slicing
            start, stop, step = idx.indices(len(self))
            indices = range(start, stop, step)
            return [self[i] for i in indices]

        traj_name, frame_idx = self.frame_index[idx]
        trajectory_names = sorted(set(idx[0] for idx in self.frame_index))
        trajectory_idx = trajectory_names.index(traj_name)

        with h5py.File(self.h5_path, "r") as f:
            traj_group = f.get(traj_name)
            if not isinstance(traj_group, h5py.Group):
                # This should not happen if the index is built correctly
                raise ValueError(f"Trajectory {traj_name} not found or not a group.")

            result: Dict[str, Union[str, int, torch.Tensor, Dict]] = {
                "trajectory_name": traj_name,
                "trajectory_idx": trajectory_idx,
                "frame_index": frame_idx,
                "frame_idx": frame_idx,  # Alias for compatibility
                "global_index": idx,
            }

            # Load camera data
            camera_data = self._load_camera_data(traj_group, frame_idx, f)
            result.update(camera_data)

            # Load timestamp
            if "metadata/timestamps_ns" in traj_group:
                timestamp = traj_group["metadata/timestamps_ns"][frame_idx]
                result["timestamp_ns"] = torch.tensor(timestamp, dtype=torch.int64)

            # Load hand data
            hand_data = self._load_hand_data(traj_group, frame_idx)
            result.update(hand_data)

            # Load skeleton data
            skeleton_data = self._load_skeleton_data(traj_group, frame_idx)
            result.update(skeleton_data)

            # Load object data
            object_data = self._load_object_data(traj_group, frame_idx, f, traj_name)
            if object_data:
                result["objects"] = object_data

            # Apply transforms
            if self.transform is not None:
                result = self.transform(result)

            for transform in self.transforms:
                result = transform(result)

            return result

    def get_trajectory_info(self) -> Dict[str, Dict]:
        """
        Get information about all trajectories in the dataset.
        """
        info = {}

        with h5py.File(self.h5_path, "r") as f:
            for traj_name in f.keys():  # type: ignore[attr-defined]
                if not traj_name.startswith("trajectory_"):
                    continue

                traj_group = f[traj_name]
                traj_info = {}

                if self.camera_streams:
                    main_camera = self.camera_streams[0]
                    pose_path = f"cameras/{main_camera}/pose_in_world"
                    if pose_path in traj_group:
                        traj_info["num_frames"] = traj_group[pose_path].shape[0]

                if "metadata" in traj_group:
                    for key, value in traj_group["metadata"].attrs.items():
                        traj_info[key] = value

                info[traj_name] = traj_info

        return info

    # ---------------------------------------------------------------------
    # Implementation of BaseDatasetReader required API
    # ---------------------------------------------------------------------

    def get_metadata(self) -> Dict[str, Dict]:
        """Alias for get_trajectory_info to satisfy the uniform API."""
        metadata = self.get_trajectory_info()
        metadata["num_trajectories"] = self.num_trajectories
        metadata["num_frames"] = self.num_frames
        return metadata

    def get_trajectory_data(self, trajectory_idx: int) -> Dict[str, torch.Tensor]:
        """Get all data for a specific trajectory."""
        if trajectory_idx >= self.num_trajectories:
            raise IndexError(f"Trajectory index {trajectory_idx} out of range")

        trajectory_names = sorted(set(idx[0] for idx in self.frame_index))
        traj_name = trajectory_names[trajectory_idx]

        # Get all frames for this trajectory
        traj_frames = [
            i for i, (name, _) in enumerate(self.frame_index) if name == traj_name
        ]

        trajectory_data = []
        for frame_idx in traj_frames:
            trajectory_data.append(self[frame_idx])

        return trajectory_data

    def get_frame_data(
        self, trajectory_idx: int, frame_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Get data for a specific frame in a specific trajectory."""
        if trajectory_idx >= self.num_trajectories:
            raise IndexError(f"Trajectory index {trajectory_idx} out of range")

        trajectory_names = sorted(set(idx[0] for idx in self.frame_index))
        traj_name = trajectory_names[trajectory_idx]

        # Find the global index for this trajectory and frame
        global_idx = None
        for i, (name, frame) in enumerate(self.frame_index):
            if name == traj_name and frame == frame_idx:
                global_idx = i
                break

        if global_idx is None:
            raise IndexError(
                f"Frame {frame_idx} not found in trajectory {trajectory_idx}"
            )

        return self[global_idx]


class LatentSequenceDataset(Dataset):
    """
    A PyTorch Dataset for loading sequences of latent vectors.

    This dataset is designed for training sequence models like transformers or
    RNNs on top of pre-encoded latent representations of egocentric data.

    Args:
        h5_path: Path to the HDF5 file.
        sequence_length: The length of the sequences to be returned.
        trajectories: Optional list of trajectory names to include.
    """

    def __init__(
        self,
        h5_path: Union[str, Path],
        sequence_length: int,
        trajectories: Optional[List[str]] = None,
    ):
        self.h5_path = Path(h5_path)
        self.sequence_length = sequence_length

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        # Build an index of valid start points for sequences
        self.sequence_index = self._build_sequence_index(trajectories)

        logging.info(
            f"Loaded LatentSequenceDataset from {self.h5_path} with "
            f"{len(self.sequence_index)} total sequences of length "
            f"{self.sequence_length}"
        )

    def _build_sequence_index(
        self, trajectories: Optional[List[str]]
    ) -> List[Tuple[str, int]]:
        """
        Build an index of (trajectory, start_frame_idx) for valid sequences.
        """
        sequence_index = []
        with h5py.File(self.h5_path, "r") as f:
            all_trajectories = sorted(
                [name for name in f.keys() if name.startswith("trajectory_")]
            )

            available_trajectories = (
                trajectories if trajectories is not None else all_trajectories
            )

            for traj_name in available_trajectories:
                if traj_name not in f:
                    logging.warning(
                        f"Requested trajectory '{traj_name}' not found, skipping."
                    )
                    continue

                if "latent/mean" in f[f"{traj_name}"]:
                    num_latents = f[f"{traj_name}/latent/mean"].shape[0]
                    # We can start a sequence from any frame that allows a full
                    # sequence to be formed
                    for i in range(num_latents - self.sequence_length + 1):
                        sequence_index.append((traj_name, i))
                else:
                    logging.warning(
                        f"No latents found for trajectory '{traj_name}', skipping."
                    )

        return sequence_index

    def __len__(self) -> int:
        return len(self.sequence_index)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        traj_name, start_idx = self.sequence_index[idx]
        start_idx + self.sequence_length

        with h5py.File(self.h5_path, "r") as f:
            latents = f[f"{traj_name}/latent/mean"][
                start_idx : start_idx + self.sequence_length
            ]

        return {"sequence": torch.from_numpy(latents).float()}
