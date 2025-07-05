"""
PyTorch Dataset classes for loading egocentric data from canonical HDF5 format.

This module provides the main interface for accessing processed egocentric datasets
in a PyTorch-compatible format.

# pyright: reportGeneralTypeIssues=false, reportIncompatibleMethodOverride=false
"""

from __future__ import annotations

import io
import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


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
        transform: Optional[callable] = None,
    ):
        self.h5_path = Path(h5_path)
        self.transform = transform

        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")

        self.camera_streams = self._resolve_camera_streams(camera_streams)

        # Build global frame index
        self.frame_index = self._build_frame_index(trajectories)

        logging.info(
            f"Loaded EgocentricH5Dataset from {self.h5_path} with "
            f"{len(self.frame_index)} total frames across "
            f"{len(set(idx[0] for idx in self.frame_index))} trajectories"
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
            if not cameras_group:
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
            all_trajectories = sorted([
                name for name in f.keys() if name.startswith("trajectory_")
            ])  # type: ignore[attr-defined]

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

                # Determine number of frames. We need at least one camera stream.
                if not self.camera_streams:
                    logging.warning(
                        f"No camera streams configured for {traj_name}, skipping."
                    )
                    continue

                # Use the first configured camera stream to determine frame count
                main_camera = self.camera_streams[0]
                pose_path = f"cameras/{main_camera}/pose_in_world"
                if pose_path in traj_group:  # type: ignore[operator]
                    num_frames = traj_group[pose_path].shape[0]
                elif "metadata/timestamps_ns" in traj_group:
                    num_frames = traj_group["metadata/timestamps_ns"].shape[0]
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

    def _load_camera_data(
        self, traj_group: h5py.Group, frame_idx: int, f: h5py.File
    ) -> Dict[str, Dict]:
        """Load camera data including RGB, poses, and intrinsics."""
        rgb_data = {}
        camera_pose_data = {}
        camera_intrinsics_data = {}

        for cam_name in self.camera_streams:
            cam_group_path = f"cameras/{cam_name}"
            if cam_group_path not in traj_group:
                continue

            cam_group = traj_group[cam_group_path]

            # Load RGB image
            rgb_path = f"{cam_group_path}/rgb"
            if str(rgb_path) in f:
                rgb_grp = f[str(rgb_path)]
                if isinstance(rgb_grp, h5py.Group) and "image_bytes" in rgb_grp:
                    num_bytes = rgb_grp["frame_sizes"][frame_idx]
                    encoded_frame = rgb_grp["image_bytes"][frame_idx, :num_bytes]
                    img_tensor = torch.from_numpy(
                        np.array(Image.open(io.BytesIO(encoded_frame)))
                    )
                    rgb_data[cam_name] = img_tensor.permute(2, 0, 1)  # HWC to CHW

            # Load camera pose
            if "pose_in_world" in cam_group:
                pose = cam_group["pose_in_world"][frame_idx]
                camera_pose_data[cam_name] = torch.from_numpy(pose).float()

            # Load camera intrinsics (static)
            if "intrinsics" in cam_group:
                intrinsics = cam_group["intrinsics"][:]
                camera_intrinsics_data[cam_name] = torch.from_numpy(intrinsics).float()

        return {
            "rgb": rgb_data,
            "camera_pose": camera_pose_data,
            "camera_intrinsics": camera_intrinsics_data,
        }

    def _load_hand_data(
        self, traj_group: h5py.Group, frame_idx: int
    ) -> Dict[str, torch.Tensor]:
        """Load hand tracking data."""
        hand_data = {}
        for hand in ["left", "right"]:
            hand_key = f"hands/{hand}/pose_in_world"
            if hand_key in traj_group:  # type: ignore[operator]
                pose = traj_group[hand_key][frame_idx]
                hand_data[f"{hand}_hand_pose"] = torch.from_numpy(pose).float()
        return hand_data

    def _load_skeleton_data(
        self, traj_group: h5py.Group, frame_idx: int
    ) -> Dict[str, Union[torch.Tensor, Any]]:
        """Load skeleton tracking data."""
        skeleton_data = {}
        if (
            "skeleton/positions" in traj_group
            and "skeleton/confidences" in traj_group
        ):
            positions = traj_group["skeleton/positions"][frame_idx]
            confidences = traj_group["skeleton/confidences"][frame_idx]
            skeleton_data["skeleton_positions"] = torch.from_numpy(positions).float()
            skeleton_data["skeleton_confidences"] = torch.from_numpy(
                confidences
            ).float()

            if "joint_names" in traj_group["skeleton"].attrs:
                skeleton_data["skeleton_joint_names"] = (
                    traj_group["skeleton"].attrs["joint_names"]
                )
        return skeleton_data

    def _load_object_data(
        self, traj_group: h5py.Group, frame_idx: int, f: h5py.File,
        traj_name: str
    ) -> Dict[str, Dict]:
        """Load object detection data."""
        objects_data = {}
        objects_path = Path(traj_name) / "objects"
        if str(objects_path) in f:
            objects_grp = f[str(objects_path)]
            if isinstance(objects_grp, h5py.Group):
                for obj_label, obj_grp in objects_grp.items():
                    if isinstance(obj_grp, h5py.Group):
                        objects_data[obj_label] = {
                            "bboxes_2d": obj_grp["bboxes_2d"][frame_idx],
                            "scores": obj_grp["scores"][frame_idx],
                        }
        return objects_data

    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, torch.Tensor, Dict]]:
        """
        Load a single frame from the dataset.
        """
        traj_name, frame_idx = self.frame_index[idx]

        with h5py.File(self.h5_path, "r") as f:
            traj_group = f[traj_name]

            result: Dict[str, Union[str, int, torch.Tensor, Dict]] = {
                "trajectory_name": traj_name,
                "frame_index": frame_idx,
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

            if self.transform is not None:
                result = self.transform(result)

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
        return self.get_trajectory_info()


class LatentSequenceDataset(Dataset):
    """
    PyTorch Dataset for loading sequences of latent vectors from HDF5 files.

    This dataset is designed for Stage 2 of the pre-training pipeline. It loads
    fixed-length sequences of latent vectors to be used for training temporal models
    like Transformers or LSTMs.

    Args:
        h5_path: Path to the HDF5 file containing latent vectors.
        sequence_length: The length of the latent vector sequences to return.
        trajectories: Optional list of trajectory names to include. If None,
            includes all.
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
            all_trajectories = sorted([
                name for name in f.keys() if name.startswith("trajectory_")
            ])

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

        with h5py.File(self.h5_path, "r") as f:
            latents = f[f"{traj_name}/latent/mean"][
                start_idx : start_idx + self.sequence_length
            ]

        return {"sequence": torch.from_numpy(latents).float()}
