"""
PyTorch Dataset classes for loading egocentric data from canonical HDF5 format.

This module provides the main interface for accessing processed egocentric datasets
in a PyTorch-compatible format.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import cv2
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset


class EgocentricH5Dataset(Dataset):
    """
    PyTorch Dataset for loading egocentric data from canonical HDF5 files.
    
    This dataset provides access to processed egocentric sequences stored in
    our canonical HDF5 format. It efficiently handles multiple trajectories
    and provides a unified interface for accessing frames across all sequences.
    
    Args:
        h5_path: Path to the HDF5 file containing processed data.
        trajectories: Optional list of trajectory names to include. If None, includes all.
        camera_streams: Optional list of camera names to load data from. 
                        If None, it will try to load from the first discovered camera.
        transform: Optional transform to apply to loaded data.
    """
    
    def __init__(
        self, 
        h5_path: Union[str, Path], 
        trajectories: Optional[List[str]] = None,
        camera_streams: Optional[List[str]] = None,
        transform: Optional[callable] = None
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
            f"{len(self.frame_index)} total frames across {len(set(idx[0] for idx in self.frame_index))} trajectories"
        )
    
    def _resolve_camera_streams(self, requested_streams: Optional[List[str]]) -> List[str]:
        """Determine which camera streams to use."""
        with h5py.File(self.h5_path, 'r') as f:
            # Check the first trajectory for available camera streams
            first_traj_name = next((key for key in f.keys() if key.startswith('trajectory_')), None)
            if not first_traj_name:
                logging.warning("No trajectories found in HDF5 file.")
                return []
            
            cameras_group = f.get(f"{first_traj_name}/cameras")
            if not cameras_group:
                logging.warning(f"No 'cameras' group found in trajectory {first_traj_name}.")
                return []
                
            available_streams = list(cameras_group.keys())
            if not available_streams:
                logging.warning(f"No camera streams found in trajectory {first_traj_name}.")
                return []
            
            if requested_streams:
                valid_streams = [s for s in requested_streams if s in available_streams]
                if len(valid_streams) != len(requested_streams):
                    missing = set(requested_streams) - set(valid_streams)
                    logging.warning(f"Requested camera streams not found: {missing}. Available: {available_streams}")
                return valid_streams
            else:
                # Default to the first available camera stream
                default_stream = available_streams[0]
                logging.info(f"No camera streams specified, defaulting to '{default_stream}'")
                return [default_stream]

    def _build_frame_index(self, trajectories: Optional[List[str]]) -> List[Tuple[str, int]]:
        """
        Build a global frame index mapping dataset indices to (trajectory, frame) pairs.
        """
        frame_index = []
        
        with h5py.File(self.h5_path, 'r') as f:
            all_trajectories = sorted([name for name in f.keys() if name.startswith('trajectory_')])
            
            available_trajectories = trajectories if trajectories is not None else all_trajectories
            
            for traj_name in available_trajectories:
                if traj_name not in f:
                    logging.warning(f"Requested trajectory '{traj_name}' not found, skipping.")
                    continue
                
                traj_group = f[traj_name]
                
                # Determine number of frames. We need at least one camera stream.
                if not self.camera_streams:
                    logging.warning(f"No camera streams configured for {traj_name}, skipping.")
                    continue

                # Use the first configured camera stream to determine frame count
                main_camera = self.camera_streams[0]
                pose_path = f"cameras/{main_camera}/pose_in_world"
                if pose_path in traj_group:
                    num_frames = traj_group[pose_path].shape[0]
                elif 'metadata/timestamps_ns' in traj_group:
                    num_frames = traj_group['metadata/timestamps_ns'].shape[0]
                else:
                    logging.warning(f"Cannot determine frame count for {traj_name}, skipping.")
                    continue
                
                for frame_idx in range(num_frames):
                    frame_index.append((traj_name, frame_idx))
        
        return frame_index
    
    def __len__(self) -> int:
        """Return the total number of frames across all trajectories."""
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[str, int, torch.Tensor, Dict]]:
        """
        Load a single frame from the dataset.
        """
        traj_name, frame_idx = self.frame_index[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            traj_group = f[traj_name]
            
            result: Dict[str, Union[str, int, torch.Tensor, Dict]] = {
                'trajectory_name': traj_name,
                'frame_index': frame_idx,
                'global_index': idx
            }

            # --- Load Camera Data ---
            result['rgb'] = {}
            result['camera_pose'] = {}
            result['camera_intrinsics'] = {}

            for cam_name in self.camera_streams:
                cam_group_path = f"cameras/{cam_name}"
                if cam_group_path not in traj_group:
                    continue

                cam_group = traj_group[cam_group_path]

                # Load RGB image
                rgb_path = f"{cam_group_path}/rgb"
                if f"{rgb_path}/image_bytes" in traj_group and f"{rgb_path}/frame_sizes" in traj_group:
                    image_bytes = traj_group[f"{rgb_path}/image_bytes"][frame_idx]
                    frame_size = traj_group[f"{rgb_path}/frame_sizes"][frame_idx]
                    actual_bytes = image_bytes[:frame_size]
                    
                    nparr = np.frombuffer(actual_bytes, np.uint8)
                    rgb_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    
                    if rgb_image is not None:
                        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                        rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
                        result['rgb'][cam_name] = rgb_tensor
                    else:
                        logging.warning(f"Failed to decode RGB for {traj_name}/{cam_name} frame {frame_idx}")

                # Load camera pose
                if "pose_in_world" in cam_group:
                    pose = cam_group["pose_in_world"][frame_idx]
                    result['camera_pose'][cam_name] = torch.from_numpy(pose).float()

                # Load camera intrinsics (static)
                if "intrinsics" in cam_group:
                    intrinsics = cam_group["intrinsics"][:]
                    result['camera_intrinsics'][cam_name] = torch.from_numpy(intrinsics).float()
            
            # --- Load Other Data Streams ---
            if 'metadata/timestamps_ns' in traj_group:
                timestamp = traj_group['metadata/timestamps_ns'][frame_idx]
                result['timestamp_ns'] = torch.tensor(timestamp, dtype=torch.int64)
            
            for hand in ['left', 'right']:
                hand_key = f'hands/{hand}/pose_in_world'
                if hand_key in traj_group:
                    pose = traj_group[hand_key][frame_idx]
                    result[f'{hand}_hand_pose'] = torch.from_numpy(pose).float()
            
            if 'skeleton/positions' in traj_group and 'skeleton/confidences' in traj_group:
                positions = traj_group['skeleton/positions'][frame_idx]
                confidences = traj_group['skeleton/confidences'][frame_idx]
                result['skeleton_positions'] = torch.from_numpy(positions).float()
                result['skeleton_confidences'] = torch.from_numpy(confidences).float()
                
                if 'joint_names' in traj_group['skeleton'].attrs:
                    result['skeleton_joint_names'] = traj_group['skeleton'].attrs['joint_names']
            
            if self.transform is not None:
                result = self.transform(result)
            
            return result
    
    def get_trajectory_info(self) -> Dict[str, Dict]:
        """
        Get information about all trajectories in the dataset.
        """
        info = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            for traj_name in f.keys():
                if not traj_name.startswith('trajectory_'):
                    continue
                
                traj_group = f[traj_name]
                traj_info = {}
                
                if self.camera_streams:
                    main_camera = self.camera_streams[0]
                    pose_path = f"cameras/{main_camera}/pose_in_world"
                    if pose_path in traj_group:
                        traj_info['num_frames'] = traj_group[pose_path].shape[0]

                if 'metadata' in traj_group:
                    for key, value in traj_group['metadata'].attrs.items():
                        traj_info[key] = value
                
                info[traj_name] = traj_info
        
        return info 