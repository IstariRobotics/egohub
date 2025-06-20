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
        h5_path: Path to the HDF5 file containing processed data
        trajectories: Optional list of trajectory names to include. If None, includes all.
        transform: Optional transform to apply to loaded data
    """
    
    def __init__(
        self, 
        h5_path: Union[str, Path], 
        trajectories: Optional[List[str]] = None,
        transform: Optional[callable] = None
    ):
        self.h5_path = Path(h5_path)
        self.transform = transform
        
        if not self.h5_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.h5_path}")
        
        # Build global frame index
        self.frame_index = self._build_frame_index(trajectories)
        
        logging.info(
            f"Loaded EgocentricH5Dataset from {self.h5_path} with "
            f"{len(self.frame_index)} total frames across {len(set(idx[0] for idx in self.frame_index))} trajectories"
        )
    
    def _build_frame_index(self, trajectories: Optional[List[str]]) -> List[Tuple[str, int]]:
        """
        Build a global frame index mapping dataset indices to (trajectory, frame) pairs.
        
        Args:
            trajectories: Optional list of trajectory names to include
            
        Returns:
            List of (trajectory_name, frame_index) tuples
        """
        frame_index = []
        
        with h5py.File(self.h5_path, 'r') as f:
            # Get all trajectory names
            all_trajectories = sorted([name for name in f.keys() if name.startswith('trajectory_')])
            
            if trajectories is not None:
                # Filter to requested trajectories
                available_trajectories = [t for t in trajectories if t in all_trajectories]
                if len(available_trajectories) != len(trajectories):
                    missing = set(trajectories) - set(available_trajectories)
                    logging.warning(f"Requested trajectories not found: {missing}")
            else:
                available_trajectories = all_trajectories
            
            # Build index
            for traj_name in available_trajectories:
                traj_group = f[traj_name]
                
                # Get number of frames from camera poses (most reliable)
                if 'camera/pose_in_world' in traj_group:
                    num_frames = traj_group['camera/pose_in_world'].shape[0]
                elif 'metadata/timestamps_ns' in traj_group:
                    num_frames = traj_group['metadata/timestamps_ns'].shape[0]
                else:
                    logging.warning(f"No frame count found for {traj_name}, skipping")
                    continue
                
                # Add all frames from this trajectory
                for frame_idx in range(num_frames):
                    frame_index.append((traj_name, frame_idx))
        
        return frame_index
    
    def __len__(self) -> int:
        """Return the total number of frames across all trajectories."""
        return len(self.frame_index)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Load a single frame from the dataset.
        
        Args:
            idx: Global frame index
            
        Returns:
            Dictionary containing loaded data as torch tensors
        """
        traj_name, frame_idx = self.frame_index[idx]
        
        with h5py.File(self.h5_path, 'r') as f:
            traj_group = f[traj_name]
            
            # Initialize result dictionary
            result = {
                'trajectory_name': traj_name,
                'frame_index': frame_idx,
                'global_index': idx
            }
            
            # Load RGB image
            if 'rgb/image_bytes' in traj_group and 'rgb/frame_sizes' in traj_group:
                image_bytes = traj_group['rgb/image_bytes'][frame_idx]
                frame_size = traj_group['rgb/frame_sizes'][frame_idx]
                
                # Extract actual frame data (remove padding)
                actual_bytes = image_bytes[:frame_size]
                
                # Decode JPG
                nparr = np.frombuffer(actual_bytes, np.uint8)
                rgb_image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if rgb_image is not None:
                    # Convert BGR to RGB and to tensor
                    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
                    rgb_tensor = torch.from_numpy(rgb_image).permute(2, 0, 1).float() / 255.0
                    result['rgb'] = rgb_tensor
                else:
                    logging.warning(f"Failed to decode RGB image for {traj_name} frame {frame_idx}")
            
            # Load camera pose
            if 'camera/pose_in_world' in traj_group:
                pose = traj_group['camera/pose_in_world'][frame_idx]
                result['camera_pose'] = torch.from_numpy(pose).float()
            
            # Load camera intrinsics
            if 'camera/intrinsics' in traj_group:
                intrinsics = traj_group['camera/intrinsics'][:]
                result['camera_intrinsics'] = torch.from_numpy(intrinsics).float()
            
            # Load timestamp
            if 'metadata/timestamps_ns' in traj_group:
                timestamp = traj_group['metadata/timestamps_ns'][frame_idx]
                result['timestamp_ns'] = torch.tensor(timestamp, dtype=torch.int64)
            
            # Load hand poses if available
            for hand in ['left', 'right']:
                hand_key = f'hands/{hand}/pose_in_world'
                if hand_key in traj_group:
                    pose = traj_group[hand_key][frame_idx]
                    result[f'{hand}_hand_pose'] = torch.from_numpy(pose).float()
            
            # Load skeleton data if available
            if 'skeleton/positions' in traj_group and 'skeleton/confidences' in traj_group:
                skeleton_positions = traj_group['skeleton/positions'][frame_idx]
                skeleton_confidences = traj_group['skeleton/confidences'][frame_idx]
                result['skeleton_positions'] = torch.from_numpy(skeleton_positions).float()
                result['skeleton_confidences'] = torch.from_numpy(skeleton_confidences).float()
                
                # Load joint names if available
                if 'joint_names' in traj_group['skeleton'].attrs:
                    result['skeleton_joint_names'] = traj_group['skeleton'].attrs['joint_names']
            
            # Apply transform if provided
            if self.transform is not None:
                result = self.transform(result)
            
            return result
    
    def get_trajectory_info(self) -> Dict[str, Dict]:
        """
        Get information about all trajectories in the dataset.
        
        Returns:
            Dictionary mapping trajectory names to their metadata
        """
        info = {}
        
        with h5py.File(self.h5_path, 'r') as f:
            for traj_name in f.keys():
                if not traj_name.startswith('trajectory_'):
                    continue
                
                traj_group = f[traj_name]
                traj_info = {}
                
                # Get frame count
                if 'camera/pose_in_world' in traj_group:
                    traj_info['num_frames'] = traj_group['camera/pose_in_world'].shape[0]
                
                # Get metadata attributes
                if 'metadata' in traj_group:
                    metadata_group = traj_group['metadata']
                    for key, value in metadata_group.attrs.items():
                        traj_info[key] = value
                
                info[traj_name] = traj_info
        
        return info 