import h5py
import numpy as np
from typing import Optional, Tuple, List

class SkeletonProcessor:
    """
    A component for processing skeleton data from a source HDF5 file.
    """
    def get_skeleton_data(self, f_in: h5py.File, num_frames: int) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
        """
        Extracts raw skeleton data (joint names, positions, and confidences).

        Args:
            f_in (h5py.File): The open source HDF5 file.
            num_frames (int): The number of frames in the sequence.

        Returns:
            An optional tuple containing:
            - A list of joint names.
            - An array of joint positions.
            - An array of joint confidences.
            Returns None if the required groups are not found.
        """
        transforms_group = f_in.get('transforms')
        confidences_group = f_in.get('confidences')

        if not (isinstance(transforms_group, h5py.Group) and isinstance(confidences_group, h5py.Group)):
            return None

        joint_names = sorted([name for name in transforms_group.keys() if name != 'camera'])
        if not joint_names:
            return None

        positions_list, confidences_list = [], []
        for joint_name in joint_names:
            joint_transform = transforms_group.get(joint_name)
            if isinstance(joint_transform, h5py.Dataset):
                positions_list.append(joint_transform[:, :3, 3])
            else:
                positions_list.append(np.zeros((num_frames, 3), dtype=np.float32))
            
            joint_conf = confidences_group.get(joint_name)
            if isinstance(joint_conf, h5py.Dataset):
                confidences_list.append(joint_conf[:])
            else:
                confidences_list.append(np.zeros(num_frames, dtype=np.float32))
        
        if not (positions_list and confidences_list):
            return None
            
        return joint_names, np.stack(positions_list, axis=1), np.stack(confidences_list, axis=1) 