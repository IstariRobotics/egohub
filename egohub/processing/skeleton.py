from typing import Dict, List, Optional, Tuple

import h5py
import numpy as np

from egohub.constants import CANONICAL_SKELETON_JOINTS


class SkeletonProcessor:
    """
    A component for processing skeleton data from a source HDF5 file.
    """

    def get_skeleton_data(
        self, f_in: h5py.File, num_frames: int
    ) -> Optional[Tuple[List[str], np.ndarray, np.ndarray]]:
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
        transforms_group = f_in.get("transforms")
        confidences_group = f_in.get("confidences")

        if not (
            isinstance(transforms_group, h5py.Group)
            and isinstance(confidences_group, h5py.Group)
        ):
            return None

        joint_names = sorted(
            [name for name in transforms_group.keys() if name != "camera"]
        )
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

        return (
            joint_names,
            np.stack(positions_list, axis=1),
            np.stack(confidences_list, axis=1),
        )


def remap_skeleton(
    source_positions: np.ndarray,
    source_confidences: np.ndarray,
    source_joint_names: List[str],
    joint_map: Optional[Dict[str, str]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Remaps a source skeleton to the canonical skeleton format.

    This function takes a skeleton with an arbitrary joint definition and maps it
    to the `CANONICAL_SKELETON_JOINTS` standard. It creates new positions and
    confidences arrays that match the canonical joint order.

    - If a canonical joint exists in the source skeleton (or is in the joint_map),
      its position and confidence are copied.
    - If a canonical joint does not exist in the source, its position is filled
      with NaNs and its confidence with 0.
    - Any joints in the source skeleton that are not in the canonical definition
      are ignored.

    Args:
        source_positions: A (num_frames, num_source_joints, 3) numpy array.
        source_confidences: A (num_frames, num_source_joints) numpy array.
        source_joint_names: A list of joint names corresponding to the second
                            dimension of `source_positions`.
        joint_map: An optional dictionary mapping source joint names to
                   canonical joint names.

    Returns:
        A tuple containing:
        - canonical_positions (np.ndarray): An array of shape
          (num_frames, num_canonical_joints, 3) with remapped positions.
        - canonical_confidences (np.ndarray): An array of shape
          (num_frames, num_canonical_joints) with remapped confidences.
    """
    num_frames = source_positions.shape[0]
    num_canonical_joints = len(CANONICAL_SKELETON_JOINTS)
    canonical_joint_indices = {
        name: i for i, name in enumerate(CANONICAL_SKELETON_JOINTS)
    }

    # Create a mapping from source joint names to their indices.
    source_joint_indices = {name: i for i, name in enumerate(source_joint_names)}

    # Initialize the new arrays.
    canonical_positions = np.full(
        (num_frames, num_canonical_joints, 3), np.nan, dtype=np.float32
    )
    canonical_confidences = np.zeros(
        (num_frames, num_canonical_joints), dtype=np.float32
    )

    if joint_map:
        # Use the provided map for remapping
        for source_name, canonical_name in joint_map.items():
            if (
                source_name in source_joint_indices
                and canonical_name in canonical_joint_indices
            ):
                source_idx = source_joint_indices[source_name]
                canonical_idx = canonical_joint_indices[canonical_name]
                canonical_positions[:, canonical_idx, :] = source_positions[
                    :, source_idx, :
                ]
                canonical_confidences[:, canonical_idx] = source_confidences[
                    :, source_idx
                ]
    else:
        # Fallback to direct name matching
        for i, joint_name in enumerate(CANONICAL_SKELETON_JOINTS):
            if joint_name in source_joint_indices:
                source_index = source_joint_indices[joint_name]
                canonical_positions[:, i, :] = source_positions[:, source_index, :]
                canonical_confidences[:, i] = source_confidences[:, source_index]

    return canonical_positions, canonical_confidences
