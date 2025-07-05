from typing import Optional

import h5py
import numpy as np


class HandProcessor:
    """
    A component for processing hand data from a source HDF5 file.
    """

    def get_hand_poses(
        self, source_group: h5py.Group, hand: str
    ) -> Optional[np.ndarray]:
        """
        Extracts raw hand poses for a specific hand.

        Args:
            source_group (h5py.Group): The source HDF5 group (e.g., f_in['transforms']).
            hand (str): Either 'left' or 'right'.

        Returns:
            Optional[np.ndarray]: The raw 4x4 pose matrices, or None if not found.
        """
        source_key = f"{hand.title()}Hand"
        hand_transforms = source_group.get(source_key)
        if isinstance(hand_transforms, h5py.Dataset):
            return hand_transforms[:]
        return None
