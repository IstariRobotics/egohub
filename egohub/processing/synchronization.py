import numpy as np
from scipy.spatial import cKDTree


def generate_indices(
    master_timestamps_ns: np.ndarray, stream_timestamps_ns: np.ndarray
) -> np.ndarray:
    """
    Generates indices that map a data stream to a master timestamp array.

    For each timestamp in the stream, this function finds the index of the
    closest timestamp in the master array. This is a robust way to synchronize
    data streams that may have slightly different timings or dropped frames.

    Args:
        master_timestamps_ns: A 1D numpy array of master timestamps (uint64).
        stream_timestamps_ns: A 1D numpy array of timestamps for the specific
                              data stream (uint64).

    Returns:
        A 1D numpy array of the same length as `stream_timestamps_ns`, where
        each value is the index into `master_timestamps_ns`.
    """
    if master_timestamps_ns.ndim != 1 or stream_timestamps_ns.ndim != 1:
        raise ValueError("Timestamp arrays must be 1-dimensional.")

    # Reshape for use with cKDTree, which expects 2D arrays
    master_reshaped = master_timestamps_ns.reshape(-1, 1)
    stream_reshaped = stream_timestamps_ns.reshape(-1, 1)

    # Build a k-d tree for efficient nearest-neighbor lookup
    tree = cKDTree(master_reshaped)

    # Query the tree to find the index of the nearest neighbor for each stream timestamp
    _, indices = tree.query(stream_reshaped, k=1)

    return indices.astype(np.uint64)
