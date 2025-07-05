import pytest
import h5py
import numpy as np
import uuid
from pathlib import Path

from egohub.constants import CANONICAL_SKELETON_JOINTS


@pytest.fixture
def hdf5_file_factory(tmpdir_factory):
    """
    A pytest fixture that returns a factory function for creating temporary,
    valid HDF5 files for testing.

    The factory creates an HDF5 file with a structure matching the canonical
    schema, populated with synthetic data.

    Usage:
        def test_something(hdf5_file_factory):
            # Create a file with 1 trajectory and 10 frames
            hdf5_path = hdf5_file_factory(num_trajectories=1, num_frames=10)
            # ... use the file ...
    """
    def _create_file(
        num_trajectories: int = 1,
        num_frames: int = 5,
        add_video: bool = True,
        add_skeleton: bool = True,
    ) -> Path:
        # Create a temporary file path
        temp_dir = Path(tmpdir_factory.mktemp("h5_data"))
        file_path = temp_dir / "test_file.h5"

        num_joints = len(CANONICAL_SKELETON_JOINTS)

        with h5py.File(file_path, "w") as f:
            for i in range(num_trajectories):
                traj_group = f.create_group(f"trajectory_{i:04d}")

                # --- Metadata ---
                meta_group = traj_group.create_group("metadata")
                meta_group.attrs["uuid"] = str(uuid.uuid4())
                meta_group.attrs["source_dataset"] = "synthetic"
                meta_group.attrs["source_identifier"] = f"synthetic_traj_{i}"
                meta_group.attrs["action_label"] = "testing"
                meta_group.create_dataset(
                    "timestamps_ns", data=np.arange(num_frames, dtype=np.uint64) * 1_000_000
                )

                # --- Cameras ---
                cameras_group = traj_group.create_group("cameras")
                cam1_group = cameras_group.create_group("camera_0")
                cam1_group.attrs["is_ego"] = True
                cam1_group.create_dataset("intrinsics", data=np.eye(3, dtype=np.float32))
                
                # Create sample poses (identity matrices)
                identity_poses = np.tile(np.eye(4, dtype=np.float32), (num_frames, 1, 1))
                cam1_group.create_dataset("pose_in_world", data=identity_poses)
                cam1_group.create_dataset("pose_indices", data=np.arange(num_frames, dtype=np.uint64))

                if add_video:
                    rgb_group = cam1_group.create_group("rgb")
                    # Store dummy byte data for images
                    dummy_frame_data = [f"frame_{j}".encode('utf-8') for j in range(num_frames)]
                    rgb_group.create_dataset("image_bytes", data=dummy_frame_data, dtype=h5py.special_dtype(vlen=bytes))
                    rgb_group.create_dataset("frame_sizes", data=np.array([len(d) for d in dummy_frame_data], dtype=np.int32))
                    rgb_group.create_dataset("frame_indices", data=np.arange(num_frames, dtype=np.uint64))

                # --- Hands ---
                hands_group = traj_group.create_group("hands")
                left_hand_group = hands_group.create_group("left_hand")
                left_hand_group.create_dataset("pose_in_world", data=identity_poses)
                left_hand_group.create_dataset("pose_indices", data=np.arange(num_frames, dtype=np.uint64))

                # --- Skeleton ---
                if add_skeleton:
                    skeleton_group = traj_group.create_group("skeleton")
                    skeleton_group.create_dataset(
                        "positions", data=np.random.rand(num_frames, num_joints, 3).astype(np.float32)
                    )
                    skeleton_group.create_dataset(
                        "confidences", data=np.random.rand(num_frames, num_joints).astype(np.float32)
                    )
                    skeleton_group.create_dataset("frame_indices", data=np.arange(num_frames, dtype=np.uint64))

        return file_path

    return _create_file
