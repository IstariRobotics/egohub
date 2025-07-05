import h5py
import pytest
from egohub.schema import Trajectory, validate_hdf5_with_schema

def test_hdf5_file_factory_creation_and_validation(hdf5_file_factory):
    """
    Tests that the hdf5_file_factory creates a valid file that passes
    our own schema validation. This ensures the fixture stays in sync.
    """
    # 1. Create a file using the factory with default parameters
    hdf5_path = hdf5_file_factory()

    assert hdf5_path.exists(), "Factory should create a file"

    # 2. Open the file and validate it against the schema
    with h5py.File(hdf5_path, "r") as f:
        # Check basic structure
        assert "trajectory_0000" in f, "Default trajectory group should exist"
        
        # Run strict validation
        try:
            validate_hdf5_with_schema(
                h5_group=f["trajectory_0000"],
                schema_cls=Trajectory,
                strict=True  # Fail the test on any validation error
            )
        except Exception as e:
            pytest.fail(f"HDF5 file from factory failed validation: {e}")

def test_hdf5_factory_parametrization(hdf5_file_factory):
    """
    Tests that the factory correctly handles different parameters.
    """
    # Create a file with more trajectories and frames
    hdf5_path = hdf5_file_factory(num_trajectories=2, num_frames=10, add_skeleton=False)

    with h5py.File(hdf5_path, "r") as f:
        assert "trajectory_0001" in f, "Should create the specified number of trajectories"
        assert len(f["trajectory_0001/metadata/timestamps_ns"]) == 10, "Should create the specified number of frames"
        assert "skeleton" not in f["trajectory_0001"], "Should respect the 'add_skeleton' flag" 