import numpy as np
from egohub.transforms.coordinates import (
    arkit_to_canonical_transform,
    transform_pose
)

def test_arkit_to_canonical_transform_properties():
    """
    Tests that the ARKit transform matrix has the correct shape and rotation.
    """
    transform = arkit_to_canonical_transform()
    assert transform.shape == (4, 4), "Transform matrix should be 4x4"

    # The transform should be a +90 degree rotation around the X-axis.
    # Applying this to the Y-axis should result in the Z-axis.
    y_axis = np.array([0, 1, 0])
    transformed_y = transform[:3, :3] @ y_axis
    z_axis = np.array([0, 0, 1])
    assert np.allclose(transformed_y, z_axis), "Y-axis should transform to Z-axis"

    # Applying this to the Z-axis should result in the -Y-axis.
    z_axis = np.array([0, 0, 1])
    transformed_z = transform[:3, :3] @ z_axis
    neg_y_axis = np.array([0, -1, 0])
    assert np.allclose(transformed_z, neg_y_axis), "Z-axis should transform to -Y-axis"


def test_pose_transformation_round_trip():
    """
    Tests that transforming a pose from ARKit to canonical and back yields the
    original pose. This confirms the transformation is bijective.
    """
    # Create a sample pose (e.g., identity)
    original_pose = np.eye(4)
    original_pose[0, 3] = 1.0  # Add some translation

    # Get the forward and inverse transformations
    arkit_to_canonical = arkit_to_canonical_transform()
    canonical_to_arkit = np.linalg.inv(arkit_to_canonical)

    # Forward transformation
    canonical_pose = transform_pose(original_pose, arkit_to_canonical)

    # Inverse (round-trip) transformation
    round_trip_pose = transform_pose(canonical_pose, canonical_to_arkit)

    # The final pose should be identical to the original
    assert np.allclose(original_pose, round_trip_pose, atol=1e-7) 