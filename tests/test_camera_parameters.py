import numpy as np
import pytest

from egohub.camera_parameters import Extrinsics


def test_extrinsics_inversion_from_world_to_cam():
    """
    Tests that given a world-to-camera transform, the computed camera-to-world
    transform is its correct inverse.
    """
    # Simple 90-degree rotation around Z axis
    world_r_cam = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    world_t_cam = np.array([1, 2, 3])

    extrinsics = Extrinsics(world_r_cam=world_r_cam, world_t_cam=world_t_cam)

    # The computed inverse transform should, when multiplied by the original,
    # yield the identity matrix.
    identity_matrix = extrinsics.world_h_cam @ extrinsics.cam_h_world
    assert np.allclose(identity_matrix, np.eye(4))


def test_extrinsics_inversion_from_cam_to_world():
    """
    Tests that given a camera-to-world transform, the computed world-to-camera
    transform is its correct inverse.
    """
    # Simple 180-degree rotation around Y axis
    cam_r_world = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    cam_t_world = np.array([-5, 0, 10])

    extrinsics = Extrinsics(cam_r_world=cam_r_world, cam_t_world=cam_t_world)

    identity_matrix = extrinsics.cam_h_world @ extrinsics.world_h_cam
    assert np.allclose(identity_matrix, np.eye(4))


def test_extrinsics_raises_error_if_insufficient_info():
    """
    Tests that Extrinsics raises a ValueError if neither transform direction
    is provided during initialization.
    """
    with pytest.raises(ValueError):
        Extrinsics()

    with pytest.raises(ValueError):
        Extrinsics(world_r_cam=np.eye(3))  # Missing translation

    with pytest.raises(ValueError):
        Extrinsics(cam_t_world=np.zeros(3))  # Missing rotation
