import pytest
import torch

from egohub.models.encoders import ImageEncoder, PoseEncoder


@pytest.mark.parametrize("batch_size, out_features", [(1, 256), (4, 128)])
def test_image_encoder_output_shape(batch_size, out_features):
    """
    Tests that the ImageEncoder runs and produces an output of the correct shape.
    """
    in_channels = 3
    height, width = 224, 224

    model = ImageEncoder(in_channels=in_channels, out_features=out_features)
    # Create a random input tensor on the CPU
    dummy_input = torch.randn(batch_size, in_channels, height, width)

    # Forward pass
    output = model(dummy_input)

    # Check output shape
    expected_shape = (batch_size, out_features)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"


@pytest.mark.parametrize(
    "batch_size, in_features, out_features",
    [(1, 204, 128), (8, 100, 64)],  # e.g., 68 landmarks * 3 coords
)
def test_pose_encoder_output_shape(batch_size, in_features, out_features):
    """
    Tests that the PoseEncoder runs and produces an output of the correct shape.
    """
    model = PoseEncoder(in_features=in_features, out_features=out_features)
    # Create a random flattened input tensor
    dummy_input = torch.randn(batch_size, in_features)

    # Forward pass
    output = model(dummy_input)

    # Check output shape
    expected_shape = (batch_size, out_features)
    assert (
        output.shape == expected_shape
    ), f"Expected output shape {expected_shape}, but got {output.shape}"
