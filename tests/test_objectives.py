import torch
import pytest
from egohub.training.objectives import VAELoss

@pytest.fixture
def vae_loss():
    """Provides a default instance of the VAELoss function."""
    return VAELoss(kl_weight=1.0)

def test_kl_divergence_zero_when_perfect(vae_loss):
    """
    Tests that the KL divergence is zero when the latent distribution
    perfectly matches the prior (a standard normal distribution).
    """
    batch_size = 4
    latent_dim = 10
    
    # mu=0 and log_var=0 (i.e., variance=1) corresponds to N(0, 1)
    mu = torch.zeros(batch_size, latent_dim)
    log_var = torch.zeros(batch_size, latent_dim)
    
    model_output = {
        'mu': mu,
        'log_var': log_var,
        # Dummy values for other keys
        'image_recon': torch.zeros(batch_size, 3, 32, 32),
        'pose_recon': torch.zeros(batch_size, 5)
    }
    
    # Dummy target data, required by the forward pass
    batch_data = {
        'rgb': {'cam1': torch.zeros(batch_size, 3, 32, 32)},
        'skeleton_positions': torch.zeros(batch_size, 5)
    }

    # The KL divergence part of the loss should be zero
    loss_dict = vae_loss(model_output, batch_data)
    
    assert torch.isclose(loss_dict['kl_divergence_loss'], torch.tensor(0.0)), \
        "KL divergence should be zero for a perfect N(0,1) latent space."

def test_reconstruction_loss_correctness(vae_loss):
    """
    Tests that the reconstruction loss is calculated correctly using simple inputs.
    """
    batch_size = 2
    
    # Create simple, non-zero data
    image_recon = torch.ones(batch_size, 1, 2, 2) * 0.5
    image_target = torch.ones(batch_size, 1, 2, 2)

    pose_recon = torch.ones(batch_size, 5) * 2.0
    pose_target = torch.ones(batch_size, 5) * 3.0

    model_output = {
        'image_recon': image_recon,
        'pose_recon': pose_recon,
        # Dummy values for KL part
        'mu': torch.zeros(batch_size, 10),
        'log_var': torch.zeros(batch_size, 10),
    }
    
    batch_data = {
        'rgb': {'cam1': image_target},
        'skeleton_positions': pose_target
    }
    
    loss_dict = vae_loss(model_output, batch_data)

    # Manual calculation
    # Image MSE = (0.5 - 1.0)^2 * (2*1*2*2 elements) = 0.25 * 8 = 2.0
    expected_image_loss = torch.tensor(2.0)
    # Pose L1 = |2.0 - 3.0| * (2*5 elements) = 1.0 * 10 = 10.0
    expected_pose_loss = torch.tensor(10.0)
    
    # Total reconstruction loss, averaged over batch size
    expected_total_recon = (expected_image_loss + expected_pose_loss) / batch_size
    
    assert torch.isclose(loss_dict['reconstruction_loss'], expected_total_recon) 