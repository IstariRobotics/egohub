import torch
import torch.nn as nn
import torch.nn.functional as functional


class VAELoss(nn.Module):
    """
    Computes the VAE loss, which is a sum of reconstruction loss and KL
    divergence loss.
    """

    def __init__(self, kl_weight: float = 1.0):
        super().__init__()
        self.kl_weight = kl_weight

    def forward(self, model_output: dict, batch_data: dict) -> dict:
        """
        Args:
            model_output (dict): The output of the VAE model, containing
                reconstructions and latent variables.
            batch_data (dict): The original input batch, used as the target for
                reconstruction.

        Returns:
            dict: A dictionary containing the total loss, reconstruction loss,
                and KL divergence loss.
        """
        # For simplicity, we'll take the first camera's RGB as the target
        image_target = next(iter(batch_data["rgb"].values()))

        # The pose target needs to be flattened and concatenated in the same way
        # as in the model's forward pass
        pose_tensors = []
        if "skeleton_positions" in batch_data:
            pose_tensors.append(batch_data["skeleton_positions"].flatten(1))
        pose_target = torch.cat(pose_tensors, dim=1)

        # 1. Reconstruction Loss
        recon_loss_image = functional.mse_loss(
            model_output["image_recon"], image_target, reduction="sum"
        )
        recon_loss_pose = functional.l1_loss(
            model_output["pose_recon"], pose_target, reduction="sum"
        )
        recon_loss = recon_loss_image + recon_loss_pose

        # 2. KL Divergence Loss
        mu = model_output["mu"]
        log_var = model_output["log_var"]
        kl_div = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

        # Total Loss
        total_loss = (recon_loss + self.kl_weight * kl_div) / image_target.size(
            0
        )  # Average by batch size

        return {
            "total_loss": total_loss,
            "reconstruction_loss": recon_loss / image_target.size(0),
            "kl_divergence_loss": kl_div / image_target.size(0),
        }
