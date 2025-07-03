import torch
import torch.nn as nn
from egohub.models.encoders import ImageEncoder, PoseEncoder

class MultimodalVAE(nn.Module):
    """
    A Multimodal Variational Autoencoder that combines image and pose data.
    """
    def __init__(self, 
                 image_feature_dim: int, 
                 pose_feature_dim: int,
                 latent_dim: int, 
                 pose_input_dim: int,
                 image_channels: int = 3):
        super().__init__()
        
        # --- Encoders ---
        self.image_encoder = ImageEncoder(in_channels=image_channels, out_features=image_feature_dim)
        self.pose_encoder = PoseEncoder(in_features=pose_input_dim, out_features=pose_feature_dim)
        
        # --- Fusion and Latent Mapping ---
        fused_dim = image_feature_dim + pose_feature_dim
        self.fc_mu = nn.Linear(fused_dim, latent_dim)
        self.fc_log_var = nn.Linear(fused_dim, latent_dim)
        
        # --- Decoder ---
        self.decoder_input = nn.Linear(latent_dim, fused_dim)
        
        # --- Decoders for each modality ---
        self.image_decoder = nn.Sequential(
            nn.Linear(image_feature_dim, 256 * 4 * 4),
            nn.ReLU(),
            nn.Unflatten(1, (256, 4, 4)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()  # To output pixel values between 0 and 1
        )
        
        self.pose_decoder = nn.Sequential(
            nn.Linear(pose_feature_dim, 256),
            nn.ReLU(),
            nn.Linear(256, pose_input_dim)
        )
        
    def encode(self, image, pose):
        image_h = self.image_encoder(image)
        pose_h = self.pose_encoder(pose)
        fused_h = torch.cat([image_h, pose_h], dim=1)
        return self.fc_mu(fused_h), self.fc_log_var(fused_h)
        
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std
        
    def decode(self, z):
        h_decoded = self.decoder_input(z)
        
        # Assuming the first part of h_decoded is for the image and the second for the pose
        image_h_decoded = h_decoded[:, :self.image_encoder.final_layer.out_features]
        pose_h_decoded = h_decoded[:, self.image_encoder.final_layer.out_features:]
        
        image_recon = self.image_decoder(image_h_decoded)
        pose_recon = self.pose_decoder(pose_h_decoded)
        
        return image_recon, pose_recon

    def forward(self, batch_data):
        # Assuming batch_data is a dictionary from our dataset
        # For simplicity, we'll take the first camera's RGB
        # and a concatenated pose vector
        
        # This part will need careful handling of the input data structure
        # from the DataLoader. This is a placeholder for the logic.
        image_tensor = next(iter(batch_data['rgb'].values()))
        
        # Flatten and concatenate all available poses for now
        pose_tensors = []
        if 'skeleton_positions' in batch_data:
            pose_tensors.append(batch_data['skeleton_positions'].flatten(1))
        # Add hand poses if they exist, assuming they are flattened similarly
        
        if not pose_tensors:
            raise ValueError("No pose data found in the batch.")
            
        pose_tensor = torch.cat(pose_tensors, dim=1)

        mu, log_var = self.encode(image_tensor, pose_tensor)
        z = self.reparameterize(mu, log_var)
        image_recon, pose_recon = self.decode(z)
        
        return {
            'image_recon': image_recon,
            'pose_recon': pose_recon,
            'mu': mu,
            'log_var': log_var
        } 