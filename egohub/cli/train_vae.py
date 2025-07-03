import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
from tqdm import tqdm
from torchvision.transforms import Resize

from egohub.datasets import EgocentricH5Dataset
from egohub.models.vae import MultimodalVAE
from egohub.training.objectives import VAELoss
from egohub.constants import AVP_ID2NAME

def main():
    parser = argparse.ArgumentParser(description="Train a Multimodal VAE on egocentric data.")
    parser.add_argument("h5_path", type=Path, help="Path to the canonical HDF5 file.")
    parser.add_argument("--latent-dim", type=int, default=128, help="Dimensionality of the latent space.")
    parser.add_argument("--num-epochs", type=int, default=20, help="Number of training epochs.")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for training.")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate for the optimizer.")
    parser.add_argument("--kl-weight", type=float, default=1.0, help="Weight for the KL divergence term in the loss.")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device for training.")
    parser.add_argument("--save-path", type=Path, default="vae_checkpoint.pth", help="Path to save the model checkpoint.")
    
    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Dataset and DataLoader
    transform = Resize((64, 64))
    dataset = EgocentricH5Dataset(args.h5_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)

    # 2. Model, Loss, and Optimizer
    # Define dimensions based on one sample from the dataset
    sample = dataset[0]
    pose_input_dim = sample['skeleton_positions'].flatten().shape[0] # Add hand poses if used
    
    model = MultimodalVAE(
        image_feature_dim=256,
        pose_feature_dim=128,
        latent_dim=args.latent_dim,
        pose_input_dim=pose_input_dim
    ).to(device)
    
    criterion = VAELoss(kl_weight=args.kl_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

    # 3. Training Loop
    print(f"Starting VAE training on device: {device}")
    for epoch in range(args.num_epochs):
        model.train()
        total_train_loss = 0
        
        for batch_data in tqdm(dataloader, desc=f"Epoch {epoch+1}/{args.num_epochs}"):
            # Move data to device
            batch_data['rgb'] = {k: v.to(device) for k, v in batch_data['rgb'].items()}
            batch_data['skeleton_positions'] = batch_data['skeleton_positions'].to(device)
            
            optimizer.zero_grad()
            
            model_output = model(batch_data)
            loss_dict = criterion(model_output, batch_data)
            loss = loss_dict['total_loss']
            
            loss.backward()
            optimizer.step()
            
            total_train_loss += loss.item()
            
        avg_train_loss = total_train_loss / len(dataloader)
        print(f"Epoch {epoch+1} finished. Average Loss: {avg_train_loss:.4f}")

    # 4. Save Model
    torch.save(model.state_dict(), args.save_path)
    print(f"Model checkpoint saved to {args.save_path}")

if __name__ == "__main__":
    main() 