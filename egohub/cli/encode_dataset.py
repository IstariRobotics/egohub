import argparse
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import Resize
from tqdm import tqdm

from egohub.datasets import EgocentricH5Dataset
from egohub.models.vae import MultimodalVAE


def main():
    parser = argparse.ArgumentParser(
        description="Encode a dataset using a trained Multimodal VAE."
    )
    parser.add_argument(
        "h5_path", type=Path, help="Path to the canonical HDF5 file to encode."
    )
    parser.add_argument(
        "checkpoint_path", type=Path, help="Path to the trained VAE model checkpoint."
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Dimensionality of the latent space (must match trained model).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for encoding."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for encoding.",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Dataset and DataLoader
    transform = Resize((64, 64))
    dataset = EgocentricH5Dataset(args.h5_path, transform=transform)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Load Model
    sample = dataset[0]
    pose_input_dim = sample["skeleton_positions"].flatten().shape[0]

    model = MultimodalVAE(
        image_feature_dim=256,
        pose_feature_dim=128,
        latent_dim=args.latent_dim,
        pose_input_dim=pose_input_dim,
    )
    model.load_state_dict(torch.load(args.checkpoint_path, map_location=device))
    model.to(device)
    model.eval()

    # 3. Encoding Loop
    print(f"Starting dataset encoding on device: {device}")

    # This dictionary will store latent vectors per trajectory
    trajectory_latents = {}

    with torch.no_grad():
        for batch_data in tqdm(dataloader, desc="Encoding dataset"):
            traj_names = batch_data["trajectory_name"]

            # Move data to device
            batch_data_device = {
                "rgb": {k: v.to(device) for k, v in batch_data["rgb"].items()},
                "skeleton_positions": batch_data["skeleton_positions"].to(device),
            }

            mu, _ = model.encode(
                next(iter(batch_data_device["rgb"].values())),
                batch_data_device["skeleton_positions"].flatten(1),
            )

            # Store latents grouped by trajectory
            for i, traj_name in enumerate(traj_names):
                if traj_name not in trajectory_latents:
                    trajectory_latents[traj_name] = []
                trajectory_latents[traj_name].append(mu[i].cpu().numpy())

    # 4. Save Latents to HDF5
    print("Saving latent vectors to HDF5 file...")
    with h5py.File(args.h5_path, "a") as f:
        for traj_name, latents in tqdm(
            trajectory_latents.items(), desc="Writing to HDF5"
        ):
            if traj_name in f:
                traj_group = f[traj_name]
                latent_group = traj_group.require_group("latent")

                # Convert list of latents to a single numpy array
                latent_array = np.array(latents)

                if "mean" in latent_group:
                    del latent_group["mean"]  # Overwrite if exists
                latent_group.create_dataset("mean", data=latent_array)
            else:
                print(
                    f"Warning: Trajectory '{traj_name}' not found in HDF5 file. "
                    "Skipping."
                )

    print("Dataset encoding complete.")


if __name__ == "__main__":
    main()
