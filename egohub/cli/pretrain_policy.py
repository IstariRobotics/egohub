import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from egohub.datasets import LatentSequenceDataset
from egohub.models.policy_model import LatentPolicyModel


def get_causal_mask(size: int) -> torch.Tensor:
    """
    Generates a causal mask for a sequence of a given size.
    """
    mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
    mask = (
        mask.float()
        .masked_fill(mask == 0, float("-inf"))
        .masked_fill(mask == 1, float(0.0))
    )
    return mask


def main():
    parser = argparse.ArgumentParser(description="Pre-train a Latent Policy Model.")
    parser.add_argument(
        "h5_path", type=Path, help="Path to the HDF5 file with latent vectors."
    )
    parser.add_argument(
        "--sequence-length", type=int, default=16, help="Sequence length for training."
    )
    parser.add_argument(
        "--latent-dim",
        type=int,
        default=128,
        help="Dimensionality of the latent space.",
    )
    parser.add_argument(
        "--nhead",
        type=int,
        default=4,
        help="Number of attention heads in the Transformer.",
    )
    parser.add_argument(
        "--d-hid",
        type=int,
        default=256,
        help="Dimension of the feedforward network in the Transformer.",
    )
    parser.add_argument(
        "--nlayers", type=int, default=3, help="Number of Transformer encoder layers."
    )
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate.")
    parser.add_argument(
        "--epochs", type=int, default=10, help="Number of training epochs."
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training."
    )
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate.")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device for training.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("policy_checkpoint.pth"),
        help="Path to save the trained model checkpoint.",
    )

    args = parser.parse_args()
    device = torch.device(args.device)

    # 1. Dataset and DataLoader
    dataset = LatentSequenceDataset(
        args.h5_path, sequence_length=args.sequence_length + 1
    )  # +1 for target
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    # 2. Model, Loss, and Optimizer
    model = LatentPolicyModel(
        latent_dim=args.latent_dim,
        nhead=args.nhead,
        d_hid=args.d_hid,
        nlayers=args.nlayers,
        dropout=args.dropout,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # 3. Training Loop
    print(f"Starting pre-training on device: {device}")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for batch in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            # Prepare data: input is sequence[:-1], target is sequence[1:]
            sequences = batch["sequence"].to(device)
            inputs = sequences[:, :-1, :]
            targets = sequences[:, 1:, :]

            # Causal mask to prevent attending to future tokens
            mask = get_causal_mask(inputs.size(1)).to(device)

            optimizer.zero_grad()

            predictions = model(inputs, src_mask=mask)

            loss = criterion(predictions, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch + 1}/{args.epochs}, Average Loss: {avg_loss:.4f}")

    # 4. Save Model
    print(f"Pre-training complete. Saving model to {args.output_path}")
    torch.save(model.state_dict(), args.output_path)
    print("Model saved.")


if __name__ == "__main__":
    main()
