import argparse
import logging
from pathlib import Path
from typing import Literal

import h5py
import numpy as np
import torch
from PIL import Image
from scipy.signal import medfilt, argrelextrema
from torchvision import transforms as T
from tqdm import tqdm


# --- Model Loading (Re-implementing UVD's preprocessor logic) ---

def get_preprocessor(
    name: Literal["dinov2", "vip"], device: str
) -> tuple[torch.nn.Module, T.Compose]:
    """
    Loads a pre-trained model and its corresponding pre-processing transform.
    This is a simplified re-implementation of the logic in the UVD library.
    """
    logging.info(f"Loading preprocessor: '{name}' on device: '{device}'")
    if name == "dinov2":
        # DINOv2 is loaded from torch.hub
        model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14")
        model = model.to(device=device)
        # The model's transform only handles normalization. Resizing must be done separately.
        transform = T.Normalize(
            mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)
        )
        return model, transform
    elif name == "vip":
        # VIP requires its own library, which is a more complex dependency.
        # For this example, we'll raise a NotImplementedError and show what's needed.
        raise NotImplementedError(
            "Loading the 'vip' model requires the 'vip' library and its specific "
            "model loading functions. This can be added by installing the library, e.g., "
            "'pip install vip-pytorch'."
        )
    else:
        raise NotImplementedError(f"Preprocessor '{name}' is not supported.")


# --- HDF5 and Video Decoding ---

def decode_video_from_hdf5(rgb_group: h5py.Group) -> np.ndarray:
    """Decodes a video from HDF5 raw bytes and converts it to a NumPy array."""
    image_bytes_dataset = rgb_group["image_bytes"]
    frame_sizes_dataset = rgb_group["frame_sizes"]
    
    frames = []
    for i in range(len(frame_sizes_dataset)):
        frame_bytes = image_bytes_dataset[i, : frame_sizes_dataset[i]].tobytes()
        from io import BytesIO
        img = Image.open(BytesIO(frame_bytes)).convert("RGB")
        frames.append(np.array(img))
    
    return np.stack(frames)


# --- Subgoal Decomposition Logic ---

def find_subgoals_by_derivative(
    embeddings: np.ndarray,
    derivative_threshold: float,
    window_length: int,
) -> tuple[list[int], np.ndarray]:
    """Identifies subgoals from a sequence of embeddings based on the derivative of the distance curve."""
    traj_length = embeddings.shape[0]
    subgoal_indices = []
    cur_subgoal_idx = traj_length - 1

    while cur_subgoal_idx > 15:
        sub_trajectory_embeddings = embeddings[: cur_subgoal_idx + 1]
        goal_embedding = embeddings[cur_subgoal_idx]
        distances = np.linalg.norm(sub_trajectory_embeddings - goal_embedding, axis=1)
        initial_distance = np.linalg.norm(embeddings[0] - goal_embedding)
        if initial_distance > 1e-6:
            distances /= initial_distance

        smoothed_distances = medfilt(distances, kernel_size=None)
        slope = np.gradient(smoothed_distances)
        significant_change_indices = np.where(np.abs(slope) >= derivative_threshold)[0]

        if len(significant_change_indices) == 0:
            # Return the full distance curve for logging, even if we break early
            final_distances = np.linalg.norm(embeddings - embeddings[-1], axis=1)
            if np.linalg.norm(embeddings[0] - embeddings[-1]) > 1e-6:
                final_distances /= np.linalg.norm(embeddings[0] - embeddings[-1])
            return sorted(list(set(subgoal_indices))), final_distances
        
        diffs = np.diff(significant_change_indices)
        break_indices = np.where(diffs > window_length + 1)[0]
        end_positions = significant_change_indices[
            np.concatenate((break_indices, [len(significant_change_indices) - 1]))
        ]
        
        subgoal_indices.append(cur_subgoal_idx)
        if len(end_positions) < 2 or end_positions[-2] < 15:
            break
        cur_subgoal_idx = end_positions[-2]
    
    if 0 not in subgoal_indices:
        subgoal_indices.append(0)
    
    final_distances = np.linalg.norm(embeddings - embeddings[-1], axis=1)
    if np.linalg.norm(embeddings[0] - embeddings[-1]) > 1e-6:
        final_distances /= np.linalg.norm(embeddings[0] - embeddings[-1])
    return sorted(list(set(subgoal_indices))), final_distances


def find_subgoals_by_peaks(
    embeddings: np.ndarray, 
    min_interval: int
) -> tuple[list[int], np.ndarray]:
    """Identifies subgoals by finding peaks in the distance curve."""
    traj_length = embeddings.shape[0]
    subgoal_indices = [traj_length - 1]
    cur_goal_idx = traj_length - 1
    
    final_distances = np.linalg.norm(embeddings - embeddings[-1], axis=1)
    if np.linalg.norm(embeddings[0] - embeddings[-1]) > 1e-6:
        final_distances /= np.linalg.norm(embeddings[0] - embeddings[-1])

    while cur_goal_idx > min_interval:
        distances = np.linalg.norm(embeddings[:cur_goal_idx] - embeddings[cur_goal_idx], axis=1)
        if np.linalg.norm(embeddings[0] - embeddings[cur_goal_idx]) > 1e-6:
            distances /= np.linalg.norm(embeddings[0] - embeddings[cur_goal_idx])

        smoothed_distances = medfilt(distances, kernel_size=None)
        
        # Find local maxima (peaks)
        extrema_indices = argrelextrema(smoothed_distances, np.greater)[0]
        
        updated = False
        for ex_idx in reversed(extrema_indices):
            if cur_goal_idx - ex_idx > min_interval and ex_idx > min_interval:
                cur_goal_idx = ex_idx
                subgoal_indices.append(cur_goal_idx)
                updated = True
                break
        
        if not updated:
            break # No more valid peaks found

    if 0 not in subgoal_indices:
        subgoal_indices.append(0)
    return sorted(list(set(subgoal_indices))), final_distances


# --- Main Task ---

def run_uvd_on_hdf5(
    hdf5_path: Path,
    preprocessor_name: Literal["dinov2", "vip"],
    device: str,
    sequence_name: str | None = None,
    decomp_method: Literal["derivative", "peaks"] = "derivative",
    derivative_threshold: float = 1e-3,
    window_length: int = 11,
    min_peak_interval: int = 15,
    visualize_subgoals: bool = False,
    force_reprocess: bool = False,
):
    """
    Runs the full UVD pipeline on an HDF5 file.
    """
    logging.info(f"Starting UVD processing on {hdf5_path} using '{preprocessor_name}'")
    model, transform = get_preprocessor(preprocessor_name, device)
    model.eval()

    with h5py.File(hdf5_path, "a") as f:
        if sequence_name:
            if sequence_name not in f:
                logging.error(f"Sequence '{sequence_name}' not found in {hdf5_path}.")
                return
            sequences_to_process = [sequence_name]
        else:
            sequences_to_process = list(f.keys())

        for seq_name in tqdm(sequences_to_process, desc="Processing sequences"):
            traj_group = f[seq_name]
            
            output_group_name = f"subgoals_{preprocessor_name}"
            if output_group_name in traj_group and not force_reprocess:
                logging.info(f"Skipping '{seq_name}', data already exists.")
                continue

            if output_group_name in traj_group:
                del traj_group[output_group_name]

            if "cameras/default_camera/rgb" not in traj_group:
                logging.warning(f"No RGB data found for sequence '{seq_name}'. Skipping.")
                continue
            
            rgb_group = traj_group["cameras/default_camera/rgb"]
            
            try:
                # 1. Decode video
                video_frames_np = decode_video_from_hdf5(rgb_group)
                
                # 2. Preprocess and Extract Embeddings
                # Ensure all frames are resized to 224x224 before any other processing.
                resize_transform = T.Resize((224, 224))
                
                video_tensor = torch.from_numpy(video_frames_np).permute(0, 3, 1, 2).float() / 255.0
                resized_video_tensor = resize_transform(video_tensor)
                normalized_video_tensor = transform(resized_video_tensor).to(device)

                all_embeddings = []
                with torch.no_grad():
                    for frame in normalized_video_tensor:
                        embedding = model(frame.unsqueeze(0))
                        all_embeddings.append(embedding.cpu().numpy().squeeze())
                
                embeddings_np = np.array(all_embeddings)
                
                # 3. Find Subgoals
                if decomp_method == "derivative":
                    subgoal_indices, distances = find_subgoals_by_derivative(
                        embeddings_np,
                        derivative_threshold=derivative_threshold,
                        window_length=window_length,
                    )
                elif decomp_method == "peaks":
                    subgoal_indices, distances = find_subgoals_by_peaks(
                        embeddings_np,
                        min_interval=min_peak_interval,
                    )
                else:
                    raise ValueError(f"Unknown decomposition method: {decomp_method}")

                # 4. Save Results
                subgoals_group = traj_group.create_group(output_group_name)
                subgoals_group.create_dataset(
                    "indices", data=np.array(subgoal_indices, dtype=np.int32)
                )
                logging.info(f"Saved {len(subgoal_indices)} subgoals for '{seq_name}'.")

                # 4b. Create UVD action boundaries for visualization
                uvd_action_group = traj_group.require_group("actions/uvd")
                # Build (start, end) pairs from consecutive subgoal indices
                if len(subgoal_indices) > 1:
                    boundaries = np.column_stack(
                        (subgoal_indices[:-1], subgoal_indices[1:])
                    ).astype(np.int32)
                    if "action_boundaries" in uvd_action_group:
                        del uvd_action_group["action_boundaries"]
                    uvd_action_group.create_dataset(
                        "action_boundaries", data=boundaries, dtype=np.int32
                    )

                # 5. Visualize if requested
                if visualize_subgoals:
                    vis_dir = hdf5_path.parent / "subgoal_visualizations" / seq_name
                    vis_dir.mkdir(parents=True, exist_ok=True)
                    for i, frame_idx in enumerate(subgoal_indices):
                        img = Image.fromarray(video_frames_np[frame_idx])
                        img.save(vis_dir / f"subgoal_{i:03d}_frame_{frame_idx:04d}.jpg")
                    logging.info(f"Saved {len(subgoal_indices)} subgoal images to {vis_dir}")

            except Exception as e:
                logging.error(f"Failed to process sequence '{seq_name}': {e}", exc_info=True)


def main():
    parser = argparse.ArgumentParser(description="Run UVD on an HDF5 dataset.")
    parser.add_argument("hdf5_path", type=Path, help="Path to the HDF5 file.")
    parser.add_argument(
        "--preprocessor",
        type=str,
        default="dinov2",
        choices=["dinov2", "vip"],
        help="UVD preprocessor model to use.",
    )
    parser.add_argument(
        "--sequence-name",
        type=str,
        default=None,
        help="Specify a single sequence to process. If not provided, all sequences are processed.",
    )
    parser.add_argument(
        "--decomp-method",
        type=str,
        default="derivative",
        choices=["derivative", "peaks"],
        help="The decomposition algorithm to use.",
    )
    parser.add_argument(
        "--derivative-threshold",
        type=float,
        default=1e-3,
        help="Sensitivity for detecting significant scene changes (for 'derivative' method).",
    )
    parser.add_argument(
        "--window-length",
        type=int,
        default=11,
        help="Minimum number of stable frames to define a subgoal boundary (for 'derivative' method).",
    )
    parser.add_argument(
        "--min-peak-interval",
        type=int,
        default=15,
        help="Minimum number of frames between detected peaks (for 'peaks' method).",
    )
    parser.add_argument(
        "--visualize-subgoals",
        action="store_true",
        help="Save the subgoal frames as images for inspection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use ('cuda' or 'cpu').",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force reprocessing of sequences.",
    )
    
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_uvd_on_hdf5(
        args.hdf5_path,
        args.preprocessor,
        args.device,
        args.sequence_name,
        args.decomp_method,
        args.derivative_threshold,
        args.window_length,
        args.min_peak_interval,
        args.visualize_subgoals,
        args.force,
    )


if __name__ == "__main__":
    main() 