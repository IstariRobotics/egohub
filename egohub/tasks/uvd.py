from __future__ import annotations

import logging

import cv2
import h5py
import numpy as np
from scipy.signal import argrelextrema
from tqdm import tqdm

from egohub.backends.base import BaseBackend
from egohub.tasks.base import BaseTask


class UVDTask(BaseTask):
    def __init__(self, output_group_name: str = "actions/uvd"):
        super().__init__(output_group_name)

    def run(self, traj_group: h5py.Group, backend: BaseBackend, **kwargs):
        """
        Runs the Universal Visual Decomposer (UVD) task.

        This task uses a visual backend to extract embeddings from video frames,
        then applies the UVD algorithm to detect subgoals and segment the
        trajectory into actions.

        The results are saved back to the HDF5 file in a new group defined by
        `self.output_group_name`.

        Args:
            traj_group: The HDF5 group for the trajectory to process.
            backend: The backend instance to use for inference.
            **kwargs: Backend-specific arguments for embedding extraction and
                      UVD algorithm parameters.
        """
        logging.info("Starting UVD Task...")

        # --- 1. Get video frames from the trajectory group ---
        video_stream = self._get_video_stream(traj_group)
        if not video_stream:
            logging.warning("No video stream found in this trajectory. Skipping.")
            return

        frames = self._decode_video_frames(video_stream)
        if not frames:
            logging.warning("Could not decode any frames. Skipping.")
            return

        logging.info(f"Decoded {len(frames)} frames.")

        # --- 2. Use the backend to get embeddings for each frame ---
        logging.info("Generating embeddings for frames...")
        embeddings = backend.get_embeddings(frames=frames, **kwargs)
        if embeddings is None:
            logging.error("Backend failed to generate embeddings. Skipping.")
            return

        logging.info(f"Generated {len(embeddings)} embeddings of dimension {embeddings.shape[1]}.")

        # --- 3. Implement and run the UVD algorithm on the embeddings ---
        logging.info("Running UVD algorithm...")
        subgoal_indices = self._run_uvd(embeddings, **kwargs)
        logging.info(f"UVD algorithm found {len(subgoal_indices)} subgoals.")

        # --- 4. Save the resulting action boundaries to the HDF5 file ---
        self._save_results(traj_group, subgoal_indices)
        logging.info("UVD Task complete.")

    def _get_video_stream(self, traj_group: h5py.Group) -> h5py.Group | None:
        """Finds the first available RGB video stream in the trajectory."""
        if "cameras" not in traj_group:
            return None

        for camera_name in traj_group["cameras"]:
            camera_group = traj_group["cameras"][camera_name]
            if "rgb" in camera_group and "image_bytes" in camera_group["rgb"]:
                logging.info(f"Found video stream in camera: {camera_name}")
                return camera_group["rgb"]
        return None

    def _decode_video_frames(self, video_stream: h5py.Group) -> list[np.ndarray]:
        """Decodes image bytes into a list of OpenCV frames."""
        frames = []
        image_bytes_dataset = video_stream["image_bytes"]
        for img_bytes in tqdm(image_bytes_dataset, desc="Decoding frames"):
            try:
                # Decode the byte string into a numpy array
                frame = cv2.imdecode(np.frombuffer(img_bytes, np.uint8), cv2.IMREAD_COLOR)
                if frame is not None:
                    # OpenCV decodes to BGR, convert to RGB for consistency
                    frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            except Exception as e:
                logging.warning(f"Could not decode frame: {e}")
        return frames

    def _run_uvd(
        self, embeddings: np.ndarray, min_interval: int = 15, smoothing_sigma: int = 5
    ) -> list[int]:
        """
        Runs the core UVD algorithm based on the paper's pseudocode.
        https://universal-visual-decomposer.github.io/

        Args:
            embeddings: A NumPy array of shape (L, d) where L is the sequence
                        length and d is the embedding dimension.
            min_interval: The minimum number of frames between two subgoals.
            smoothing_sigma: The standard deviation of the Gaussian kernel for smoothing.

        Returns:
            A list of frame indices corresponding to the detected subgoals.
        """
        from scipy.ndimage import gaussian_filter1d

        def _smooth_fn(d):
            return gaussian_filter1d(d, sigma=smoothing_sigma)

        num_frames = embeddings.shape[0]
        # The last frame is always the final subgoal
        cur_goal_idx = num_frames - 1
        # Store subgoal indices in reverse order
        goal_indices = [cur_goal_idx]
        
        # Normalize embeddings to have unit norm
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=-1, keepdims=True) + 1e-6)

        current_embeddings = embeddings
        while cur_goal_idx > min_interval:
            # Calculate smoothed L2 distance to the current goal embedding
            dist = np.linalg.norm(current_embeddings - current_embeddings[-1], axis=-1)
            dist_smoothed = _smooth_fn(dist)

            # Find local maxima, which indicate a phase shift
            extremas = argrelextrema(dist_smoothed, np.greater)[0]

            # Filter extremas to ensure they are far enough from the current goal
            valid_extremas = [e for e in extremas if cur_goal_idx - e > min_interval]

            if valid_extremas:
                # The new subgoal is the frame before the last valid extrema
                cur_goal_idx = valid_extremas[-1]
                goal_indices.append(cur_goal_idx)
                current_embeddings = embeddings[: cur_goal_idx + 1]
            else:
                # No more subgoals found
                break
        
        # Reverse to get chronological order and add the start frame
        final_indices = sorted(list(set([0] + goal_indices)))
        return final_indices

    def _save_results(self, traj_group: h5py.Group, subgoal_indices: list[int]):
        """
        Saves the detected subgoal indices and action boundaries to the HDF5 file.
        """
        # Ensure the output group exists
        output_group = traj_group.require_group(self.output_group_name)

        # --- Save Subgoal Indices ---
        subgoal_dset_name = "subgoal_indices"
        if subgoal_dset_name in output_group:
            del output_group[subgoal_dset_name]
        output_group.create_dataset(
            subgoal_dset_name, data=np.array(subgoal_indices, dtype=np.uint64)
        )
        logging.info(f"Saved subgoal indices to {output_group.name}/{subgoal_dset_name}")

        # --- Save Action Boundaries ---
        # Create (start, end) pairs for each action
        action_boundaries = list(zip(subgoal_indices[:-1], subgoal_indices[1:]))
        action_dset_name = "action_boundaries"
        if action_dset_name in output_group:
            del output_group[action_dset_name]
        
        if action_boundaries:
            output_group.create_dataset(
                action_dset_name,
                data=np.array(action_boundaries, dtype=np.uint64),
                dtype=np.uint64,
            )
            logging.info(
                f"Saved {len(action_boundaries)} action boundaries to "
                f"{output_group.name}/{action_dset_name}"
            )
        else:
            logging.warning("No action boundaries were generated.") 