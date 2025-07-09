from __future__ import annotations

import logging
from collections import defaultdict

import cv2
import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm

from egohub.backends.base import BaseBackend
from egohub.utils.video_utils import hdf5_to_cv2_video

logger = logging.getLogger(__name__)


def _extract_video_data(traj_group: h5py.Group) -> tuple[list[np.ndarray], np.ndarray]:
    """Extract video frames and timestamps from trajectory group."""
    cameras_group = traj_group.get("cameras")
    if not isinstance(cameras_group, h5py.Group) or "ego_camera" not in cameras_group:
        raise ValueError("No 'ego_camera' found in cameras group")

    ego_camera_group = cameras_group.get("ego_camera")
    if not isinstance(ego_camera_group, h5py.Group):
        raise ValueError("'ego_camera' is not a group")

    rgb_group = ego_camera_group.get("rgb")
    if not isinstance(rgb_group, h5py.Group):
        raise ValueError("No 'rgb' group found for 'ego_camera'")

    # Extract frames
    video_frames = list(hdf5_to_cv2_video(rgb_group))

    # Extract timestamps
    metadata_group = traj_group.get("metadata")
    if isinstance(metadata_group, h5py.Group) and "timestamps_ns" in metadata_group:
        timestamps_ns = np.array(metadata_group["timestamps_ns"])
    else:
        logger.warning("No timestamps found, using frame indices")
        timestamps_ns = np.arange(len(video_frames), dtype=np.uint64)

    return video_frames, timestamps_ns


class HuggingFaceBackend(BaseBackend):
    """A backend for running models from the Hugging Face Hub."""

    def __init__(self, model_name: str, task_name: str, **kwargs):
        """
        Initializes the backend with a Hugging Face model.

        Args:
            model_name: The name of the model on the HF Hub.
            task_name: The name of the task to perform (e.g., 'object-detection').
        """
        self.model_name = model_name
        self.task_name = task_name
        self.model_kwargs = kwargs

        logger.info(
            f"Initializing HF Backend for model: {self.model_name}, "
            f"task: {self.task_name}"
        )

        if self.task_name == "object-detection":
            from transformers.pipelines import pipeline

            self.detector = pipeline(self.task_name, model=self.model_name)
        elif self.task_name == "pose-estimation":
            try:
                from mmpose.apis import MMPoseInferencer  # type: ignore

                self.detector = MMPoseInferencer(self.model_name)
            except ImportError:
                raise ImportError(
                    "mmpose is required for pose estimation. "
                    "Install it with: pip install egohub[pose] or pip install mmpose"
                )
        else:
            raise ValueError(
                f"Unsupported task for HuggingFaceBackend: {self.task_name}"
            )

    def run(self, traj_group: h5py.Group, **kwargs) -> dict:
        """
        Runs the specified task on the trajectory's video.

        Args:
            traj_group: The HDF5 group for the trajectory to process.
            **kwargs: Additional arguments for the task.

        Returns:
            A dictionary containing the results of the inference.
        """
        if self.task_name == "object-detection":
            return self._run_object_detection(traj_group, **kwargs)
        elif self.task_name == "pose-estimation":
            return self._run_pose_estimation(traj_group, **kwargs)
        else:
            # This should not be reachable due to the __init__ check
            raise ValueError(f"Unsupported task: {self.task_name}")

    def _run_pose_estimation(self, traj_group: h5py.Group, **kwargs) -> dict:
        """Runs pose estimation and returns results."""
        video_frames, _ = _extract_video_data(traj_group)
        if not video_frames:
            logger.warning("No video frames found, skipping pose estimation.")
            return {}

        all_keypoints = []
        all_confidences = []
        frame_indices = []

        for i, frame_np in enumerate(tqdm(video_frames, desc="Pose Estimation")):
            # Convert BGR to RGB for mmpose
            frame_rgb = cv2.cvtColor(frame_np, cv2.COLOR_BGR2RGB)
            # mmpose inferencer expects a list of images
            result_generator = self.detector([frame_rgb], show=False)
            results = next(result_generator)

            if results["predictions"]:
                person_result = results["predictions"][0][0]
                keypoints = person_result["keypoints"]
                confidences = person_result["keypoint_scores"]

                all_keypoints.append(keypoints)
                all_confidences.append(confidences)
                frame_indices.append(i)

        return {
            "keypoints": all_keypoints,
            "confidences": all_confidences,
            "frame_indices": frame_indices,
        }

    def _run_object_detection(self, traj_group: h5py.Group, **kwargs) -> dict:
        """Runs object detection and returns results."""
        cameras_group = traj_group.get("cameras")
        if (
            not isinstance(cameras_group, h5py.Group)
            or "ego_camera" not in cameras_group
        ):
            logger.warning("No 'ego_camera' found, skipping object detection.")
            return {}

        ego_camera_group = cameras_group.get("ego_camera")
        if not isinstance(ego_camera_group, h5py.Group):
            logger.warning("'ego_camera' is not a group, skipping object detection.")
            return {}

        rgb_group = ego_camera_group.get("rgb")
        if not isinstance(rgb_group, h5py.Group):
            logger.warning("No 'rgb' group found for 'ego_camera', skipping.")
            return {}

        frames_iterator = hdf5_to_cv2_video(rgb_group)
        frame_sizes_dset = rgb_group.get("frame_sizes")
        if not isinstance(frame_sizes_dset, h5py.Dataset):
            logger.warning("No 'frame_sizes' dataset found, skipping.")
            return {}

        num_frames = len(frame_sizes_dset)
        all_detections = defaultdict(list)

        logger.info("Running object detection on video frames...")
        for i, frame_cv2 in enumerate(
            tqdm(frames_iterator, total=num_frames, desc="Object Detection")
        ):
            # transformers pipeline expects a PIL image
            image = Image.fromarray(frame_cv2[:, :, ::-1])  # BGR to RGB
            predictions = self.detector(image)

            for pred in predictions:
                box = pred["box"]
                bbox = (
                    box["xmin"],
                    box["ymin"],
                    box["xmax"] - box["xmin"],
                    box["ymax"] - box["ymin"],
                )
                all_detections[pred["label"]].append((i, bbox, pred["score"]))

        # The backend's responsibility is just to return the raw data.
        # The Task will be responsible for writing it to HDF5.
        return {"detections": all_detections, "num_frames": num_frames}
