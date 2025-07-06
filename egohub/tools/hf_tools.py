"""Tools that use the Hugging Face transformers library."""

from __future__ import annotations

import logging
from collections import defaultdict

import h5py
import numpy as np
from PIL import Image
from tqdm import tqdm
from transformers import pipeline

from egohub.tools.base import BaseTool
from egohub.video_utils import get_video_frames_from_hdf5, hdf5_to_cv2_video

logger = logging.getLogger(__name__)


class SapiensPoseTool(BaseTool):
    def __init__(self, model_name: str = "sapiens-pose-1b"):
        from mmpose.apis import MMPoseInferencer

        self.model_name = model_name
        self.inferencer = MMPoseInferencer(model_name)

    def __call__(self, traj_group: h5py.Group) -> h5py.Group:
        """
        Processes a trajectory group to add full-body pose data.

        Args:
            traj_group (h5py.Group): The trajectory group to process.

        Returns:
            The processed trajectory group with skeleton data.
        """
        logger.info(f"Running Sapiens Pose Estimation on {traj_group.name}")
        video_frames, timestamps_ns = get_video_frames_from_hdf5(
            traj_group, "ego_camera"
        )

        if not video_frames:
            logger.warning("No video frames found, skipping pose estimation.")
            return traj_group

        all_keypoints = []
        all_confidences = []
        frame_indices = []

        for i, frame_np in enumerate(tqdm(video_frames, desc="Processing frames")):
            result_generator = self.inferencer(frame_np, show=False)
            results = next(result_generator)

            if results["predictions"]:
                # Assume the first detected person is the one we care about
                person_result = results["predictions"][0][0]
                keypoints = person_result["keypoints"]
                confidences = person_result["keypoint_scores"]

                all_keypoints.append(keypoints)
                all_confidences.append(confidences)
                frame_indices.append(i)

        if "skeleton" in traj_group:
            del traj_group["skeleton"]
        skel_group = traj_group.create_group("skeleton")

        if not all_keypoints:
            logger.warning("No poses detected in the video.")
            return traj_group

        skel_group.create_dataset(
            "positions", data=np.array(all_keypoints, dtype=np.float32)
        )
        skel_group.create_dataset(
            "confidences", data=np.array(all_confidences, dtype=np.float32)
        )
        skel_group.create_dataset(
            "frame_indices", data=np.array(frame_indices, dtype=np.int64)
        )

        logger.info(f"Saved {len(all_keypoints)} poses to {skel_group.name}")
        return traj_group


class HuggingFaceObjectDetectionTool(BaseTool):
    """A tool for running object detection using a Hugging Face model."""

    def __init__(
        self,
        model_name: str = "facebook/detr-resnet-50",
        output_group_name: str = "objects",
    ):
        """Initializes the tool with a Hugging Face model.

        Args:
            model_name: The name of the object detection model on the HF Hub.
            output_group_name: The name of the HDF5 group to save results to.
        """
        self.model_name = model_name
        self.output_group_name = output_group_name
        logging.info(f"Initializing HF pipeline for model: {self.model_name}")
        self.detector = pipeline("object-detection", model=self.model_name)

    def __call__(self, traj_group: h5py.Group) -> None:
        """Runs object detection on the trajectory's video and saves results."""
        if "ego_camera" not in traj_group["cameras"]:
            logging.warning("No 'ego_camera' found, skipping object detection.")
            return

        rgb_group = traj_group["cameras/ego_camera/rgb"]
        frames_iterator = hdf5_to_cv2_video(rgb_group)
        num_frames = len(rgb_group["frame_sizes"])

        # This will store all detections across all frames, grouped by label
        # e.g., {"cup": [(frame_idx, box, score), ...], "remote": [...]}
        all_detections = defaultdict(list)

        logging.info("Running object detection on video frames...")
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
                )  # (x, y, w, h)
                all_detections[pred["label"]].append((i, bbox, pred["score"]))

        # --- Write results to HDF5 ---
        if self.output_group_name in traj_group:
            logging.warning(f"Overwriting existing '{self.output_group_name}' group.")
            del traj_group[self.output_group_name]
        objects_group = traj_group.create_group(self.output_group_name)

        logging.info(
            f"Writing {len(all_detections)} detected object classes to HDF5 group "
            f"'{self.output_group_name}'."
        )
        for label, detections in all_detections.items():
            obj_group = objects_group.create_group(label)

            # For each detected object class, create datasets aligned by frame
            bboxes_2d = np.zeros((num_frames, 4), dtype=np.float32)
            scores = np.zeros(num_frames, dtype=np.float32)

            for frame_idx, bbox, score in detections:
                bboxes_2d[frame_idx] = bbox
                scores[frame_idx] = score

            obj_group.create_dataset("bboxes_2d", data=bboxes_2d)
            obj_group.create_dataset("scores", data=scores)
