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
from egohub.video_utils import hdf5_to_cv2_video


class HuggingFaceObjectDetectionTool(BaseTool):
    """A tool for running object detection using a Hugging Face model."""

    def __init__(self, model_name: str = "facebook/detr-resnet-50"):
        """Initializes the tool with a Hugging Face model.

        Args:
            model_name: The name of the object detection model on the HF Hub.
        """
        self.model_name = model_name
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
        for i, frame_cv2 in enumerate(tqdm(frames_iterator, total=num_frames, desc="Object Detection")):
            # transformers pipeline expects a PIL image
            image = Image.fromarray(frame_cv2[:, :, ::-1]) # BGR to RGB
            predictions = self.detector(image)
            
            for pred in predictions:
                box = pred["box"]
                bbox = (box['xmin'], box['ymin'], box['xmax'] - box['xmin'], box['ymax'] - box['ymin']) # (x, y, w, h)
                all_detections[pred["label"]].append(
                    (i, bbox, pred["score"])
                )
        
        # --- Write results to HDF5 ---
        if "objects" in traj_group:
            logging.warning("Overwriting existing 'objects' group.")
            del traj_group["objects"]
        objects_group = traj_group.create_group("objects")

        logging.info(f"Writing {len(all_detections)} detected object classes to HDF5.")
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