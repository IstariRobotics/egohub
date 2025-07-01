import logging
from pathlib import Path
from typing import Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import h5py

from egohub.datasets import EgocentricH5Dataset
from egohub.constants import AVP_ID2NAME, AVP_IDS, AVP_LINKS

logger = logging.getLogger(__name__)

class RerunExporter:
    """Exports egocentric data to the Rerun visualizer."""

    def __init__(self, max_frames: int | None = None):
        self.max_frames = max_frames

    def export(self, dataset: EgocentricH5Dataset):
        """
        Visualize HDF5 data using Rerun.
        
        Args:
            dataset: The loaded EgocentricH5Dataset instance.
        """
        rr.init("egocentric-h5-viewer", spawn=True)
        
        if len(dataset) == 0:
            logger.error("No data found in the specified dataset/trajectories")
            return
        
        camera_names = dataset.camera_streams
        logger.info(f"Visualizing {len(dataset.frame_index)} frames with cameras: {camera_names}")
        
        blueprint = self._create_blueprint(camera_names)
        rr.send_blueprint(blueprint)
        
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        self._set_pose_annotation_context()
        self._log_static_data(dataset)
        
        frame_range = len(dataset)
        if self.max_frames:
            frame_range = min(frame_range, self.max_frames)
        
        for i in range(frame_range):
            logger.info(f"Processing frame {i+1}/{frame_range}")
            self._log_temporal_data(dataset[i], camera_names)
        
        logger.info("Visualization complete! Check the Rerun viewer.")

    def _create_blueprint(self, camera_names: list[str]) -> rrb.Blueprint:
        spatial2d_views = [
            rrb.Spatial2DView(origin=f"world/cameras/{name}/image", name=f"{name} View")
            for name in camera_names
        ]
        return rrb.Blueprint(
            rrb.Horizontal(
                rrb.Spatial3DView(origin="world"),
                rrb.Vertical(
                    rrb.TextDocumentView(origin="llm_description"),
                    rrb.Tabs(*spatial2d_views), row_shares=[1, 10]
                ), column_shares=[2, 1]
            ), collapse_panels=True
        )

    def _set_pose_annotation_context(self) -> None:
        rr.log("world/skeleton", rr.AnnotationContext(
            [rr.ClassDescription(
                info=rr.AnnotationInfo(id=0, label="AVP Skeleton"),
                keypoint_annotations=[rr.AnnotationInfo(id=id, label=name) for id, name in AVP_ID2NAME.items()],
                keypoint_connections=AVP_LINKS,
            )]
        ), static=True)

    def _log_static_data(self, dataset: EgocentricH5Dataset):
        first_frame_data = dataset[0]
        with h5py.File(dataset.h5_path, 'r') as f:
            first_traj_name = dataset.frame_index[0][0]
            traj_group = f[first_traj_name]
            if 'metadata' in traj_group and 'action_label' in traj_group['metadata'].attrs:
                action_label = traj_group['metadata'].attrs['action_label']
                rr.log("llm_description", rr.TextDocument(text=action_label), static=True)

            for cam_name in dataset.camera_streams:
                img_path = Path("world/cameras") / cam_name / "image"
                intrinsics = first_frame_data['camera_intrinsics'][cam_name].numpy()
                height, width = first_frame_data['rgb'][cam_name].shape[1:3]
                rr.log(str(img_path), rr.Pinhole(
                    image_from_camera=intrinsics,
                    width=width, height=height,
                    camera_xyz=rr.ViewCoordinates.RDF,
                ), static=True)

    def _log_temporal_data(self, frame_data: dict, camera_names: list[str]):
        rr.set_time("frame", timestamp=int(frame_data['timestamp_ns']))
        for cam_name in camera_names:
            cam_path = Path("world/cameras") / cam_name
            if cam_name in frame_data.get('camera_pose', {}):
                camera_pose = frame_data['camera_pose'][cam_name].numpy()
                rr.log(str(cam_path), rr.Transform3D(
                    translation=camera_pose[:3, 3], mat3x3=camera_pose[:3, :3]
                ))
            if cam_name in frame_data.get('rgb', {}):
                rgb_hwc = frame_data['rgb'][cam_name].permute(1, 2, 0)
                rr.log(str(cam_path / "image"), rr.Image(rgb_hwc))
        
        if 'skeleton_positions' in frame_data and 'skeleton_confidences' in frame_data:
            positions = frame_data['skeleton_positions'].numpy()
            confidences = frame_data['skeleton_confidences'].numpy()
            colors = self._confidence_scores_to_rgb(confidences)
            rr.log("world/skeleton", rr.Points3D(
                positions=positions, colors=colors, radii=0.01,
                class_ids=0, keypoint_ids=AVP_IDS
            ))

    def _confidence_scores_to_rgb(self, confidence_scores: np.ndarray) -> np.ndarray:
        n_kpts = confidence_scores.shape[0]
        clipped_confidences = np.clip(confidence_scores, a_min=0.0, a_max=1.0)
        colors = np.zeros((n_kpts, 3), dtype=np.uint8)
        mask_low = clipped_confidences <= 0.5
        if mask_low.any():
            t_low = clipped_confidences[mask_low] * 2.0
            colors[mask_low, 0] = 255
            colors[mask_low, 1] = (t_low * 255).astype(np.uint8)
        mask_high = ~mask_low
        if mask_high.any():
            t_high = (clipped_confidences[mask_high] - 0.5) * 2.0
            colors[mask_high, 0] = ((1.0 - t_high) * 255).astype(np.uint8)
            colors[mask_high, 1] = 255
        return colors 