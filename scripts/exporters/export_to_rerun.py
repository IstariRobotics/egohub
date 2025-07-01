#!/usr/bin/env python3
"""
Rerun visualization script for egocentric HDF5 data.

This script loads data from our canonical HDF5 format and visualizes it using Rerun,
providing the ultimate validation that our data pipeline works end-to-end.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import h5py

# Add the project root to the path so we can import egohub
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from egohub.datasets import EgocentricH5Dataset
from egohub.exporters.base import BaseExporter
from egohub.camera_parameters import PinholeParameters, Intrinsics, Extrinsics
from egohub.rerun_log_utils import log_pinhole
from egohub.constants import AVP_ID2NAME, AVP_IDS, AVP_LINKS

# Set up logging - will be configured in run_from_main
logger = logging.getLogger(__name__)


def create_optimal_blueprint(camera_names: list[str]) -> rrb.Blueprint:
    """Create an optimal blueprint layout for egocentric visualization."""
    
    # Create a 2D view for each camera stream
    spatial2d_views = [
        rrb.Spatial2DView(origin=f"world/cameras/{name}/image", name=f"{name} View")
        for name in camera_names
    ]

    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="world"),
            rrb.Vertical(
                rrb.TextDocumentView(origin="llm_description"),
                rrb.Tabs(*spatial2d_views),
                row_shares=[1, 10],
            ),
            column_shares=[2, 1],
        ),
        collapse_panels=True,
    )


def confidence_scores_to_rgb(confidence_scores: np.ndarray) -> np.ndarray:
    """Convert confidence scores to RGB colors using a Red-Yellow-Green gradient."""
    n_kpts = confidence_scores.shape[0]
    clipped_confidences = np.clip(confidence_scores, a_min=0.0, a_max=1.0)
    
    colors = np.zeros((n_kpts, 3), dtype=np.uint8)
    
    # Segment A: red → yellow for conf ≤ 0.5
    mask_low = clipped_confidences <= 0.5
    if mask_low.any():
        t_low = clipped_confidences[mask_low] * 2.0  # 0‥1
        colors[mask_low, 0] = 255  # red fixed
        colors[mask_low, 1] = (t_low * 255).astype(np.uint8)
    
    # Segment B: yellow → green for conf > 0.5
    mask_high = ~mask_low
    if mask_high.any():
        t_high = (clipped_confidences[mask_high] - 0.5) * 2.0
        colors[mask_high, 0] = ((1.0 - t_high) * 255).astype(np.uint8)
        colors[mask_high, 1] = 255  # green fixed
    
    return colors


def set_pose_annotation_context() -> None:
    """Log the annotation context for the AVP skeleton to Rerun."""
    rr.log(
        "world/skeleton",
        rr.AnnotationContext(
            [
                rr.ClassDescription(
                    info=rr.AnnotationInfo(id=0, label="AVP Skeleton"),
                    keypoint_annotations=[
                        rr.AnnotationInfo(id=id, label=name) for id, name in AVP_ID2NAME.items()
                    ],
                    keypoint_connections=AVP_LINKS,
                ),
            ]
        ),
        static=True,
    )


class RerunExporter(BaseExporter):
    """Exports egocentric data to the Rerun visualizer."""

    def get_arg_parser(self) -> argparse.ArgumentParser:
        """Adds Rerun-specific arguments to the parser."""
        parser = super().get_arg_parser()
        parser.add_argument("--max-frames", type=int, help="Maximum number of frames to visualize")
        parser.add_argument("--frame-rate", type=int, default=30, help="Frame rate for playback")
        return parser

    def export(self, dataset: EgocentricH5Dataset, output_path: Path | None, **kwargs: Any):
        """
        Visualize HDF5 data using Rerun.
        
        Args:
            dataset: The loaded EgocentricH5Dataset instance.
            output_path: Not used for this exporter.
            **kwargs: Additional arguments from the parser (e.g., max_frames).
        """
        max_frames = kwargs.get('max_frames')
        
        # Initialize Rerun
        rr.init("egocentric-h5-viewer", spawn=True)
        
        if len(dataset) == 0:
            logger.error("No data found in the specified dataset/trajectories")
            return
        
        camera_names = dataset.camera_streams
        logger.info(f"Visualizing {len(dataset.frame_index)} frames with cameras: {camera_names}")
        
        # Create optimal blueprint layout
        blueprint = create_optimal_blueprint(camera_names)
        
        # Set up coordinate system
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        
        # Send blueprint
        rr.send_blueprint(blueprint)
        
        # Set up the annotation context for the skeleton
        set_pose_annotation_context()
        
        # --- Log static data once ---
        first_frame_data = dataset[0]

        # Log camera intrinsics (static) and action label
        with h5py.File(dataset.h5_path, 'r') as f:
            first_traj_name = dataset.frame_index[0][0]
            traj_group = f[first_traj_name]
            if 'metadata' in traj_group and 'action_label' in traj_group['metadata'].attrs:
                action_label = traj_group['metadata'].attrs['action_label']
                rr.log("llm_description", rr.TextDocument(text=action_label), static=True)

            for cam_name in camera_names:
                img_path = Path("world/cameras") / cam_name / "image"
                intrinsics = first_frame_data['camera_intrinsics'][cam_name].numpy()
                height, width = first_frame_data['rgb'][cam_name].shape[1:3]

                rr.log(
                    str(img_path),
                    rr.Pinhole(
                        image_from_camera=intrinsics,
                        width=width,
                        height=height,
                        camera_xyz=rr.ViewCoordinates.RDF,  # X=Right, Y=Down, Z=Forward
                    ),
                    static=True
                )
        
        # Determine frame range
        frame_range = len(dataset)
        if max_frames:
            frame_range = min(frame_range, max_frames)
        
        # Visualize frames
        for i in range(frame_range):
            logger.info(f"Processing frame {i+1}/{frame_range}")
            
            # Load frame data
            frame_data = dataset[i]
            
            # Set timeline
            rr.set_time("frame", timestamp=int(frame_data['timestamp_ns']))

            # Log data for each camera
            for cam_name in camera_names:
                # Set up camera logging paths
                cam_path = Path("world/cameras") / cam_name
                img_path = cam_path / "image"

                # Log camera pose (extrinsics) - this changes every frame
                if cam_name in frame_data.get('camera_pose', {}):
                    camera_pose = frame_data['camera_pose'][cam_name].numpy()
                    rr.log(
                        str(cam_path),
                        rr.Transform3D(
                            translation=camera_pose[:3, 3],
                            mat3x3=camera_pose[:3, :3]
                        )
                    )
            
                # Log RGB image
                if cam_name in frame_data.get('rgb', {}):
                    rgb = frame_data['rgb'][cam_name]
                    rgb_hwc = rgb.permute(1, 2, 0)
                    rr.log(str(img_path), rr.Image(rgb_hwc))
            
            # Log skeleton data (remains top-level)
            if 'skeleton_positions' in frame_data and 'skeleton_confidences' in frame_data:
                positions = frame_data['skeleton_positions'].numpy()
                confidences = frame_data['skeleton_confidences'].numpy()
                
                colors = confidence_scores_to_rgb(confidences)
                
                rr.log(
                    "world/skeleton",
                    rr.Points3D(
                        positions=positions,
                        colors=colors,
                        radii=0.01,
                        class_ids=0,
                        keypoint_ids=AVP_IDS,
                    )
                )
                
                if 'skeleton_joint_names' in frame_data:
                    joint_names = frame_data['skeleton_joint_names']
                    for i, (pos, conf, name) in enumerate(zip(positions, confidences, joint_names)):
                        if conf > 0.5:
                            rr.log(
                                f"world/skeleton/labels/{i}",
                                rr.TextDocument(f"{name}: {conf:.2f}")
                            )
        
        logger.info(f"Visualization complete! Check the Rerun viewer.")


def run_from_main():
    RerunExporter().run_from_main()


if __name__ == "__main__":
    run_from_main() 