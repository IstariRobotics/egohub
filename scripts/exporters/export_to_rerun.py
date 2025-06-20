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
from typing import Optional

import numpy as np
import rerun as rr
import rerun.blueprint as rrb
import torch
from scipy.spatial.transform import Rotation as R
import h5py

# Add the project root to the path so we can import egohub
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from egohub.datasets import EgocentricH5Dataset
from egohub.camera_parameters import PinholeParameters, Intrinsics, Extrinsics
from egohub.rerun_log_utils import log_pinhole
from egohub.constants import AVP_ID2NAME, AVP_IDS, AVP_LINKS

# Set up logging
logging.basicConfig(level=logging.INFO)


def create_optimal_blueprint() -> rrb.Blueprint:
    """Create an optimal blueprint layout for egocentric visualization."""
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(),
            rrb.Vertical(
                rrb.TextDocumentView(origin="llm_description"),
                rrb.Spatial2DView(origin="world/camera/pinhole/video"),
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


def resolve_trajectory_name(h5_path: str, user_input: Optional[str]) -> Optional[str]:
    """Resolve user trajectory input (index or name) to a valid group name in the HDF5 file."""
    with h5py.File(h5_path, 'r') as f:
        all_trajectories = sorted([name for name in f.keys() if name.startswith('trajectory_')])
        if not all_trajectories:
            print("No trajectory groups found in the file.")
            return None
        if user_input is None:
            return all_trajectories[0]
        # Try numeric index
        try:
            idx = int(user_input)
            if 0 <= idx < len(all_trajectories):
                return all_trajectories[idx]
            else:
                print(f"Numeric trajectory index {idx} out of range. Available: 0 to {len(all_trajectories)-1}")
                print(f"Available groups: {all_trajectories}")
                return None
        except ValueError:
            # Not an int, treat as string
            if user_input in all_trajectories:
                return user_input
            else:
                print(f"Trajectory '{user_input}' not found. Available: {all_trajectories}")
                return None


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


def visualize_h5_data(
    h5_path: str,
    trajectory_name: Optional[str] = None,
    max_frames: Optional[int] = None,
    frame_rate: int = 30
):
    """
    Visualize HDF5 data using Rerun.
    
    Args:
        h5_path: Path to the HDF5 file
        trajectory_name: Specific trajectory to visualize (if None, uses first available)
        max_frames: Maximum number of frames to visualize (if None, uses all)
        frame_rate: Frame rate for playback
    """
    
    # Initialize Rerun
    rr.init("egocentric-h5-viewer", spawn=True)
    
    # Resolve trajectory name
    resolved_traj = resolve_trajectory_name(h5_path, trajectory_name)
    if resolved_traj is None:
        logging.error("No valid trajectory selected. Exiting.")
        return

    dataset = EgocentricH5Dataset(h5_path, trajectories=[resolved_traj])
    if len(dataset) == 0:
        logging.error("No data found in the specified trajectory")
        return
    
    print(f"Visualizing trajectory: {resolved_traj}")
    print(f"Total frames: {len(dataset)}")
    
    # Create optimal blueprint layout
    blueprint = create_optimal_blueprint()
    
    # Set up coordinate system
    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    
    # Send blueprint
    rr.send_blueprint(blueprint=blueprint)
    
    # Set up the annotation context for the skeleton
    set_pose_annotation_context()
    
    # Log action label from metadata if available
    with h5py.File(h5_path, 'r') as f:
        traj_group = f[resolved_traj]
        if 'metadata' in traj_group and 'action_label' in traj_group['metadata'].attrs:
            action_label = traj_group['metadata'].attrs['action_label']
            rr.log("llm_description", rr.TextDocument(text=action_label), static=True)
    
    # Set up camera logging paths
    cam_log_path = Path("world/camera")
    pinhole_log_path = cam_log_path / "pinhole"
    video_log_path = pinhole_log_path / "video"
    
    # Determine frame range
    frame_range = len(dataset)
    if max_frames:
        frame_range = min(frame_range, max_frames)
    
    # Visualize frames
    for i in range(frame_range):
        print(f"Processing frame {i+1}/{frame_range}")
        
        # Load frame data
        frame_data = dataset[i]
        
        # Set timeline
        rr.set_time_sequence("frame", i)
        
        # Log camera pose if available
        if 'camera_pose' in frame_data and 'camera_intrinsics' in frame_data:
            camera_pose = frame_data['camera_pose'].numpy()  # 4x4 matrix
            camera_intrinsics = frame_data['camera_intrinsics'].numpy()  # 3x3 matrix
            
            # Extract rotation and translation
            rotation = camera_pose[:3, :3]
            translation = camera_pose[:3, 3]
            
            # Create pinhole camera parameters
            fl_x = float(camera_intrinsics[0, 0])
            fl_y = float(camera_intrinsics[1, 1])
            cx = float(camera_intrinsics[0, 2])
            cy = float(camera_intrinsics[1, 2])
            
            pinhole = PinholeParameters(
                name="Canonical Camera",
                intrinsics=Intrinsics(fl_x=fl_x, fl_y=fl_y, cx=cx, cy=cy, camera_conventions="RDF"),
                extrinsics=Extrinsics(world_R_cam=rotation, world_t_cam=translation),
            )
            
            # Log pinhole camera
            log_pinhole(
                pinhole,
                cam_log_path=cam_log_path,
                image_plane_distance=0.1,
                static=False,
            )
        
        # Log RGB image
        if 'rgb' in frame_data:
            rgb = frame_data['rgb']
            # Convert from CHW to HWC for Rerun
            rgb_hwc = rgb.permute(1, 2, 0)
            rr.log(str(video_log_path), rr.Image(rgb_hwc))
        
        # Log skeleton data
        if 'skeleton_positions' in frame_data and 'skeleton_confidences' in frame_data:
            positions = frame_data['skeleton_positions'].numpy()
            confidences = frame_data['skeleton_confidences'].numpy()
            
            # Generate confidence-based colors
            colors = confidence_scores_to_rgb(confidences)
            
            # Log skeleton points
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
            
            # Log confidence values as text if available
            if 'skeleton_joint_names' in frame_data:
                joint_names = frame_data['skeleton_joint_names']
                for i, (pos, conf, name) in enumerate(zip(positions, confidences, joint_names)):
                    if conf > 0.5:  # Only show high-confidence joints
                        rr.log(
                            f"world/skeleton/labels/{i}",
                            rr.TextDocument(f"{name}: {conf:.2f}")
                        )
        
        # Log timestamp
        if 'timestamp_ns' in frame_data:
            timestamp = frame_data['timestamp_ns'].item()
            rr.set_time_nanos("timestamp", timestamp)
    
    print(f"Visualization complete! Check the Rerun viewer.")
    print(f"Timeline: {frame_range} frames at {frame_rate} FPS")


def main():
    parser = argparse.ArgumentParser(description="Visualize egocentric HDF5 data with Rerun")
    parser.add_argument("h5_path", help="Path to the HDF5 file")
    parser.add_argument("--trajectory", help="Specific trajectory to visualize")
    parser.add_argument("--max-frames", type=int, help="Maximum number of frames to visualize")
    parser.add_argument("--frame-rate", type=int, default=30, help="Frame rate for playback")
    
    args = parser.parse_args()
    
    if not Path(args.h5_path).exists():
        print(f"Error: HDF5 file not found at {args.h5_path}")
        return 1
    
    try:
        visualize_h5_data(
            args.h5_path,
            trajectory_name=args.trajectory,
            max_frames=args.max_frames,
            frame_rate=args.frame_rate
        )
        return 0
    except Exception as e:
        logging.error(f"Error during visualization: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main()) 