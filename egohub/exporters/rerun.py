import io
import logging
from pathlib import Path
from typing import List, Optional

import h5py
import numpy as np
import rerun as rr
import rerun.blueprint as rrb
from PIL import Image

from egohub.constants import (
    CANONICAL_SKELETON_HIERARCHY,
    CANONICAL_SKELETON_JOINTS,
)

logger = logging.getLogger(__name__)


class RerunExporter:
    """Exports egocentric data directly from an HDF5 file to Rerun."""

    def __init__(self, max_frames: Optional[int] = None):
        self.max_frames = max_frames

    def export(self, h5_path: Path, output_path: Optional[Path] = None):
        """Visualize HDF5 data using Rerun by reading the file directly."""
        if not h5_path.exists():
            logger.error(f"Input file not found: {h5_path}")
            return

        if output_path:
            rr.init("egocentric-h5-viewer", spawn=False)
            rr.save(str(output_path))
        else:
            rr.init("egocentric-h5-viewer", spawn=True)

        with h5py.File(h5_path, "r") as f:
            traj_keys = sorted(
                [key for key in f.keys() if key.startswith("trajectory_")]
            )
            if not traj_keys:
                logger.error("No trajectories found in the HDF5 file.")
                return

            traj_key = traj_keys[0]
            traj_group = f[traj_key]
            if isinstance(traj_group, h5py.Group):
                self._visualize_trajectory(traj_group, traj_key)
            else:
                logger.warning(f"'{traj_key}' is not a group, skipping visualization.")

    def _visualize_trajectory(self, traj_group: h5py.Group, traj_name: str):
        """Logs a single trajectory to Rerun."""
        cameras_group = traj_group.get("cameras")
        cam_keys = []
        if isinstance(cameras_group, h5py.Group):
            cam_keys = list(cameras_group.keys())

        if not cam_keys:
            logger.warning(f"No cameras found for trajectory {traj_name}, skipping.")
            return

        blueprint = self._create_blueprint(cam_keys)
        rr.send_blueprint(blueprint)
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)

        self._log_static_metadata(traj_group)
        self._log_static_camera_data(traj_group, cam_keys)
        self._set_pose_annotation_context()

        timestamps_dset = traj_group.get("metadata/timestamps_ns")
        if not isinstance(timestamps_dset, h5py.Dataset):
            logger.error("No 'timestamps_ns' dataset found. Cannot visualize.")
            return

        master_timestamps_ns = timestamps_dset[:]
        num_frames = len(master_timestamps_ns)

        if self.max_frames:
            num_frames = min(num_frames, self.max_frames)

        logger.info(f"Visualizing {num_frames} frames for {traj_name}...")
        for i in range(num_frames):
            timestamp_ns = master_timestamps_ns[i]
            rr.set_time_sequence("frame", i)
            rr.set_time_nanos("timestamp", timestamp_ns)
            self._log_temporal_camera_data(traj_group, cam_keys, i)
            self._log_temporal_object_data(traj_group, cam_keys, i)
            self._log_temporal_skeleton_data(traj_group, i)

    def _create_blueprint(self, camera_names: List[str]) -> rrb.Blueprint:
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

    def _log_static_metadata(self, traj_group: h5py.Group):
        metadata_group = traj_group.get("metadata")
        if (
            isinstance(metadata_group, h5py.Group)
            and "action_label" in metadata_group.attrs
        ):
            action_label = metadata_group.attrs["action_label"]
            rr.log(
                "llm_description", rr.TextDocument(text=str(action_label)), static=True
            )

    def _log_static_camera_data(self, traj_group: h5py.Group, cam_keys: List[str]):
        for cam_key in cam_keys:
            cam_group = traj_group.get(f"cameras/{cam_key}")
            if not isinstance(cam_group, h5py.Group):
                continue

            img_path = f"world/cameras/{cam_key}/image"
            intr_dset = cam_group.get("intrinsics")
            rgb_grp = cam_group.get("rgb")

            if isinstance(intr_dset, h5py.Dataset) and isinstance(rgb_grp, h5py.Group):
                intr = intr_dset[:]
                frame_sizes = rgb_grp.get("frame_sizes")
                image_bytes = rgb_grp.get("image_bytes")

                if (
                    isinstance(frame_sizes, h5py.Dataset)
                    and isinstance(image_bytes, h5py.Dataset)
                    and len(frame_sizes) > 0
                ):
                    num_bytes = frame_sizes[0]
                    encoded_frame = image_bytes[0, :num_bytes]
                    image = Image.open(io.BytesIO(encoded_frame))
                    width, height = image.size
                    rr.log(
                        img_path,
                        rr.Pinhole(
                            image_from_camera=intr,
                            width=width,
                            height=height,
                            camera_xyz=rr.ViewCoordinates.RDF,
                        ),
                        static=True,
                    )

    def _set_pose_annotation_context(self) -> None:
        keypoint_connections = []
        for child, parent in CANONICAL_SKELETON_HIERARCHY.items():
            if (
                child in CANONICAL_SKELETON_JOINTS
                and parent in CANONICAL_SKELETON_JOINTS
            ):
                child_id = CANONICAL_SKELETON_JOINTS.index(child)
                parent_id = CANONICAL_SKELETON_JOINTS.index(parent)
                keypoint_connections.append((parent_id, child_id))

        rr.log(
            "world/skeleton",
            rr.AnnotationContext(
                [
                    rr.ClassDescription(
                        info=rr.AnnotationInfo(id=0, label="Canonical Skeleton"),
                        keypoint_annotations=[
                            rr.AnnotationInfo(id=i, label=name)
                            for i, name in enumerate(CANONICAL_SKELETON_JOINTS)
                        ],
                        keypoint_connections=keypoint_connections,
                    )
                ]
            ),
            static=True,
        )

    def _log_temporal_camera_data(
        self, traj_group: h5py.Group, cam_keys: List[str], frame_idx: int
    ):
        for cam_key in cam_keys:
            cam_group = traj_group.get(f"cameras/{cam_key}")
            if not isinstance(cam_group, h5py.Group):
                continue

            pose_dset = cam_group.get("pose_in_world")
            pose_indices_dset = cam_group.get("pose_indices")
            if isinstance(pose_dset, h5py.Dataset) and isinstance(
                pose_indices_dset, h5py.Dataset
            ):
                pose_indices = pose_indices_dset[:]
                current_indices = np.where(pose_indices == frame_idx)[0]
                for idx in current_indices:
                    pose = pose_dset[idx]
                    rr.log(
                        f"world/cameras/{cam_key}",
                        rr.Transform3D(translation=pose[:3, 3], mat3x3=pose[:3, :3]),
                    )

            rgb_grp = cam_group.get("rgb")
            if isinstance(rgb_grp, h5py.Group):
                frame_sizes = rgb_grp.get("frame_sizes")
                image_bytes = rgb_grp.get("image_bytes")
                frame_indices_dset = rgb_grp.get("frame_indices")
                if (
                    isinstance(frame_sizes, h5py.Dataset)
                    and isinstance(image_bytes, h5py.Dataset)
                    and isinstance(frame_indices_dset, h5py.Dataset)
                ):
                    frame_indices = frame_indices_dset[:]
                    current_indices = np.where(frame_indices == frame_idx)[0]
                    for idx in current_indices:
                        num_bytes = frame_sizes[idx]
                        encoded_frame = image_bytes[idx, :num_bytes]

                        image = Image.open(io.BytesIO(encoded_frame))
                        image_np = np.array(image)

                        rr.log(f"world/cameras/{cam_key}/image", rr.Image(image_np))

    def _log_temporal_object_data(
        self, traj_group: h5py.Group, cam_keys: List[str], frame_idx: int
    ):
        # This function will need to be updated if/when objects get temporal indices
        objects_group = traj_group.get("objects")
        if not isinstance(objects_group, h5py.Group):
            return

        all_boxes, all_labels = [], []
        for label, obj_grp in objects_group.items():
            if isinstance(obj_grp, h5py.Group):
                scores_dset = obj_grp.get("scores")
                bboxes_dset = obj_grp.get("bboxes_2d")
                if isinstance(scores_dset, h5py.Dataset) and isinstance(
                    bboxes_dset, h5py.Dataset
                ):
                    if frame_idx < len(scores_dset):
                        score = scores_dset[frame_idx]
                        if score > 0:
                            all_boxes.append(bboxes_dset[frame_idx])
                            all_labels.append(f"{label}: {score:.2f}")

        if all_boxes and cam_keys:
            img_path = f"world/cameras/{cam_keys[0]}/image"
            rr.log(
                img_path,
                rr.Boxes2D(
                    array=np.array(all_boxes),
                    array_format=rr.Box2DFormat.XYWH,
                    labels=all_labels,
                ),
            )

    def _log_temporal_skeleton_data(self, traj_group: h5py.Group, frame_idx: int):
        skeleton_group = traj_group.get("skeleton")
        if not isinstance(skeleton_group, h5py.Group):
            return

        pos_dset = skeleton_group.get("positions")
        conf_dset = skeleton_group.get("confidences")
        frame_indices_dset = skeleton_group.get("frame_indices")
        if (
            isinstance(pos_dset, h5py.Dataset)
            and isinstance(conf_dset, h5py.Dataset)
            and isinstance(frame_indices_dset, h5py.Dataset)
        ):
            frame_indices = frame_indices_dset[:]
            current_indices = np.where(frame_indices == frame_idx)[0]

            for idx in current_indices:
                positions = pos_dset[idx]
                confidences = conf_dset[idx]
                colors = self._confidence_scores_to_rgb(confidences)

                rr.log(
                    "world/skeleton",
                    rr.Points3D(
                        positions=positions,
                        colors=colors,
                        radii=0.01,
                        class_ids=0,
                        keypoint_ids=list(range(len(CANONICAL_SKELETON_JOINTS))),
                    ),
                )

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
