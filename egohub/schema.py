"""
Canonical Data Schema for the Egocentric Data Pipeline.

This file defines the standard HDF5 structure for all processed datasets.
Using a canonical schema ensures that the data is consistent and can be
consumed by a single, unified dataloader and visualization tools.
"""

# --- Coordinate System Definitions ---
# The pipeline uses a right-handed, Z-up coordinate system for the 'world' frame.
#   +Z is up
#   +Y is forward
#   +X is to the right
#
# The 'camera' frame follows the standard OpenCV pinhole camera model:
#   +Z is forward (into the scene)
#   +Y is down
#   +X is to the right
#
# All poses are stored as 4x4 homogenous transformation matrices that transform
# a point from the entity's local coordinate frame to the 'world' frame.
# For example, `camera_pose_in_world` transforms a point from the camera's
# frame to the world frame.

CANONICAL_SCHEMA = {
    "description": "Root group for a single, continuous egocentric recording.",
    "trajectory_{:04d}": {
        "description": "A group containing all data for a single trajectory.",
        "metadata": {
            "description": "High-level information about the trajectory.",
            "attributes": {
                "uuid": "A unique identifier for the trajectory.",
                "source_dataset": "Name of the original dataset (e.g., 'EgoDex').",
                "source_identifier": "The original file/sequence name from the source.",
                "action_label": "A short, high-level description of the action performed.",
            },
            "datasets": {
                "timestamps_ns": "Synchronized timestamps for all data streams (uint64, in nanoseconds).",
            },
        },
        "rgb": {
            "description": "RGB image data.",
            "datasets": {
                "image_bytes": "Variable-length dataset of JPG-compressed image bytes.",
                "frame_indices": "Index mapping timestamps to image frames.",
            },
        },
        "depth": {
            "description": "Depth map data.",
            "attributes": {
                "scale_factor": "Multiplier to convert depth values to meters (e.g., 1000.0 for 16-bit millimeter data)."
            },
            "datasets": {
                "image_bytes": "Variable-length dataset of lossless PNG-compressed 16-bit depth maps.",
                "frame_indices": "Index mapping timestamps to depth frames.",
            },
        },
        "camera": {
            "description": "Camera information, including pose and intrinsics.",
            "datasets": {
                "pose_in_world": "4x4 homogenous transformation matrix (world <- camera).",
                "intrinsics": "3x3 pinhole camera intrinsic matrix.",
                "frame_indices": "Index mapping timestamps to camera data.",
            },
        },
        "hands": {
            "description": "Hand tracking data.",
            "left": {
                "description": "Data for the left hand.",
                "datasets": {
                    "pose_in_world": "4x4 homogenous transformation matrix for the wrist (world <- hand).",
                    "mano_betas": "10 MANO shape parameters.",
                    "mano_thetas": "48 MANO pose parameters (3 global rotation + 45 joint rotations).",
                    "frame_indices": "Index mapping timestamps to hand data.",
                },
            },
            "right": {
                "description": "Data for the right hand.",
                "datasets": {
                    "pose_in_world": "4x4 homogenous transformation matrix for the wrist (world <- hand).",
                    "mano_betas": "10 MANO shape parameters.",
                    "mano_thetas": "48 MANO pose parameters (3 global rotation + 45 joint rotations).",
                    "frame_indices": "Index mapping timestamps to hand data.",
                },
            },
        },
        "objects": {
            "description": "Data for tracked objects in the scene.",
            "object_{:02d}": {
                "description": "Data for a specific object instance.",
                "attributes": {
                    "label": "Semantic label for the object (e.g., 'cup', 'screwdriver')."
                },
                "datasets": {
                    "pose_in_world": "4x4 homogenous transformation matrix (world <- object).",
                    "frame_indices": "Index mapping timestamps to object data.",
                },
            },
        },
    },
} 