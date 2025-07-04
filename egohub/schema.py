"""
Canonical Data Schema for the Egocentric Data Pipeline.

This file defines the standard HDF5 structure for all processed datasets.
Using a canonical schema ensures that the data is consistent and can be
consumed by a single, unified dataloader and visualization tools.
"""
from __future__ import annotations
from dataclasses import dataclass, field, fields, is_dataclass
import logging
from typing import List, Optional, Dict, get_args, get_origin, Any, Type

import h5py
import numpy as np

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

# --- Type Aliases for Clarity ---
# class PoseInWorld(TypedDict):
#     shape: tuple[int, int]
#     dtype: np.dtype
#     data: np.ndarray

# --- Schema Definition using dataclasses ---

@dataclass
class Rgb:
    image_bytes: Any = field(metadata={"shape": (-1,), "dtype": h5py.special_dtype(vlen=bytes)})
    frame_sizes: np.ndarray = field(metadata={"shape": (-1,), "dtype": np.int32})

@dataclass
class Camera:
    intrinsics: np.ndarray = field(metadata={"shape": (3, 3), "dtype": np.float32})
    pose_in_world: np.ndarray = field(metadata={"shape": (-1, 4, 4), "dtype": np.float32})
    is_ego: bool = field(metadata={"h5_type": "attr"})
    rgb: Optional['Rgb'] = None

@dataclass
class Hand:
    pose_in_world: np.ndarray = field(metadata={"shape": (-1, 4, 4), "dtype": np.float32})

@dataclass
class Skeleton:
    positions: np.ndarray = field(metadata={"shape": (-1, -1, 3), "dtype": np.float32})
    confidences: np.ndarray = field(metadata={"shape": (-1, -1), "dtype": np.float32})
    joint_names: List[str] = field(metadata={"h5_type": "attr"})

@dataclass
class Metadata:
    timestamps_ns: np.ndarray = field(metadata={"shape": (-1,), "dtype": np.uint64})
    uuid: str = field(metadata={"h5_type": "attr"})
    source_dataset: str = field(metadata={"h5_type": "attr"})
    source_identifier: str = field(metadata={"h5_type": "attr"})
    action_label: str = field(metadata={"h5_type": "attr"})

@dataclass
class Trajectory:
    metadata: Metadata
    cameras: Dict[str, Camera]
    hands: Dict[str, Hand]
    skeleton: Optional[Skeleton] = None

# --- Helper function to create HDF5 structure from schema ---
# Removing this function to simplify the pipeline and resolve errors.
# The adapter will be responsible for creating the HDF5 structure directly.

def validate_hdf5_with_schema(h5_group: h5py.Group, schema_cls: Type[Any], path: str = ""):
    """
    Recursively validates an HDF5 group against a dataclass schema.
    """
    schema_fields = {f.name: f for f in fields(schema_cls)}

    # Check for unexpected members in the HDF5 group
    for member_name in h5_group.keys():
        if member_name not in schema_fields:
            logging.warning(f"Validation: Unexpected group/dataset '{path}/{member_name}' found.")

    # Check for missing members and validate existing ones
    for f in fields(schema_cls):
        field_name = f.name
        field_type = f.type
        current_path = f"{path}/{field_name}"

        # Attributes are validated on the parent group
        if f.metadata.get("h5_type") == "attr":
            if field_name not in h5_group.attrs:
                logging.warning(f"Validation: Missing attribute '{field_name}' in group '{path}'")
            continue
        
        # Check if required members are present
        if field_name not in h5_group:
            if not (get_origin(field_type) is Optional or get_origin(field_type) is dict):
                logging.warning(f"Validation: Missing required group/dataset '{current_path}'")
            continue

        member = h5_group[field_name]

        # Recurse for nested groups
        if is_dataclass(field_type) and isinstance(member, h5py.Group):
            validate_hdf5_with_schema(member, field_type, current_path)
        
        # Validate datasets
        elif isinstance(member, h5py.Dataset):
            expected_dtype = f.metadata.get("dtype")
            expected_shape_meta = f.metadata.get("shape")

            if expected_dtype:
                if f.type == h5py.special_dtype(vlen=bytes):
                    if not h5py.check_vlen_dtype(member.dtype) == bytes:
                        logging.warning(f"Validation: Mismatched dtype for vlen dataset '{current_path}'.")
                elif member.dtype != expected_dtype:
                    logging.warning(f"Validation: Mismatched dtype for dataset '{current_path}'. Got {member.dtype}, expected {expected_dtype}")

            if expected_shape_meta and len(member.shape) != len(expected_shape_meta):
                logging.warning(f"Validation: Mismatched rank for dataset '{current_path}'. Got {len(member.shape)}, expected rank {len(expected_shape_meta)}")

    # Validate dictionary groups (like 'cameras' or 'hands')
    for f in fields(schema_cls):
        if get_origin(f.type) is dict:
            dict_group = h5_group.get(f.name)
            if not dict_group: continue

            value_type = get_args(f.type)[1]
            if not is_dataclass(value_type) or not isinstance(dict_group, h5py.Group):
                continue
            
            for child_name, child_group in dict_group.items():
                if isinstance(child_group, h5py.Group):
                    validate_hdf5_with_schema(child_group, value_type, f"{path}/{f.name}/{child_name}")


# --- Deprecated Schema and Template ---

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
        "cameras": {
            "description": "Group containing data for all cameras.",
            "{camera_name}": {
                "description": "Data for a single camera, identified by a unique name.",
                "attributes": {
                    "is_ego": "Boolean attribute, true if the camera is egocentric.",
                },
                "datasets": {
                    "pose_in_world": "4x4 homogenous transformation matrix (world <- camera).",
                    "intrinsics": "3x3 pinhole camera intrinsic matrix.",
                    "frame_indices": "Index mapping timestamps to camera data.",
                },
                "rgb": {
                    "description": "RGB image data for this camera.",
                    "datasets": {
                        "image_bytes": "Variable-length dataset of JPG-compressed image bytes.",
                        "frame_indices": "Index mapping timestamps to image frames.",
                    },
                },
                "depth": {
                    "description": "Depth map data for this camera.",
                    "attributes": {
                        "scale_factor": "Multiplier to convert depth values to meters (e.g., 1000.0 for 16-bit millimeter data)."
                    },
                    "datasets": {
                        "image": "Dataset of depth images, typically 16-bit.",
                        "frame_indices": "Index mapping timestamps to depth frames.",
                    },
                },
            }
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

# A definitive list of all possible canonical data streams for validation.
# This is now a template. '{camera_name}' should be replaced with actual camera names.
CANONICAL_DATA_STREAMS_TEMPLATE = [
    "metadata/timestamps_ns",
    "cameras/{camera_name}/intrinsics",
    "cameras/{camera_name}/pose_in_world",
    "cameras/{camera_name}/rgb/image_bytes",
    "cameras/{camera_name}/depth/image",
    "hands/left/pose_in_world",
    "hands/right/pose_in_world",
    "objects/{object_name}/pose_in_world",
    "skeleton/positions",
    "skeleton/confidences",
] 