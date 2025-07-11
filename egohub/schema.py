"""
Canonical Data Schema for the Egocentric Data Pipeline.

This file defines the standard HDF5 structure for all processed datasets.
Using a canonical schema ensures that data is consistent and can be
consumed by a single, unified dataloader and visualization tools.

---
Canonical Coordinate System Definition:
- **Handedness**: Right-handed
- **World Frame**:
    - `+Z` is up
    - `+Y` is forward (away from the camera)
    - `+X` is to the right
- **Camera Frame (OpenCV Pinhole Model)**:
    - `+Z` is forward (into the scene)
    - `+Y` is down
    - `+X` is to the right
- **Pose Convention**:
    - All poses are stored as 4x4 homogeneous transformation matrices.
    - These matrices represent the transform `T_world_local`, which transforms
      a point from the entity's local frame into the world frame.
---
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field, fields, is_dataclass
from typing import Any, Dict, Optional, Type, get_args, get_origin

import h5py
import numpy as np


# --- Custom Exception for Schema Validation ---
class SchemaValidationError(ValueError):
    """Custom exception raised for HDF5 schema validation errors."""

    pass


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
    image_bytes: Any = field(
        metadata={"shape": (-1,), "dtype": h5py.special_dtype(vlen=bytes)}
    )
    frame_sizes: np.ndarray = field(metadata={"shape": (-1,), "dtype": np.int32})
    frame_indices: np.ndarray = field(metadata={"shape": (-1,), "dtype": np.uint64})


@dataclass
class Camera:
    intrinsics: np.ndarray = field(metadata={"shape": (3, 3), "dtype": np.float32})
    pose_in_world: np.ndarray = field(
        metadata={"shape": (-1, 4, 4), "dtype": np.float32}
    )
    pose_indices: np.ndarray = field(metadata={"shape": (-1,), "dtype": np.uint64})
    is_ego: bool = field(metadata={"h5_type": "attr"})
    rgb: Optional[Rgb] = None


@dataclass
class Hand:
    pose_in_world: np.ndarray = field(
        metadata={"shape": (-1, 4, 4), "dtype": np.float32}
    )
    pose_indices: np.ndarray = field(metadata={"shape": (-1,), "dtype": np.uint64})


@dataclass
class Skeleton:
    positions: np.ndarray = field(metadata={"shape": (-1, -1, 3), "dtype": np.float32})
    confidences: np.ndarray = field(metadata={"shape": (-1, -1), "dtype": np.float32})
    frame_indices: np.ndarray = field(metadata={"shape": (-1,), "dtype": np.uint64})


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


def _validate_unexpected_members(
    h5_group: h5py.Group, schema_fields: dict, path: str, strict: bool
) -> None:
    """Check for unexpected members in the HDF5 group."""
    for member_name in h5_group.keys():
        if member_name not in schema_fields:
            msg = f"Validation: Unexpected group/dataset '{path}/{member_name}' found."
            if strict:
                raise SchemaValidationError(msg)
            logging.warning(msg)


def _validate_attribute(
    field_name: str, h5_group: h5py.Group, path: str, strict: bool
) -> None:
    """Validate attributes on the parent group."""
    if field_name not in h5_group.attrs:
        msg = f"Validation: Missing attribute '{field_name}' in group '{path}'"
        if strict:
            raise SchemaValidationError(msg)
        logging.warning(msg)


def _validate_dataset_dtype(
    member: h5py.Dataset, field_type: Any, current_path: str, strict: bool
) -> None:
    """Validate dataset dtype."""
    expected_dtype = field_type.metadata.get("dtype")
    if not expected_dtype:
        return

    if field_type.type == h5py.special_dtype(vlen=bytes):
        if not h5py.check_vlen_dtype(member.dtype) is not bytes:
            msg = f"Validation: Mismatched dtype for vlen dataset '{current_path}'."
            if strict:
                raise SchemaValidationError(msg)
            logging.warning(msg)
    elif member.dtype != expected_dtype:
        msg = (
            f"Validation: Mismatched dtype for dataset '{current_path}'. "
            f"Got {member.dtype}, expected {expected_dtype}"
        )
        if strict:
            raise SchemaValidationError(msg)
        logging.warning(msg)


def _validate_dataset_shape(
    member: h5py.Dataset, field_type: Any, current_path: str, strict: bool
) -> None:
    """Validate dataset shape."""
    expected_shape_meta = field_type.metadata.get("shape")
    if not expected_shape_meta:
        return

    if len(member.shape) != len(expected_shape_meta):
        # We only validate rank, not specific dimension sizes
        # (which can be variable, e.g., -1)
        msg = (
            "Validation: Mismatched rank for dataset '"
            f"{current_path}'. Got rank {len(member.shape)}, expected rank "
            f"{len(expected_shape_meta)}"
        )
        if strict:
            raise SchemaValidationError(msg)
        logging.warning(msg)


def _validate_dict_groups(
    h5_group: h5py.Group, schema_cls: Type[Any], path: str, strict: bool
) -> None:
    """Validate dictionary groups (like 'cameras' or 'hands')."""
    for f in fields(schema_cls):
        if get_origin(f.type) is dict:
            dict_group = h5_group.get(f.name)
            if not dict_group:
                continue

            value_type = get_args(f.type)[1]
            if not is_dataclass(value_type) or not isinstance(dict_group, h5py.Group):
                continue

            for child_name, child_group in dict_group.items():
                if isinstance(child_group, h5py.Group):
                    validate_hdf5_with_schema(
                        child_group,
                        value_type,
                        f"{path}/{f.name}/{child_name}",
                        strict=strict,
                    )


def validate_hdf5_with_schema(
    h5_group: h5py.Group,
    schema_cls: Type[Any] | Any,
    path: str = "",
    strict: bool = False,
):
    """
    Recursively validates an HDF5 group against a dataclass schema.

    Args:
        h5_group (h5py.Group): The HDF5 group to validate.
        schema_cls (Type[Any] | Any): The dataclass schema to validate against.
        path (str): The current path within the HDF5 file (for logging).
        strict (bool): If True, raises SchemaValidationError on failure.
                       Otherwise, logs a warning.
    """
    schema_fields = {f.name: f for f in fields(schema_cls)}

    # Check for unexpected members in the HDF5 group
    _validate_unexpected_members(h5_group, schema_fields, path, strict)

    # Check for missing members and validate existing ones
    for f in fields(schema_cls):
        field_name = f.name
        field_type = f.type
        current_path = f"{path}/{field_name}"

        # Attributes are validated on the parent group
        if f.metadata.get("h5_type") == "attr":
            _validate_attribute(field_name, h5_group, path, strict)
            continue

        # Check if required members are present
        if field_name not in h5_group:
            if not (
                get_origin(field_type) is Optional or get_origin(field_type) is dict
            ):
                msg = f"Validation: Missing required group/dataset '{current_path}'"
                if strict:
                    raise SchemaValidationError(msg)
                logging.warning(msg)
            continue

        member = h5_group[field_name]

        # Recurse for nested groups
        if is_dataclass(field_type) and isinstance(member, h5py.Group):
            validate_hdf5_with_schema(member, field_type, current_path, strict=strict)

        # Validate datasets
        elif isinstance(member, h5py.Dataset):
            _validate_dataset_dtype(member, f, current_path, strict)
            _validate_dataset_shape(member, f, current_path, strict)

    # Validate dictionary groups
    _validate_dict_groups(h5_group, schema_cls, path, strict)
