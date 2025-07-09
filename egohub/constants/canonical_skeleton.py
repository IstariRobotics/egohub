# egohub/constants/canonical_skeleton.py
"""
This file contains constants related to the canonical skeleton definition.
"""

# --- Canonical Skeleton Definition ---
# The following constants define the "Universal Human Rig" for a full-body skeleton.
# This definition is based on the standard 22-joint body model from SMPL-X, which
# provides a robust and widely-used standard for humanoid robotics research.
#
# All skeletons from all source datasets must be remapped to this canonical format
# during the adapter's processing phase. This ensures that downstream models can
# consume skeleton data in a consistent and unambiguous way.
#
# It now includes a full hand model based on the AVP skeleton.

CANONICAL_SKELETON_JOINTS: list[str] = [
    # Body (SMPL-X Base)
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    # Left Hand Fingers
    "left_hand_index_metacarpal",
    "left_hand_index_knuckle",
    "left_hand_index_intermediate_base",
    "left_hand_index_intermediate_tip",
    "left_hand_index_tip",
    "left_hand_little_finger_metacarpal",
    "left_hand_little_finger_knuckle",
    "left_hand_little_finger_intermediate_base",
    "left_hand_little_finger_intermediate_tip",
    "left_hand_little_finger_tip",
    "left_hand_middle_finger_metacarpal",
    "left_hand_middle_finger_knuckle",
    "left_hand_middle_finger_intermediate_base",
    "left_hand_middle_finger_intermediate_tip",
    "left_hand_middle_finger_tip",
    "left_hand_ring_finger_metacarpal",
    "left_hand_ring_finger_knuckle",
    "left_hand_ring_finger_intermediate_base",
    "left_hand_ring_finger_intermediate_tip",
    "left_hand_ring_finger_tip",
    "left_hand_thumb_knuckle",
    "left_hand_thumb_intermediate_base",
    "left_hand_thumb_intermediate_tip",
    "left_hand_thumb_tip",
    # Right Hand Fingers
    "right_hand_index_metacarpal",
    "right_hand_index_knuckle",
    "right_hand_index_intermediate_base",
    "right_hand_index_intermediate_tip",
    "right_hand_index_tip",
    "right_hand_little_finger_metacarpal",
    "right_hand_little_finger_knuckle",
    "right_hand_little_finger_intermediate_base",
    "right_hand_little_finger_intermediate_tip",
    "right_hand_little_finger_tip",
    "right_hand_middle_finger_metacarpal",
    "right_hand_middle_finger_knuckle",
    "right_hand_middle_finger_intermediate_base",
    "right_hand_middle_finger_intermediate_tip",
    "right_hand_middle_finger_tip",
    "right_hand_ring_finger_metacarpal",
    "right_hand_ring_finger_knuckle",
    "right_hand_ring_finger_intermediate_base",
    "right_hand_ring_finger_intermediate_tip",
    "right_hand_ring_finger_tip",
    "right_hand_thumb_knuckle",
    "right_hand_thumb_intermediate_base",
    "right_hand_thumb_intermediate_tip",
    "right_hand_thumb_tip",
]

# Defines the kinematic tree by specifying the parent of each joint.
# The parent is referenced by its string name from the list above.
CANONICAL_SKELETON_HIERARCHY: dict[str, str] = {
    # Body
    "left_hip": "pelvis",
    "right_hip": "pelvis",
    "spine1": "pelvis",
    "left_knee": "left_hip",
    "right_knee": "right_hip",
    "spine2": "spine1",
    "left_ankle": "left_knee",
    "right_ankle": "right_knee",
    "spine3": "spine2",
    "left_foot": "left_ankle",
    "right_foot": "right_ankle",
    "neck": "spine3",
    "left_collar": "spine3",
    "right_collar": "spine3",
    "head": "neck",
    "left_shoulder": "left_collar",
    "right_shoulder": "right_collar",
    "left_elbow": "left_shoulder",
    "right_elbow": "right_shoulder",
    "left_wrist": "left_elbow",
    "right_wrist": "right_elbow",
    # Left Hand
    "left_hand_thumb_knuckle": "left_wrist",
    "left_hand_thumb_intermediate_base": "left_hand_thumb_knuckle",
    "left_hand_thumb_intermediate_tip": "left_hand_thumb_intermediate_base",
    "left_hand_thumb_tip": "left_hand_thumb_intermediate_tip",
    "left_hand_index_metacarpal": "left_wrist",
    "left_hand_index_knuckle": "left_hand_index_metacarpal",
    "left_hand_index_intermediate_base": "left_hand_index_knuckle",
    "left_hand_index_intermediate_tip": "left_hand_index_intermediate_base",
    "left_hand_index_tip": "left_hand_index_intermediate_tip",
    "left_hand_middle_finger_metacarpal": "left_wrist",
    "left_hand_middle_finger_knuckle": "left_hand_middle_finger_metacarpal",
    "left_hand_middle_finger_intermediate_base": "left_hand_middle_finger_knuckle",
    "left_hand_middle_finger_intermediate_tip": (
        "left_hand_middle_finger_intermediate_base"
    ),
    "left_hand_middle_finger_tip": "left_hand_middle_finger_intermediate_tip",
    "left_hand_ring_finger_metacarpal": "left_wrist",
    "left_hand_ring_finger_knuckle": "left_hand_ring_finger_metacarpal",
    "left_hand_ring_finger_intermediate_base": "left_hand_ring_finger_knuckle",
    "left_hand_ring_finger_intermediate_tip": "left_hand_ring_finger_intermediate_base",
    "left_hand_ring_finger_tip": "left_hand_ring_finger_intermediate_tip",
    "left_hand_little_finger_metacarpal": "left_wrist",
    "left_hand_little_finger_knuckle": "left_hand_little_finger_metacarpal",
    "left_hand_little_finger_intermediate_base": "left_hand_little_finger_knuckle",
    "left_hand_little_finger_intermediate_tip": (
        "left_hand_little_finger_intermediate_base"
    ),
    "left_hand_little_finger_tip": "left_hand_little_finger_intermediate_tip",
    # Right Hand
    "right_hand_thumb_knuckle": "right_wrist",
    "right_hand_thumb_intermediate_base": "right_hand_thumb_knuckle",
    "right_hand_thumb_intermediate_tip": "right_hand_thumb_intermediate_base",
    "right_hand_thumb_tip": "right_hand_thumb_intermediate_tip",
    "right_hand_index_metacarpal": "right_wrist",
    "right_hand_index_knuckle": "right_hand_index_metacarpal",
    "right_hand_index_intermediate_base": "right_hand_index_knuckle",
    "right_hand_index_intermediate_tip": "right_hand_index_intermediate_base",
    "right_hand_index_tip": "right_hand_index_intermediate_tip",
    "right_hand_middle_finger_metacarpal": "right_wrist",
    "right_hand_middle_finger_knuckle": "right_hand_middle_finger_metacarpal",
    "right_hand_middle_finger_intermediate_base": "right_hand_middle_finger_knuckle",
    "right_hand_middle_finger_intermediate_tip": (
        "right_hand_middle_finger_intermediate_base"
    ),
    "right_hand_middle_finger_tip": "right_hand_middle_finger_intermediate_tip",
    "right_hand_ring_finger_metacarpal": "right_wrist",
    "right_hand_ring_finger_knuckle": "right_hand_ring_finger_metacarpal",
    "right_hand_ring_finger_intermediate_base": "right_hand_ring_finger_knuckle",
    "right_hand_ring_finger_intermediate_tip": (
        "right_hand_ring_finger_intermediate_base"
    ),
    "right_hand_ring_finger_tip": "right_hand_ring_finger_intermediate_tip",
    "right_hand_little_finger_metacarpal": "right_wrist",
    "right_hand_little_finger_knuckle": "right_hand_little_finger_metacarpal",
    "right_hand_little_finger_intermediate_base": "right_hand_little_finger_knuckle",
    "right_hand_little_finger_intermediate_tip": (
        "right_hand_little_finger_intermediate_base"
    ),
    "right_hand_little_finger_tip": "right_hand_little_finger_intermediate_tip",
}
