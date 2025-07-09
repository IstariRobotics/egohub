# egohub/constants/__init__.py
"""
This module centralizes all the constants used in the egohub project,
making them easily accessible from a single, organized location.
"""
from .canonical_skeleton import (
    CANONICAL_SKELETON_HIERARCHY,
    CANONICAL_SKELETON_JOINTS,
)
from .egodex_skeleton import (
    AVP_ID2NAME,
    AVP_IDS,
    AVP_LINKS,
    AVP_NAME2ID,
    EGODEX_SKELETON_HIERARCHY,
    EGODEX_SKELETON_JOINTS,
)

__all__ = [
    "CANONICAL_SKELETON_JOINTS",
    "CANONICAL_SKELETON_HIERARCHY",
    "EGODEX_SKELETON_JOINTS",
    "EGODEX_SKELETON_HIERARCHY",
    "AVP_ID2NAME",
    "AVP_NAME2ID",
    "AVP_IDS",
    "AVP_LINKS",
]
