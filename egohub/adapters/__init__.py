# This file makes the 'adapters' directory a Python package.
from .base import BaseAdapter
from .egodex.egodex import EgoDexAdapter

# Note: AriaDigitalTwinAdapter is temporarily disabled
# due to missing implementation file.
# from .aria_digital_twin.aria_digital_twin import AriaDigitalTwinAdapter

__all__ = [
    "BaseAdapter",
    "EgoDexAdapter",
    # "AriaDigitalTwinAdapter",
]
