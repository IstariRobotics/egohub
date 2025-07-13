# Dataset Adapters

This directory contains dataset adapters that convert raw dataset formats into the canonical EgoHub HDF5 format.

## Structure

Each adapter is organized in its own folder named after the adapter:

```
adapters/
├── base.py                    # Base adapter class
├── egodex/                    # EgoDex dataset adapter
│   └── egodex.py             # EgoDex adapter implementation
└── [other_adapters]/         # Other dataset adapters
    └── [adapter_name].py     # Adapter implementation
```

## Configuration

Adapter configurations are stored in the project root `configs/` directory:

```
configs/
├── egodex.yaml              # EgoDex adapter configuration
└── [other_adapters].yaml    # Other adapter configurations
```

The `BaseAdapter` class automatically loads configuration files by:
1. First looking in the adapter's own folder (`adapters/[adapter_name]/[adapter_name].yaml`)
2. Falling back to the project root configs directory (`configs/[adapter_name].yaml`)

## Adding a New Adapter

To add a new dataset adapter:

1. Create a new folder in `adapters/` named after your adapter
2. Create the adapter implementation file (e.g., `my_adapter.py`)
3. Create a configuration file in `configs/my_adapter.yaml`
4. Implement the required methods from `BaseAdapter`

Example:
```python
from egohub.adapters.base import BaseAdapter
from egohub.adapters.dataset_info import DatasetInfo

class MyAdapter(BaseAdapter):
    name = "my_adapter"

    # ------------------------------------------------------------------
    # Required skeletal metadata
    # ------------------------------------------------------------------

    @property
    def source_joint_names(self) -> List[str]:
        # Return list of joint names for your dataset
        ...

    @property
    def source_skeleton_hierarchy(self) -> Dict[str, str]:
        # Return skeleton hierarchy for your dataset
        ...

    # ------------------------------------------------------------------
    # New unified metadata interface (required)
    # ------------------------------------------------------------------

    def get_camera_intrinsics(self) -> Dict[str, Any]:
        # Return camera intrinsics as a 3×3 matrix or a dict containing one
        ...

    def get_dataset_info(self) -> DatasetInfo:
        # Provide dataset-wide metadata (frame-rate, joint remaps, etc.)
        ...

    # ------------------------------------------------------------------
    # Sequence discovery & processing
    # ------------------------------------------------------------------

    def discover_sequences(self) -> List[Dict[str, Any]]:
        # Discover sequences in raw data
        ...

    def process_sequence(self, seq_info: Dict[str, Any], traj_group: h5py.Group):
        # Process a single sequence
        ...
```

### Skeleton Remapping

Adapters **must** remap their source skeleton joints to the project-wide canonical
Mediapipe skeleton (33 pose landmarks + 21 hand landmarks per hand, plus a
virtual `pelvis` root).  The canonical joint list and hierarchy live in
`egohub/constants/canonical_skeleton.py`.  Your adapter should expose a
`joint_remap` dict in `DatasetInfo` (see `EgoDexAdapter` for an example) that
maps each *source* joint name to its corresponding canonical joint.

Failing to provide a complete remap will raise an error during processing. 