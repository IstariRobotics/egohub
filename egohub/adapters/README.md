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

class MyAdapter(BaseAdapter):
    name = "my_adapter"
    
    @property
    def source_joint_names(self) -> List[str]:
        # Return list of joint names for your dataset
        pass
    
    @property
    def source_skeleton_hierarchy(self) -> Dict[str, str]:
        # Return skeleton hierarchy for your dataset
        pass
    
    def discover_sequences(self) -> List[Dict[str, Any]]:
        # Discover sequences in raw data
        pass
    
    def process_sequence(self, seq_info: Dict[str, Any], traj_group: h5py.Group):
        # Process a single sequence
        pass
``` 