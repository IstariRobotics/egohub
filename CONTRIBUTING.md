# Contributing to egohub

We welcome contributions! This guide provides instructions for extending `egohub` with new dataset adapters.

## How to Add a New Dataset Adapter

Adding support for a new egocentric dataset involves creating a new **Adapter** class and a corresponding **configuration file**. The adapter is responsible for converting the raw, dataset-specific format into our canonical HDF5 schema, while the config file manages dataset-specific parameters.

Here is the step-by-step process, using a hypothetical `AwesomeEgo` dataset as an example.

### Step 1: Create the Configuration File

First, create a YAML configuration file in the `configs/` directory. The filename must be the snake-cased name of the dataset. This name will be used to link the adapter to its configuration.

-   **File:** `configs/awesome_ego.yaml`

```yaml
# configs/awesome_ego.yaml

# Frame rate of the source videos
frame_rate: 60.0

# Default camera intrinsics if not present in the data
default_intrinsics:
  - [1000.0, 0.0, 960.0]
  - [0.0, 1000.0, 540.0]
  - [0.0, 0.0, 1.0]

# RGB encoding settings
rgb_encoding:
  format: .jpg
  jpeg_quality: 95
```

### Step 2: Create the Adapter File

Create a new Python file in the `egohub/adapters/` directory.

-   **File:** `egohub/adapters/awesome_ego.py`

### Step 3: Implement the Adapter Class

Inside your new file, create a class that inherits from `egohub.adapters.base.BaseAdapter`. You must implement the following:
1.  A `name` class attribute that **must** match the name of your configuration file (without the `.yaml` extension). The command-line interface uses this name to find your adapter.
2.  The `discover_sequences` and `process_sequence` methods.
3.  The `source_joint_names` and `source_skeleton_hierarchy` properties.

```python
# egohub/adapters/awesome_ego.py

import logging
from pathlib import Path
import h5py
import numpy as np

from egohub.adapters.base import BaseAdapter

class AwesomeEgoAdapter(BaseAdapter):
    """Adapter for the AwesomeEgo dataset."""
    name = "awesome_ego" # This MUST match the config file name

    @property
    def source_joint_names(self) -> list[str]:
        """
        Returns the list of joint names from the source dataset.
        If the dataset has no skeleton data, return an empty list.
        """
        return []

    @property
    def source_skeleton_hierarchy(self) -> dict[str, str]:
        """
        Returns the kinematic hierarchy for the source skeleton.
        If the dataset has no skeleton data, return an empty dict.
        """
        return {}

    def discover_sequences(self) -> list[dict]:
        """
        Scan the raw data directory and return a list of all processable sequences.
        Each item in the list should be a dictionary containing enough information
        to process a single sequence (e.g., paths to video and annotation files).
        """
        logging.info(f"Discovering AwesomeEgo sequences in {self.raw_dir}...")
        sequences = []
        # Example: Find all .mp4 files
        for video_file in self.raw_dir.glob("**/*.mp4"):
            annotation_file = video_file.with_suffix(".json")
            if annotation_file.exists():
                sequences.append({
                    "video_path": video_file,
                    "annotation_path": annotation_file,
                })
        return sequences

    def process_sequence(self, seq_info: dict, traj_group: h5py.Group):
        """
        Process a single sequence and write its data to the provided HDF5 group.
        
        This is where you'll read the raw data, perform any necessary transformations,
        and write to the canonical HDF5 structure. You can access your config
        via `self.config`. The schema is defined via dataclasses in `egohub/schema.py`.
        """
        video_path = seq_info["video_path"]
        logging.info(f"Processing sequence from {video_path}...")
        
        # Example: Access a value from your config file
        frame_rate = self.config.get("frame_rate", 30.0)
        logging.info(f"Using frame rate: {frame_rate}")
        
        # Example: Create a group for some data
        metadata_group = traj_group.create_group("metadata")
        metadata_group.attrs["source_dataset"] = self.name
        
        # ... (rest of your processing logic)
```

### Step 4: Handling Skeleton Data

If your dataset includes full-body skeleton data, you must remap it to the library's canonical skeleton format. This involves two parts.

1.  **Define the Source Skeleton**: In your adapter class, implement `source_joint_names` and `source_skeleton_hierarchy` to describe the skeleton provided by your dataset. These properties are **required** for all adapters, so if your dataset does not include skeletons, simply return an empty list and dictionary, respectively.

    ```python
    # In your AwesomeEgoAdapter class:
    @property
    def source_joint_names(self) -> list[str]:
        # Return the list of joint names from your dataset
        return ["root", "neck", "head", "left_shoulder", "left_elbow", ...]

    @property
    def source_skeleton_hierarchy(self) -> dict[str, str]:
        # Return the kinematic tree for your dataset's skeleton
        return {"neck": "root", "head": "neck", ...}
    ```

2.  **Use the Remapping Utility**: In your `process_sequence` method, after loading the source skeleton data, use the `egohub.processing.skeleton.remap_skeleton` utility to perform the conversion.

    ```python
    from egohub.processing.skeleton import remap_skeleton
    
    # In your process_sequence method:
    # ...
    source_skeleton_positions = ... # Load your (N, num_source_joints, 3) data
    source_skeleton_confidences = ... # Load your (N, num_source_joints) data
    
    canonical_positions, canonical_confidences = remap_skeleton(
        source_positions=source_skeleton_positions,
        source_confidences=source_skeleton_confidences,
        source_joint_names=self.source_joint_names
    )
    
    # Now, save the remapped data to the HDF5 file.
    skeleton_group = traj_group.create_group("skeleton")
    skeleton_group.create_dataset("positions", data=canonical_positions)
    skeleton_group.create_dataset("confidences", data=canonical_confidences)
    # ...
    ```

### Step 5: Test Your Adapter

Your adapter is now ready to be used. The `egohub` command-line tool will automatically discover any `BaseAdapter` subclasses in the `egohub/adapters/` directory, so no manual registration is needed.

You can run your adapter from the command line, using the `name` you defined in the adapter class as the identifier:

```bash
egohub convert awesome-ego \
    --raw-dir path/to/raw/AwesomeEgo \
    --output-file data/processed/awesome-ego.h5
```

---

We encourage contributions for new datasets! Please follow the guide above and open a pull request. 