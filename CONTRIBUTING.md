# Contributing to egohub

We welcome contributions! This guide provides instructions for extending `egohub` with new dataset adapters.

## How to Add a New Dataset Adapter

Adding support for a new egocentric dataset involves creating a new **Adapter** class. The adapter is responsible for converting the raw, dataset-specific format into our canonical HDF5 schema.

Here is the step-by-step process, using a hypothetical `AwesomeEgo` dataset as an example.

### Step 1: Create the Adapter File

Create a new Python file in the `egohub/adapters/` directory. The filename should be the snake-cased name of the dataset.

-   **File:** `egohub/adapters/awesome_ego.py`

### Step 2: Implement the Adapter Class

Inside your new file, create a class that inherits from `egohub.adapters.base.BaseAdapter`. You must implement two key methods: `discover_sequences` and `process_sequence`.

```python
# egohub/adapters/awesome_ego.py

import logging
from pathlib import Path
import h5py
import numpy as np

from egohub.adapters.base import BaseAdapter
from egohub.schema import create_trajectory_group_from_template

class AwesomeEgoAdapter(BaseAdapter):
    """Adapter for the AwesomeEgo dataset."""

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
        and write to the canonical HDF5 structure.
        """
        video_path = seq_info["video_path"]
        logging.info(f"Processing sequence from {video_path}...")
        
        # Example: Write placeholder data
        num_frames = 100 # Replace with actual frame count
        
        # Get a pre-configured group based on our schema template
        # create_trajectory_group_from_template(traj_group, num_frames)

        # Write camera pose data (e.g., from JSON annotations)
        # traj_group["cameras/ego_camera/pose_in_world"][:] = ...

        # Write JPG-encoded video frames
        # video_stream = traj_group["cameras/ego_camera/rgb/image_bytes"]
        # for i, frame in enumerate(read_video_frames(video_path)):
        #     encoded_frame = encode_as_jpg(frame)
        #     video_stream[i] = np.frombuffer(encoded_frame, dtype='uint8')
```

### Step 3: Register the Adapter in the CLI

Finally, make your new adapter accessible through the `egohub` command-line interface.

Open `egohub/cli/main.py` and add your new adapter class to the `ADAPTER_MAP` dictionary.

```python
# egohub/cli/main.py

# ... other imports
from egohub.adapters.egodex import EgoDexAdapter
from egohub.adapters.awesome_ego import AwesomeEgoAdapter # 1. Import your new adapter

# ...

# 2. Add your adapter to the map
ADAPTER_MAP = {
    "egodex": EgoDexAdapter,
    "awesome-ego": AwesomeEgoAdapter, # The key is the name used on the command line
}

# ... rest of the file
```

### Step 4: Test Your Adapter

You can now run your adapter from the command line:

```bash
egohub convert awesome-ego \
    --raw-dir path/to/raw/AwesomeEgo \
    --output-file data/processed/awesome-ego.h5
```

---

We encourage contributions for new datasets! Please follow the guide above and open a pull request. 