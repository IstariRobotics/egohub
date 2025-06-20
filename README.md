# Egocentric Data Hub (egohub)

An end-to-end Python pipeline for ingesting, normalizing, and serving egocentric datasets for humanoid robotics research.

## Core Workflow: Ingest → Re-express → Re-serve

This project implements a complete pipeline for processing egocentric data:

1. **Ingest**: Read raw, heterogeneous egocentric datasets (currently EgoDex)
2. **Re-express**: Convert to a canonical HDF5 format with standardized coordinate systems
3. **Re-serve**: Provide PyTorch datasets and Rerun visualizations for easy access and validation

## Quick Start

### Prerequisites

- Python 3.9+
- `uv` package manager (`brew install uv` on macOS)

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd egohub

# Set up environment
uv venv
source .venv/bin/activate
uv pip install -e .
```

### Usage Examples

#### 1. **Canonical Format Pipeline** (Recommended)

Convert EgoDex data to our canonical format and visualize:

```bash
# Convert EgoDex to canonical HDF5 format
python scripts/adapters/adapter_egodex.py \
    --raw_dir data/EgoDex/test/add_remove_lid/ \
    --output_file data/processed/test_lid.h5

# Visualize the canonical format
python scripts/visualizers/view_h5_rerun.py \
    data/processed/test_lid.h5 \
    --trajectory trajectory_0000 \
    --max-frames 50
```

#### 2. **Direct EgoDex Visualization** (Reference Implementation)

Visualize EgoDex data directly without conversion (based on Pablo's working implementation):

```bash
python scripts/visualizers/view_egodex_direct.py \
    data/EgoDex/test/add_remove_lid \
    --max-frames 50
```

## Project Structure

```
egohub/
├── egohub/                    # Core library
│   ├── datasets.py           # PyTorch dataset classes
│   ├── schema.py             # Canonical HDF5 schema definition
│   ├── camera_parameters.py  # Camera intrinsics/extrinsics utilities
│   └── transforms/           # Data transformation utilities
│       ├── coordinates.py    # Coordinate frame transformations
│       └── __init__.py
├── scripts/
│   ├── adapters/             # Data format converters
│   │   └── adapter_egodex.py # EgoDex to canonical format
│   └── visualizers/          # Rerun visualization scripts
│       ├── view_h5_rerun.py  # Canonical format visualization
│       └── view_egodex_direct.py # Direct EgoDex visualization
├── tests/                    # Test suite
│   ├── unit/                 # Unit tests
│   └── integration/          # Integration tests
└── data/                     # Data storage (gitignored)
    ├── raw/                  # Raw datasets
    └── processed/            # Processed canonical format
```

## Key Features

### **Canonical Data Format**
- **Standardized HDF5 structure** with consistent coordinate systems
- **Z-up world frame** with proper camera conventions
- **Efficient storage** with JPG-encoded video frames
- **Comprehensive metadata** including action labels and timestamps

### **PyTorch Integration**
- **`EgocentricH5Dataset`** class for seamless data loading
- **Global frame indexing** across multiple trajectories
- **Automatic tensor conversion** and data decoding
- **Trajectory filtering** and transform support

### **Advanced Visualization**
- **Rerun-based visualization** with 3D camera poses
- **Timeline synchronization** across all data streams
- **Multiple visualization modes** (canonical vs. direct)
- **Real-time playback** with proper coordinate systems

### **Robust Data Processing**
- **Coordinate transformations** from ARKit to canonical frame
- **Error handling** for missing data (hand poses, intrinsics)
- **Video optimization** with efficient encoding
- **Comprehensive logging** and validation

## Canonical Data Schema

Our canonical HDF5 format organizes data into groups, with each group representing a distinct trajectory from a source dataset. All spatial data is in our **right-handed, Z-up** world coordinate frame.

The following table details the standardized paths within each trajectory group and their expected data types. It also indicates whether scripts exist to generate a data stream if it's missing from a source dataset.

| Canonical Path | Data Type & Shape | Description | Generation Script | Generation Dependencies |
| :--- | :--- | :--- | :--- | :--- |
| **`metadata/timestamps_ns`** | `uint64` (N,) | Nanosecond timestamps for each frame. | N/A | Frame rate (e.g., 30Hz) |
| **`camera/intrinsics`** | `float32` (3, 3) | Camera intrinsic matrix (fx, fy, cx, cy). | N/A | - |
| **`camera/pose_in_world`** | `float32` (N, 4, 4) | 4x4 pose matrix of the camera in the world. | N/A | - |
| **`rgb/image_bytes`** | `uint8` (N, S) | Variable-length, JPG-encoded image bytes. | N/A | - |
| **`rgb/frame_sizes`** | `int32` (N,) | Size of each encoded frame in `image_bytes`. | N/A | `rgb/image_bytes` |
| **`hands/left/pose_in_world`**| `float32` (N, 4, 4) | 4x4 pose of the left hand. | No | - |
| **`hands/right/pose_in_world`**| `float32` (N, 4, 4) | 4x4 pose of the right hand. | No | - |
| **`skeleton/joint_names`** | `string[]` (J,) | Attribute list of joint names. | No | - |
| **`skeleton/positions`** | `float32` (N, J, 3) | 3D position of each joint in the world. | No | - |
| **`skeleton/confidences`** | `float32` (N, J) | Confidence value for each joint detection. | No | - |
| **`depth/image`** | `uint16` (N, H, W) | Per-pixel depth image (e.g., in mm). | No | - |

_**Notes:** N = number of frames, S = max encoded image size, J = number of joints, H/W = image height/width._

## Coordinate Systems

- **World Frame**: Right-handed, Z-up, Y-forward, X-right (meters)
- **Camera Frame**: Standard OpenCV model (Z-forward, Y-down, X-right)
- **Transformations**: Automatic conversion from ARKit to canonical frame

## Testing

```bash
# Run unit tests
pytest tests/unit/

# Run integration tests
pytest tests/integration/

# Test data loading
python -c "
from egohub.datasets import EgocentricH5Dataset
dataset = EgocentricH5Dataset('data/processed/test_lid.h5')
print(f'Loaded {len(dataset)} frames from {len(set(idx[0] for idx in dataset.frame_index))} trajectories')
"
```

## Comparison: Canonical vs. Direct Approaches

| Feature | Canonical Format | Direct EgoDex |
|---------|------------------|---------------|
| **Data Portability** | Universal format | EgoDex-specific |
| **Coordinate Systems** | Standardized | Original format |
| **Performance** | Optimized storage | Original size |
| **Extensibility** | Easy to add datasets | Hard to extend |
| **Validation** | End-to-end tested | Reference implementation |

## Development Status

### Completed
- [x] EgoDex adapter with coordinate transformations
- [x] Canonical HDF5 format with proper schema
- [x] PyTorch dataset class with global indexing
- [x] Rerun visualization for canonical format
- [x] Direct EgoDex visualization (reference)
- [x] Camera parameter utilities
- [x] Comprehensive testing framework

### In Progress
- [ ] Depth data support
- [ ] Hand mesh reconstruction
- [ ] Additional dataset adapters
- [ ] Advanced visualization features

### Planned
- [ ] Real-time data streaming
- [ ] Multi-modal fusion
- [ ] Training utilities
- [ ] Performance optimization


