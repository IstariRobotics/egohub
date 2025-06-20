# Egocentric Data Hub (egohub)

An end-to-end Python pipeline for ingesting, normalizing, and serving egocentric datasets for humanoid robotics research.

## 🎯 **Core Workflow: Ingest → Re-express → Re-serve**

This project implements a complete pipeline for processing egocentric data:

1. **Ingest**: Read raw, heterogeneous egocentric datasets (currently EgoDex)
2. **Re-express**: Convert to a canonical HDF5 format with standardized coordinate systems
3. **Re-serve**: Provide PyTorch datasets and Rerun visualizations for easy access and validation

## 🚀 **Quick Start**

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

## 📁 **Project Structure**

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

## 🔧 **Key Features**

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

## 📊 **Data Schema**

Our canonical HDF5 format organizes data as follows:

```
trajectory_XXXX/
├── camera/
│   ├── intrinsics           # 3x3 camera matrix
│   └── pose_in_world        # 4x4 transformation matrices
├── hands/
│   ├── left/pose_in_world   # Left hand poses (if available)
│   └── right/pose_in_world  # Right hand poses (if available)
├── metadata/
│   ├── timestamps_ns        # Frame timestamps
│   └── action_label         # Human-readable action description
└── rgb/
    ├── image_bytes          # JPG-encoded video frames
    └── frame_sizes          # Actual frame sizes for reconstruction
```

## 🎯 **Coordinate Systems**

- **World Frame**: Right-handed, Z-up, Y-forward, X-right (meters)
- **Camera Frame**: Standard OpenCV model (Z-forward, Y-down, X-right)
- **Transformations**: Automatic conversion from ARKit to canonical frame

## 🧪 **Testing**

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

## 🔄 **Comparison: Canonical vs. Direct Approaches**

| Feature | Canonical Format | Direct EgoDex |
|---------|------------------|---------------|
| **Data Portability** | ✅ Universal format | ❌ EgoDex-specific |
| **Coordinate Systems** | ✅ Standardized | ⚠️ Original format |
| **Performance** | ✅ Optimized storage | ⚠️ Original size |
| **Extensibility** | ✅ Easy to add datasets | ❌ Hard to extend |
| **Validation** | ✅ End-to-end tested | ✅ Reference implementation |

## 🚧 **Development Status**

### ✅ **Completed**
- [x] EgoDex adapter with coordinate transformations
- [x] Canonical HDF5 format with proper schema
- [x] PyTorch dataset class with global indexing
- [x] Rerun visualization for canonical format
- [x] Direct EgoDex visualization (reference)
- [x] Camera parameter utilities
- [x] Comprehensive testing framework

### 🔄 **In Progress**
- [ ] Depth data support
- [ ] Hand mesh reconstruction
- [ ] Additional dataset adapters
- [ ] Advanced visualization features

### 📋 **Planned**
- [ ] Real-time data streaming
- [ ] Multi-modal fusion
- [ ] Training utilities
- [ ] Performance optimization

## 🤝 **Contributing**

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Update documentation
6. Submit a pull request

## 📄 **License**

[Add your license information here]

## 🙏 **Acknowledgments**

- **Pablo's Implementation**: Reference implementation for direct EgoDex visualization
- **EgoDex Dataset**: Original egocentric dataset
- **Rerun**: 3D visualization framework
- **PyTorch**: Deep learning framework 