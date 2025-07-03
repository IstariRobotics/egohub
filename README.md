# Egocentric Data Hub (egohub)

An end-to-end Python pipeline for ingesting, normalizing, and serving egocentric datasets for humanoid robotics research in a unified state and action format.

## Core
This project implements a complete pipeline for processing egocentric data, designed for extensibility.

1.  **Ingest (Adapters)**: Raw, heterogeneous egocentric datasets (e.g., EgoDex, HOT3D, Ego4D) are converted into a canonical format using dedicated, class-based **Adapters** that inherit from `egohub.adapters.BaseAdapter`.
2.  **Re-express (Universal Human Rig)**: The data is re-expressed into a canonical schema that represents a **universal human rig**. This critical step standardizes the data by mapping different dataset conventions (e.g., coordinate systems, skeleton joint definitions) onto a single, consistent format. The result is a multi-modal HDF5 file organized by trajectory and data stream (e.g., `cameras/{camera_name}/rgb`, `hands/left`).
3.  **Consume (Training & Visualization)**: Once the data is in the canonical format, it can be consumed for multiple purposes in parallel:
    *   **Training**: A two-stage pipeline inspired by research like [Latent Action Diffusion](https://arxiv.org/pdf/2506.14608v1) is provided to pre-train powerful policy models. This involves first training a `MultimodalVAE` to create a unified latent representation of the data, and then using those latent vectors to pre-train a temporal `LatentPolicyModel`.
    *   **Visualization & Serving**: The data can be immediately visualized using the built-in `RerunExporter`. The library also provides a `EgocentricH5Dataset` class to serve the data for custom PyTorch models or other applications.

This architecture ensures that the canonical HDF5 file is the central, reusable asset for all downstream tasks.

## Architectural Diagram

```mermaid
graph TD
    %% Define Styles
    classDef built fill:#d4edda,stroke:#155724,stroke-width:2px
    classDef notBuilt fill:#f8f9fa,stroke:#6c757d,stroke-width:2px,stroke-dasharray: 5 5

    subgraph "Stage 1: Ingest"
        direction LR
        RawData["Raw Data Source<br/>(e.g., EgoDex)"] --> Adapter["Ingestion Adapter<br/>(EgoDexAdapter)"]
    end

    subgraph "Stage 2: Re-express"
        Adapter --> CanonicalH5["Canonical HDF5 File<br/><b>(Universal Human Rig)</b>"]
    end

    subgraph "Stage 3: Consume"
        direction TB
        subgraph Downstream Applications
            AppPytorch["PyTorch Datasets"]
            AppRerun["Rerun Visualizer"]
            AppSim["Robotics Simulator"]
        end
        subgraph "Enrichment Toolbox (Future Work)"
            ToolHand["Hand Pose Estimator"]
            ToolDepth["Depth Estimator"]
            ToolSeg["Video Segmenter"]
            ToolLabel["Action Labeler VLM"]
        end
    end

    %% Define Flows
    CanonicalH5 --> AppPytorch
    CanonicalH5 --> AppRerun
    CanonicalH5 --> AppSim

    CanonicalH5 -->|Reads video| ToolHand
    ToolHand -->|Writes back| CanonicalH5
    CanonicalH5 -->|Reads video| ToolDepth
    ToolDepth -->|Writes back| CanonicalH5

    %% Apply Styles to Nodes
    class RawData,Adapter,CanonicalH5,AppPytorch,AppRerun built
    class ToolHand,ToolDepth,ToolSeg,ToolLabel,AppSim notBuilt
```
_Completed modules marked in green; future work marked in grey._

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

### Usage Workflows

The following steps outline the key workflows supported by the library. You can run the visualization workflow independently, or follow the full pre-training pipeline.

#### Visualization Workflow
At any point after converting your data, you can visualize its contents with Rerun. This is useful for inspecting the raw data, camera poses, and skeleton data.

```bash
# First, convert a raw dataset (if you haven't already)
egohub convert egodex \
    --raw-dir path/to/raw/EgoDex \
    --output-file data/processed/egodex.h5

# Now, visualize the canonical HDF5 file
egohub visualize data/processed/egodex.h5 --max-frames 100
```

#### Full Pre-Training Workflow
This workflow uses the canonical data to train and save a pre-trained policy model.

```bash
# 1. Convert Raw Data (if not already done)
egohub convert egodex \
    --raw-dir path/to/raw/EgoDex \
    --output-file data/processed/egodex.h5 \
    --num-sequences 10

# 2. Train the Multimodal VAE
# This will save a vae_checkpoint.pth file
python -m egohub.cli.train_vae data/processed/egodex.h5 --epochs 20

# 3. Encode the Dataset
# This saves the latent vectors back into the HDF5 file
python -m egohub.cli.encode_dataset data/processed/egodex.h5 vae_checkpoint.pth

# 4. Pre-train the Policy Model
# This will save a policy_checkpoint.pth file
python -m egohub.cli.pretrain_policy data/processed/egodex.h5 --epochs 10
```

## Supported Datasets

This table lists the datasets currently supported by `egohub`. We welcome contributions for new adapters! See `CONTRIBUTING.md` for a guide on how to add one.

| Dataset Name | CLI Identifier | Adapter Class | Notes |
| :--- | :--- | :--- | :--- |
| **EgoDex** | `egodex` | `EgoDexAdapter` | Supports video, camera pose, and hand/skeleton poses. |

## Project Structure

```
egohub/
├── egohub/                    # Core, installable library
│   ├── adapters/             # Data ingestion classes (EgoDexAdapter, etc.)
│   │   └── base.py
│   ├── cli/                  # Argparse-based CLI application
│   │   └── main.py
│   ├── models/               # VAE and Policy models
│   ├── training/             # Loss objectives and training utilities
│   ├── datasets.py           # PyTorch dataset classes
│   ├── exporters/            # Data exporting classes (RerunExporter, etc.)
│   │   └── rerun.py
│   ├── schema.py             # Canonical HDF5 schema definition
│   └── transforms/           # Data transformation utilities
│       ├── coordinates.py
│       └── pipeline.py
├── tests/                    # Test suite
└── data/                     # Data storage (gitignored)
    ├── raw/
    └── processed/
```

## Canonical Data Schema

Our canonical HDF5 format is designed to be flexible and extensible, particularly for multi-camera and multi-modal data. All spatial data is transformed into a single **right-handed, Z-up** world coordinate frame. Our plan is to continue to build upon this file format and standard as we introduce and incorporate more egocentric (and exocentric) datasets into the library - building a rich standardization framework for a unified 'rig' of human to robot data.

Each HDF5 file can contain multiple trajectories, identified as `trajectory_{:04d}`. Within each trajectory, data is organized into logical groups:

| Group Path | Description |
| :--- | :--- |
| **`metadata/`** | Contains high-level information about the trajectory, such as `uuid`, `source_dataset`, and a synchronized master `timestamps_ns` dataset. |
| **`cameras/{camera_name}/`** | A group for each camera, where `{camera_name}` is a unique identifier (e.g., `ego_camera_left`, `external_gopro_1`). This is the core of our multi-camera support. |
| **`hands/{left,right}/`** | Contains data related to hand tracking, such as `pose_in_world` and MANO parameters. |
| **`objects/{object_name}/`** | Holds data for tracked objects in the scene, including their `pose_in_world`. |
| **`skeleton/`** | Stores full-body skeleton tracking data, like joint `positions` and `confidences`. |
| **`imu/`** | Placeholder for raw Inertial Measurement Unit data. |
| **`gaze/`** | Placeholder for eye gaze tracking data. |

### Example: Camera Data Structure

Within each `cameras/{camera_name}/` group, the data is further organized. This structure ensures all data associated with a single camera is co-located.

| Path within `cameras/{camera_name}/` | Data Type & Shape | Description |
| :--- | :--- | :--- |
| `pose_in_world` | `float32` (N, 4, 4) | 4x4 pose matrix of this camera in the world frame. |
| `intrinsics` | `float32` (3, 3) | 3x3 pinhole camera intrinsic matrix. |
| `rgb/image_bytes` | `uint8` (N, S) | JPG-encoded RGB image bytes. |
| `depth/image` | `uint16` (N, H, W) | Per-pixel depth images. |

## Coordinate Systems

- **World Frame**: Right-handed, Z-up, Y-forward, X-right (units are in meters).
- **Camera Frame**: Standard OpenCV model (Z-forward, Y-down, X-right).
- **Transformations**: Poses are stored as `T_world_local`, representing the transform from the entity's local frame to the world frame.

## Testing

```bash
# Run all tests
pytest
```

## Pipeline Scope and Future Work

This section clarifies what this library currently supports and where we plan to go next.

### What is Supported
- **End-to-End Pre-Training**: A full pipeline from raw EgoDex data to a pre-trained `LatentPolicyModel`.
- **Unified Data Format**: A canonical HDF5 schema that supports multimodal and multi-camera data streams.
- **Rerun.io Visualization**: Built-in support for visualizing the canonical HDF5 data.
- **Two-Stage Training Architecture**: A VAE-based approach to learn a latent representation, followed by self-supervised pre-training of a temporal Transformer model.

### Future Work
- **Downstream Task Fine-Tuning**: While this pipeline produces a pre-trained model, it does not yet include scripts for fine-tuning on specific tasks like imitation learning or reinforcement learning.
- **Additional Data Enrichment**: The architecture is designed for future integration of tools for depth estimation, video segmentation, and automated labeling, as shown in the diagram.
- **Broader Dataset Support**: We welcome contributions of new `Adapters` to support more egocentric datasets like Ego4D, Epic-Kitchens, etc.
- **Robotics Integration**: The ultimate goal is to bridge this data to robotics simulators and real-world hardware, a feature that is still in the planning phase.


