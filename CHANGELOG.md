# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- **Advanced Camera Parameter System**: Complete rewrite of camera parameters module with:
  - Automatic transformation matrix computation
  - Support for different camera conventions (RDF/RUB)
  - Distortion models (Brown Conrady, Fisheye)
  - Advanced projection functions
  - Comprehensive type annotations with jaxtyping
- **Sophisticated Rerun Integration**: New `rerun_log_utils` module with:
  - Custom component batches for confidence scores
  - Advanced video logging with fallback mechanisms
  - Optimal blueprint layouts
  - Tyro integration for configuration
- **Video Optimization Utilities**: New `video_utils` module with:
  - AV1 encoding for optimal Rerun performance
  - NVIDIA GPU acceleration support
  - Frame extraction and video information utilities
  - Quality presets and optimization options
- **Enhanced Type Safety**: Upgraded to use jaxtyping for precise array shape annotations
- **Advanced Dependencies**: Added beartype, icecream, pyserde, tyro for better development experience
- **Improved EgoDex Visualization**: Enhanced direct visualization script with:
  - Video optimization
  - Confidence-based color coding
  - Advanced error handling
  - Configurable frame limits

### Changed
- **Dependency Management**: Updated to setuptools build system
- **Python Version**: Increased minimum Python version to 3.9
- **Type Annotations**: Migrated to jaxtyping for better type safety
- **Camera Parameters**: Complete API redesign for better usability and performance

### Fixed
- **Type Safety Issues**: Resolved linter errors in camera parameters module
- **Video Performance**: Optimized video handling for better Rerun integration
- **Error Handling**: Improved robustness with proper fallbacks and exception handling

## [0.1.0] - 2025-06-20

### Added

- **Initial Project Setup:**
    - Created core project structure with `egohub` library and `scripts` directories.
    - Set up Python environment using `uv` with dependencies in `pyproject.toml`.
- **Canonical Data Format:**
    - Defined a canonical HDF5 schema for egocentric data (`egohub/schema.py`).
    - Established a canonical coordinate system (Right-Handed, Z-up).
- **EgoDex Adapter:**
    - Created `scripts/adapters/adapter_egodex.py` to convert raw EgoDex data into the canonical HDF5 format.
    - Implemented correct coordinate transformation from ARKit's Y-up system to our canonical Z-up system.
    - Script now processes video frames, camera intrinsics/poses, hand poses, and full 68-joint skeleton data with confidences.
- **PyTorch Dataset:**
    - Implemented `egohub.datasets.EgocentricH5Dataset` for easy loading of canonical HDF5 files.
- **Rerun Exporter:**
    - Created `scripts/exporters/export_to_rerun.py` to visualize the canonical data.
    - Visualizer displays the camera view, video stream, and a full wireframe skeleton with bones, all correctly oriented.
- **Project Constants:**
    - Added `egohub/constants.py` to store skeleton definitions (joint names, IDs, connections).

### Changed
- N/A

### Fixed
- N/A