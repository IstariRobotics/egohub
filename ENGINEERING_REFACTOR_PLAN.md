# Engineering Refactoring Plan

This document outlines a plan to refactor the `egohub` codebase to improve modularity, scalability, and maintainability.

## 1. Externalize Hardcoded Configurations using Adapter-Specific Configs

**Goal:** Move hardcoded values from adapters into dedicated, per-adapter configuration files. This improves modularity by co-locating configuration with the specific adapter that uses it, enhancing clarity and scalability.

- [x] Create a `configs/` directory at the project root to store adapter configurations.
- [x] Create `configs/egodex.yaml` and move the hardcoded values (frame rate, default intrinsics, JPEG quality) from `egohub/adapters/egodex.py` into it.
- [x] Modify the `BaseAdapter` to find and load a YAML configuration file corresponding to the adapter's name (e.g., `EgoDexAdapter` loads `egodex.yaml`). The loaded config should be stored in an instance attribute like `self.config`.
- [x] Refactor `EgoDexAdapter` to use the values from `self.config` instead of hardcoded constants.
- [x] Update the documentation or `README` to explain how to add a new adapter and its corresponding configuration file.

## 2. Centralize and Formalize Data Schema

**Goal:** Establish a single source of truth for the HDF5 data structure to ensure consistency across all data adapters.

- [x] Fully define the canonical HDF5 structure in `egohub/schema.py`. This should include group names, dataset names, expected dtypes, and attributes. The existing `CANONICAL_DATA_STREAMS_TEMPLATE` can be expanded upon.
- [x] Create helper functions or a class in `egohub/schema.py` to create the HDF5 structure based on the schema definition.
- [x] Refactor `EgoDexAdapter` to use the centralized schema definition instead of hardcoded strings for creating HDF5 groups and datasets.
- [x] Validate the output of the adapter against the schema to ensure compliance.

## 3. Modularize Data Processing Logic

**Goal:** Break down the monolithic `process_sequence` method into smaller, reusable, and testable components.

- [x] Create a dedicated module for data processing components (e.g., `egohub/processing/`).
- [x] Implement a `PoseTransformer` component responsible for handling all coordinate transformations. The existing `TransformPipeline` could be the basis for this.
- [x] Implement a `VideoProcessor` component to handle video decoding and frame extraction.
- [x] Implement components for specific data types like `HandProcessor`, `SkeletonProcessor`.
- [x] Refactor `EgoDexAdapter` to use a pipeline of these new processing components instead of containing all the logic in one method.
- [x] Add unit tests for each new processing component to ensure they work correctly in isolation. 