# Egohub Tools Development Plan

This document outlines the engineering checklist for developing a modular and extensible tool system within `egohub`. The primary goal is to allow for post-processing of canonical HDF5 datasets with a variety of external tools, ensuring a clear separation of concerns and maintaining data integrity.

## Phase 1: Core Architecture & Scaffolding

- [x] **Create `egohub/tools/` directory:** Establish a dedicated module for all data processing tools.
- [x] **Define `BaseTool` Abstract Class:**
  - **File:** `egohub/tools/base.py`
  - **Details:** Create an abstract base class `BaseTool` using `abc.ABC`.
  - **Interface:** It must define a single abstract method, `__call__(self, traj_group: h5py.Group)`. This forces a consistent, callable interface for all tools.

- [x] **Create CLI Orchestrator:**
  - **File:** `egohub/cli/process_data.py`
  - **Purpose:** Develop a command-line script to apply one or more tools to an existing canonical HDF5 file.
  - **Functionality:** It can dynamically load tool classes and pass arguments from the command line.

## Phase 2: Hugging Face Tool Implementation

- [x] **Add Hugging Face Dependencies:**
  - **File:** `pyproject.toml`
  - **Action:** Add `transformers`, `torch`, and `timm` to the project dependencies.

- [x] **Implement `HuggingFaceObjectDetectionTool`:**
  - **File:** `egohub/tools/hf_tools.py`
  - **Class:** `HuggingFaceObjectDetectionTool(BaseTool)`
  - **Details:**
    - [x] The constructor (`__init__`) should accept a Hugging Face model name (e.g., `"facebook/detr-resnet-50"`).
    - The `__call__` method will implement the core logic:
      - [x] **Data Extraction:** Read RGB video frames from the `traj_group`.
      - [x] **Model Inference:** Use the `transformers` pipeline to run object detection on each frame.
      - [x] **Format Conversion:** Convert the raw bounding boxes and labels from the model into a canonical format.
      - [x] **Data Writing:** Write the canonical object detection data back into the `traj_group` under an `objects` group.

## Phase 3: Documentation & Testing

- [ ] **Update Documentation:**
  - **File:** `README.md` & `CONTRIBUTING.md`
  - **Action:** Add sections explaining the new "Tools" system, how to use `process_data.py`, and how to create a new tool.

- [ ] **Implement Testing:**
  - **Unit Tests:** Write a test for `HuggingFaceObjectDetectionTool` using a mock model.
  - **Integration Tests:** Write a test for `process_data.py` to verify it can correctly apply the HF tool to a sample HDF5 file. 