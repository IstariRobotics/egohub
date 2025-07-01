# Engineering Plan: Building a Professional Data Toolkit

**Objective:** To architect and build `egohub` as a professional-grade, extensible Python toolkit for handling egocentric data, incorporating best practices in data validation, user interface design, and modularity.

## 1. Guiding Architectural Principles

Our design will adhere to the following principles, inspired by `simplecv` and other best-in-class data tools:

1.  **Class-Based Data Readers:** Each raw dataset format is encapsulated in its own class, providing a consistent, high-level interface for data access.
2.  **Transformation Pipeline:** Data conversion is not monolithic. It's a configurable pipeline of discrete, reusable transformation steps. This provides maximum flexibility and separation of concerns.
3.  **Unified Command-Line Interface (CLI):** All tools are exposed through a single, polished CLI entry point (`egohub`), creating a cohesive user experience.
4.  **Data Integrity First:** We provide tools to programmatically validate that our data products conform to the canonical schema, ensuring reliability.

---

## Engineering Checklist: Improving Modularity & Extensibility (Inspired by simplecv)

### 1. Class-Based Data Readers
- [x] Refactor each dataset adapter (e.g., EgoDex) into a class inheriting from a common `BaseDatasetReader`.
- [ ] Ensure all readers expose a uniform API (`__getitem__`, `__len__`, `get_metadata()`, etc.).
- [x] Move all dataset-specific logic out of scripts and into these classes.

### 2. Transformation Pipeline as First-Class Objects
- [x] Move all transformation logic into `egohub/transforms/` as stateless, testable functions or classes.
- [x] Implement a `TransformPipeline` class or similar to compose transformations.
- [x] Refactor adapters/exporters to use the pipeline, not inline transformation code.
- [x] Add unit tests for each transformation step.

### 3. Unified, Extensible CLI
- [x] Implement a single CLI entry point (e.g., `egohub`) using `argparse` or `tyro`.
- [x] Add subcommands for `ingest`, `convert`, `validate`, `visualize`, etc.
- [x] Register each adapter/exporter/validator as a subcommand for easy extensibility.
- [x] Document how to add new CLI tools.

### 4. Separation of Concerns & Directory Structure
- [x] Adopt a clear directory structure:
    - [x] `egohub/datasets/` for data readers
    - [x] `egohub/transforms/` for transformations
    - [x] `egohub/exporters/` for writers/exporters
    - [x] `egohub/schema.py` for the canonical schema
    - [x] `egohub/cli.py` for the CLI entry point
    - [x] `scripts/adapters/` for legacy/one-off scripts
- [x] Audit and refactor codebase to ensure:
    - [x] No transformation logic in adapters/exporters.
    - [x] No I/O logic in transformation modules.
    - [x] No visualization code in the core library.
- [x] Update imports and documentation to match new structure.

---

## 2. Implementation Strategy: A Phased Approach

This section outlines a concrete, step-by-step plan to execute the refactoring. We will tackle this in phases to ensure stability and create a clear path forward.

### **Phase 1: Foundational Directory & Base Class Setup**

The goal of this phase is to create the skeleton of our new, modular architecture.

1.  **Create New Directory Structure:**
    - Create the following empty directories:
        - `egohub/datasets/`
        - `egohub/exporters/`
        - `egohub/cli/`
    - Add `__init__.py` to each new directory to make them packages.

2.  **Define the `BaseAdapter` Class:**
    - Create `egohub/adapters/base.py`.
    - Define an abstract base class that all dataset *readers* will inherit from. It should define the core interface for discovering and processing data sequences.
    - **Proposed `BaseAdapter` in `egohub/adapters/base.py`:**
      ```python
      from abc import ABC, abstractmethod
      from pathlib import Path
      import h5py

      class BaseAdapter(ABC):
          """Abstract base class for all dataset adapters."""
          def __init__(self, raw_dir: Path, output_file: Path):
              self.raw_dir = raw_dir
              self.output_file = output_file
              # ...

          @abstractmethod
          def discover_sequences(self) -> list[dict]:
              """Discover all processable data sequences in the raw_dir."""
              pass

          @abstractmethod
          def process_sequence(self, seq_info: dict, traj_group: h5py.Group):
              """Process a single data sequence and write to the HDF5 group."""
              pass

          def run(self, num_sequences: int | None = None):
              """Main loop to discover, process, and save sequences."""
              # This can contain the shared logic for opening the HDF5 file
              # and iterating through sequences.
              # ...
      ```

### **Phase 2: Refactor `EgoDexAdapter` as a Reference Implementation**

Now, we'll refactor a single, concrete adapter to fit the new model. This will serve as the template for all future adapters.

1.  **Move and Refactor `adapter_egodex.py`:**
    - Move the `EgoDexAdapter` class from `scripts/adapters/adapter_egodex.py` into a new file: `egohub/adapters/egodex.py`.
    - Make `EgoDexAdapter` inherit from `egohub.adapters.base.BaseAdapter`.
    - Ensure it correctly implements `discover_sequences` and `process_sequence`.
    - The `main()` function from the old script will now be handled by the new CLI.

2.  **Isolate Transformations:**
    - Identify all data transformation functions within `EgoDexAdapter.process_sequence` (e.g., `arkit_to_canonical_poses`).
    - Move these functions to their appropriate module, like `egohub/transforms/coordinates.py`.
    - The adapter should now *import* and *call* these functions, not define them.

### **Phase 3: Build the Unified CLI**

With a refactored adapter, we can build the user-facing tool to run it.

1.  **Create the CLI Entry Point:**
    - Create a new file: `egohub/cli/main.py`.
    - Use a library like `tyro` or `argparse` to create the main CLI application. `tyro` is recommended for its modern, type-hint-based approach, which reduces boilerplate.

2.  **Create the `convert` Subcommand:**
    - Implement a `convert` function within `egohub/cli/main.py`.
    - This function will be responsible for:
        - Taking arguments like `--dataset`, `--raw_dir`, `--output_file`.
        - Dynamically finding and instantiating the correct adapter class (e.g., `EgoDexAdapter`) based on the `--dataset` argument.
        - Calling the `.run()` method on the adapter instance.
    - **Proposed CLI structure in `egohub/cli/main.py`:**
      ```