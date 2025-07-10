# Dataset Acquisition and Setup

This document provides instructions for downloading the raw datasets required for this project. The scripts below will download the data into the `data/` directory, which is ignored by Git.

| Dataset | Description | License | Dataset Details | Paper Link | Setup Command |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EgoDex** | 800+ hours of 30 Hz, 1080p egocentric video with paired 3D pose annotations for the upper body, hands, and camera. The test set provides a representative sample. [Source](https://ml-site.cdn-apple.com/datasets/egodex/README.md) | CC-by-NC-ND | 829 hours of 1080p, 30 Hz egocentric video with 338,000 episodes across 194 tasks. 90 million frames total. 2.0 TB storage. Features language annotations, camera extrinsics, and dexterous annotations. | [EgoDex: A Large-Scale Egocentric Dataset for Dexterous Manipulation](https://arxiv.org/abs/2505.11709) | `mkdir -p data/EgoDex && curl -L https://ml-site.cdn-apple.com/datasets/egodex/test.zip -o data/EgoDex/test.zip && unzip data/EgoDex/test.zip -d data/EgoDex/ && rm data/EgoDex/test.zip` |
| | | | | | | 
