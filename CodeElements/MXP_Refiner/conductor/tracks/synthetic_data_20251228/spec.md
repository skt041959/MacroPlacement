# Spec: Synthetic Data Generation & Geometric Verification

## Overview
This track focuses on Phase 1 of the implementation strategy: creating a reliable source of synthetic data that emphasizes macro alignment and packing, and ensuring the geometric kernels used for verification (overlap and alignment scoring) are robust and well-tested.

## Requirements
- **SyntheticDataGenerator**: A class to generate datasets of macro cells with controlled parameters (count, size range, canvas size).
- **Alignment Patterns**: Support for generating macros in specific patterns (e.g., grids, rows, or clusters) to test the GNN's ability to recognize and preserve regularity.
- **Overlap Detection**: A highly optimized kernel to detect and measure overlap area between macros.
- **Alignment Scoring**: Implementation of metrics to quantify how well macros are aligned to grids or to each other.
- **Testing**: 100% test coverage for all geometric calculation functions.

## Technical Constraints
- Must be compatible with existing HeteroData structure in data_builder.py.
- Must support reproducibility via seeding.
