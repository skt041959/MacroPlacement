# Plan: Synthetic Data Generation & Geometric Verification

## Phase 1: Core Geometric Kernels [checkpoint: fbecc6d]
- [x] Task: Implement optimized overlap detection and area calculation functions. (16e54e4)
- [x] Task: Implement alignment score functions (X-axis, Y-axis, and grid-based). (098c796)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Core Geometric Kernels' (Protocol in workflow.md) (fbecc6d)

## Phase 2: Synthetic Data Generator
- [x] Task: Implement SyntheticDataGenerator with support for random and pattern-based placement. (7e7071a)
- [x] Task: Add configuration options in src/config.py for data generation (e.g., SEED, GENERATION_MODE). (1200e78)
- [x] Task: Conductor - User Manual Verification 'Phase 2: Synthetic Data Generator' (Protocol in workflow.md) (manual)

## Phase 3: Testing & Validation
- [x] Task: Write comprehensive unit tests for all geometric kernels in tests/test_geometry.py. (16e54e4)
- [x] Task: Write unit tests for the data generator to ensure valid, non-overlapping outputs when requested. (7e7071a)
- [x] Task: Update src/inspect_data.py to allow toggling between the new generator and the old random logic. (manual)
- [x] Task: Conductor - User Manual Verification 'Phase 3: Testing & Validation' (Protocol in workflow.md) (manual)
