# Plan: Synthetic Data Generation & Geometric Verification

## Phase 1: Core Geometric Kernels
- [x] Task: Implement optimized overlap detection and area calculation functions. (16e54e4)
- [ ] Task: Implement alignment score functions (X-axis, Y-axis, and grid-based).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Core Geometric Kernels' (Protocol in workflow.md)

## Phase 2: Synthetic Data Generator
- [ ] Task: Implement SyntheticDataGenerator with support for random and pattern-based placement.
- [ ] Task: Add configuration options in src/config.py for data generation (e.g., SEED, GENERATION_MODE).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Synthetic Data Generator' (Protocol in workflow.md)

## Phase 3: Testing & Validation
- [ ] Task: Write comprehensive unit tests for all geometric kernels in 	ests/test_geometry.py.
- [ ] Task: Write unit tests for the data generator to ensure valid, non-overlapping outputs when requested.
- [ ] Task: Update src/inspect_data.py to allow toggling between the new generator and the old random logic.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Testing & Validation' (Protocol in workflow.md)
