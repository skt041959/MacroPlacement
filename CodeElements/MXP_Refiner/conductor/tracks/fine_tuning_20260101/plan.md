# Plan: Restoration Model Fine-tuning & Data Diversity

## Phase 1: Enhanced Data Generation
- [x] Task: Implement 'Mixed' generation mode in `SyntheticDataGenerator`. (7ee60c0)
- [~] Task: Update `SyntheticDataGenerator` to support variable noise levels (sampled per layout).
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Enhanced Data Generation' (Protocol in workflow.md)

## Phase 2: Architecture & Training Optimization
- [ ] Task: Implement Learning Rate Scheduler in src/train_restorer.py.
- [ ] Task: Tune model hyperparameters in src/config.py (e.g., Hidden Dim 128, 4 Layers).
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Architecture & Training Optimization' (Protocol in workflow.md)

## Phase 3: Advanced Evaluation Metrics
- [ ] Task: Implement Alignment Recovery Score in src/geometry.py.
- [ ] Task: Update src/evaluate_model.py to compute and report metrics (MSE, Overlap, Alignment).
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Advanced Evaluation Metrics' (Protocol in workflow.md)
