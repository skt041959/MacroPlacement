# Plan: Restoration Model Fine-tuning & Data Diversity

## Phase 1: Enhanced Data Generation [checkpoint: 7e79d9d]
- [x] Task: Implement 'Mixed' generation mode in `SyntheticDataGenerator`. (7ee60c0)
- [~] Task: Update `SyntheticDataGenerator` to support variable noise levels (sampled per layout).
- [x] Task: Conductor - User Manual Verification 'Phase 1: Enhanced Data Generation' (Protocol in workflow.md) (7e79d9d)

## Phase 2: Architecture & Training Optimization [checkpoint: 106cba6]
- [x] Task: Implement Learning Rate Scheduler in src/train_restorer.py. (106cba6)
- [ ] Task: Tune model hyperparameters in src/config.py (e.g., Hidden Dim 128, 4 Layers).
- [x] Task: Conductor - User Manual Verification 'Phase 2: Architecture & Training Optimization' (Protocol in workflow.md) (106cba6)

## Phase 3: Advanced Evaluation Metrics
- [x] Task: Implement Alignment Recovery Score in `src/geometry.py`. (a53188a)
- [x] Task: Update `src/evaluate_model.py` to compute and report metrics (MSE, Overlap, Alignment). (494484a)
- [x] Task: Update `src/train_restorer.py` to log Overlap and Alignment metrics during validation. (14dc02c)
- [x] Task: Refactor `src/evaluate_model.py` to remove legacy snapshot generation. (0f07df0)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Advanced Evaluation Metrics' (Protocol in workflow.md)
