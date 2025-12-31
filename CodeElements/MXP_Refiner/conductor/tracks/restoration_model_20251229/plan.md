# Plan: Graph-to-Sequence Restoration Model

## Phase 1: Model Architecture
- [x] Task: Implement the FloorplanRestorer class with a GNN Encoder and a Sequence Decoder. (572b635)
- [ ] Task: Integrate the model with the existing HeteroGATRefiner logic where applicable.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Model Architecture' (Protocol in workflow.md)

## Phase 2: Supervised Training Pipeline
- [ ] Task: Implement a dataset loader that wraps SyntheticDataGenerator for PyTorch training.
- [ ] Task: Write the supervised training script (src/train_restorer.py) with MSE loss.
- [ ] Task: Add alignment-aware loss terms to the training objective.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Supervised Training Pipeline' (Protocol in workflow.md)

## Phase 3: Evaluation & Visualization
- [ ] Task: Implement an inference script to generate restored layouts.
- [ ] Task: Update the visualizer to show Reference, Disturbed, and Restored layouts side-by-side.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Evaluation & Visualization' (Protocol in workflow.md)
