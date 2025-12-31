# Plan: Graph-to-Sequence Restoration Model

## Phase 1: Model Architecture [checkpoint: 24268a2]
- [x] Task: Implement the FloorplanRestorer class with a GNN Encoder and a Sequence Decoder. (572b635)
- [x] Task: Integrate the model with the existing HeteroGATRefiner logic where applicable. (572b635)
- [x] Task: Conductor - User Manual Verification 'Phase 1: Model Architecture' (Protocol in workflow.md) (24268a2)

## Phase 2: Supervised Training Pipeline [checkpoint: 0d2cdb8]
- [x] Task: Implement a dataset loader that wraps SyntheticDataGenerator for PyTorch training. (43b44f4)
- [x] Task: Write the supervised training script (src/train_restorer.py) with MSE loss. (7cf363e)
- [x] Task: Add alignment-aware loss terms to the training objective. (7cf363e)
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Supervised Training Pipeline' (Protocol in workflow.md)

## Phase 3: Evaluation & Visualization
- [x] Task: Implement an inference script to generate restored layouts. (b4b8f09)
- [x] Task: Update the visualizer to show Reference, Disturbed, and Restored layouts side-by-side. (f0651f0)
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Evaluation & Visualization' (Protocol in workflow.md)
