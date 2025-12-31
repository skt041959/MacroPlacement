# Spec: Graph-to-Sequence Restoration Model

## Overview
Develop an Encoder-Decoder architecture to restore perturbed floorplans to their aligned "ground truth" states. The encoder will use a Heterogeneous Graph Neural Network (GNN) to capture macro relationships, and the decoder will generate a sequence of corrected coordinates or refinement offsets.

## Requirements
- **Encoder**: A HeteroGAT-based encoder that processes the perturbed graph observation.
- **Decoder**: A sequence-based decoder (e.g., GRU/LSTM or Transformer-style) that outputs the restored coordinates for each macro.
- **Supervised Training Loop**: A pipeline to train the model using (Disturbed, Aligned) pairs from SyntheticDataGenerator.
- **Loss Function**: Weighted combination of Coordinate MSE loss and Alignment consistency loss.
- **Inference Script**: A script to evaluate the model on unseen disturbed layouts.

## Technical Constraints
- Must utilize HeteroData from src/data_builder.py.
- Should be flexible to handle varying macro counts.
