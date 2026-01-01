# Spec: Restoration Model Fine-tuning & Data Diversity

## Overview
This track aims to improve the robustness and accuracy of the floorplan restoration model. We will enhance the synthetic data generator to produce more varied and realistic training samples, and optimize the training process with advanced techniques.

## Requirements
- **Data Generator Enhancements**:
    - Support 'Mixed' generation mode (combining clusters and random placements).
    - Variable noise injection per sample (e.g., Uniform(0, MAX_NOISE)).
    - Diverse macro aspect ratios.
- **Training Enhancements**:
    - Learning Rate Scheduler (Cosine Annealing).
    - Model hyperparameter tuning (Hidden Dim, Layers).
- **Evaluation**:
    - Alignment Recovery Score metric.
    - Automated metric comparison (Overlap, Alignment) for Reference vs. Restored.

## Technical Constraints
- Maintain compatibility with HeteroData pipeline.
- Training should be runnable on both CPU and GPU.
