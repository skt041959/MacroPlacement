# Specification: Graph Diffusion Restoration Model (Next-Gen)

## Overview
This track implements a Generative AI approach to macro floorplan restoration, transitioning from the current supervised sequence-based restoration to a **Graph Diffusion Model** with a **Graph U-Net** architecture. This upgrade introduces a manual DDPM (Denoising Diffusion Probabilistic Model) loop and explicit **Alignment Guidance** during sampling to ensure physical validity and high regularity.

## Functional Requirements
### 1. Graph U-Net Architecture (Denoising Model)
- **Encoder**:
    - Implement a series of Graph Attention (GAT) layers.
    - Integrate **Top-K Pooling (gPool)** to extract hierarchical global features and simulate layout downsampling.
- **Bottleneck**: Deep GNN layers to process the most compressed graph state.
- **Decoder**:
    - Implement **Graph Unpooling (gUnpool)** to restore graph resolution.
    - Use **Skip Connections** to concatenate encoder features into the decoder path for precise spatial information retention.
- **Output Head**: Predict the noise (\(\epsilon\)) or the original coordinates (\(x_0\)).

### 2. Manual DDPM Pipeline
- **Forward Process**: Implement the SDE logic to add Gaussian noise to macro coordinates over \(T\) steps while keeping graph topology (\(E\)) and macro dimensions (\(w, h\)) constant.
- **Reverse Process (Inference)**: Implement the iterative denoising loop.
- **Alignment Guidance**:
    - Implement an **Alignment Energy Function** (\(E_{align}\)) with two components:
        - **Snap-to-Grid**: Penalizes coordinates that deviate from a defined pitch.
        - **Channel Alignment**: Encourages shared edges between neighboring macros in the Channel Graph.
    - Inject the gradient of the energy function (\(\nabla_{x_t} E_{align}\)) into each sampling step to guide the model toward aligned states.

### 3. Integration & Tooling
- Replace the existing `GraphToSeqRestorer` in the restoration pipeline.
- Expose guidance parameters (`guidance_scale`, `grid_k`, `lambda_1`, `lambda_2`, `energy_threshold`) in `src/config.py`.
- Maintain compatibility with the current `HeteroData` and `GraphBuilder` infrastructure.

## Non-Functional Requirements
- **Performance**: Denoising loop should be optimized for reasonable inference times (e.g., \(T=50\) or \(T=100\) steps).
- **Scalability**: The pooling/unpooling logic must support variable numbers of macros.

## Acceptance Criteria
- Successful training of the Graph U-Net on existing validation sets with decreasing MSE loss.
- Inference results show noticeably higher alignment regularity (quantified by alignment scores) compared to the previous model.
- Guidance parameters effectively control the "strength" of regularity in the output.

## Out of Scope
- Implementing richer node features (`type`, `connectivity_density`)—this will be a follow-up track.
- Real-world DEF/LEF parsing (remaining within the current synthetic dataset flow).
