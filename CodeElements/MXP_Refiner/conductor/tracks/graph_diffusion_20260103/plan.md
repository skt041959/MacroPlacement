# Implementation Plan: Graph Diffusion Restoration Model

## Phase 1: Graph U-Net Core Components
- [x] Task: Implement `GraphUNet` Architecture in `src/model.py` (565eed3)
    - [ ] Sub-task: Implement `gPool` (Top-K Pooling) and `gUnpool` (Unpooling) layers.
    - [ ] Sub-task: Define the `GraphUNet` class with Encoder, Bottleneck, and Decoder paths.
    - [ ] Sub-task: Implement Skip Connections (Concat logic) between Encoder and Decoder.
- [ ] Task: Unit Test Graph U-Net Connectivity
    - [ ] Sub-task: Write tests in `tests/test_model_unet.py` to verify output shapes and skip connection logic with variable node counts.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Graph U-Net Core Components' (Protocol in workflow.md)

## Phase 2: DDPM Training Pipeline
- [ ] Task: Implement Diffusion Scheduler and Forward SDE
    - [ ] Sub-task: Add `DiffusionConfig` to `src/config.py` (timesteps, beta schedule).
    - [ ] Sub-task: Implement `add_noise` function in `src/utils.py` or new `src/diffusion.py`.
- [ ] Task: Update Supervised Training for Diffusion
    - [ ] Sub-task: Modify `src/train_restorer.py` to sample random timesteps $t$ and train the model to predict noise $\epsilon$ or $x_0$.
    - [ ] Sub-task: Implement MSE loss between predicted and ground truth noise.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: DDPM Training Pipeline' (Protocol in workflow.md)

## Phase 3: Alignment-Guided Inference
- [ ] Task: Implement Reverse Denoising Loop
    - [ ] Sub-task: Implement the standard DDPM sampling loop in `src/restore_floorplan.py`.
- [ ] Task: Implement Alignment Energy Guidance
    - [ ] Sub-task: Define `alignment_energy_function` in `src/geometry.py`.
    - [ ] Sub-task: Use `torch.autograd.grad` to compute $\nabla_{x_t} E_{align}$ during the sampling loop.
    - [ ] Sub-task: Apply guidance update to $x_{t-1}$ using `guidance_scale`.
- [ ] Task: Integration & Validation
    - [ ] Sub-task: Update `src/evaluate_model.py` to use the new diffusion-based restorer.
    - [ ] Sub-task: Compare alignment scores with and without guidance.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Alignment-Guided Inference' (Protocol in workflow.md)
