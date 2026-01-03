# MXP Refiner - Project Context

## Project Overview
**MXP Refiner** is a macro placement optimization engine for chip design. It utilizes deep learning techniques to solve two primary tasks:
1.  **Layout Restoration:** A Supervised Learning model (`GraphToSeqRestorer`) that learns to recover a clean, aligned floorplan from a perturbed or noisy initial state.
2.  **Placement Refinement:** A Reinforcement Learning (RL) agent (`HeteroGATRefiner`) that iteratively optimizes macro positions to minimize wirelength (HPWL) and overlap while maximizing alignment.

The system models the chip layout as a **Heterogeneous Graph**, capturing:
-   **Physical Relations:** Spatial proximity and Delaunay triangulation edges for overlap avoidance.
-   **Logical Relations:** Netlist connectivity for wirelength optimization.
-   **Alignment Relations:** Geometric alignment for regularity.

## Key Technologies
-   **Language:** Python 3.10+
-   **Deep Learning:** PyTorch, PyTorch Geometric (PyG)
-   **Dependency Management:** `uv`
-   **Graph Processing:** Scipy (Delaunay), NetworkX (optional)
-   **Visualization:** Matplotlib, HTML5 Canvas (Dashboard)

## Usage & Development

### Prerequisites
Ensure `uv` is installed on your system.
```bash
# Install dependencies
uv sync
```

### 1. Layout Restoration (Supervised Learning)
The restoration model learns to "snap" macros back to a valid grid or aligned state.

*   **Generate Categorized Datasets:**
    Generates multiple layout patterns (grid, rows, clustered) and performs a train/val split.
    ```bash
    uv run src/generate_dataset.py
    ```
    *   **Output:** Saves per-category data, a combined dataset, and split sets (`train_dataset.pt`, `val_dataset.pt`).

*   **Train Model:**
    ```bash
    uv run src/train_restorer.py
    ```
    *   **Output:** Saves model to `restorer_model.pth` and logs to `training.log`.
    *   **Config:** Adjust `src/config.py` (e.g., `CATEGORIES`, `NUM_TRAIN_SAMPLES`, `NOISE_LEVEL`).

### 2. Placement Refinement (Reinforcement Learning)
The RL agent interacts with a custom environment (`MacroLayoutEnv`) to optimize placement.

*   **Train Agent:**
    ```bash
    uv run src/train.py
    ```
    *   **Output:** Saves model to `model.pth` and generates `dashboard.html` for visualization.

### 3. Verification & Testing
*   **Run Unit Tests:**
    ```bash
    uv run pytest
    ```
*   **Verify Model Architecture:**
    ```bash
    uv run verify_model.py
    ```

### 4. Dataset Viewer
An interactive web-based tool to browse and analyze restoration results.

*   **Generate Results:**
    Runs inference on the validation set and prepares data for the viewer.
    ```bash
    uv run src/generate_val_results.py
    ```
*   **Launch Server:**
    Starts the Flask server (default: http://127.0.0.1:5000).
    ```bash
    uv run src/viewer_server.py
    ```

## Directory Structure
*   **`src/`**: Core application logic.
    *   `generate_dataset.py`: Primary script for categorized data generation and train/val splitting.
    *   `train_restorer.py`: Main entry point for restoration model training.
    *   `train.py`: Main entry point for RL agent training.
    *   `viewer_server.py`: Flask backend for the dataset viewer.
    *   `generate_val_results.py`: Script to pre-compute restoration results for the viewer.
    *   `model.py`: PyTorch Geometric models (`GraphToSeqRestorer`, `HeteroGATRefiner`).
    *   `generator.py`: Synthetic data generation (Random, Grid, Mixed clusters).
    *   `data_builder.py`: Converts raw macro data into Heterogeneous Graph objects.
    *   `dataset.py`: PyTorch Dataset implementation for loading/generating data.
    *   `config.py`: Centralized configuration for hyperparameters and constants.
*   **`data/`**: Stores generated datasets (e.g., `restoration_dataset_10k_mixed.pt`).
*   **`conductor/`**: Project documentation, plans, and specifications.
*   **`tests/`**: Unit tests.

## Development Conventions
*   **Code Style:** Follow PEP 8.
*   **Type Hinting:** Use standard Python type hints for function arguments and return values.
*   **Configuration:** All hyperparams (LR, batch size, dimensions) should be in `src/config.py`, not hardcoded.
*   **Data Generation:** Use `SyntheticDataGenerator` in `src/generator.py` for consistent testing scenarios.
