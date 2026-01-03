# Specification: Dataset Viewer Server

## Overview
A dynamic web-based application to visualize and analyze the validation dataset for the MXP Refiner project. It provides a user interface to compare Reference (Aligned), Disturbed (Noisy), and Restored (Model Output) layouts, along with associated metrics and graph connectivity.

## Functional Requirements
### 1. Backend Server (Python/Flask)
- Load the validation dataset and pre-computed restoration results from disk.
- Provide API endpoints for:
    - Fetching paginated lists of samples.
    - Fetching detailed data for a specific sample.
    - Filtering and sorting samples based on metrics (e.g., MSE, category).

### 2. Frontend Interface (HTML5/JS/Canvas)
- **Visual Comparison:** Display three side-by-side Canvas elements showing:
    - **Reference:** Original aligned layout.
    - **Disturbed:** Perturbed input layout.
    - **Restored:** Pre-computed output from the restoration model.
- **Graph Visualization:**
    - Render Physical, Logical, and Alignment edges with toggle controls.
- **Interactive Inspection:**
    - Display macro properties (ID, dimensions, coordinates) when hovering over a macro in the Canvas.
- **Navigation & Control:**
    - Pagination controls (Next/Previous/Page selector).
    - Search and filter by sample ID, category (grid, rows, clustered), or error threshold.
    - Sort samples by metric values (e.g., highest MSE first).

### 3. Data Integration
- Read from `data/val_dataset.pt` and a results file (e.g., `data/val_results.pt`).
- Calculate/Display metrics: MSE, Overlap Area, and HPWL.

## Non-Functional Requirements
- **Performance:** Efficiently handle the 1500+ validation samples by using paginated data fetching to avoid browser lag.
- **Usability:** Clean, professional layout using Bootstrap or similar CSS framework.

## Acceptance Criteria
- A standalone server can be launched (e.g., `uv run src/viewer_server.py`).
- The user can browse the entire validation set via pagination.
- Layout comparisons and metrics match the stored data.
- Filtering by category (e.g., "clustered") correctly updates the view.

## Out of Scope
- Real-time model inference (all "Restored" data must be pre-generated).
- Editing macro positions or saving changes back to the dataset.
- Training initiation from the web interface.