# Implementation Plan - Dataset Viewer

## Phase 1: Data Preparation & Backend Setup
- [x] Task: Create `src/generate_val_results.py` (a74bfad)
    - [ ] Sub-task: Load `restorer_model.pth` and `data/val_dataset.pt`.
    - [ ] Sub-task: Run inference on the entire validation set.
    - [ ] Sub-task: Save combined results (Reference, Disturbed, Restored, Metrics) to `data/val_results.pt`.
- [x] Task: Setup Flask Backend (`src/viewer_server.py`) (d581976)
    - [ ] Sub-task: Initialize Flask app and define static file serving.
    - [ ] Sub-task: Implement `DataLoader` class to handle `val_results.pt`.
    - [ ] Sub-task: Create API `/api/samples` with pagination (page, limit).
    - [ ] Sub-task: Create API `/api/filter` for category and metric thresholds.
- [ ] Task: Conductor - User Manual Verification 'Phase 1: Data Preparation & Backend Setup' (Protocol in workflow.md)

## Phase 2: Frontend Implementation
- [ ] Task: Create Basic HTML Structure (`src/templates/viewer.html`)
    - [ ] Sub-task: Layout containers for "Filter/Sort", "Sample List", and "Detailed View".
    - [ ] Sub-task: Import Bootstrap/CSS and necessary JS libraries.
- [ ] Task: Implement Canvas Visualization (`src/static/js/visualizer.js`)
    - [ ] Sub-task: Port drawing logic from `visualizer.py` (macros, edges) to HTML5 Canvas API.
    - [ ] Sub-task: Implement "Layers" toggle for different edge types.
    - [ ] Sub-task: Add mouse hover event listeners for tooltips/inspection.
- [ ] Task: Implement Navigation & Filtering (`src/static/js/app.js`)
    - [ ] Sub-task: Fetch data from `/api/samples` and render the pagination controls.
    - [ ] Sub-task: Render sample metrics and connect filter inputs to API calls.
- [ ] Task: Conductor - User Manual Verification 'Phase 2: Frontend Implementation' (Protocol in workflow.md)

## Phase 3: Integration & Polish
- [ ] Task: Connect Backend and Frontend
    - [ ] Sub-task: Ensure API responses correctly populate the UI.
    - [ ] Sub-task: Handle edge cases (empty results, loading states).
- [ ] Task: Final Polish
    - [ ] Sub-task: Refine CSS styling for a cleaner look.
    - [ ] Sub-task: Add "Launch Viewer" command to `GEMINI.md`.
- [ ] Task: Conductor - User Manual Verification 'Phase 3: Integration & Polish' (Protocol in workflow.md)