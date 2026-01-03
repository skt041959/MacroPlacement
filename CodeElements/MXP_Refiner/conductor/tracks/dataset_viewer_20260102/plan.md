# Implementation Plan - Dataset Viewer

## Phase 1: Data Preparation & Backend Setup [checkpoint: 8a37b62]
- [x] Task: Create `src/generate_val_results.py` (a74bfad)
    - [x] Sub-task: Load `restorer_model.pth` and `data/val_dataset.pt`.
    - [x] Sub-task: Run inference on the entire validation set.
    - [x] Sub-task: Save combined results (Reference, Disturbed, Restored, Metrics) to `data/val_results.pt`.
- [x] Task: Setup Flask Backend (`src/viewer_server.py`) (d581976)
    - [x] Sub-task: Initialize Flask app and define static file serving.
    - [x] Sub-task: Implement `DataLoader` class to handle `val_results.pt`.
    - [x] Sub-task: Create API `/api/samples` with pagination (page, limit).
    - [x] Sub-task: Create API `/api/filter` for category and metric thresholds.
- [x] Task: Conductor - User Manual Verification 'Phase 1: Data Preparation & Backend Setup' (Protocol in workflow.md) (8a37b62)

## Phase 2: Frontend Implementation [checkpoint: 1730043]
- [x] Task: Create Basic HTML Structure (`src/templates/viewer.html`) (d02a8e4)
    - [x] Sub-task: Layout containers for "Filter/Sort", "Sample List", and "Detailed View".
    - [x] Sub-task: Import Bootstrap/CSS and necessary JS libraries.
- [x] Task: Implement Canvas Visualization (`src/static/js/visualizer.js`) (a896093)
    - [x] Sub-task: Port drawing logic from `visualizer.py` (macros, edges) to HTML5 Canvas API.
    - [x] Sub-task: Implement "Layers" toggle for different edge types.
    - [x] Sub-task: Add mouse hover event listeners for tooltips/inspection.
- [x] Task: Implement Navigation & Filtering (`src/static/js/app.js`) (dd03804)
    - [x] Sub-task: Fetch data from `/api/samples` and render the pagination controls.
    - [x] Sub-task: Render sample metrics and connect filter inputs to API calls.
- [x] Task: Conductor - User Manual Verification 'Phase 2: Frontend Implementation' (Protocol in workflow.md) (1730043)

## Phase 3: Integration & Polish [checkpoint: 905876d]
- [x] Task: Connect Backend and Frontend (905876d)
    - [x] Sub-task: Ensure API responses correctly populate the UI.
    - [x] Sub-task: Handle edge cases (empty results, loading states).
- [x] Task: Final Polish (905876d)
    - [x] Sub-task: Refine CSS styling for a cleaner look.
    - [x] Sub-task: Add "Launch Viewer" command to `GEMINI.md`.
- [x] Task: Conductor - User Manual Verification 'Phase 3: Integration & Polish' (Protocol in workflow.md) (905876d)