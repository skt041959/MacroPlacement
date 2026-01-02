# Tech Stack - MXP Refiner

## Core Language & Runtime
- **Python 3.10+**: Leveraging modern type hinting and async features.
- **uv**: Utilized as the primary package manager and virtual environment orchestrator for speed and reliability.

## Machine Learning Frameworks
- **PyTorch**: Used for the foundational neural network implementation and automatic differentiation.
- **PyTorch Geometric (PyG)**: The core library for implementing Heterogeneous Graph Attention Networks (GATv2), specifically handling `HeteroData` and `HeteroConv`.

## Environment & Algorithms
- **Custom Placement Environment**: Dedicated logic in `env.py` for managing macro coordinates, computing rewards (HPWL, overlap, alignment), and handling geometric transitions.
- **NumPy & SciPy**: Essential for high-performance numerical computing and geometric algorithms (e.g., Delaunay Triangulation for physical adjacency).

## Data Integration & Tooling
- **Protocol Buffers (Protobuf)**: Used for efficient, structured serialization of placement data when importing from industry-standard formats (LEF/DEF).
- **Matplotlib**: Generates static training metrics and visualization snapshots.
- **HTML5/JS (Canvas)**: Powers the interactive refinement dashboard for real-time playback of macro movements.

## Web Server
- **Flask**: Lightweight Python web framework used for serving the dataset viewer and visualization dashboard. [Added 2026-01-02]
