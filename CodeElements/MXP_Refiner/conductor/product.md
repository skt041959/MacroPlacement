# Product Guide - MXP Refiner

## Initial Concept
A macro placement refinement engine that uses Reinforcement Learning and Heterogeneous Graph Neural Networks to optimize chip layout.

## Vision and Goals
The MXP Refiner aims to provide a high-performance, ML-driven alternative for macro placement refinement. By leveraging Heterogeneous Graph Attention Networks (GAT), it captures complex geometric and logical relationships to produce layouts that are optimized for both wirelength and manufacturing regularity.

## Implementation Strategy
The development will follow a phased approach:
1.  **Phase 1: Synthetic Data Generation (Completed):** Established a robust synthetic data generator (`SyntheticDataGenerator`) capable of creating aligned clusters of macros and perturbing them to simulate initial placement states. Implemented geometric kernels for overlap detection and alignment scoring.
2.  **Phase 2: Graph-to-Sequence Restoration Model (Completed):** Developed a Graph-to-Sequence model (`FloorplanRestorer`) using a HeteroGAT Encoder and GRU Decoder. Implemented a supervised training pipeline with a combined MSE and Alignment loss, trained on 100,000 synthetic samples.
3.  **Phase 3: RL-Driven Optimization:** Integrate Reinforcement Learning (PPO/A2C) to optimize for wirelength (HPWL) and complex placement objectives using real-world connectivity constraints.
4.  **Phase 4: Integration & PPA Validation:** Validate final layouts with commercial EDA flows.

## Target Users
- **EDA Tool Developers:** Looking to integrate advanced RL-based refinement techniques into production-grade physical design flows.
- **Researchers in ML for EDA:** Exploring new graph representation learning and reinforcement learning paradigms for electronic design automation.

## Core Features
- **Intelligent Physical Constraint Management:** Automated macro movement utilizing Delaunay-based physical edges to prevent overlaps.
- **Geometric Alignment Optimization:** Enhances layout regularity through specialized alignment edges and snapping actions (Snap_X/Snap_Y).
- **Logical Connectivity Awareness:** Minimizes Half-Perimeter Wire Length (HPWL) by accounting for netlist connectivity via logic edges.
- **Interactive Training Dashboard:** Real-time visualization of layout evolution and training metrics for easier model debugging and evaluation.
- **Advanced Dataset Inspection:** Interactive web-based viewer for analyzing restoration performance across the entire validation set, featuring side-by-side comparison and metric filtering.
- **Protobuf-based Data Interface:** Seamlessly imports placement data using Protobuf from LEF/DEF format data, integrated directly within the repository.

## Success Metrics
- **HPWL Reduction:** Significant decrease in total wirelength compared to initial placement.
- **Regularity & Alignment:** High alignment scores indicating the formation of regular macro structures.
- **PPA Qualification:** Final placement quality verified through commercial EDA Place and Route (P&R) tools to ensure superior Power, Performance, and Area (PPA).
