# Product Guidelines - MXP Refiner

## Documentation & Prose Style
- **Practical & Engineering-focused:** Documentation and code comments must be concise, clear, and optimized for developers and EDA engineers. Focus on "how-to" and "why" rather than just describing the code.
- **Illustrative & Diagrammatic:** Use diagrams, flowcharts, and concrete examples liberally to explain complex geometric algorithms and GNN architectures.

## Visual Identity & UX
- **Data-Rich Aesthetic:** Visualizations and the interactive dashboard must prioritize technical detail. Provide multiple data-driven views and overlays, such as congestion heatmaps, macro density maps, and connectivity force-directed graphs.
- **CAD-inspired Professionalism:** Use a professional, high-contrast color palette that mirrors modern physical design software for clarity and familiarity.

## Code Quality & Engineering Principles
- **Strict Modularity:** Adhere to a modular architecture with a clean separation of concerns between the Environment (geometry/physics), the Model (GNN architecture), and the Agent (RL strategy).
- **Comprehensive Testing:** Maintain a high standard for unit testing, targeting >90% coverage for core geometric kernels and graph construction algorithms to ensure robustness in critical placement logic.
- **Efficiency:** Prioritize performance in graph construction and inference to ensure the refiner remains practical for large-scale designs.

## Brand Messaging & Positioning
- **AI-First Physical Design:** Position the tool as a pioneering "AI-first" solution that redefines traditional placement refinement with deep learning.
- **Seamless Integration:** Emphasize the "drop-in" potential and compatibility with standard industry data formats (LEF/DEF/Protobuf), facilitating adoption into existing EDA flows.
