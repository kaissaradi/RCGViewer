# Axolotl Project Plan

This document outlines the development roadmap for the Axolotl GUI, tracking completed tasks and future goals.

## ✅ Current Status (Phase 2)

We have successfully completed a major refactoring of the application, resulting in a more modular and maintainable codebase. The foundational logic for loading Vision files (`.ei`, `.sta`, `.params`) has been fully integrated.

- **Modular Codebase**: The application is now split into logical components (`data_manager`, `analysis_core`, `gui/` modules), making it easier to extend.
- **Vision Data Loading**: The backend can now load and store Vision data alongside Kilosort data.

---

## 🚀 Immediate Next Steps (Finish Phase 2)

The immediate goal is to visualize the Vision data that we can now load.

- **[ ] Connect Vision Data to UI:**
    - **Goal**: Display the loaded receptive field (RF) in the "Spatial Analysis" tab.
    - **File to Modify**: `gui/callbacks.py`.
    - **Action**: In the `on_tab_changed` function, check if Vision data exists for the selected cluster. If it does, call the `plotting.draw_summary_plot` function, which is now ready to handle both Vision and Kilosort data types.

---

## 💡 Future Work & Ideas (Phase 3 and Beyond)

Once the Vision data visualization is complete, we can move on to the major UI/UX enhancements.

- **[ ] Multi-Dataset Tabs (Phase 3):**
    - **Goal**: Allow users to load multiple Kilosort/Vision datasets in a top-level tabbed interface for comparison.
    - **Impact**: High. This is a core feature for comparative analysis.

- **[ ] Collapsible Panels & Multi-Select (Phase 3):**
    - **Goal**: Make the left-hand cluster list collapsible and allow users to select multiple clusters at once.
    - **Impact**: High. Improves screen real estate and enables group analysis.

- **[ ] Grouped Visualizations (Phase 3):**
    - **Goal**: When multiple cells are selected, update the plots to show overlaid waveforms and a grid of receptive fields.
    - **Impact**: High. Essential for comparing neurons.

- **[ ] Hierarchical Cell Classification (Phase 4):**
    - **Goal**: Implement a tree view for users to classify and organize neurons into a nested hierarchy (e.g., by cell type, layer, etc.).
    - **Impact**: High. Adds powerful data organization capabilities.

- **[ ] Performance & Quality of Life:**
    - **Goal**: Add more data quality metrics to the UI.
    - **Goal**: Profile the application to ensure it remains snappy, especially with large datasets.
    - **Goal**: Allow saving and loading of the application "session."
