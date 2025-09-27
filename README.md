# RCGViewer
# - Neural Spike Sorting Cluster Refinement GUI

A high-performance GUI for refining and analyzing neural spike sorting clusters from Kilosort output.

## Installation and Usage

1.  **Clone the repository:**
    ```bash
    git clone <your-repository-url>
    cd axolotl-wrapper
    ```

2.  **Set up a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Run the application:**
    ```bash
    python main.py
    ```

## Project Log

### ✅ Completed

* **Phase 1: Code Modularization**
    * The original `gui.py` and `cleaning_utils_cpu.py` files have been successfully refactored into a more organized and scalable structure.
    * The new structure includes:
        * `main.py`: Application entry point.
        * `data_manager.py`: Handles all data loading and management.
        * `analysis_core.py`: Contains core scientific and refinement algorithms.
        * `gui/`: A dedicated directory for all UI components.

### 📝 To-Do

* **Phase 2: Vision File Integration**
    * Implement a "Load Vision Files" action to import `.ei`, `.sta`, and `.params` files.
    * Integrate loaded Vision data into the `DataManager`.
    * Update the "Spatial Analysis" tab to display STA/RF data from Vision files.

* **Phase 3: UI/UX Enhancements**
    * Implement a top-level tabbed interface to manage multiple datasets.
    * Make the left-hand cluster list panel collapsible.
    * Enable multi-cell selection in the cluster list.
    * Update plotting views to display data for multiple selected cells (overlaid waveforms, grid of RFs).

* **Phase 4: Future Features**
    * Add a hierarchical `QTreeView` for advanced cell classification.
