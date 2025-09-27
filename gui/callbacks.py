from pathlib import Path
from qtpy.QtWidgets import QFileDialog, QMessageBox, QApplication
from qtpy.QtCore import QThread

from data_manager import DataManager
from gui.workers import RefinementWorker, SpatialWorker
from gui.widgets import PandasModel
import gui.plotting as plotting

def load_directory(main_window):
    """Handles the logic for loading a Kilosort directory."""
    ks_dir_name = QFileDialog.getExistingDirectory(main_window, "Select Kilosort Output Directory")
    if not ks_dir_name:
        return
        
    main_window.status_bar.showMessage("Loading Kilosort files...")
    QApplication.processEvents()
    
    main_window.data_manager = DataManager(ks_dir_name)
    success, message = main_window.data_manager.load_kilosort_data()
    
    if not success:
        QMessageBox.critical(main_window, "Loading Error", message)
        main_window.status_bar.showMessage("Loading failed.", 5000)
        return
        
    main_window.status_bar.showMessage("Kilosort files loaded. Please select the raw data file.")
    QApplication.processEvents()
    
    dat_file, _ = QFileDialog.getOpenFileName(main_window, "Select Raw Data File (.dat or .bin)",
                                              str(main_window.data_manager.dat_path_suggestion.parent),
                                              "Binary Files (*.dat *.bin)")
    if not dat_file:
        main_window.status_bar.showMessage("Data loading cancelled by user.", 5000)
        return
        
    main_window.data_manager.dat_path = Path(dat_file)
    main_window.status_bar.showMessage("Building cluster dataframe...")
    QApplication.processEvents()
    
    main_window.data_manager.build_cluster_dataframe()
    main_window.pandas_model = PandasModel(main_window.data_manager.cluster_df)
    main_window.table_view.setModel(main_window.pandas_model)
    main_window.table_view.horizontalHeader().setSectionResizeMode(main_window.table_view.horizontalHeader().ResizeToContents)
    main_window.table_view.selectionModel().selectionChanged.connect(main_window.on_cluster_selection_changed)
    
    start_worker(main_window)
    main_window.central_widget.setEnabled(True)
    main_window.load_vision_action.setEnabled(True)
    main_window.status_bar.showMessage(f"Successfully loaded {len(main_window.data_manager.cluster_df)} clusters.", 5000)

def load_vision_directory(main_window):
    """Handles the logic for loading a Vision analysis directory."""
    if not main_window.data_manager:
        QMessageBox.warning(main_window, "No Kilosort Data", "Please load a Kilosort directory first.")
        return

    vision_dir_name = QFileDialog.getExistingDirectory(main_window, "Select Vision Analysis Directory")
    if not vision_dir_name:
        return

    main_window.status_bar.showMessage(f"Loading Vision files from {Path(vision_dir_name).name}...")
    QApplication.processEvents()

    success, message = main_window.data_manager.load_vision_data(vision_dir_name)

    if success:
        main_window.status_bar.showMessage(message, 5000)
        # Trigger a refresh of the currently selected cluster to show new data
        if main_window._get_selected_cluster_id() is not None:
            on_cluster_selection_changed(main_window)
    else:
        QMessageBox.critical(main_window, "Vision Loading Error", message)
        main_window.status_bar.showMessage("Vision loading failed.", 5000)

def on_cluster_selection_changed(main_window):
    """Handles the UI updates when a new cluster is selected."""
    cluster_id = main_window._get_selected_cluster_id()
    if cluster_id is None:
        return
        
    main_window.status_bar.showMessage(f"Loading data for Cluster ID: {cluster_id}", 2000)
    QApplication.processEvents()
    
    lightweight_features = main_window.data_manager.get_lightweight_features(cluster_id)
    if lightweight_features is None:
        main_window.status_bar.showMessage(f"Could not generate EI for cluster {cluster_id}.", 3000)
        main_window.waveform_plot.clear()
        main_window.isi_plot.clear()
        main_window.fr_plot.clear()
        main_window.summary_canvas.fig.clear()
        main_window.summary_canvas.fig.text(0.5, 0.5, "Select a cluster", ha='center', va='center', color='gray')
        main_window.summary_canvas.draw()
        return
        
    plotting.update_waveform_plot(main_window, cluster_id, lightweight_features)
    plotting.update_isi_plot(main_window, cluster_id)
    plotting.update_fr_plot(main_window, cluster_id)
    
    main_window.spatial_plot_dirty = True
    main_window.summary_canvas.fig.clear()
    main_window.summary_canvas.fig.text(0.5, 0.5, "Click 'Spatial Analysis' tab to load", ha='center', va='center', color='gray')
    main_window.summary_canvas.draw()
    
    if main_window.spatial_worker:
        main_window.spatial_worker.add_to_queue(cluster_id, high_priority=False)
        
    on_tab_changed(main_window, main_window.analysis_tabs.currentIndex())
    main_window.status_bar.showMessage("Ready.", 2000)

def on_tab_changed(main_window, index):
    """Handles logic when the user switches between analysis tabs."""
    is_summary_tab = main_window.analysis_tabs.widget(index) == main_window.summary_tab
    if is_summary_tab and main_window.spatial_plot_dirty:
        cluster_id = main_window._get_selected_cluster_id()
        if cluster_id is None:
            return
        
        # --- MODIFICATION START ---
        # The plotting function now handles the logic of what to draw.
        # We just need to call it. This completes the Vision data connection.
        plotting.draw_summary_plot(main_window, cluster_id)
        main_window.spatial_plot_dirty = False
        
        # If we drew the Vision RF, we don't need the worker. 
        # If not, the plot function will have used the heavyweight cache.
        # If that was empty too, we need to queue the worker.
        if not (main_window.data_manager.vision_stas and cluster_id in main_window.data_manager.vision_stas) and \
           cluster_id not in main_window.data_manager.heavyweight_cache:
            
            main_window.status_bar.showMessage(f"Requesting spatial analysis for C{cluster_id}...", 3000)
            main_window.summary_canvas.fig.clear()
            main_window.summary_canvas.fig.text(0.5, 0.5, f"Loading C{cluster_id}...", ha='center', va='center', color='white')
            main_window.summary_canvas.draw()
            QApplication.processEvents()
            
            if main_window.spatial_worker:
                main_window.spatial_worker.add_to_queue(cluster_id, high_priority=True)
        # --- MODIFICATION END ---


def on_spatial_data_ready(main_window, cluster_id, features):
    """Callback for when heavyweight spatial features are ready from the worker."""
    current_id = main_window._get_selected_cluster_id()
    current_tab_widget = main_window.analysis_tabs.currentWidget()
    if cluster_id == current_id and current_tab_widget == main_window.summary_tab:
        plotting.draw_summary_plot(main_window, cluster_id)
        main_window.status_bar.showMessage("Spatial analysis complete.", 2000)

def on_refine_cluster(main_window):
    """Starts the cluster refinement process in a background thread."""
    cluster_id = main_window._get_selected_cluster_id()
    if cluster_id is None:
        QMessageBox.warning(main_window, "No Cluster Selected", "Please select a cluster from the table to refine.")
        return
        
    main_window.refine_button.setEnabled(False)
    main_window.status_bar.showMessage(f"Starting refinement for Cluster {cluster_id}...")
    
    main_window.refine_thread = QThread()
    main_window.refinement_worker = RefinementWorker(main_window.data_manager, cluster_id)
    main_window.refinement_worker.moveToThread(main_window.refine_thread)
    main_window.refinement_worker.finished.connect(main_window.handle_refinement_results)
    main_window.refinement_worker.error.connect(main_window.handle_refinement_error)
    main_window.refinement_worker.progress.connect(lambda msg: main_window.status_bar.showMessage(msg, 3000))
    main_window.refine_thread.started.connect(main_window.refinement_worker.run)
    main_window.refine_thread.start()

def handle_refinement_results(main_window, parent_id, new_clusters):
    """Handles the results from a successful refinement operation."""
    main_window.status_bar.showMessage(f"Refinement of C{parent_id} complete. Found {len(new_clusters)} sub-clusters.", 5000)
    main_window.data_manager.update_after_refinement(parent_id, new_clusters)
    main_window.pandas_model.set_dataframe(main_window.data_manager.cluster_df)
    main_window.refine_button.setEnabled(True)
    main_window.save_action.setEnabled(True)
    main_window.setWindowTitle("*axolotl (unsaved changes)")
    main_window.refine_thread.quit()
    main_window.refine_thread.wait()

def handle_refinement_error(main_window, error_message):
    """Handles an error from the refinement worker."""
    QMessageBox.critical(main_window, "Refinement Error", error_message)
    main_window.status_bar.showMessage("Refinement failed.", 5000)
    main_window.refine_button.setEnabled(True)
    main_window.refine_thread.quit()
    main_window.refine_thread.wait()

def on_save_action(main_window):
    """Handles the save action from the menu."""
    if main_window.data_manager:
        if main_window.data_manager.info_path:
            original_path = main_window.data_manager.info_path
            suggested_path = str(original_path.parent / f"{original_path.stem}_refined.tsv")
        else:
            suggested_path = str(main_window.data_manager.kilosort_dir / "cluster_group_refined.tsv")

        save_path, _ = QFileDialog.getSaveFileName(main_window, "Save Refined Cluster Info",
            suggested_path, "TSV Files (*.tsv)")
        
        if save_path:
            save_results(main_window, save_path)

def save_results(main_window, output_path):
    """Saves the refined cluster data to a TSV file."""
    try:
        col = 'KSLabel' if 'KSLabel' in main_window.data_manager.cluster_info.columns else 'group'
        final_df = main_window.data_manager.cluster_df[['cluster_id', 'KSLabel']].copy()
        final_df.rename(columns={'KSLabel': col}, inplace=True)
        final_df.to_csv(output_path, sep='\t', index=False)
        main_window.data_manager.is_dirty = False
        main_window.setWindowTitle("axolotl")
        main_window.save_action.setEnabled(False)
        main_window.status_bar.showMessage(f"Results saved to {output_path}", 5000)
    except Exception as e:
        QMessageBox.critical(main_window, "Save Error", f"Could not save the file: {e}")
        main_window.status_bar.showMessage("Save failed.", 5000)

def apply_good_filter(main_window):
    """Filters the table view to show only 'good' clusters."""
    if main_window.data_manager is None:
        return
    filtered_df = main_window.data_manager.original_cluster_df[
        main_window.data_manager.original_cluster_df['KSLabel'] == 'good'
    ].copy()
    main_window.pandas_model.set_dataframe(filtered_df)

def reset_table_view(main_window):
    """Resets the table view to its original, unfiltered state."""
    if main_window.data_manager is None:
        return
    main_window.pandas_model.set_dataframe(main_window.data_manager.original_cluster_df)

def start_worker(main_window):
    """Starts the background spatial worker thread."""
    if main_window.worker_thread is not None:
        stop_worker(main_window)
    main_window.worker_thread = QThread()
    main_window.spatial_worker = SpatialWorker(main_window.data_manager)
    main_window.spatial_worker.moveToThread(main_window.worker_thread)
    main_window.worker_thread.started.connect(main_window.spatial_worker.run)
    main_window.spatial_worker.result_ready.connect(main_window.on_spatial_data_ready)
    main_window.worker_thread.start()

def stop_worker(main_window):
    """Stops the background spatial worker thread."""
    if main_window.worker_thread and main_window.worker_thread.isRunning():
        main_window.spatial_worker.stop()
        main_window.worker_thread.quit()
        main_window.worker_thread.wait()
