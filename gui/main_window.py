from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget, QTableView, QTreeView, QAbstractItemView, QSlider, QLabel, QMenu, QInputDialog, QStackedWidget, QLineEdit
)
from qtpy.QtCore import Qt, QItemSelectionModel
from qtpy.QtGui import QFont, QStandardItemModel, QStandardItem
import pyqtgraph as pg
import analysis_core
# Custom GUI Modules
from gui.widgets import MplCanvas, PandasModel
import gui.callbacks as callbacks
import gui.plotting as plotting

# Global pyqtgraph configuration
pg.setConfigOption('background', '#1f1f1f')
pg.setConfigOption('foreground', 'd')

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("axolotl")
        self.setGeometry(50, 50, 1800, 1000)

        # --- Application State ---
        self.data_manager = None
        self.pandas_model = None
        self.tree_model = QStandardItemModel()
        self.refine_thread = None
        self.refinement_worker = None
        self.worker_thread = None
        self.spatial_worker = None
        self.spatial_plot_dirty = False
        self.current_spatial_features = None
        # --- Timer for EI Animation ---
        self.ei_animation_timer = None  # To prevent garbage collection
        # --- Current STA View ---
        self.current_sta_view = "rf"  # Default to RF plot
        # --- STA Animation State ---
        self.current_sta_data = None
        self.current_sta_cluster_id = None
        self.current_frame_index = 0
        self.total_sta_frames = 0
        self.sta_animation_timer = None
        self._is_syncing = False
        self.last_left_width = 450
        # --- Raw Trace Update State ---
        self._raw_trace_updating = False
        # --- Raw Trace Buffer for Seamless Panning ---
        self.raw_trace_buffer = None
        self.raw_trace_buffer_start_time = 0.0
        self.raw_trace_buffer_end_time = 0.0
        self.raw_trace_current_cluster_id = None
        # --- Raw Trace Manual Loading Flag ---
        self._raw_trace_manual_load = False

        # --- UI Setup ---
        self._setup_style()
        self._setup_ui()
        self.central_widget.setEnabled(False)
        self.status_bar.showMessage("Welcome to axolotl. Please load a Kilosort directory to begin.")

    def _setup_style(self):
        """Sets the application's stylesheet."""
        self.setFont(QFont("Segoe UI", 9))
        self.setStyleSheet("""
            QWidget { color: white; background-color: #2D2D2D; }
            QTableView { background-color: #191919; alternate-background-color: #252525; gridline-color: #454545; }
            QHeaderView::section { background-color: #353535; padding: 4px; border: 1px solid #555555; }
            QPushButton { background-color: #353535; border: 1px solid #555555; padding: 5px; border-radius: 4px; }
            QPushButton:hover { background-color: #454545; }
            QPushButton:pressed { background-color: #252525; }
            QTabWidget::pane { border: 1px solid #4282DA; }
            QTabBar::tab { color: white; background: #353535; padding: 8px; border-top-left-radius: 4px; border-top-right-radius: 4px; }
            QTabBar::tab:selected { background: #4282DA; }
            QStatusBar { color: white; }
        """)

    def _setup_ui(self):
        """Initializes and lays out all the UI widgets."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        main_layout = QHBoxLayout(self.central_widget)

        # --- Left Pane ---
        self.left_pane = QWidget()
        left_layout = QVBoxLayout(self.left_pane)
        
        # Add a toggle button for collapsing/expanding the sidebar
        self.sidebar_toggle_button = QPushButton("◀")
        self.sidebar_toggle_button.setFixedWidth(20)
        self.sidebar_toggle_button.clicked.connect(self.toggle_sidebar)
        self.sidebar_collapsed = False
        
        # Create a widget to contain the filter box and views
        left_content = QWidget()
        left_content_layout = QVBoxLayout(left_content)

        # Add filter controls
        filter_box = QHBoxLayout()
        self.filter_button = QPushButton("Filter 'Good'")
        self.reset_button = QPushButton("Reset View")
        filter_box.addWidget(self.filter_button)
        filter_box.addWidget(self.reset_button)

        # --- View Switcher ---
        view_switch_layout = QHBoxLayout()
        self.tree_view_button = QPushButton("Tree View")
        self.table_view_button = QPushButton("Table View")
        self.tree_view_button.clicked.connect(lambda: self._switch_left_view(0))
        self.table_view_button.clicked.connect(lambda: self._switch_left_view(1))
        view_switch_layout.addWidget(self.tree_view_button)
        view_switch_layout.addWidget(self.table_view_button)

        # --- View Stack (Tree and Table) ---
        self.view_stack = QStackedWidget()
        
        # Tree View
        self.tree_view = QTreeView()
        self.tree_view.setHeaderHidden(True)
        self.tree_view.setDragEnabled(True)
        self.tree_view.setAcceptDrops(True)
        self.tree_view.setDropIndicatorShown(True)
        self.tree_view.setDragDropMode(QAbstractItemView.DragDropMode.InternalMove)
        self.tree_view.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree_view.customContextMenuRequested.connect(self.open_tree_context_menu)
        
        # Table View
        self.table_view = QTableView()
        self.table_view.setSortingEnabled(True)
        self.table_view.setAlternatingRowColors(True)
        self.table_view.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

        self.view_stack.addWidget(self.tree_view)
        self.view_stack.addWidget(self.table_view)

        self.refine_button = QPushButton("Refine Selected Cluster")
        self.refine_button.setFixedHeight(40)
        self.refine_button.setStyleSheet("font-size: 14px; font-weight: bold; color: #aeffe3; background-color: #005230;")

        left_content_layout.addLayout(filter_box)
        left_content_layout.addLayout(view_switch_layout)
        left_content_layout.addWidget(self.view_stack)
        left_content_layout.addWidget(self.refine_button)
        
        # Add the toggle button and content to the left pane
        left_layout.addWidget(self.sidebar_toggle_button)
        left_layout.addWidget(left_content)
        # Store reference to content widget for collapsing/expanding
        self.left_content = left_content

        # --- Right Pane ---
        right_pane = QWidget()
        right_layout = QVBoxLayout(right_pane)
        self.analysis_tabs = QTabWidget()
        
        # Waveforms Tab
        self.waveforms_tab = QWidget()
        waveforms_layout = QVBoxLayout(self.waveforms_tab)
        wf_splitter = QSplitter(Qt.Orientation.Vertical)
        self.waveform_plot = pg.PlotWidget(title="Waveforms (Sampled)")
        bottom_panel_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.isi_plot = pg.PlotWidget(title="Inter-Spike Interval (ISI) Histogram")
        self.fr_plot = pg.PlotWidget(title="Smoothed Firing Rate")
        bottom_panel_splitter.addWidget(self.isi_plot)
        bottom_panel_splitter.addWidget(self.fr_plot)
        wf_splitter.addWidget(self.waveform_plot)
        wf_splitter.addWidget(bottom_panel_splitter)
        wf_splitter.setSizes([600, 400])
        waveforms_layout.addWidget(wf_splitter)
        self.analysis_tabs.addTab(self.waveforms_tab, "Waveform Details")
        
        # Spatial Analysis Tab
        self.summary_tab = QWidget()
        summary_layout = QVBoxLayout(self.summary_tab)
        
        # Add controls for EI animation
        ei_control_layout = QHBoxLayout()
        self.ei_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.ei_frame_slider.setMinimum(0)
        self.ei_frame_slider.setMaximum(29)  # Default to 30 frames
        self.ei_frame_slider.setValue(0)
        self.ei_frame_slider.setEnabled(False)  # Only enabled when EI data is loaded
        self.ei_frame_label = QLabel("Frame: 0/30")
        self.ei_play_button = QPushButton("Play")
        self.ei_pause_button = QPushButton("Pause")
        self.ei_prev_frame_button = QPushButton("<< Prev")
        self.ei_next_frame_button = QPushButton("Next >>")
        
        # Connect slider and buttons
        self.ei_frame_slider.valueChanged.connect(self.update_ei_frame_manual)
        self.ei_play_button.clicked.connect(self.start_ei_animation)
        self.ei_pause_button.clicked.connect(self.pause_ei_animation)
        self.ei_prev_frame_button.clicked.connect(self.prev_ei_frame)
        self.ei_next_frame_button.clicked.connect(self.next_ei_frame)
        
        # Add controls to layout
        ei_control_layout.addWidget(self.ei_prev_frame_button)
        ei_control_layout.addWidget(self.ei_frame_slider)
        ei_control_layout.addWidget(self.ei_next_frame_button)
        ei_control_layout.addWidget(self.ei_frame_label)
        ei_control_layout.addWidget(self.ei_play_button)
        ei_control_layout.addWidget(self.ei_pause_button)
        
        # Add controls and canvas to layout
        summary_layout.addLayout(ei_control_layout)
        self.summary_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        summary_layout.addWidget(self.summary_canvas)
        self.analysis_tabs.addTab(self.summary_tab, "Spatial Analysis")
        
        # STA Analysis Tab
        self.sta_tab = QWidget()
        sta_layout = QVBoxLayout(self.sta_tab)
        
        # Add buttons to select different STA views
        sta_control_layout = QHBoxLayout()
        self.sta_rf_button = QPushButton("RF Plot")
        self.sta_population_rfs_button = QPushButton("Population RFs")
        self.sta_timecourse_button = QPushButton("Timecourse")
        self.sta_animation_button = QPushButton("Animate STA")
        self.sta_animation_stop_button = QPushButton("Stop Animation")
        
        # Set up button functionality
        self.sta_rf_button.clicked.connect(lambda: self.select_sta_view("rf"))
        self.sta_population_rfs_button.clicked.connect(lambda: self.select_sta_view("population_rfs"))
        self.sta_timecourse_button.clicked.connect(lambda: self.select_sta_view("timecourse"))
        self.sta_animation_button.clicked.connect(lambda: self.select_sta_view("animation"))
        self.sta_animation_stop_button.clicked.connect(lambda: plotting.stop_sta_animation(self))
        
        # Add buttons to layout
        sta_control_layout.addWidget(self.sta_rf_button)
        sta_control_layout.addWidget(self.sta_population_rfs_button)
        sta_control_layout.addWidget(self.sta_timecourse_button)
        sta_control_layout.addWidget(self.sta_animation_button)
        sta_control_layout.addWidget(self.sta_animation_stop_button)
        
        # Add frame control elements
        sta_frame_layout = QHBoxLayout()
        self.sta_frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.sta_frame_slider.setMinimum(0)
        self.sta_frame_slider.setMaximum(29)  # Default to 30 frames
        self.sta_frame_slider.setValue(0)
        self.sta_frame_slider.setEnabled(False)  # Only enabled during manual animation
        self.sta_frame_label = QLabel("Frame: 0/30")
        self.sta_frame_prev_button = QPushButton("<< Prev")
        self.sta_frame_next_button = QPushButton("Next >>")
        
        # Connect slider and buttons
        self.sta_frame_slider.valueChanged.connect(self.update_sta_frame_manual)
        self.sta_frame_prev_button.clicked.connect(self.prev_sta_frame)
        self.sta_frame_next_button.clicked.connect(self.next_sta_frame)
        
        # Add frame controls to layout
        sta_frame_layout.addWidget(self.sta_frame_prev_button)
        sta_frame_layout.addWidget(self.sta_frame_slider)
        sta_frame_layout.addWidget(self.sta_frame_next_button)
        sta_frame_layout.addWidget(self.sta_frame_label)
        sta_control_layout.addLayout(sta_frame_layout)
        
        # Add controls and canvas to layout
        sta_layout.addLayout(sta_control_layout)
        self.sta_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        sta_layout.addWidget(self.sta_canvas)
        self.analysis_tabs.addTab(self.sta_tab, "STA Analysis")
        
        # Raw Trace Tab
        self.raw_trace_tab = QWidget()
        raw_trace_layout = QVBoxLayout(self.raw_trace_tab)
        
        # Add controls for time navigation
        time_control_layout = QHBoxLayout()
        self.time_input_hours = QLineEdit()
        self.time_input_hours.setPlaceholderText("HH")
        self.time_input_hours.setFixedWidth(40)
        self.time_input_minutes = QLineEdit()
        self.time_input_minutes.setPlaceholderText("MM")
        self.time_input_minutes.setFixedWidth(40)
        self.time_input_seconds = QLineEdit()
        self.time_input_seconds.setPlaceholderText("SS")
        self.time_input_seconds.setFixedWidth(40)
        
        time_control_layout.addWidget(QLabel("Time:"))
        time_control_layout.addWidget(self.time_input_hours)
        time_control_layout.addWidget(QLabel(":"))
        time_control_layout.addWidget(self.time_input_minutes)
        time_control_layout.addWidget(QLabel(":"))
        time_control_layout.addWidget(self.time_input_seconds)
        
        self.go_button = QPushButton("Go")
        time_control_layout.addWidget(self.go_button)
        
        # Add buttons to load next/previous 10 seconds of data
        self.load_prev_10s_button = QPushButton("Load Prev 10s")
        self.load_next_10s_button = QPushButton("Load Next 10s")
        time_control_layout.addWidget(self.load_prev_10s_button)
        time_control_layout.addWidget(self.load_next_10s_button)
        
        time_control_layout.addStretch()  # Add space to push controls to the left
        
        # Connect the Go button to the time navigation function
        self.go_button.clicked.connect(lambda: callbacks.on_go_to_time(self))
        # Connect the Load Next/Prev 10s buttons
        self.load_next_10s_button.clicked.connect(self.load_next_10s_data)
        self.load_prev_10s_button.clicked.connect(self.load_prev_10s_data)
        
        raw_trace_layout.addLayout(time_control_layout)
        
        # Create the main plot widget for raw traces
        self.raw_trace_plot = pg.PlotWidget(title="Raw Traces with Spike Templates")
        # Lock the Y-axis to only allow horizontal panning
        self.raw_trace_plot.plotItem.getViewBox().setMouseEnabled(y=False)
        raw_trace_layout.addWidget(self.raw_trace_plot)
        
        self.analysis_tabs.addTab(self.raw_trace_tab, "Raw Trace")
        right_layout.addWidget(self.analysis_tabs)

        # --- Main Splitter and Layout ---
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal)
        self.main_splitter.addWidget(self.left_pane)
        self.main_splitter.addWidget(right_pane)
        self.main_splitter.setSizes([450, 1350])
        main_layout.addWidget(self.main_splitter)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- Menu Bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        load_ks_action = file_menu.addAction("&Load Kilosort Directory...")
        self.load_vision_action = file_menu.addAction("&Load Vision Files...")
        self.load_vision_action.setEnabled(False)
        self.load_classification_action = file_menu.addAction("&Load Classification File...")
        self.load_classification_action.setEnabled(False)
        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)
        
        # Connect Signals to Callback Functions ---
        load_ks_action.triggered.connect(self.load_directory)
        self.load_vision_action.triggered.connect(self.load_vision_directory)
        self.load_classification_action.triggered.connect(self.load_classification_file)
        self.save_action.triggered.connect(self.on_save_action)
        self.filter_button.clicked.connect(self.apply_good_filter)
        self.reset_button.clicked.connect(self.reset_views)
        self.refine_button.clicked.connect(self.on_refine_cluster)
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)
        self.summary_canvas.fig.canvas.mpl_connect('motion_notify_event', self.on_summary_plot_hover)
        
        # Connect the raw trace plot's x-axis range change to update the plot
        self.raw_trace_plot.sigXRangeChanged.connect(self.on_raw_trace_zoom)

    def _switch_left_view(self, index):
        """Switches between the tree and table views in the left pane."""
        self.view_stack.setCurrentIndex(index)

    # --- Helper Method ---
    def _get_selected_cluster_id(self):
        """Returns the cluster_id of the currently selected item from the active view."""
        current_view_index = self.view_stack.currentIndex()

        # Case 1: Tree View is active
        if current_view_index == 0:
            if not self.tree_view.selectionModel().hasSelection():
                return None
            
            index = self.tree_view.selectionModel().selectedIndexes()[0]
            item = self.tree_model.itemFromIndex(index)
            
            # Only leaf nodes (cells) have a cluster ID stored. Groups will return None.
            cluster_id = item.data(Qt.ItemDataRole.UserRole)
            return cluster_id

        # Case 2: Table View is active
        elif current_view_index == 1:
            if not self.table_view.selectionModel().hasSelection() or self.pandas_model is None:
                return None
            
            selected_row = self.table_view.selectionModel().selectedIndexes()[0].row()
            
            # Check if the model has mapToSource method (for proxy models)
            model = self.table_view.model()
            if hasattr(model, 'mapToSource'):
                # The pandas model can be sorted, so we must map the view's row to the model's row
                source_index = model.mapToSource(model.index(selected_row, 0))
                cluster_id = model._data.iloc[source_index.row()]['cluster_id']
            else:
                # If no proxy model, use the row directly
                cluster_id = model._data.iloc[selected_row]['cluster_id']
            return cluster_id
        
        return None

    def setup_tree_model(self, model):
        """Sets up the tree view model and connects the selection changed signal."""
        self.tree_view.setModel(model)
        try:
            self.tree_view.selectionModel().selectionChanged.disconnect(self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.tree_view.selectionModel().selectionChanged.connect(self.on_view_selection_changed)

    def setup_table_model(self, model):
        """Sets up the table view model and connects the selection changed signal."""
        self.table_view.setModel(model)
        try:
            self.table_view.selectionModel().selectionChanged.disconnect(self.on_view_selection_changed)
        except (TypeError, RuntimeError):
            pass
        self.table_view.selectionModel().selectionChanged.connect(self.on_view_selection_changed)
        
    # --- Methods to bridge UI signals to callback functions ---
    def load_directory(self):
        callbacks.load_directory(self)

    def load_vision_directory(self):
        callbacks.load_vision_directory(self)

    def on_view_selection_changed(self, selected, deselected):
        """
        Handles a selection change in either view, synchronizes the other view,
        and then triggers the main plot update callback.
        """
        if self._is_syncing:
            return

        self._is_syncing = True
        
        cluster_id = self._get_selected_cluster_id()
        sender = self.sender()

        if cluster_id is not None:
            # Sync from Tree to Table
            if sender == self.tree_view.selectionModel():
                model = self.table_view.model()
                if hasattr(model, '_data'):
                    df = model._data
                    if cluster_id in df['cluster_id'].values:
                        row_indices = df.index[df['cluster_id'] == cluster_id].tolist()
                        if row_indices:
                            model_row = df.index.get_loc(row_indices[0])
                            source_index = model.index(model_row, 0)
                            # This assumes the model is a proxy model if sorting is enabled
                            view_index = model.mapFromSource(source_index) if hasattr(model, 'mapFromSource') else source_index
                            if view_index.isValid():
                                self.table_view.selectionModel().select(view_index, QItemSelectionModel.ClearAndSelect | QItemSelectionModel.Rows)
                                self.table_view.scrollTo(view_index, QAbstractItemView.ScrollHint.PositionAtCenter)

            # Sync from Table to Tree
            elif sender == self.table_view.selectionModel():
                for row in range(self.tree_model.rowCount()):
                    group_item = self.tree_model.item(row)
                    if not group_item: continue
                    for child_row in range(group_item.rowCount()):
                        child_item = group_item.child(child_row)
                        if child_item and child_item.data(Qt.ItemDataRole.UserRole) == cluster_id:
                            index = self.tree_model.indexFromItem(child_item)
                            self.tree_view.selectionModel().select(index, QItemSelectionModel.ClearAndSelect)
                            self.tree_view.scrollTo(index, QAbstractItemView.ScrollHint.PositionAtCenter)
                            break
                    else:
                        continue
                    break
        
        # Now that views are synced, trigger the update callbacks
        callbacks.on_cluster_selection_changed(self)
        
        self._is_syncing = False

    def on_cluster_selection_changed(self, *args):
        callbacks.on_cluster_selection_changed(self)
        
    def on_tab_changed(self, index):
        callbacks.on_tab_changed(self, index)

    def on_spatial_data_ready(self, cluster_id, features):
        callbacks.on_spatial_data_ready(self, cluster_id, features)
        
    def on_refine_cluster(self):
        callbacks.on_refine_cluster(self)

    def handle_refinement_results(self, parent_id, new_clusters):
        callbacks.handle_refinement_results(self, parent_id, new_clusters)

    def handle_refinement_error(self, error_message):
        callbacks.handle_refinement_error(self, error_message)

    def on_save_action(self):
        callbacks.on_save_action(self)
        
    def apply_good_filter(self):
        callbacks.apply_good_filter(self)

    def reset_views(self):
        callbacks.reset_views(self)

    def select_sta_view(self, view_type):
        """Select the STA view to display."""
        self.current_sta_view = view_type
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            return
        
        # Call the appropriate plotting function based on the selected view
        if view_type == "rf":
            plotting.draw_sta_plot(self, cluster_id)
        elif view_type == "population_rfs":
            print(f"--- 1. DEBUG (MainWindow): Got selected_cell_id = {cluster_id}. Passing to plotting function. ---")
            plotting.draw_population_rfs_plot(self, selected_cell_id=cluster_id)
        elif view_type == "timecourse":
            plotting.draw_sta_timecourse_plot(self, cluster_id)
        elif view_type == "animation":
            plotting.draw_sta_animation_plot(self, cluster_id)

    def update_sta_frame_manual(self, frame_index):
        """Updates the STA visualization to a specific frame manually."""
        if hasattr(self, 'current_sta_data') and self.current_sta_data is not None:
            # Stop any running animation
            plotting.stop_sta_animation(self)
            
            # Update the frame index
            self.current_frame_index = frame_index
            
            # Update the label
            self.sta_frame_label.setText(f"Frame: {frame_index+1}/{self.total_sta_frames}")
            
            # Update the STA canvas with the new frame
            self.sta_canvas.fig.clear()
            analysis_core.animate_sta_movie(
                self.sta_canvas.fig,
                self.current_sta_data,
                frame_index=frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            cluster_id = self.current_sta_cluster_id - 1  # Convert back to 0-indexed
            self.sta_canvas.fig.suptitle(f"Cluster {cluster_id} - STA Frame {frame_index+1}/{self.total_sta_frames}", color='white', fontsize=16)
            self.sta_canvas.draw()

    # In gui/main_window.py

    def prev_sta_frame(self):
        """Go to the previous frame in the STA animation."""
        if hasattr(self, 'current_sta_data') and self.current_sta_data is not None:
            plotting.stop_sta_animation(self)
            self.current_frame_index = (self.current_frame_index - 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.sta_canvas.fig.clear()
            analysis_core.animate_sta_movie(
                self.sta_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit, # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            self.sta_canvas.draw()

    def next_sta_frame(self):
        """Go to the next frame in the STA animation."""
        if hasattr(self, 'current_sta_data') and self.current_sta_data is not None:
            plotting.stop_sta_animation(self)
            self.current_frame_index = (self.current_frame_index + 1) % self.total_sta_frames
            self.sta_frame_slider.setValue(self.current_frame_index)
            self.sta_frame_label.setText(f"Frame: {self.current_frame_index+1}/{self.total_sta_frames}")
            self.sta_canvas.fig.clear()
            analysis_core.animate_sta_movie(
                self.sta_canvas.fig,
                self.current_sta_data,
                stafit=self.current_stafit, # <-- Pass the stored fit
                frame_index=self.current_frame_index,
                sta_width=self.data_manager.vision_sta_width,
                sta_height=self.data_manager.vision_sta_height
            )
            self.sta_canvas.draw()

    def on_summary_plot_hover(self, event):
        plotting.on_summary_plot_hover(self, event)
        
    def on_raw_trace_zoom(self):
        """Handle zoom/pan events in the raw trace plot by updating the visualization."""
        # Prevent recursion
        if self._raw_trace_updating:
            return
        # If a manual load is in progress, don't override it with auto-loading
        if getattr(self, '_raw_trace_manual_load', False):
            return
        self._raw_trace_updating = True
        
        try:
            # Only update if we're currently on the raw trace tab
            if self.analysis_tabs.currentWidget() == self.raw_trace_tab:
                cluster_id = self._get_selected_cluster_id()
                if cluster_id is not None:
                    # Check if we need to load new data based on the visible range
                    view_range = self.raw_trace_plot.viewRange()
                    x_range = view_range[0]  # [min_x, max_x] in seconds
                    
                    # Define threshold for loading new data (30 seconds from buffer edge)
                    buffer_threshold = 30.0  # seconds
                    
                    # Check if we need to load new data
                    if (cluster_id != self.raw_trace_current_cluster_id or  # New cluster selected
                        x_range[0] < (self.raw_trace_buffer_start_time + buffer_threshold) or  # Near start of buffer
                        x_range[1] > (self.raw_trace_buffer_end_time - buffer_threshold)):  # Near end of buffer
                    
                        # Update the current cluster ID
                        self.raw_trace_current_cluster_id = cluster_id
                        
                        # Determine direction of movement to load the next/previous segment
                        if x_range[1] > (self.raw_trace_buffer_end_time - buffer_threshold):
                            # Moving forward, load the next segment
                            buffer_start = self.raw_trace_buffer_end_time
                            buffer_end = buffer_start + 180  # 3 minutes = 180 seconds
                        elif x_range[0] < (self.raw_trace_buffer_start_time + buffer_threshold):
                            # Moving backward, load the previous segment
                            buffer_end = self.raw_trace_buffer_start_time
                            buffer_start = max(0, buffer_end - 180)
                        else:
                            # New cluster selected or centered view - load centered buffer
                            center_time = (x_range[0] + x_range[1]) / 2
                            buffer_duration = 180  # 3 minutes = 180 seconds
                            buffer_start = max(0, center_time - buffer_duration/2)
                            buffer_end = buffer_start + buffer_duration
                        
                        # Ensure we don't exceed the recording duration
                        max_duration = self.data_manager.n_samples / self.data_manager.sampling_rate
                        if buffer_end > max_duration:
                            buffer_end = max_duration
                            buffer_start = max(0, buffer_end - 180)
                        
                        self.raw_trace_buffer_start_time = buffer_start
                        self.raw_trace_buffer_end_time = buffer_end
                        
                        # Update the plot with the new buffer data
                        plotting.update_raw_trace_plot(self, cluster_id)
                    else:
                        # Just update the display with the existing buffer
                        plotting.update_raw_trace_plot(self, cluster_id)
        finally:
            self._raw_trace_updating = False
            
    def load_next_10s_data(self):
        """Load the next 10 seconds of raw trace data."""
        if self.data_manager.raw_data_memmap is None:
            self.status_bar.showMessage("No raw data file loaded.")
            return
            
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            self.status_bar.showMessage("No cluster selected.")
            return
        
        # Set the manual load flag to prevent on_raw_trace_zoom from interfering
        self._raw_trace_manual_load = True
        
        # Calculate the new buffer range: from end of current buffer to 10 seconds ahead
        buffer_start = self.raw_trace_buffer_end_time
        buffer_end = buffer_start + 10  # Load the next 10 seconds
        
        # Ensure we don't exceed the recording duration
        max_duration = self.data_manager.n_samples / self.data_manager.sampling_rate
        if buffer_end > max_duration:
            buffer_end = max_duration
            
        self.raw_trace_buffer_start_time = buffer_start
        self.raw_trace_buffer_end_time = buffer_end
        
        # Update the plot with the new buffer data
        plotting.update_raw_trace_plot(self, cluster_id)
        
        # Update plot view to show the new data
        self.raw_trace_plot.setXRange(buffer_start, buffer_end, padding=0)
        
        # Reset the manual load flag after a short delay to allow UI updates
        from qtpy.QtCore import QTimer
        QTimer.singleShot(100, lambda: setattr(self, '_raw_trace_manual_load', False))
        
    def load_prev_10s_data(self):
        """Load the previous 10 seconds of raw trace data."""
        if self.data_manager.raw_data_memmap is None:
            self.status_bar.showMessage("No raw data file loaded.")
            return
            
        cluster_id = self._get_selected_cluster_id()
        if cluster_id is None:
            self.status_bar.showMessage("No cluster selected.")
            return
        
        # Set the manual load flag to prevent on_raw_trace_zoom from interfering
        self._raw_trace_manual_load = True
        
        # Calculate the new buffer range: from 10 seconds before current start to current start
        buffer_end = self.raw_trace_buffer_start_time
        buffer_start = max(0, buffer_end - 10)  # Load the previous 10 seconds
        
        self.raw_trace_buffer_start_time = buffer_start
        self.raw_trace_buffer_end_time = buffer_end
        
        # Update the plot with the new buffer data
        plotting.update_raw_trace_plot(self, cluster_id)
        
        # Update plot view to show the new data
        self.raw_trace_plot.setXRange(buffer_start, buffer_end, padding=0)
        
        # Reset the manual load flag after a short delay to allow UI updates
        from qtpy.QtCore import QTimer
        QTimer.singleShot(100, lambda: setattr(self, '_raw_trace_manual_load', False))
        
    def load_classification_file(self):
        callbacks.load_classification_file(self)
        
    def open_tree_context_menu(self, position):
        menu = QMenu()
        add_group_action = menu.addAction("Add New Group")
        
        action = menu.exec(self.tree_view.viewport().mapToGlobal(position))
        
        if action == add_group_action:
            text, ok = QInputDialog.getText(self, 'New Group', 'Enter group name:')
            if ok and text:
                callbacks.add_new_group(self, text)
    
    def toggle_sidebar(self):
        """Collapses or expands the left sidebar by manipulating the main splitter."""
        if self.sidebar_collapsed:
            # --- EXPAND --- 
            self.sidebar_toggle_button.setText("◀")
            widths = self.main_splitter.sizes()
            total_width = sum(widths)
            self.main_splitter.setSizes([self.last_left_width, total_width - self.last_left_width])
            self.sidebar_collapsed = False
        else:
            # --- COLLAPSE ---
            self.sidebar_toggle_button.setText("▶")
            widths = self.main_splitter.sizes()
            # Save the current width if it's not already collapsed
            if widths[0] > 35:
                self.last_left_width = widths[0]
            total_width = sum(widths)
            self.main_splitter.setSizes([35, total_width - 35])
            self.sidebar_collapsed = True

    def update_ei_frame_manual(self, frame_index):
        """Updates the EI visualization to a specific frame manually."""
        if hasattr(self, 'current_ei_data') and self.current_ei_data is not None:
            # Stop any running animation
            if hasattr(self, 'ei_animation_timer') and self.ei_animation_timer and self.ei_animation_timer.isActive():
                self.ei_animation_timer.stop()
            
            # Update the frame index
            self.current_frame = frame_index
            
            # Update the label
            self.ei_frame_label.setText(f"Frame: {frame_index+1}/{self.n_frames}")
            
            # Update the EI canvas with the new frame
            self.summary_canvas.fig.clear()
            from gui import plotting
            plotting.draw_vision_ei_frame(
                self, 
                self.current_ei_data[:, frame_index], 
                frame_index, 
                self.n_frames
            )
            self.summary_canvas.draw()

    def start_ei_animation(self):
        """Start the EI animation."""
        if hasattr(self, 'current_ei_data') and self.current_ei_data is not None:
            # If timer doesn't exist, create it
            if self.ei_animation_timer is None:
                from qtpy.QtCore import QTimer
                self.ei_animation_timer = QTimer()
                self.ei_animation_timer.timeout.connect(lambda: self.update_ei_frame())
            
            # Start the timer
            if not self.ei_animation_timer.isActive():
                self.ei_animation_timer.start(100)  # 100ms per frame (10 fps)

    def pause_ei_animation(self):
        """Pause the EI animation."""
        if hasattr(self, 'ei_animation_timer') and self.ei_animation_timer and self.ei_animation_timer.isActive():
            self.ei_animation_timer.stop()

    def prev_ei_frame(self):
        """Go to the previous frame in the EI animation."""
        if hasattr(self, 'current_ei_data') and self.current_ei_data is not None:
            # Stop any running animation
            if hasattr(self, 'ei_animation_timer') and self.ei_animation_timer and self.ei_animation_timer.isActive():
                self.ei_animation_timer.stop()
            
            # Calculate previous frame index with wrap-around
            self.current_frame = (self.current_frame - 1) % self.n_frames
            self.ei_frame_slider.setValue(self.current_frame)
            self.update_ei_frame_manual(self.current_frame)

    def next_ei_frame(self):
        """Go to the next frame in the EI animation."""
        if hasattr(self, 'current_ei_data') and self.current_ei_data is not None:
            # Stop any running animation
            if hasattr(self, 'ei_animation_timer') and self.ei_animation_timer and self.ei_animation_timer.isActive():
                self.ei_animation_timer.stop()
            
            # Calculate next frame index with wrap-around
            self.current_frame = (self.current_frame + 1) % self.n_frames
            self.ei_frame_slider.setValue(self.current_frame)
            self.update_ei_frame_manual(self.current_frame)

    def update_ei_frame(self):
        """Updates the EI visualization to the next frame in the animation."""
        if self.current_frame >= self.n_frames - 1:
            self.current_frame = 0
        else:
            self.current_frame += 1
        
        self.ei_frame_slider.setValue(self.current_frame)
        self.update_ei_frame_manual(self.current_frame)

    def closeEvent(self, event):
        """Handles the window close event."""
        if self.data_manager and self.data_manager.is_dirty:
            reply = QMessageBox.question(self, 'Unsaved Changes',
                "You have unsaved refinement changes. Do you want to save before exiting?",
                QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel)
            if reply == QMessageBox.StandardButton.Save:
                self.on_save_action()
            elif reply == QMessageBox.StandardButton.Cancel:
                event.ignore()
                return
        callbacks.stop_worker(self)
        event.accept()
