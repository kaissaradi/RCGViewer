from qtpy.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QSplitter, QStatusBar,
    QHeaderView, QMessageBox, QTabWidget, QTableView
)
from qtpy.QtCore import Qt
from qtpy.QtGui import QFont
import pyqtgraph as pg

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
        self.refine_thread = None
        self.refinement_worker = None
        self.worker_thread = None
        self.spatial_worker = None
        self.spatial_plot_dirty = False
        self.current_spatial_features = None

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
        left_pane = QWidget()
        left_layout = QVBoxLayout(left_pane)
        left_pane.setFixedWidth(450)
        
        filter_box = QHBoxLayout()
        self.filter_button = QPushButton("Filter 'Good'")
        self.reset_button = QPushButton("Reset View")
        filter_box.addWidget(self.filter_button)
        filter_box.addWidget(self.reset_button)
        
        self.table_view = QTableView()
        self.table_view.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table_view.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.table_view.setSortingEnabled(True)
        
        self.refine_button = QPushButton("Refine Selected Cluster")
        self.refine_button.setFixedHeight(40)
        self.refine_button.setStyleSheet("font-size: 14px; font-weight: bold; color: #aeffe3; background-color: #005230;")
        
        left_layout.addLayout(filter_box)
        left_layout.addWidget(self.table_view)
        left_layout.addWidget(self.refine_button)

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
        self.summary_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        summary_layout.addWidget(self.summary_canvas)
        self.analysis_tabs.addTab(self.summary_tab, "Spatial Analysis")
        right_layout.addWidget(self.analysis_tabs)

        # --- Main Splitter and Layout ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(left_pane)
        splitter.addWidget(right_pane)
        splitter.setSizes([450, 1350])
        main_layout.addWidget(splitter)
        
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)

        # --- Menu Bar ---
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        load_ks_action = file_menu.addAction("&Load Kilosort Directory...")
        self.load_vision_action = file_menu.addAction("&Load Vision Files...")
        self.load_vision_action.setEnabled(False)
        self.save_action = file_menu.addAction("&Save Results...")
        self.save_action.setEnabled(False)
        
        # --- Connect Signals to Callback Functions ---
        load_ks_action.triggered.connect(self.load_directory)
        self.load_vision_action.triggered.connect(self.load_vision_directory)
        self.save_action.triggered.connect(self.on_save_action)
        self.filter_button.clicked.connect(self.apply_good_filter)
        self.reset_button.clicked.connect(self.reset_table_view)
        self.refine_button.clicked.connect(self.on_refine_cluster)
        self.analysis_tabs.currentChanged.connect(self.on_tab_changed)
        self.summary_canvas.fig.canvas.mpl_connect('motion_notify_event', self.on_summary_plot_hover)

    # --- Helper Method ---
    def _get_selected_cluster_id(self):
        """Returns the cluster_id of the currently selected row."""
        if not self.table_view.selectionModel().hasSelection():
            return None
        row = self.table_view.selectionModel().selectedRows()[0].row()
        return self.pandas_model._dataframe.iloc[row]['cluster_id']

    # --- Methods to bridge UI signals to callback functions ---
    def load_directory(self):
        callbacks.load_directory(self)

    def load_vision_directory(self):
        callbacks.load_vision_directory(self)

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

    def reset_table_view(self):
        callbacks.reset_table_view(self)

    def on_summary_plot_hover(self, event):
        plotting.on_summary_plot_hover(self, event)
        
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
