from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel, QSplitter
from qtpy.QtCore import Qt, QTimer
import numpy as np
from gui.widgets import MplCanvas
import matplotlib.pyplot as plt

class EIPanel(QWidget):
    """
    Panel for spatial/EI analysis, including controls and matplotlib canvas.
    """
    def __init__(self, main_window, parent=None):
        super().__init__(parent)
        self.main_window = main_window  # Needed for data access and callbacks

         # --- Splitter for spatial (left) and temporal (right) EI ---
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(0)

        # --- Spatial EI Canvas ---
        self.spatial_canvas = MplCanvas(self, width=10, height=8, dpi=120)
        left_layout.addWidget(self.spatial_canvas)
        self.spatial_canvas.fig.canvas.mpl_connect('motion_notify_event', self.on_canvas_hover)
        splitter.addWidget(left_widget)

        # --- Temporal EI Canvas (right) ---
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(0)
        self.temporal_canvas = MplCanvas(self, width=7, height=6, dpi=120)
        right_layout.addWidget(self.temporal_canvas)
        splitter.addWidget(right_widget)

        # --- Main Layout ---
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.addWidget(splitter)

        self.current_ei_data = None
        self.current_cluster_ids = None
        self.n_frames = 0
        self.current_frame = 0

    # --- Public API ---

    def on_canvas_hover(self, event):
        # Handles hover events on the summary plot for tooltips.
        if event.inaxes is None or self.main_window.data_manager is None or self.current_ei_data is None:
            return
        if event.inaxes in self.spatial_canvas.fig.axes:
            positions = self.main_window.data_manager.channel_positions
            mouse_pos = np.array([[event.xdata, event.ydata]])
            distances = np.linalg.norm(positions - mouse_pos, axis=1)
            if distances.min() < 20:
                closest_idx = distances.argmin()
                self.main_window.status_bar.showMessage(f"Channel ID {closest_idx}")
            else:
                self.main_window.status_bar.clearMessage()
    
    def update_ei(self, cluster_ids):
        """
        Main entry point: update the EI panel for one or more clusters.
        """
        cluster_ids = np.array(cluster_ids)
        if cluster_ids.ndim == 0:
            cluster_ids = np.array([cluster_ids])
        vision_cluster_ids = cluster_ids + 1

        # Check for Vision EI
        has_vision_ei = self.main_window.data_manager.vision_eis and any(
            cid in self.main_window.data_manager.vision_eis for cid in vision_cluster_ids
        )

        if has_vision_ei:
            self._load_and_draw_vision_ei(cluster_ids)
        else:
            self._load_and_draw_ks_ei(cluster_ids)

    def clear(self):
        self.spatial_canvas.fig.clear()
        self.spatial_canvas.draw()
        self.temporal_canvas.fig.clear()
        self.temporal_canvas.draw()

    # --- Internal: Vision EI ---

    def _load_and_draw_vision_ei(self, cluster_ids):
        vision_cluster_ids = cluster_ids + 1
        ei_data_list = []
        for cid in vision_cluster_ids:
            if cid in self.main_window.data_manager.vision_eis:
                ei_data_list.append(self.main_window.data_manager.vision_eis[cid].ei)
        if not ei_data_list:
            self.clear()
            return

        self.current_ei_data = ei_data_list
        self.current_cluster_ids = cluster_ids
        self.n_frames = ei_data_list[0].shape[1]
        self.current_frame = 0

        # Draw peak summary frame
        summary_frames = [self._create_peak_summary_frame(ei) for ei in ei_data_list]
        self._draw_vision_ei_frame(summary_frames, cluster_ids)
        # self._draw_vision_ei_temporal(ei_data_list, cluster_ids)

    def _draw_vision_ei_temporal(self, ei_data_list, cluster_ids, channels):
        """
        Example: Plot temporal EI traces for the given clusters.
        """
        self.temporal_canvas.fig.clear()
        n_clusters = len(ei_data_list)
        axes = self.temporal_canvas.fig.subplots(nrows=n_clusters, ncols=1, squeeze=False)[0]
        for i, ei_data in enumerate(ei_data_list):
            ax = axes[i]
            time = np.arange(ei_data.shape[1]) / self.main_window.data_manager.sampling_rate * 1000  # ms
            for ch in range(ei_data.shape[0]):
                ax.plot(time, ei_data[ch, :], alpha=0.3)
            ax.set_title(f"Cluster {cluster_ids[i]} Temporal EI")
            ax.set_xlabel("Time (ms)")
            ax.set_ylabel("Amplitude (µV)")
    
    def _draw_vision_ei_frame(self, ls_summary, cluster_ids):
        n_clusters = len(ls_summary)
        self.spatial_canvas.fig.clear()
        axes = self.spatial_canvas.fig.subplots(nrows=1, ncols=n_clusters, squeeze=False)[0]
        for i, frame_data in enumerate(ls_summary):
            ax = axes[i]
            ax.set_title(f"{cluster_ids[i]} EI")
            ei_map = self._reshape_ei(
                frame_data[:, np.newaxis],
                self.main_window.data_manager.sorted_channels
            )
            ei_map = np.log10(ei_map + 1e-6)
            im = ax.imshow(ei_map, cmap='hot', aspect='auto', origin='lower')
            self.spatial_canvas.fig.colorbar(im, ax=ax, label='Log10 Amplitude (µV)')
        self.spatial_canvas.fig.suptitle("Spatial Analysis (EI)", color='white', fontsize=16)
        self.spatial_canvas.draw()

    # --- Internal: Kilosort EI ---

    def _load_and_draw_ks_ei(self, cluster_ids):
        lightweight_features = self.main_window.data_manager.get_lightweight_features(cluster_ids)
        heavyweight_features = self.main_window.data_manager.get_heavyweight_features(cluster_ids)
        if lightweight_features is None or heavyweight_features is None:
            self.clear()
            self.spatial_canvas.fig.text(0.5, 0.5, "Error generating features.", ha='center', va='center', color='red')
            self.spatial_canvas.draw()
            return

        self.spatial_canvas.fig.clear()
        import analysis_core
        analysis_core.plot_rich_ei(
            self.spatial_canvas.fig, lightweight_features['median_ei'],
            self.main_window.data_manager.channel_positions,
            heavyweight_features, self.main_window.data_manager.sampling_rate, pre_samples=20
        )
        self.spatial_canvas.fig.suptitle(f"Cluster {cluster_ids} Spatial Analysis", color='white', fontsize=16)
        self.spatial_canvas.draw()

    # --- Utility functions (copied from plotting.py) ---

    def _reshape_ei(self, ei: np.ndarray, sorted_electrodes: np.ndarray, n_rows: int=16) -> np.ndarray:
        if ei.shape[0] != 512:
            print(f'Warning: Expected EI shape (512, 201), got {ei.shape}')
        n_electrodes = ei.shape[0]
        n_frames = ei.shape[1]
        n_cols = n_electrodes // n_rows
        if n_cols * n_rows != n_electrodes:
            raise ValueError(f"Number of electrodes {n_electrodes} is not compatible with {n_rows} rows and {n_cols} columns.")
        sorted_ei = ei[sorted_electrodes]
        reshaped_ei = sorted_ei.reshape(n_rows, n_cols, n_frames)
        return reshaped_ei

    def _create_peak_summary_frame(self, ei_data):
        return np.max(np.abs(ei_data), axis=1)