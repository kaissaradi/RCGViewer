from qtpy.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QSlider, QLabel
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

        layout = QVBoxLayout(self)
        # --- Controls ---
        control_layout = QHBoxLayout()
        self.prev_frame_button = QPushButton("<< Prev")
        self.next_frame_button = QPushButton("Next >>")
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(29)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_label = QLabel("Frame: 0/30")
        self.play_button = QPushButton("Play")
        self.pause_button = QPushButton("Pause")

        control_layout.addWidget(self.prev_frame_button)
        control_layout.addWidget(self.frame_slider)
        control_layout.addWidget(self.next_frame_button)
        control_layout.addWidget(self.frame_label)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        layout.addLayout(control_layout)

        # --- Canvas ---
        self.canvas = MplCanvas(self, width=10, height=8, dpi=120)
        layout.addWidget(self.canvas)
        self.canvas.fig.canvas.mpl_connect('motion_notify_event', self.on_canvas_hover)

        # --- Animation State ---
        self.ei_animation_timer = QTimer()
        self.ei_animation_timer.setInterval(100)
        self.ei_animation_timer.timeout.connect(self._on_timer_tick)
        self.current_ei_data = None
        self.current_cluster_ids = None
        self.n_frames = 0
        self.current_frame = 0

        # --- Connect Controls ---
        self.frame_slider.valueChanged.connect(self._on_slider_changed)
        self.prev_frame_button.clicked.connect(self._on_prev_frame)
        self.next_frame_button.clicked.connect(self._on_next_frame)
        self.play_button.clicked.connect(self._on_play)
        self.pause_button.clicked.connect(self._on_pause)

    # --- Public API ---

    def on_canvas_hover(self, event):
        # Handles hover events on the summary plot for tooltips.
        # if (event.inaxes is None or main_window.data_manager is None or main_window.current_spatial_features is None):
        #     return
        # if event.inaxes == main_window.summary_canvas.fig.axes[0]:
        #     positions = main_window.data_manager.channel_positions
        #     ptp_amps = main_window.current_spatial_features.get('ptp_amps')
        #     if ptp_amps is None:
        #         return
        #     mouse_pos = np.array([[event.xdata, event.ydata]])
        #     distances = cdist(mouse_pos, positions)[0]
        #     if distances.min() < 20:
        #         closest_idx = distances.argmin()
        #         ptp = ptp_amps[closest_idx]
        #         main_window.status_bar.showMessage(f"Channel ID {closest_idx}: PTP = {ptp:.2f} µV")
        if event.inaxes is None or self.main_window.data_manager is None or self.current_ei_data is None:
            return
        if event.inaxes in self.canvas.fig.axes:
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
            self._load_vision_ei(cluster_ids)
        else:
            self._load_kilosort_ei(cluster_ids)

    def clear(self):
        self.canvas.fig.clear()
        self.canvas.draw()
        self.frame_slider.setEnabled(False)
        self.frame_label.setText("Frame: 0/0")

    # --- Internal: Vision EI ---

    def _load_vision_ei(self, cluster_ids):
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

        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(self.n_frames - 1)
        self.frame_slider.setValue(0)
        self.frame_label.setText(f"Frame: Peak Summary")
        self.frame_slider.setEnabled(True)

        # Draw peak summary frame
        summary_frames = [self._create_peak_summary_frame(ei) for ei in ei_data_list]
        self._draw_vision_ei_frame(summary_frames, -1)

    def _draw_vision_ei_frame(self, ls_summary, frame_index):
        n_clusters = len(ls_summary)
        self.canvas.fig.clear()
        axes = self.canvas.fig.subplots(nrows=1, ncols=n_clusters, squeeze=False)[0]
        for i, frame_data in enumerate(ls_summary):
            ax = axes[i]
            if frame_index == -1:
                ax.set_title("Vision EI - Peak Summary")
                ei_map = self._reshape_ei(
                    frame_data[:, np.newaxis],
                    self.main_window.data_manager.sorted_channels
                )
                ei_map = np.log10(ei_map + 1e-6)
                im = ax.imshow(ei_map, cmap='hot', aspect='auto', origin='lower')
                self.canvas.fig.colorbar(im, ax=ax, label='Log10 Amplitude (µV)')
            else:
                # For frame-by-frame animation, you can implement as needed
                pass
        self.canvas.fig.suptitle("Spatial Analysis (EI)", color='white', fontsize=16)
        self.canvas.draw()

    def _on_timer_tick(self):
        if self.current_ei_data is None:
            return
        self.current_frame = (self.current_frame + 1) % self.n_frames
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(self.current_frame)
        self.frame_slider.blockSignals(False)
        # Draw the current frame (implement as needed)
        # For now, just update the label
        self.frame_label.setText(f"Frame: {self.current_frame+1}/{self.n_frames}")

    def _on_slider_changed(self, value):
        self.current_frame = value
        self.frame_label.setText(f"Frame: {value+1}/{self.n_frames}")
        # Optionally, draw the selected frame

    def _on_prev_frame(self):
        if self.n_frames == 0:
            return
        self.current_frame = (self.current_frame - 1) % self.n_frames
        self.frame_slider.setValue(self.current_frame)

    def _on_next_frame(self):
        if self.n_frames == 0:
            return
        self.current_frame = (self.current_frame + 1) % self.n_frames
        self.frame_slider.setValue(self.current_frame)

    def _on_play(self):
        if not self.ei_animation_timer.isActive():
            self.ei_animation_timer.start()

    def _on_pause(self):
        if self.ei_animation_timer.isActive():
            self.ei_animation_timer.stop()

    # --- Internal: Kilosort EI ---

    def _load_kilosort_ei(self, cluster_ids):
        lightweight_features = self.main_window.data_manager.get_lightweight_features(cluster_ids)
        heavyweight_features = self.main_window.data_manager.get_heavyweight_features(cluster_ids)
        if lightweight_features is None or heavyweight_features is None:
            self.clear()
            self.canvas.fig.text(0.5, 0.5, "Error generating features.", ha='center', va='center', color='red')
            self.canvas.draw()
            return

        self.canvas.fig.clear()
        import analysis_core
        analysis_core.plot_rich_ei(
            self.canvas.fig, lightweight_features['median_ei'],
            self.main_window.data_manager.channel_positions,
            heavyweight_features, self.main_window.data_manager.sampling_rate, pre_samples=20
        )
        self.canvas.fig.suptitle(f"Cluster {cluster_ids} Spatial Analysis", color='white', fontsize=16)
        self.canvas.draw()

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