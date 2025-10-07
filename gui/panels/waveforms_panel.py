from qtpy.QtWidgets import QWidget, QVBoxLayout, QSplitter
import pyqtgraph as pg
import numpy as np
from qtpy.QtCore import Qt
from scipy.ndimage import gaussian_filter1d

class WaveformPanel(QWidget):
    """
    Panel for displaying cluster waveforms, ISI histogram, and firing rate.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        splitter = QSplitter()
        splitter.setOrientation(pg.QtCore.Qt.Vertical)

        # Top: Waveform plot
        self.waveform_plot = pg.PlotWidget(title="Waveforms (Sampled)")
        splitter.addWidget(self.waveform_plot)

        # Bottom: ISI and FR side by side
        bottom_splitter = QSplitter()
        bottom_splitter.setOrientation(pg.QtCore.Qt.Horizontal)
        self.isi_plot = pg.PlotWidget(title="Inter-Spike Interval (ISI) Histogram")
        self.fr_plot = pg.PlotWidget(title="Smoothed Firing Rate")
        bottom_splitter.addWidget(self.isi_plot)
        bottom_splitter.addWidget(self.fr_plot)
        splitter.addWidget(bottom_splitter)

        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

    def update_waveforms(self, cluster_ids, features_dict, color_map=None):
        """
        Update the waveform plot for the given cluster(s).
        Args:
            cluster_ids: list of cluster IDs to plot
            features_dict: dict mapping cluster_id -> features (must include 'raw_snippets')
            color_map: optional dict mapping cluster_id -> color (tuple or string)
        """
        self.waveform_plot.clear()
        if not cluster_ids or not features_dict:
            self.waveform_plot.setTitle("Waveforms (No data)")
            return

        if color_map is None:
            color_cycle = [(200, 200, 255), (255, 180, 180), (180, 255, 180), (255, 255, 180), (180, 255, 255)]
            color_map = {cid: color_cycle[i % len(color_cycle)] for i, cid in enumerate(cluster_ids)}

        for cid in cluster_ids:
            features = features_dict.get(cid)
            if not features or 'raw_snippets' not in features:
                continue
            snippets = features['raw_snippets']  # shape: (n_channels, n_samples, n_snips)
            if snippets.ndim == 3:
                p2p = snippets.max(axis=1) - snippets.min(axis=1)
                dom_chan = np.argmax(p2p.mean(axis=1))
                mean_waveform = snippets[dom_chan].mean(axis=1)
                x = np.arange(mean_waveform.size)
                self.waveform_plot.plot(x, mean_waveform, pen=pg.mkPen(color=color_map[cid], width=2), name=f"Cluster {cid}")
            elif snippets.ndim == 2:
                mean_waveform = snippets.mean(axis=1)
                x = np.arange(mean_waveform.size)
                self.waveform_plot.plot(x, mean_waveform, pen=pg.mkPen(color=color_map[cid], width=2), name=f"Cluster {cid}")

        self.waveform_plot.setTitle("Waveforms (Sampled)")
        self.waveform_plot.setLabel('bottom', 'Sample')
        self.waveform_plot.setLabel('left', 'uV')

    def update_isi(self, cluster_id, spike_times, sampling_rate):
        """
        Update the ISI histogram for the given cluster.
        Args:
            cluster_id: cluster ID
            spike_times: numpy array of spike times (in samples)
            sampling_rate: sampling rate in Hz
        """
        self.isi_plot.clear()
        if spike_times is None or len(spike_times) < 2:
            self.isi_plot.setTitle("ISI (No data)")
            return
        isi_ms = np.diff(spike_times) / sampling_rate * 1000
        bins = np.linspace(0, 50, 101)
        y, x = np.histogram(isi_ms, bins=bins)
        self.isi_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0, 163, 224, 150))
        self.isi_plot.addLine(x=2.0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2))
        self.isi_plot.setTitle(f"ISI Histogram (Cluster {cluster_id})")
        self.isi_plot.setLabel('bottom', 'ISI (ms)')
        self.isi_plot.setLabel('left', 'Count')

    def update_fr(self, cluster_id, spike_times, sampling_rate):
        """
        Update the firing rate plot for the given cluster.
        Args:
            cluster_id: cluster ID
            spike_times: numpy array of spike times (in samples)
            sampling_rate: sampling rate in Hz
        """
        self.fr_plot.clear()
        if spike_times is None or len(spike_times) < 2:
            self.fr_plot.setTitle("Firing Rate (No data)")
            return
        spike_times_sec = spike_times / sampling_rate
        duration = spike_times_sec[-1] - spike_times_sec[0]
        if duration <= 0:
            self.fr_plot.setTitle("Firing Rate (No data)")
            return
        # bin_size = 0.1  # 100 ms bins
        # bins = np.arange(spike_times_sec[0], spike_times_sec[-1] + bin_size, bin_size)
        bins = np.arange(0, duration+1,1)
        counts, _ = np.histogram(spike_times_sec, bins=bins)
        rate = gaussian_filter1d(counts.astype(float), sigma=5)
        self.fr_plot.plot(bins[:-1], rate, pen='y')
        self.fr_plot.setTitle(f"Firing Rate (Cluster {cluster_id})")
        self.fr_plot.setLabel('bottom', 'Time (s)')
        self.fr_plot.setLabel('left', 'Rate (Hz)')

    def clear(self):
        self.waveform_plot.clear()
        self.isi_plot.clear()
        self.fr_plot.clear()
        self.waveform_plot.setTitle("Waveforms (Sampled)")
        self.isi_plot.setTitle("ISI Histogram")
        self.fr_plot.setTitle("Firing Rate")