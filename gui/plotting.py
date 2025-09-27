import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import analysis_core

def draw_summary_plot(main_window, cluster_id):
    """Draws the main spatial analysis plot, switching between custom EI and Vision RF."""
    # --- MODIFICATION: Check for Vision Data First ---
    if main_window.data_manager.vision_stas and cluster_id in main_window.data_manager.vision_stas:
        sta_data = main_window.data_manager.vision_stas[cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(cluster_id)
        
        main_window.summary_canvas.fig.clear()
        analysis_core.plot_vision_rf(main_window.summary_canvas.fig, sta_data, stafit)
        main_window.summary_canvas.fig.suptitle(f"Cluster {cluster_id} Receptive Field", color='white', fontsize=16)
        main_window.summary_canvas.draw()
        main_window.current_spatial_features = None # Clear features from other plot types
    else:
        # --- Fallback to original EI-based spatial plot ---
        lightweight_features = main_window.data_manager.get_lightweight_features(cluster_id)
        heavyweight_features = main_window.data_manager.get_heavyweight_features(cluster_id)
        main_window.current_spatial_features = heavyweight_features
        if lightweight_features is None or heavyweight_features is None:
            main_window.summary_canvas.fig.clear()
            main_window.summary_canvas.fig.text(0.5, 0.5, "Error generating features.", ha='center', va='center', color='red')
            main_window.summary_canvas.draw()
            return
            
        main_window.summary_canvas.fig.clear()
        analysis_core.plot_rich_ei(
            main_window.summary_canvas.fig, lightweight_features['mean_ei'], main_window.data_manager.channel_positions,
            heavyweight_features, main_window.data_manager.sampling_rate, pre_samples=20)
        main_window.summary_canvas.fig.suptitle(f"Cluster {cluster_id} Spatial Analysis", color='white', fontsize=16)
        main_window.summary_canvas.draw()

def on_summary_plot_hover(main_window, event):
    """Handles hover events on the summary plot for tooltips."""
    if (event.inaxes is None or main_window.data_manager is None or main_window.current_spatial_features is None):
        return
    if event.inaxes == main_window.summary_canvas.fig.axes[0]:
        positions = main_window.data_manager.channel_positions
        ptp_amps = main_window.current_spatial_features.get('ptp_amps')
        if ptp_amps is None:
            return
        mouse_pos = np.array([[event.xdata, event.ydata]])
        distances = cdist(mouse_pos, positions)[0]
        if distances.min() < 20:
            closest_idx = distances.argmin()
            ptp = ptp_amps[closest_idx]
            main_window.status_bar.showMessage(f"Channel ID {closest_idx}: PTP = {ptp:.2f} µV")

def update_waveform_plot(main_window, cluster_id, lightweight_features):
    """Updates the waveform plot with data for the selected cluster."""
    main_window.waveform_plot.clear()
    main_window.waveform_plot.setTitle(f"Cluster {cluster_id} | Waveforms (Sampled)")
    mean_ei = lightweight_features['mean_ei']
    snippets = lightweight_features['raw_snippets']
    p2p = mean_ei.max(axis=1) - mean_ei.min(axis=1)
    dom_chan = np.argmax(p2p)
    pre_peak_samples = 20
    time_axis = (np.arange(mean_ei.shape[1]) - pre_peak_samples) / main_window.data_manager.sampling_rate * 1000
    for i in range(snippets.shape[2]):
        main_window.waveform_plot.plot(time_axis, snippets[dom_chan, :, i], pen=pg.mkPen(color=(200, 200, 200, 30)))
    main_window.waveform_plot.plot(time_axis, mean_ei[dom_chan], pen=pg.mkPen('#00A3E0', width=2.5))
    main_window.waveform_plot.setLabel('bottom', 'Time (ms)')
    main_window.waveform_plot.setLabel('left', 'Amplitude (uV)')

def update_isi_plot(main_window, cluster_id):
    """Updates the ISI histogram plot."""
    main_window.isi_plot.clear()
    violation_rate = main_window.data_manager._calculate_isi_violations(cluster_id)
    main_window.isi_plot.setTitle(f"Cluster {cluster_id} | ISI | Violations: {violation_rate:.2f}%")
    spikes = main_window.data_manager.get_cluster_spikes(cluster_id)
    if len(spikes) < 2:
        return
    isis_ms = np.diff(np.sort(spikes)) / main_window.data_manager.sampling_rate * 1000
    y, x = np.histogram(isis_ms, bins=np.linspace(0, 50, 101))
    main_window.isi_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0, 163, 224, 150))
    main_window.isi_plot.addLine(x=2.0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2))
    main_window.isi_plot.setLabel('bottom', 'ISI (ms)')
    main_window.isi_plot.setLabel('left', 'Count')

def update_fr_plot(main_window, cluster_id):
    """Updates the smoothed firing rate plot."""
    main_window.fr_plot.clear()
    main_window.fr_plot.setTitle(f"Cluster {cluster_id} | Firing Rate")
    spikes_sec = main_window.data_manager.get_cluster_spikes(cluster_id) / main_window.data_manager.sampling_rate
    if len(spikes_sec) == 0:
        return
    total_duration = main_window.data_manager.spike_times.max() / main_window.data_manager.sampling_rate
    bins = np.arange(0, total_duration + 1, 1)
    counts, _ = np.histogram(spikes_sec, bins=bins)
    rate = gaussian_filter1d(counts.astype(float), sigma=5)
    main_window.fr_plot.plot(bins[:-1], rate, pen='y')
    main_window.fr_plot.setLabel('bottom', 'Time (s)')
    main_window.fr_plot.setLabel('left', 'Firing Rate (Hz)')
