# --- Standard Library Imports ---
import os
from pathlib import Path
from typing import Tuple

# --- Third-Party Imports ---
import numpy as np
import torch
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.patches import Ellipse
import networkx as nx
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from scipy.interpolate import griddata

# --- Interactive Plotting Imports ---
import ipywidgets as widgets
from ipywidgets import HBox
from IPython.display import display

def compute_per_spike_features(
    snippets: np.ndarray,
    channel_positions: np.ndarray,
    n_pcs: int = 3
) -> np.ndarray:
    """
    Computes waveform and spatial features for each spike in a vectorized manner.
    """
    n_channels, n_samples, n_spikes = snippets.shape
    waveforms_flat = snippets.reshape(n_channels * n_samples, n_spikes).T

    pca = PCA(n_components=n_pcs)
    waveform_features = pca.fit_transform(waveforms_flat)

    ptp_amplitudes = np.ptp(snippets, axis=1)
    sum_of_masses = np.sum(ptp_amplitudes, axis=0)
    sum_of_masses[sum_of_masses == 0] = 1e-9

    weighted_positions = ptp_amplitudes.T @ channel_positions
    spatial_features = weighted_positions / sum_of_masses[:, np.newaxis]

    all_features = np.hstack((waveform_features, spatial_features))
    return all_features

def extract_snippets(dat_path, spike_times, window=(-20, 60), n_channels=512, dtype='int16'):
    """
    Extracts snippets of raw data using memory-mapping for high efficiency.
    """
    snip_len = window[1] - window[0]
    spike_count = len(spike_times)
    
    if spike_count == 0:
        return np.zeros((n_channels, snip_len, 0), dtype=np.float32)

    raw_data = np.memmap(dat_path, dtype=dtype, mode='r').reshape(-1, n_channels)
    total_samples = raw_data.shape[0]

    snips = np.zeros((spike_count, n_channels, snip_len), dtype=np.float32)

    for i, spike_time in enumerate(spike_times):
        start_sample = spike_time.astype(np.int64) + window[0]
        end_sample = start_sample + snip_len

        if start_sample >= 0 and end_sample < total_samples:
            snippet = raw_data[start_sample:end_sample, :]
            snips[i, :, :] = snippet.T
            
    return snips.transpose(1, 2, 0)

def compare_eis(eis, ei_template=None, max_lag=3):
    """
    Compare a list of EIs to each other or to a template.
    """
    k = len(eis)
    if ei_template is not None:
        sim = np.zeros((k, 1))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]
            trace_t = ei_template[dom_chan, :]

            lags = np.arange(-max_lag, max_lag + 1)
            xc = correlate(trace_i, trace_t, mode='full', method='auto')
            center = len(xc) // 2
            xc_window = xc[center - max_lag:center + max_lag + 1]
            shift = lags[np.argmax(xc_window)]

            aligned_t = np.roll(ei_template, shift, axis=1)
            sim[i] = np.dot(ei_i.flatten(), aligned_t.flatten()) / (
                np.linalg.norm(ei_i) * np.linalg.norm(aligned_t))
        return sim

    else:
        sim = np.zeros((k, k))
        for i in range(k):
            ei_i = eis[i]
            dom_chan = np.argmax(np.max(np.abs(ei_i), axis=1))
            trace_i = ei_i[dom_chan, :]

            for j in range(i, k):
                ei_j = eis[j]
                trace_j = ei_j[dom_chan, :]

                lags = np.arange(-max_lag, max_lag + 1)
                xc = correlate(trace_i, trace_j, mode='full', method='auto')
                center = len(xc) // 2
                xc_window = xc[center - max_lag:center + max_lag + 1]
                shift = lags[np.argmax(xc_window)]

                aligned_j = np.roll(ei_j, shift, axis=1)
                val = np.dot(ei_i.flatten(), aligned_j.flatten()) / (
                    np.linalg.norm(ei_i) * np.linalg.norm(aligned_j))
                sim[i, j] = val
                sim[j, i] = val
        return sim

def baseline_correct(snips, pre_samples=20):
    """Corrects for baseline shifts in an EI."""
    if snips.ndim == 3:
        baseline = snips[:, :pre_samples, :].mean(axis=1)
        return snips - baseline[:, np.newaxis, :]
    else:
        return snips - snips[:, :pre_samples].mean(axis=1, keepdims=True)

def compute_ei(snips, pre_samples=20):
    """Computes the Electrical Image (average waveform) from snippets."""
    snips = baseline_correct(snips, pre_samples=pre_samples)
    snips_torch = torch.from_numpy(snips)
    ei = torch.mean(snips_torch, dim=2).numpy()
    return ei

def select_channels(ei, min_chan=30, max_chan=80, threshold=15):
    """Selects the most significant channels from an EI based on peak-to-peak amplitude."""
    p2p = ei.max(axis=1) - ei.min(axis=1)
    selected = np.where(p2p > threshold)[0]
    if len(selected) > max_chan:
        selected = np.argsort(p2p)[-max_chan:]
    elif len(selected) < min_chan and len(p2p) > min_chan:
        selected = np.argsort(p2p)[-min_chan:]
    return np.sort(selected)

def find_merge_groups(sim, threshold):
    """Finds groups of clusters to merge based on a similarity matrix."""
    G = nx.Graph()
    k = sim.shape[0]
    G.add_nodes_from(range(k))
    for i in range(k):
        for j in range(i + 1, k):
            if sim[i, j] > threshold:
                G.add_edge(i, j)
    return list(nx.connected_components(G))

def calculate_isi_violations(spike_times_samples, sampling_rate, refractory_period_ms=2.0):
    """
    Calculates the rate of refractory period violations for a spike train.
    """
    if len(spike_times_samples) < 2:
        return 0.0
    refractory_period_samples = (refractory_period_ms / 1000.0) * sampling_rate
    sorted_spikes = np.sort(spike_times_samples)
    isis_samples = np.diff(sorted_spikes)
    violation_count = np.sum(isis_samples < refractory_period_samples)
    violation_rate = violation_count / len(isis_samples)
    return violation_rate

def refine_cluster_v2(spike_times, dat_path, channel_positions, params):
    """
    Recursively refines neural spike clusters using PCA+KMeans clustering.
    """
    print(f"\n=== Starting Cluster Refinement ===")
    print(f"Input spikes: {len(spike_times)}")
    
    window = params.get('window', (-20, 60))
    min_spikes = params.get('min_spikes', 500)
    k_start = params.get('k_start', 3)
    k_refine = params.get('k_refine', 2)
    ei_sim_threshold = params.get('ei_sim_threshold', 0.9)
    max_depth = params.get('max_depth', 10)
    
    if isinstance(dat_path, np.ndarray):
        snips = dat_path
    elif isinstance(dat_path, str):
        snips = extract_snippets(dat_path, spike_times, window)
    else:
        return []

    full_inds = np.arange(snips.shape[2])
    cluster_pool = [{'inds': full_inds, 'depth': 0}]
    final_clusters = []

    while cluster_pool:
        cl = cluster_pool.pop(0)
        inds = cl['inds']
        depth = cl['depth']
        
        if depth >= max_depth:
            final_clusters.append({'inds': inds})
            continue

        if len(inds) < min_spikes:
            continue

        k = k_start if depth == 0 else k_refine
        
        snips_cl = snips[:, :, inds]
        ei = compute_ei(snips_cl)
        selected = select_channels(ei)
        snips_sel = snips[np.ix_(selected, np.arange(snips.shape[1]), inds)]
        snips_centered = snips_sel - snips_sel.mean(axis=1, keepdims=True)
        flat = snips_centered.transpose(2, 0, 1).reshape(len(inds), -1)
        pcs = PCA(n_components=5).fit_transform(flat)
        labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(pcs)

        subclusters = [{'inds': inds[np.where(labels == i)[0]]} for i in range(k)]
        
        large_subclusters = [sc for sc in subclusters if len(sc['inds']) >= min_spikes]
        
        if len(large_subclusters) <= 1:
            final_clusters.append({'inds': inds})
            continue

        eis_large = [compute_ei(snips[:, :, sc['inds']]) for sc in large_subclusters]
        sim = compare_eis(eis_large)
        groups = find_merge_groups(sim, ei_sim_threshold)

        for group in groups:
            all_inds = np.concatenate([large_subclusters[j]['inds'] for j in group])
            if len(all_inds) >= min_spikes:
                cluster_pool.append({'inds': all_inds, 'depth': depth + 1})

    print(f"\n=== Refinement Complete: {len(final_clusters)} final clusters ===")
    return final_clusters

def _spatial_smooth(values, positions, sigma=30):
    """Spatially smooth values based on channel positions."""
    smoothed = np.zeros_like(values)
    for i in range(len(values)):
        distances = np.sqrt(np.sum((positions - positions[i])**2, axis=1))
        weights = np.exp(-distances**2 / (2 * sigma**2))
        smoothed[i] = np.sum(values * weights) / np.sum(weights)
    return smoothed

def compute_spatial_features(ei, channel_positions, sampling_rate=20000, pre_samples=40):
    """
    Computes rich spatial features for the EI.
    """
    peak_negative = ei.min(axis=1)
    peak_times = ei.argmin(axis=1)
    peak_times_ms = (peak_times - pre_samples) / sampling_rate * 1000
    peak_negative_smooth = _spatial_smooth(peak_negative, channel_positions)
    amplitude_threshold = np.percentile(np.abs(peak_negative), 80)
    active_channels = np.abs(peak_negative) > amplitude_threshold
    ptp_amps = np.ptp(ei, axis=1)

    grid_x, grid_y = np.mgrid[
        channel_positions[:, 0].min():channel_positions[:, 0].max():200j,
        channel_positions[:, 1].min():channel_positions[:, 1].max():200j
    ]
    grid_z = griddata(channel_positions, peak_negative_smooth, (grid_x, grid_y), method='cubic')

    return {
        'peak_negative_smooth': peak_negative_smooth,
        'peak_times_ms': peak_times_ms,
        'active_channels': active_channels,
        'ptp_amps': ptp_amps,
        'grid_x': grid_x,
        'grid_y': grid_y,
        'grid_z': grid_z,
    }

# =============================================================================
# Plotting Functions
# =============================================================================

def plot_rich_ei(fig, ei, channel_positions, spatial_features, sampling_rate=20000, pre_samples=20):
    """
    Creates a rich, multi-panel EI visualization.
    """
    fig.clear()
    peak_negative_smooth = spatial_features['peak_negative_smooth']
    ptp_amps = spatial_features['ptp_amps']
    grid_x, grid_y, grid_z = spatial_features['grid_x'], spatial_features['grid_y'], spatial_features['grid_z']
    peak_times_ms = spatial_features['peak_times_ms']
    active_channels = spatial_features['active_channels']

    gs = fig.add_gridspec(2, 2, height_ratios=[3, 2], width_ratios=[1, 1], hspace=0.4, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, :])

    ax1.set_title('Spatial Amplitude', color='white')
    v_min, v_max = np.percentile(peak_negative_smooth, [5, 95])
    contour_fill = ax1.contourf(grid_x, grid_y, grid_z, levels=20, cmap='RdBu_r', alpha=0.7, vmin=v_min, vmax=v_max)
    ax1.contour(grid_x, grid_y, grid_z, levels=20, colors='white', linewidths=0.5, alpha=0.4)
    max_ptp = ptp_amps.max()
    scaled_sizes = 10 + (ptp_amps / max_ptp) * 250 if max_ptp > 0 else np.full_like(ptp_amps, 10)
    ax1.scatter(channel_positions[:, 0], channel_positions[:, 1], s=scaled_sizes, c=peak_negative_smooth,
                cmap='RdBu_r', edgecolor='black', linewidth=0.7, zorder=2, vmin=v_min, vmax=v_max)
    fig.colorbar(contour_fill, ax=ax1, label='Smoothed Peak Amp (µV)', shrink=0.8)

    ax2.set_title('Spike Propagation', color='white')
    scatter2 = ax2.scatter(channel_positions[active_channels, 0], channel_positions[active_channels, 1],
                           c=peak_times_ms[active_channels], cmap='viridis', s=80, edgecolor='white', linewidth=0.5)
    fig.colorbar(scatter2, ax=ax2, label='Time to Peak (ms)', shrink=0.8)

    ax3.set_title('Waveform Heatmap', color='white')
    time_axis_ms = (np.arange(ei.shape[1]) - pre_samples) / sampling_rate * 1000
    if active_channels.sum() > 0:
        active_idx = np.where(active_channels)[0]
        sorted_channel_idx = active_idx[np.argsort(ei.argmin(axis=1)[active_idx])]
        waveform_matrix = ei[sorted_channel_idx]
        im = ax3.imshow(waveform_matrix, aspect='auto', cmap='RdBu_r',
                        vmin=-np.percentile(np.abs(waveform_matrix), 98),
                        vmax=np.percentile(np.abs(waveform_matrix), 98),
                        extent=[time_axis_ms[0], time_axis_ms[-1], len(sorted_channel_idx), 0])
        ax3.axvline(0, color='black', linestyle='--', alpha=0.8)
        fig.colorbar(im, ax=ax3, label='Amplitude (µV)', shrink=0.8, orientation='horizontal', pad=0.25)

    for ax in [ax1, ax2, ax3]:
        ax.set_facecolor('#1f1f1f')
        ax.tick_params(colors='gray')
        for spine in ax.spines.values(): spine.set_edgecolor('gray')
    ax1.axis('equal')
    ax2.axis('equal')

# --- New Plotting Function for Vision RF ---
def plot_vision_rf(fig, sta_data, stafit):
    """
    Visualizes the receptive field from loaded Vision STA and parameter data.
    
    Args:
        fig (matplotlib.figure.Figure): The figure object to draw on.
        sta_data (STAContainer): Named tuple containing the raw STA movie.
        stafit (STAFit): Named tuple containing the Gaussian fit parameters.
    """
    fig.clear()
    ax = fig.add_subplot(111)
    
    # Combine RGB channels to get a single grayscale STA for simplicity
    # A more advanced version could plot each channel or the most significant one.
    sta_rgb = np.stack([sta_data.red, sta_data.green, sta_data.blue], axis=-1)
    sta_gray = np.mean(sta_rgb, axis=-1)
    
    # Find the peak frame (time index with the largest deviation from zero)
    peak_frame_idx = np.argmax(np.max(np.abs(sta_gray), axis=(0, 1)))
    peak_frame = sta_gray[:, :, peak_frame_idx]
    
    # Display the peak STA frame
    ax.imshow(peak_frame.T, cmap='gray', origin='lower',
              extent=[0, peak_frame.shape[0], 0, peak_frame.shape[1]])
    
    # Overlay the Gaussian fit ellipse
    ellipse = Ellipse(
        xy=(stafit.center_x, stafit.center_y),
        width=2 * stafit.std_x,
        height=2 * stafit.std_y,
        angle=np.rad2deg(stafit.rot),
        edgecolor='cyan',
        facecolor='none',
        lw=2
    )
    ax.add_patch(ellipse)
    
    # Style the plot
    ax.set_title("Receptive Field (STA + Fit)", color='white')
    ax.set_xlabel("X (stixels)", color='gray')
    ax.set_ylabel("Y (stixels)", color='gray')
    ax.set_facecolor('#1f1f1f')
    ax.tick_params(colors='gray')
    for spine in ax.spines.values():
        spine.set_edgecolor('gray')
    
    fig.tight_layout()

