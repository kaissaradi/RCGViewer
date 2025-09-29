import numpy as np
import pyqtgraph as pg
from qtpy.QtCore import Qt, QTimer
from scipy.spatial.distance import cdist
from scipy.ndimage import gaussian_filter1d
import analysis_core
import matplotlib.pyplot as plt

def draw_summary_plot(main_window, cluster_id):
    # Draws the main spatial analysis plot, switching between custom EI and Vision EI.
    vision_cluster_id = cluster_id + 1
    
    # Check if we have Vision EI data
    has_vision_ei = main_window.data_manager.vision_eis and vision_cluster_id in main_window.data_manager.vision_eis
    
    if has_vision_ei:
        if hasattr(main_window, 'ei_animation_timer') and main_window.ei_animation_timer and main_window.ei_animation_timer.isActive():
            main_window.ei_animation_timer.stop()
        draw_vision_ei_animation(main_window, cluster_id)
        main_window.current_spatial_features = None
    else:
        # Fallback to original Kilosort EI-based spatial plot
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
            main_window.summary_canvas.fig, lightweight_features['median_ei'], main_window.data_manager.channel_positions,
            heavyweight_features, main_window.data_manager.sampling_rate, pre_samples=20)
        main_window.summary_canvas.fig.suptitle(f"Cluster {cluster_id} Spatial Analysis", color='white', fontsize=16)
        main_window.summary_canvas.draw()

def on_summary_plot_hover(main_window, event):
    # Handles hover events on the summary plot for tooltips.
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
    # Updates the waveform plot with data for the selected cluster.
    main_window.waveform_plot.clear()
    main_window.waveform_plot.setTitle(f"Cluster {cluster_id} | Waveforms (Sampled)")
    median_ei = lightweight_features['median_ei']
    snippets = lightweight_features['raw_snippets']
    p2p = median_ei.max(axis=1) - median_ei.min(axis=1)
    dom_chan = np.argmax(p2p)
    pre_peak_samples = 20
    time_axis = (np.arange(median_ei.shape[1]) - pre_peak_samples) / main_window.data_manager.sampling_rate * 1000
    for i in range(snippets.shape[2]):
        main_window.waveform_plot.plot(time_axis, snippets[dom_chan, :, i], pen=pg.mkPen(color=(200, 200, 200, 30)))
    main_window.waveform_plot.plot(time_axis, median_ei[dom_chan], pen=pg.mkPen('#00A3E0', width=2.5))
    main_window.waveform_plot.setLabel('bottom', 'Time (ms)')
    main_window.waveform_plot.setLabel('left', 'Amplitude (uV)')
    main_window.waveform_plot.enableAutoRange(axis=pg.ViewBox.XYAxes)

def update_isi_plot(main_window, cluster_id):
    # Updates the ISI histogram plot.
    main_window.isi_plot.clear()
    violation_rate = main_window.data_manager._calculate_isi_violations(cluster_id)
    main_window.data_manager.update_cluster_isi(cluster_id, violation_rate)
    main_window.isi_plot.setTitle(f"Cluster {cluster_id} | ISI | Violations: {violation_rate:.2f}%")
    spikes = main_window.data_manager.get_cluster_spikes(cluster_id)
    if len(spikes) < 2: return
    isis_ms = np.diff(np.sort(spikes)) / main_window.data_manager.sampling_rate * 1000
    y, x = np.histogram(isis_ms, bins=np.linspace(0, 50, 101))
    main_window.isi_plot.plot(x, y, stepMode="center", fillLevel=0, brush=(0, 163, 224, 150))
    main_window.isi_plot.addLine(x=2.0, pen=pg.mkPen('r', style=Qt.PenStyle.DashLine, width=2))
    main_window.isi_plot.setLabel('bottom', 'ISI (ms)')
    main_window.isi_plot.setLabel('left', 'Count')

def update_fr_plot(main_window, cluster_id):
    # Updates the smoothed firing rate plot.
    main_window.fr_plot.clear()
    main_window.fr_plot.setTitle(f"Cluster {cluster_id} | Firing Rate")
    spikes_sec = main_window.data_manager.get_cluster_spikes(cluster_id) / main_window.data_manager.sampling_rate
    if len(spikes_sec) == 0: return
    total_duration = main_window.data_manager.spike_times.max() / main_window.data_manager.sampling_rate
    bins = np.arange(0, total_duration + 1, 1)
    counts, _ = np.histogram(spikes_sec, bins=bins)
    rate = gaussian_filter1d(counts.astype(float), sigma=5)
    main_window.fr_plot.plot(bins[:-1], rate, pen='y')
    main_window.fr_plot.setLabel('bottom', 'Time (s)')
    main_window.fr_plot.setLabel('left', 'Firing Rate (Hz)')

def draw_vision_ei_animation(main_window, cluster_id):
    """Draws an animated visualization of the Vision EI."""
    vision_cluster_id = cluster_id + 1
    if not main_window.data_manager.vision_eis or vision_cluster_id not in main_window.data_manager.vision_eis:
        main_window.summary_canvas.fig.clear()
        main_window.summary_canvas.fig.text(0.5, 0.5, "No Vision EI data available", ha='center', va='center', color='gray')
        main_window.summary_canvas.draw()
        # Disable controls if no data
        main_window.ei_frame_slider.setEnabled(False)
        return
    ei_data = main_window.data_manager.vision_eis[vision_cluster_id].ei
    main_window.current_ei_data = ei_data
    main_window.current_ei_cluster_id = cluster_id
    main_window.n_frames = ei_data.shape[1]
    
    # Find the peak frame (frame with maximum absolute amplitude)
    frame_energies = np.max(np.abs(ei_data), axis=0)  # Max amplitude for each frame
    peak_frame_index = np.argmax(frame_energies)
    main_window.current_frame = peak_frame_index

    # Update the slider properties
    main_window.ei_frame_slider.setMinimum(0)
    main_window.ei_frame_slider.setMaximum(main_window.n_frames - 1)
    main_window.ei_frame_slider.setValue(peak_frame_index)
    main_window.ei_frame_label.setText(f"Frame: {peak_frame_index + 1}/{main_window.n_frames}")
    main_window.ei_frame_slider.setEnabled(True)
    
    # Initially display the peak frame instead of auto-playing
    draw_vision_ei_frame(main_window, ei_data[:, peak_frame_index], peak_frame_index, main_window.n_frames)

def draw_vision_ei_frame(main_window, frame_data, frame_index, total_frames):
    """Draws a single frame of the Vision EI animation."""
    # Calculate size based on amplitude
    size_data = np.abs(frame_data)
    
    # Scale the sizes to make them more visible (user can adjust this)
    size_multiplier = 50  # This could be made adjustable by user later
    
    main_window.summary_canvas.fig.clear()
    ax = main_window.summary_canvas.fig.add_subplot(111)
    ax.set_facecolor('#1f1f1f')
    scatter = ax.scatter(
        main_window.data_manager.channel_positions[:, 0],
        main_window.data_manager.channel_positions[:, 1],
        c=frame_data, 
        s=size_data * size_multiplier, 
        cmap='RdBu_r',
        edgecolor='white', 
        linewidth=0.5,
        vmin=-np.max(np.abs(frame_data)),  # Normalize colors across the frame
        vmax=np.max(np.abs(frame_data))
    )
    ax.set_title(f"Vision EI - Frame {frame_index + 1}/{total_frames}")
    
    # Add a colorbar for reference
    cbar = main_window.summary_canvas.fig.colorbar(scatter, ax=ax)
    cbar.set_label('Amplitude', color='white')
    # Update colorbar text color to match theme
    cbar.ax.yaxis.set_tick_params(color='white')
    cbar.outline.set_edgecolor('#444444')
    plt = main_window.summary_canvas.fig
    for tick_label in cbar.ax.yaxis.get_ticklabels():
        tick_label.set_color('white')
    
    main_window.summary_canvas.draw()

def update_ei_frame(main_window):
    """Updates the EI visualization to the next frame in the animation."""
    if main_window.current_frame >= main_window.n_frames - 1:
        main_window.current_frame = 0
    else:
        main_window.current_frame += 1
    
    frame_data = main_window.current_ei_data[:, main_window.current_frame]
    
    # Use the same drawing function for consistency
    draw_vision_ei_frame(main_window, frame_data, main_window.current_frame, main_window.n_frames)

# In gui/plotting.py

def draw_sta_plot(main_window, cluster_id):
    """
    MODIFIED: Fetches STAFit data and passes it to the plotting function.
    """
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas
    
    if has_vision_sta:
        stop_sta_animation(main_window)

        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        # --- ADDED: Get STAFit data and store it for other functions to use ---
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_id)
        main_window.current_sta_data = sta_data
        main_window.current_stafit = stafit # <-- Store the fit
        main_window.current_sta_cluster_id = vision_cluster_id
        
        n_frames = sta_data.red.shape[2]
        main_window.total_sta_frames = n_frames
        main_window.sta_frame_slider.setMaximum(n_frames - 1)
        
        all_channels = np.stack([sta_data.red, sta_data.green, sta_data.blue], axis=0)
        frame_energies = np.max(np.abs(all_channels), axis=(0, 1, 2))
        peak_frame_index = np.argmax(frame_energies)
        main_window.current_frame_index = peak_frame_index

        main_window.sta_frame_slider.setValue(peak_frame_index)
        main_window.sta_frame_label.setText(f"Frame: {peak_frame_index + 1}/{n_frames}")
        main_window.sta_frame_slider.setEnabled(True)

        main_window.sta_canvas.fig.clear()
        analysis_core.animate_sta_movie(
            main_window.sta_canvas.fig,
            sta_data,
            stafit=stafit, # <-- Pass the fit to the plotting function
            frame_index=peak_frame_index,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height
        )
        main_window.sta_canvas.draw()
    else:
        # No Vision STA data available
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)



def draw_population_rfs_plot(main_window, selected_cell_id=None):
    """Draws the population receptive field plot showing all cell RFs."""
    print(f"--- 2. DEBUG (Plotting): Received selected_cell_id = {selected_cell_id}. Passing to analysis_core. ---")
    # MODIFIED: This function now accepts 'selected_cell_id'
    has_vision_params = main_window.data_manager.vision_params
    
    if has_vision_params:
        main_window.sta_canvas.fig.clear()
        
        analysis_core.plot_population_rfs(
            main_window.sta_canvas.fig,
            main_window.data_manager.vision_params,
            sta_width=main_window.data_manager.vision_sta_width,
            sta_height=main_window.data_manager.vision_sta_height,
            selected_cell_id=selected_cell_id # Pass the ID along to the core plotting function
        )
        main_window.sta_canvas.draw()
    else:
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision parameters available", 
                                       ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()

def draw_sta_timecourse_plot(main_window, cluster_id):
    # Draws the STA timecourse plot for a specific cell.
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas
    if has_vision_sta:
        sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        stafit = main_window.data_manager.vision_params.get_stafit_for_cell(vision_cluster_id)
        main_window.sta_canvas.fig.clear()
        analysis_core.plot_sta_timecourse(
            main_window.sta_canvas.fig,
            sta_data,
            stafit,
            main_window.data_manager.vision_params,
            vision_cluster_id
        )
        main_window.sta_canvas.draw()
    else:
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()

def draw_sta_animation_plot(main_window, cluster_id):
    # Draws the STA animation plot for a specific cell.
    vision_cluster_id = cluster_id + 1
    has_vision_sta = main_window.data_manager.vision_stas and vision_cluster_id in main_window.data_manager.vision_stas
    if has_vision_sta:
        main_window.current_sta_data = main_window.data_manager.vision_stas[vision_cluster_id]
        main_window.current_sta_cluster_id = vision_cluster_id
        main_window.current_frame_index = 0
        n_frames = main_window.current_sta_data.red.shape[2]
        main_window.total_sta_frames = n_frames
        
        main_window.sta_frame_slider.setMinimum(0)
        main_window.sta_frame_slider.setMaximum(n_frames - 1)
        main_window.sta_frame_slider.setValue(0)
        main_window.sta_frame_label.setText(f"Frame: 1/{n_frames}")
        main_window.sta_frame_slider.setEnabled(True)
        
        if main_window.sta_animation_timer is None:
            main_window.sta_animation_timer = QTimer()
            main_window.sta_animation_timer.timeout.connect(lambda: update_sta_frame(main_window))
        
        if not main_window.sta_animation_timer.isActive():
            main_window.sta_animation_timer.start(100)
    else:
        main_window.sta_canvas.fig.clear()
        main_window.sta_canvas.fig.text(0.5, 0.5, "No Vision STA data available", ha='center', va='center', color='gray')
        main_window.sta_canvas.draw()
        main_window.sta_frame_slider.setEnabled(False)

def update_sta_frame(main_window):
    # Updates the STA visualization to the next frame in the animation.
    if not hasattr(main_window, 'current_sta_data') or main_window.current_sta_data is None:
        return
    
    main_window.current_frame_index = (main_window.current_frame_index + 1) % main_window.total_sta_frames
    main_window.sta_frame_slider.setValue(main_window.current_frame_index)
    main_window.sta_frame_label.setText(f"Frame: {main_window.current_frame_index + 1}/{main_window.total_sta_frames}")

    main_window.sta_canvas.fig.clear()
    analysis_core.animate_sta_movie(
        main_window.sta_canvas.fig,
        main_window.current_sta_data,
        stafit=main_window.current_stafit, # <-- Pass the stored fit during animation
        frame_index=main_window.current_frame_index,
        sta_width=main_window.data_manager.vision_sta_width,
        sta_height=main_window.data_manager.vision_sta_height
    )
    main_window.sta_canvas.draw()

def stop_sta_animation(main_window):
    # Stops the STA animation if running.
    if hasattr(main_window, 'sta_animation_timer') and main_window.sta_animation_timer and main_window.sta_animation_timer.isActive():
        main_window.sta_animation_timer.stop()


def update_raw_trace_plot(main_window, cluster_id):
    """
    Draw the raw trace plot with the nearest channels and spike templates overlaid.
    """
    # Guard against recursive updates
    if getattr(main_window, '_raw_trace_updating', False):
        return

    main_window._raw_trace_updating = True
    try:
        if main_window.data_manager.raw_data_memmap is None:
            main_window.raw_trace_plot.clear()
            main_window.raw_trace_plot.setTitle("Raw Trace View - No raw data file loaded")
            return

        # Get the dominant channel for the cluster
        lightweight_features = main_window.data_manager.get_lightweight_features(cluster_id)
        if not lightweight_features:
            main_window.raw_trace_plot.clear()
            main_window.raw_trace_plot.setTitle(f"Raw Trace View - No data available for cluster {cluster_id}")
            return

        # Find the dominant channel (the one with the largest peak-to-peak amplitude)
        median_ei = lightweight_features['median_ei']
        if median_ei.size == 0 or len(median_ei.shape) < 2:
            main_window.raw_trace_plot.clear()
            main_window.raw_trace_plot.setTitle(f"Raw Trace View - Invalid data for cluster {cluster_id}")
            return

        p2p = median_ei.max(axis=1) - median_ei.min(axis=1)
        dom_chan = np.argmax(p2p)

        # Get the nearest 3 channels (dominant channel and its 2 closest neighbors)
        nearest_channels = main_window.data_manager.get_nearest_channels(dom_chan, n_channels=3)
        
        # Determine the time range to plot (either the current view or the buffer range)
        if (hasattr(main_window, 'raw_trace_buffer_start_time') and 
            main_window.raw_trace_buffer_start_time != main_window.raw_trace_buffer_end_time):
            # Use the buffered range if it exists
            start_time = main_window.raw_trace_buffer_start_time
            end_time = main_window.raw_trace_buffer_end_time
        else:
            # Otherwise, determine from the visible range
            try:
                view_range = main_window.raw_trace_plot.viewRange()
                x_range = view_range[0]  # [min_x, max_x] in seconds
            except:
                x_range = (0, 1)  # Default if viewRange fails
            
            # If plot is not initialized yet, try to navigate to the first spike for the cluster
            if x_range[0] == 0 and x_range[1] == 1:
                # Get spikes for the cluster and find the first one
                cluster_spikes = main_window.data_manager.get_cluster_spikes(cluster_id)
                if len(cluster_spikes) > 0:
                    # Find the time of the first spike in seconds
                    first_spike_sample = cluster_spikes[0]
                    first_spike_time = first_spike_sample / main_window.data_manager.sampling_rate
                    
                    # Show a 5-second window starting from the first spike
                    start_time = first_spike_time
                    end_time = start_time + 5.0
                else:
                    # If no spikes, default to first 10 seconds
                    start_time = 0
                    end_time = 10
            else:
                start_time = x_range[0]
                end_time = x_range[1]
        
        # Convert time range from seconds to samples
        start_sample = int(start_time * main_window.data_manager.sampling_rate)
        end_sample = int(end_time * main_window.data_manager.sampling_rate)
        
        # Ensure we stay within bounds
        start_sample = max(0, start_sample)
        end_sample = min(main_window.data_manager.n_samples, end_sample)
        
        if start_sample >= end_sample:
            main_window.raw_trace_plot.clear()
            main_window.raw_trace_plot.setTitle(f"Raw Trace View - Invalid time range for cluster {cluster_id}")
            return
        
        # Get the raw trace data for the nearest channels
        raw_trace_data = main_window.data_manager.get_raw_trace_snippet(
            nearest_channels, start_sample, end_sample
        )
        
        if raw_trace_data is None or raw_trace_data.size == 0 or len(raw_trace_data.shape) < 2:
            main_window.raw_trace_plot.clear()
            main_window.raw_trace_plot.setTitle(f"Raw Trace View - No data available in time range for cluster {cluster_id}")
            return

        # Clear the plot
        main_window.raw_trace_plot.clear()
        
        # Set up time axis (convert samples to seconds)
        if raw_trace_data.shape[1] > 0:  # Make sure we have time points
            time_axis = np.linspace(
                start_sample / main_window.data_manager.sampling_rate,
                end_sample / main_window.data_manager.sampling_rate,
                raw_trace_data.shape[1]
            )
        else:
            # If no data points, just return with a message
            main_window.raw_trace_plot.clear()
            main_window.raw_trace_plot.setTitle(f"Raw Trace View - No time points to display for cluster {cluster_id}")
            return
        
        # Define vertical offset for each channel so they don't overlap
        if raw_trace_data.size > 0:
            vertical_offset = max(np.max(np.abs(raw_trace_data)), 1) * 2.0  # Space channels apart more for 3 channels
        else:
            vertical_offset = 100  # Default offset if no data
        
        # Plot each of the 3 channels with a vertical offset, ensuring dominant channel is in the center
        for i, (chan_idx, trace) in enumerate(zip(nearest_channels, raw_trace_data)):
            # Apply vertical offset based on channel index
            offset_trace = trace + i * vertical_offset
            
            # Apply refined visual styling
            if chan_idx == dom_chan:
                # Raw trace for the dominant channel: thinner, slightly less transparent
                pen = pg.mkPen(color=(150, 150, 150, 150), width=1)  # Transparent grey-blue for dominant channel
            else:
                # Raw trace for neighbor channels: thinner, more transparent
                pen = pg.mkPen(color=(120, 120, 150, 100), width=1)  # More transparent grey-blue
            
            main_window.raw_trace_plot.plot(time_axis, offset_trace, pen=pen)

        # Add title with cluster info
        main_window.raw_trace_plot.setTitle(f"Raw Traces for Cluster {cluster_id} - 3 channels: dominant {dom_chan} with neighbors")
        main_window.raw_trace_plot.setLabel('bottom', 'Time (s)')
        main_window.raw_trace_plot.setLabel('left', 'Amplitude (µV)')
        
        # Get spike times for the selected cluster that fall within the current time window
        all_spikes = main_window.data_manager.get_cluster_spikes(cluster_id)
        if len(all_spikes) > 0:
            window_spikes = all_spikes[(all_spikes >= start_sample) & (all_spikes <= end_sample)]
            window_spikes_sec = window_spikes / main_window.data_manager.sampling_rate
            
            # Determine zoom level to decide whether to show templates or just spike lines
            visible_duration = end_time - start_time
            
            # If zoomed in (showing less than 0.05 seconds), show the actual template
            if visible_duration < 0.05 and len(window_spikes) > 0:
                # Only process templates if we have valid median_ei data
                template_len = len(median_ei[dom_chan])
                if template_len > 0:
                    for spike_time_sec in window_spikes_sec:
                        template_duration = template_len / main_window.data_manager.sampling_rate
                        
                        # Create time axis for the template centered on the spike time
                        template_time = np.linspace(
                            spike_time_sec - template_duration/2,
                            spike_time_sec + template_duration/2,
                            template_len
                        )
                        
                        # Find which position corresponds to the dominant channel in our nearest channels list
                        try:
                            dom_idx_in_list = nearest_channels.index(dom_chan)
                        except ValueError:
                            # If dom_chan is not in nearest_channels, default to position 0
                            dom_idx_in_list = 0
                            
                        offset_template = median_ei[dom_chan] + dom_idx_in_list * vertical_offset
                        
                        # Draw the template with refined styling
                        main_window.raw_trace_plot.plot(template_time, offset_template, 
                                                      pen=pg.mkPen(color='#FFA500', width=2.5))  # Orange for templates
            else:
                # If zoomed out, just show vertical lines at spike times
                for spike_time_sec in window_spikes_sec:
                    main_window.raw_trace_plot.addLine(x=spike_time_sec, 
                                                     pen=pg.mkPen('#FFFF00', width=1, style=Qt.PenStyle.DotLine))  # Thin, dotted bright yellow lines for spike markers
    finally:
        main_window._raw_trace_updating = False