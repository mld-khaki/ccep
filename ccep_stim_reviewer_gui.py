import json
import tkinter as tk
from tkinter import filedialog, messagebox
import numpy as np
import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
import matplotlib.pyplot as plt
import pyedflib
from scipy.signal import find_peaks
import pandas as pd
from ccep_lib import decode_events_rev1, decode_events_rev2

import os





def load_events_from_files(edf_path, subject_id, session_id, run, read_from_tsv=True):
    """
    Load events from TSV file or EDF annotations.
    Returns dict: {channel_pair: [(timestamp, current), ...]}
    """
    folder = os.path.dirname(edf_path)
    
    if read_from_tsv:
        # Try to read from TSV file
        tsv_path = os.path.join(
            folder,
            f"sub-{subject_id}_{session_id}_task-ccep_run-{run:02d}_events.tsv"
        )
        
        if os.path.exists(tsv_path):
            print(f"Reading events from TSV: {tsv_path}")
            df = pd.read_csv(tsv_path, sep="\t")
            annot_timestamps = df["onset"].values
            annot_labels = df["event"].values
        else:
            print(f"TSV file not found: {tsv_path}")
            print("Falling back to EDF annotations...")
            read_from_tsv = False
    
    if not read_from_tsv:
        # Read from EDF annotations
        print("Reading events from EDF annotations...")
        edf = pyedflib.EdfReader(edf_path)
        ann = edf.readAnnotations()
        annot_timestamps = ann[0]  # seconds from recording start
        annot_labels = ann[2]
        edf.close()
    
    # Try both decoding methods
    events = decode_events_rev1(annot_timestamps, annot_labels)
    if not events:
        events = decode_events_rev2(annot_timestamps, annot_labels)
    
    if not events:
        print("WARNING: No events found in annotations/TSV!")
    
    return events


# =============================================================
# Load stim channels from CCEP summary JSON
# =============================================================
def load_stim_channels_from_summary(summary_json_path):
    with open(summary_json_path, "r") as f:
        data = json.load(f)

    stim_dict = data.get("stim_electrodes", {})
    stim_channels = sorted(list(stim_dict.keys()))
    return stim_channels, data["stim_electrodes"]


# =============================================================
# Detect stim pulses using pipeline method
# =============================================================
def detect_stim_pulses(signal, fs):
    """
    Detect stim pulses using the same method as in pipeline18:
    - Calculate mean + std as height threshold
    - Use distance of ~0.9 seconds between pulses
    """
    # Calculate threshold based on signal statistics
    sig_mean = np.nanmean(signal)
    sig_std = np.nanstd(signal)
    height_threshold = sig_mean + sig_std
    
    # Distance between pulses (assuming at least 0.9 s apart)
    min_distance = int(0.9 * fs)
    
    # Find peaks using scipy's find_peaks
    pks, properties = find_peaks(
        signal,
        height=height_threshold,
        distance=min_distance
    )
    
    return pks, properties


# =============================================================
# Main GUI class
# =============================================================
class StimReviewGUI:
    def __init__(self, root, edf_path, summary_json_path, fs, subject_id, session_id, run, read_from_tsv=True):
        self.root = root
        self.root.title("CCEP Stim Pulse Reviewer (Enhanced)")

        self.edf_path = edf_path
        self.summary_json_path = summary_json_path
        self.fs = fs
        self.subject_id = subject_id
        self.session_id = session_id
        self.run = run

        # Load events from TSV or EDF annotations
        print("\n" + "="*60)
        print("Loading stimulation events...")
        print("="*60)
        self.events_by_channel = load_events_from_files(
            edf_path, subject_id, session_id, run, read_from_tsv
        )
        
        # Load stim channels from JSON (if it exists)
        if os.path.exists(summary_json_path):
            self.stim_channels_json, self.stim_meta = load_stim_channels_from_summary(summary_json_path)
            print(f"Loaded {len(self.stim_channels_json)} channels from summary JSON")
        else:
            self.stim_channels_json = []
            self.stim_meta = {}
            print("No summary JSON found")
        
        # Get actual stimulated channels from events
        self.stim_channels = sorted(list(self.events_by_channel.keys()))
        
        if len(self.stim_channels) == 0:
            messagebox.showerror("Error", "No stim channels found in events file/annotations.")
            return
        
        print(f"\nFound {len(self.stim_channels)} stimulated channel pairs:")
        for ch_pair in self.stim_channels:
            n_events = len(self.events_by_channel[ch_pair])
            currents = set([curr for _, curr in self.events_by_channel[ch_pair]])
            print(f"  {ch_pair}: {n_events} events, currents={sorted(currents)}")

        # Data storage
        self.accepted_pulses = {}  # channel -> [sample indices]
        self.current_channel_index = 0

        # Signal display options
        self.window_sec = 2.0  # window around stim train
        self.flip_signal = False  # polarity flipping

        # Selection mode
        self.add_mode = False

        # Current channel data (updated each time we plot)
        self.current_sig = None
        self.current_t = None
        self.current_global_peaks = []
        self.current_base_idx = 0
        self.selected_peaks = []
        self.manually_added_peaks = []  # Indices of peaks that were manually added
        
        # For persistent manual peaks across redraws
        self.manual_peaks_by_channel = {}  # channel -> [global_sample_indices]

        # Open EDF once
        self.edf = pyedflib.EdfReader(edf_path)

        # === BUILD GUI ===
        main_frame = tk.Frame(root)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top: controls
        ctrl_frame = tk.Frame(main_frame)
        ctrl_frame.pack(side=tk.TOP, fill=tk.X, padx=5, pady=5)

        tk.Label(ctrl_frame, text="Window (s):").grid(row=0, column=0, sticky="w", padx=5)
        self.window_var = tk.StringVar(value=str(self.window_sec))
        tk.Entry(ctrl_frame, textvariable=self.window_var, width=8).grid(row=0, column=1, padx=5)
        tk.Button(ctrl_frame, text="Update Window", command=self.update_window).grid(row=0, column=2, padx=5)

        tk.Button(ctrl_frame, text="Flip Signal", command=self.toggle_flip).grid(row=0, column=3, padx=5)

        # Add mode checkbox
        self.add_mode_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ctrl_frame, text="Add Mode (click to add peaks)", 
                      variable=self.add_mode_var, command=self.toggle_add_mode).grid(row=0, column=4, padx=5)

        tk.Button(ctrl_frame, text="Select All", command=self.select_all).grid(row=0, column=5, padx=5)
        tk.Button(ctrl_frame, text="Deselect All", command=self.deselect_all).grid(row=0, column=6, padx=5)

        # Channel label
        self.ch_label = tk.Label(main_frame, text="", font=("Arial", 12, "bold"), anchor="w", justify="left")
        self.ch_label.pack(side=tk.TOP, fill=tk.X, padx=10, pady=5)

        # Plot area
        plot_frame = tk.Frame(main_frame)
        plot_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=5, pady=5)

        self.fig, self.ax = plt.subplots(figsize=(12, 6))
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # Add toolbar for zoom/pan
        toolbar = NavigationToolbar2Tk(self.canvas, plot_frame)
        toolbar.update()
        toolbar.pack(side=tk.TOP, fill=tk.X)

        # Connect click event
        self.canvas.mpl_connect("button_press_event", self.on_click)

        # Bottom: buttons
        btn_frame = tk.Frame(main_frame)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=5, pady=5)

        tk.Button(btn_frame, text="Previous", command=self.previous_channel, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Accept Pulses", command=self.accept_pulses, width=12, bg='lightgreen').pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Skip Channel", command=self.skip_channel, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Next", command=self.next_channel, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Load Session", command=self.load_session, width=12).pack(side=tk.LEFT, padx=5)
        tk.Button(btn_frame, text="Save JSON", command=self.save_json, width=12, bg='lightblue').pack(side=tk.RIGHT, padx=5)

        # Initialize first channel
        self.plot_current_channel()

    # =============================================================
    # Plotting
    # =============================================================
    def plot_current_channel(self):
        """Load and plot the current channel's signal and detected peaks"""
        if self.current_channel_index >= len(self.stim_channels):
            return

        ch_pair = self.stim_channels[self.current_channel_index]
        ch_a, ch_b = ch_pair.split("-")

        # Get list of indices for both channels
        signal_labels = self.edf.getSignalLabels()
        try:
            ch_a_idx = signal_labels.index(ch_a)
        except ValueError:
            messagebox.showerror("Error", f"Channel {ch_a} not found in EDF")
            return

        # Read data
        sig_a = self.edf.readSignal(ch_a_idx)
        fs = self.edf.getSampleFrequency(ch_a_idx)

        # Get events for this channel pair
        events = self.events_by_channel[ch_pair]
        if len(events) == 0:
            messagebox.showwarning("Warning", f"No events for {ch_pair}")
            return

        # Get time window around stim train
        first_event_time = events[0][0]
        last_event_time = events[-1][0]
        
        start_time = first_event_time - self.window_sec
        end_time = last_event_time + self.window_sec
        
        start_idx = max(0, int(start_time * fs))
        end_idx = min(len(sig_a), int(end_time * fs))
        
        self.current_base_idx = start_idx
        
        sig = sig_a[start_idx:end_idx]
        t = np.arange(len(sig)) / fs + start_time

        # Flip if requested
        if self.flip_signal:
            sig = -sig

        # Store current signal for interactive addition
        self.current_sig = sig
        self.current_t = t

        # Detect peaks using pipeline method
        pks, properties = detect_stim_pulses(sig, fs)
        
        # Convert local peaks to global sample indices
        auto_detected_peaks = [int(start_idx + p) for p in pks]
        
        # Get manually added peaks for this channel (persistent across redraws)
        manual_peaks_global = self.manual_peaks_by_channel.get(ch_pair, [])
        
        # Combine auto-detected and manual peaks
        all_peaks_global = auto_detected_peaks + manual_peaks_global
        all_peaks_global = sorted(list(set(all_peaks_global)))  # Remove duplicates and sort
        
        # Track which peaks are manual (indices in all_peaks_global)
        self.manually_added_peaks = []
        for i, peak in enumerate(all_peaks_global):
            if peak in manual_peaks_global:
                self.manually_added_peaks.append(i)
        
        # Convert global peaks back to local indices for plotting
        pks = [p - start_idx for p in all_peaks_global]
        
        # Store global peaks for acceptance
        self.current_global_peaks = all_peaks_global

        # Check if we have accepted pulses for this channel
        self.selected_peaks = []
        if ch_pair in self.accepted_pulses:
            # Load previously accepted selections
            accepted_samples = self.accepted_pulses[ch_pair]
            
            # Match accepted samples to detected peaks
            for idx, peak_sample in enumerate(self.current_global_peaks):
                if peak_sample in accepted_samples:
                    self.selected_peaks.append(idx)
            
            print(f"Loaded {len(self.selected_peaks)} previously selected peaks for {ch_pair}")
        else:
            # Initialize: all detected peaks are selected by default
            self.selected_peaks = list(range(len(pks)))

        # Get event info
        events = self.events_by_channel[ch_pair]
        currents = sorted(set([curr for _, curr in events]))
        n_events = len(events)
        
        # Calculate threshold for display
        sig_mean = np.nanmean(sig)
        sig_std = np.nanstd(sig)
        height_threshold = sig_mean + sig_std

        # Clear and plot
        self.ax.clear()
        self.peak_artists = []
        self.peak_positions = []

        # Plot signal
        self.ax.plot(t, sig, 'k-', alpha=0.4, linewidth=0.8, label='Signal')

        # Plot detection threshold
        self.ax.axhline(height_threshold, color='blue', linestyle='--', alpha=0.6, label='Detection Threshold')

        # Plot peaks
        if len(pks) > 0:
            for idx, p in enumerate(pks):
                x = t[p]
                y = sig[p]
                self.peak_positions.append((x, y, idx))
                
                # Determine color: green for manually added, red/gray for auto-detected
                if idx in self.manually_added_peaks:
                    # Manually added peak
                    if idx in self.selected_peaks:
                        artist = self.ax.plot(x, y, "o", color='limegreen', markersize=8, 
                                            markeredgewidth=1.5, markeredgecolor='darkgreen')[0]
                    else:
                        artist = self.ax.plot(x, y, "o", color='lightgreen', markersize=8, 
                                            markeredgewidth=1.5, markeredgecolor='green', alpha=0.5)[0]
                else:
                    # Auto-detected peak
                    if idx in self.selected_peaks:
                        artist = self.ax.plot(x, y, "ro", markersize=8, 
                                            markeredgewidth=1.5, markeredgecolor='darkred')[0]
                    else:
                        artist = self.ax.plot(x, y, "o", color='gray', markersize=8, 
                                            markeredgewidth=1.5, markeredgecolor='darkgray', alpha=0.5)[0]
                self.peak_artists.append(artist)

        flip_status = " [FLIPPED]" if self.flip_signal else ""
        add_status = " | ADD MODE ON" if self.add_mode else ""
        self.ax.set_title(f"{ch_pair}: ±{self.window_sec}s around stim train{flip_status}{add_status}")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Amplitude (µV)")
        self.ax.grid(True, linestyle=":", alpha=0.5)
        self.ax.legend(loc='upper right')

        self.update_channel_label(ch_pair, n_events, currents, len(self.selected_peaks), 
                                 len(pks), len(self.manually_added_peaks), height_threshold)

        self.canvas.draw()

    def update_channel_label(self, ch_pair, n_events, currents, n_selected, n_total, n_manual, threshold):
        """Update the channel label with current selection status"""
        manual_str = f" ({n_manual} manual)" if n_manual > 0 else ""
        self.ch_label.config(
            text=f"Channel: {ch_pair} | Events: {n_events} | Currents: {currents} | Progress {self.current_channel_index + 1}/{len(self.stim_channels)}\n"
            f"Selected: {n_selected}/{n_total}{manual_str} | Threshold: {threshold:.2f} µV"
        )

    # =============================================================
    # Interactive selection and addition
    # =============================================================
    def on_click(self, event):
        """Handle mouse clicks on the plot"""
        if event.inaxes != self.ax:
            return
        
        click_x = event.xdata
        click_y = event.ydata
        
        if click_x is None or click_y is None:
            return
        
        if self.add_mode:
            # Add mode: find peak near click and add it
            self.add_peak_at_position(click_x)
        else:
            # Selection mode: toggle existing peaks
            if len(self.peak_positions) == 0:
                return
            
            # Find the closest peak to the click
            distances = []
            for x, y, idx in self.peak_positions:
                # Normalize by axes ranges for better distance calculation
                x_range = self.ax.get_xlim()[1] - self.ax.get_xlim()[0]
                y_range = self.ax.get_ylim()[1] - self.ax.get_ylim()[0]
                dx = (x - click_x) / x_range
                dy = (y - click_y) / y_range
                dist = np.sqrt(dx**2 + dy**2)
                distances.append((dist, idx, x, y))
            
            # Find closest peak
            distances.sort()
            closest_dist, closest_idx, peak_x, peak_y = distances[0]
            
            # Only toggle if click is reasonably close (within 5% of axis range)
            if closest_dist < 0.05:
                self.toggle_peak(closest_idx, peak_x, peak_y)

    def add_peak_at_position(self, click_x):
        """Add a new peak at the clicked position by finding nearby peak in signal"""
        if self.current_sig is None or self.current_t is None:
            return
        
        ch_pair = self.stim_channels[self.current_channel_index]
        
        # Find the time index closest to click
        time_idx = np.argmin(np.abs(self.current_t - click_x))
        
        # Search for peak in a small window around this position
        search_window = int(0.05 * self.fs)  # 50ms window
        start_idx = max(0, time_idx - search_window)
        end_idx = min(len(self.current_sig), time_idx + search_window)
        
        # Find local maximum in this window
        window_sig = self.current_sig[start_idx:end_idx]
        if len(window_sig) == 0:
            return
        
        local_peak_idx = np.argmax(np.abs(window_sig))
        peak_idx = start_idx + local_peak_idx
        
        # Convert to global sample index
        global_peak = int(self.current_base_idx + peak_idx)
        
        # Check if this peak already exists (within 10ms)
        tolerance_samples = int(0.01 * self.fs)
        
        for existing_peak in self.current_global_peaks:
            if abs(global_peak - existing_peak) < tolerance_samples:
                messagebox.showinfo("Info", "Peak already exists at this location")
                return
        
        # Add this peak to persistent manual peaks for this channel
        if ch_pair not in self.manual_peaks_by_channel:
            self.manual_peaks_by_channel[ch_pair] = []
        self.manual_peaks_by_channel[ch_pair].append(global_peak)
        
        print(f"Added manual peak at global sample {global_peak} for {ch_pair}")
        
        # Redraw to show the new peak
        self.plot_current_channel()

    def toggle_peak(self, peak_idx, x, y):
        """Toggle selection state of a peak"""
        if peak_idx in self.selected_peaks:
            # Deselect
            self.selected_peaks.remove(peak_idx)
            # Update artist
            if peak_idx in self.manually_added_peaks:
                self.peak_artists[peak_idx].set_color('lightgreen')
                self.peak_artists[peak_idx].set_markeredgecolor('green')
            else:
                self.peak_artists[peak_idx].set_color('gray')
                self.peak_artists[peak_idx].set_markeredgecolor('darkgray')
            self.peak_artists[peak_idx].set_alpha(0.5)
        else:
            # Select
            self.selected_peaks.append(peak_idx)
            # Update artist
            if peak_idx in self.manually_added_peaks:
                self.peak_artists[peak_idx].set_color('limegreen')
                self.peak_artists[peak_idx].set_markeredgecolor('darkgreen')
            else:
                self.peak_artists[peak_idx].set_color('red')
                self.peak_artists[peak_idx].set_markeredgecolor('darkred')
            self.peak_artists[peak_idx].set_alpha(1.0)
        
        # Update label
        ch_pair = self.stim_channels[self.current_channel_index]
        events = self.events_by_channel[ch_pair]
        currents = sorted(set([curr for _, curr in events]))
        n_events = len(events)
        sig_mean = np.nanmean(self.current_sig)
        sig_std = np.nanstd(self.current_sig)
        height_threshold = sig_mean + sig_std
        
        self.update_channel_label(ch_pair, n_events, currents, len(self.selected_peaks), 
                                 len(self.peak_positions), len(self.manually_added_peaks), height_threshold)
        
        self.canvas.draw()

    def select_all(self):
        """Select all detected peaks"""
        self.selected_peaks = list(range(len(self.peak_positions)))
        self.replot_with_current_selection()

    def deselect_all(self):
        """Deselect all peaks"""
        self.selected_peaks = []
        self.replot_with_current_selection()

    def replot_with_current_selection(self):
        """Redraw the plot with current selection state"""
        if len(self.peak_positions) == 0:
            return
        
        # Update all artists
        for idx, artist in enumerate(self.peak_artists):
            if idx in self.selected_peaks:
                if idx in self.manually_added_peaks:
                    artist.set_color('limegreen')
                    artist.set_markeredgecolor('darkgreen')
                else:
                    artist.set_color('red')
                    artist.set_markeredgecolor('darkred')
                artist.set_alpha(1.0)
            else:
                if idx in self.manually_added_peaks:
                    artist.set_color('lightgreen')
                    artist.set_markeredgecolor('green')
                else:
                    artist.set_color('gray')
                    artist.set_markeredgecolor('darkgray')
                artist.set_alpha(0.5)
        
        # Update label
        ch_pair = self.stim_channels[self.current_channel_index]
        events = self.events_by_channel[ch_pair]
        currents = sorted(set([curr for _, curr in events]))
        n_events = len(events)
        sig_mean = np.nanmean(self.current_sig)
        sig_std = np.nanstd(self.current_sig)
        height_threshold = sig_mean + sig_std
        
        self.update_channel_label(ch_pair, n_events, currents, len(self.selected_peaks), 
                                 len(self.peak_positions), len(self.manually_added_peaks), height_threshold)
        
        self.canvas.draw()

    # =============================================================
    # Control functions
    # =============================================================
    def update_window(self):
        """Update the window length and replot"""
        try:
            new_window = float(self.window_var.get())
            if new_window <= 0:
                messagebox.showerror("Error", "Window length must be positive")
                return
            self.window_sec = new_window
            self.plot_current_channel()
        except ValueError:
            messagebox.showerror("Error", "Invalid window length")

    def toggle_flip(self):
        """Toggle signal polarity and replot"""
        self.flip_signal = not self.flip_signal
        self.plot_current_channel()

    def toggle_add_mode(self):
        """Toggle add mode"""
        self.add_mode = self.add_mode_var.get()
        # Update plot title to show mode
        self.plot_current_channel()

    # =============================================================
    # Session management
    # =============================================================
    def load_session(self):
        """Load a previously saved session JSON"""
        path = filedialog.askopenfilename(
            filetypes=[("JSON files", "*.json")],
            title="Load Previous Session"
        )
        if not path:
            return
        
        try:
            with open(path, "r") as f:
                loaded_data = json.load(f)
            
            # Clear current accepted pulses
            self.accepted_pulses = {}
            
            # Convert loaded data to accepted_pulses format
            for ch_pair, data in loaded_data.items():
                if isinstance(data, dict) and "sample_indices" in data:
                    self.accepted_pulses[ch_pair] = data["sample_indices"]
                elif isinstance(data, list):
                    self.accepted_pulses[ch_pair] = data
            
            messagebox.showinfo("Loaded", 
                              f"Loaded session with {len(self.accepted_pulses)} channels\n" +
                              "Navigate through channels to review loaded selections")
            
            # Refresh current channel to apply loaded selections
            self.plot_current_channel()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load session: {str(e)}")

    # =============================================================
    # Buttons
    # =============================================================
    def accept_pulses(self):
        ch_pair = self.stim_channels[self.current_channel_index]
        # Only save selected peaks
        selected_global_peaks = [self.current_global_peaks[i] for i in self.selected_peaks]
        self.accepted_pulses[ch_pair] = selected_global_peaks
        
        n_manual = len([i for i in self.selected_peaks if i in self.manually_added_peaks])
        manual_str = f" ({n_manual} manual)" if n_manual > 0 else ""
        
        messagebox.showinfo("Accepted", 
                           f"Accepted {len(selected_global_peaks)} selected pulses{manual_str} for {ch_pair}")
        self.next_channel()

    def skip_channel(self):
        ch_pair = self.stim_channels[self.current_channel_index]
        messagebox.showinfo("Skipped", f"Skipped {ch_pair}")
        self.next_channel()

    def next_channel(self):
        if self.current_channel_index < len(self.stim_channels) - 1:
            self.current_channel_index += 1
            self.plot_current_channel()
        else:
            self.current_channel_index += 1  # Move past last channel
            self.ch_label.config(text="All channels reviewed. Click 'Save JSON' to save results.")
            self.ax.clear()
            self.ax.text(0.5, 0.5, "Review Complete!", 
                        ha='center', va='center', fontsize=20, transform=self.ax.transAxes)
            self.canvas.draw()

    def previous_channel(self):
        """Go back to the previous channel"""
        if self.current_channel_index > 0:
            self.current_channel_index -= 1
            self.plot_current_channel()
        else:
            messagebox.showinfo("Info", "Already at first channel")

    # =============================================================
    # Save results
    # =============================================================
    def save_json(self):
        if not self.accepted_pulses:
            messagebox.showwarning("Warning", "No pulses accepted. Nothing to save.")
            return

        fs = self.fs
        out = {}

        for ch, samples in self.accepted_pulses.items():
            # Convert numpy integers to Python integers to ensure JSON serialization
            samples_list = [int(s) for s in samples]
            out[ch] = {
                "sample_indices": samples_list,
                "timestamps_sec": [float(s / fs) for s in samples_list],
                "n_pulses": len(samples_list)
            }

        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json")],
            title="Save Accepted Pulses"
        )
        if path:
            try:
                with open(path, "w") as f:
                    json.dump(out, f, indent=4)
                messagebox.showinfo("Saved", f"Saved {len(out)} channels to:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save JSON: {str(e)}")

    def __del__(self):
        # Clean up EDF reader
        if hasattr(self, 'edf'):
            try:
                self.edf.close()
            except:
                pass


# =============================================================
# Runner
# =============================================================
def run_reviewer(edf_path, summary_json_path, fs, subject_id, session_id, run, read_from_tsv=True):
    root = tk.Tk()
    try:
        gui = StimReviewGUI(root, edf_path, summary_json_path, fs, 
                           subject_id, session_id, run, read_from_tsv)
        root.mainloop()
    except Exception as e:
        messagebox.showerror("Error", f"Failed to initialize GUI: {str(e)}")
        import traceback
        traceback.print_exc()
        root.destroy()
    finally:
        # Ensure matplotlib figures are closed
        plt.close('all')


# =============================================================
# Example usage
# =============================================================
if __name__ == "__main__":
    # Subject parameters
    subject_id = "114"
    session_id = "ses-007"
    run = 2
    fs = 2048
    
    # Paths
    root_path = "o:/Other_Datasets_phis/CCEP-DB/sub-114-nc/ses-007/ieeg/"
    edf_path = root_path + f"sub-{subject_id}_{session_id}_task-ccep_run-{run:02d}_ieeg.edf"
    summary_json = root_path + f"sub-{subject_id}_{session_id}_run-{run:02d}_ccep_summary.json"
    
    # Read events from TSV file (set to False to use EDF annotations)
    read_from_tsv = True
    
    run_reviewer(edf_path, summary_json, fs, subject_id, session_id, run, read_from_tsv)
