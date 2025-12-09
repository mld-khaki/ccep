# -*- coding: utf-8 -*-
"""
Modified on Sun Oct 5 12:59:16 2025

Original Author:   Alaina Mahalanobis
Modified by:       Milad Khaki
Enhanced with ERDetect integration

Current update:
- Return to annotation-based absolute EDF synchronization
- Use annotation onsets -> absolute sample indices as ground truth
- Epoch and polarity detection use full-length filtered signals
- 10 s window now only for visualization, not for time indexing
- v5p6: Per-pulse orientation-based flipping (compare stim vs evoked channel derivatives)
- NEW: Support for JSON file with valid pulse indices for selective averaging
"""

import numpy as np
import pyedflib
from mne.filter import filter_data, notch_filter
import matplotlib.pyplot as plt
import pandas as pd
from types import SimpleNamespace

from ccep_lib import (load_valid_pulses,
                      decode_events_rev1,
                      decode_events_rev2,
                      epochs_no_flip,
                      find_peaks,
                      epochs_with_orientation_flip)

# Import ERDetect modules

# -----------------------
# User Parameters
# -----------------------
subject_id = "114"
session_fold = "ses-007"
session_id   = "ses-007"
run = 2

# stimulated_channel = "LPCg3"
# evoked_channel     = "LAIn4"

# stimulated_channel = "LOFr1"
# evoked_channel     = "LPIn1"

stimulated_channel_a = "LPHc1"
stimulated_channel_b = "LPHc2"

evoked_channel     = "LAIn9"

current = '1'
session_name_trailer = ""

# NOTE: main_folder points to the subject/session root that contains /ieeg/ folder
main_folder = f"o:/Other_Datasets_phis/CCEP-DB/sub-{subject_id}-nc/{session_fold}/ieeg/"
# main_folder = f"o:/Other_Datasets_phis/CCEP-DB/MOVED_TO_Sharepoint/sub-{subject_id}/{session_fold}/ieeg/"

# NEW: Path to JSON file with valid pulse indices
# Set to None to use all pulses, or provide path to JSON file
valid_pulses_json = "sub114_selected_stim_peaks_for_ccep.json"

# ----------------------------------------
# Selectors / Options
# ----------------------------------------
# Which events determine alignment in LABELS / text (data always uses annotations)?
#   "stim"   → conceptually: stim channel
#   "evoked" → conceptually: evoked channel
#   (actual timing is from annotations either way)
sync_source = "stim"

# Which signal do we epoch/average/plot in panes 3–4?
# "stim" or "evoked"
avg_source = "evoked"

# Polarity: use flipping logic before averaging?
enable_polarity_flip = True
final_polarity = -1

apply_hamming = False
hamming_power = 1
skip_first_stim = False

bpf_high_freq = 500
bpf_low_freq = 0.5

# Filters
en_band_pass_filter = True
en_notch = True
en_detrend = True
en_median = True
med_val = 7  # kernel size for medfilt; must be odd

en_movavg = True
avg_val = 1
read_events_edf0_tsv1 = True

window_pre_s = 7.0
window_len_s = 15.0

# ----------------------------------------
# ERDetect-inspired Parameters
# ----------------------------------------
# Detection windows (in seconds relative to stim onset)
peak_search_epoch = (0.0, 0.5)          # Wide window to find all peaks
response_search_epoch = (0.01, 0.050)   # Narrow window where ER is expected
N1_search_window = (0.01, 0.100)   # Narrow window where ER is expected
N2_search_window = (0.075, 0.300)   # Narrow window where ER is expected

baseline_epoch = (-1.0, -0.1)           # Pre-stim baseline

# Detection thresholds
baseline_threshold_factor = 0.9         # Factor applied to baseline std
baseline_minimum_std = 5.0              # Minimum baseline std in signal units

# Auto-adjust minimum std based on signal scale
use_adaptive_threshold = True

# ----------------------------------------
# Orientation-based flip parameters
# ----------------------------------------
orientation_window_ms = 50.0  # Window size for derivative averaging (ms)

# ----------------------------------------
# Epoching / polarity args
# ----------------------------------------
args = SimpleNamespace()
args.sampling_freq = None
args.SEG_MS = (-100, 500)
args.BASELINE_MS = (-25.0, -5.0)
args.PEAK_SEARCH_MS = (20.0, 190.0)
args.PEAK_PROMINENCE_Z = 7.0
args.enable_detrend = en_detrend
args.DETREND_MAX_CURV = 1e-2
args.enable_median = en_median
args.MED_FILT_K = int(med_val)
args.enable_moving_avg = en_movavg
args.MOV_AVG_K = int(avg_val)
args.enable_polarity_flip = bool(enable_polarity_flip)



# -----------------------
# Main Execution
# -----------------------

# Construct stim pair key for JSON lookup
stim_pair_key = f"{stimulated_channel_a}-{stimulated_channel_b}"



# Build file path
edf_path = f"{main_folder}/sub-{subject_id}_{session_id}_task-ccep_run-{run:02.0f}_ieeg{session_name_trailer}.edf"
review_path = f"{main_folder}/" + valid_pulses_json

# Load valid pulse indices if JSON file is provided
valid_pulse_indices = load_valid_pulses(review_path, stim_pair_key)

print(f"\n{'='*80}")
print(f"Processing: {edf_path}")
print(f"Stim pair: {stim_pair_key}")
print(f"Evoked channel: {evoked_channel}")
print(f"{'='*80}\n")

# Read EDF
edf_file = pyedflib.EdfReader(edf_path)

# Read annotations or TSV events
if not read_events_edf0_tsv1:
    ann = edf_file.readAnnotations()
    annot_timestamps = ann[0]  # seconds from recording start
    annot_labels = ann[2]
else:
    df = pd.read_csv(
        f"{main_folder}/sub-{subject_id}_{session_id}_task-ccep_run-{run:02.0f}_events{session_name_trailer}.tsv",
        sep="\t"
    )
    annot_timestamps = df["onset"].values
    annot_labels = df["event"].values

# Decode events for this stimulated channel + current
events_by_channel = decode_events_rev1(annot_timestamps, annot_labels)
if not events_by_channel:
    events_by_channel = decode_events_rev2(annot_timestamps, annot_labels)

if not events_by_channel:
    edf_file.close()
    raise Exception("No stimulation events found.")

# Filter for the specific stim pair and current
if stim_pair_key not in events_by_channel:
    edf_file.close()
    raise Exception(f"No stimulation events found for pair {stim_pair_key}.")

# Filter events by current
filtered_events = [(t, c) for t, c in events_by_channel[stim_pair_key] if c == current]

if not filtered_events:
    edf_file.close()
    raise Exception(f"No stimulation events found for pair {stim_pair_key} with current {current} mA.")


# Channel indices & sampling rate
labels = edf_file.getSignalLabels()

stim_idx_a = labels.index(stimulated_channel_a)
stim_idx_b = labels.index(stimulated_channel_b)
ev_idx = labels.index(evoked_channel)
fs = edf_file.getSampleFrequency(stim_idx_a)
args.sampling_freq = fs

print(f"Sampling frequency: {fs} Hz")
print(f"Total channels: {edf_file.signals_in_file}")

# Read full signals (absolute indexing space)
stim_raw = edf_file.readSignal(stim_idx_a) - edf_file.readSignal(stim_idx_b)
evok_raw = edf_file.readSignal(ev_idx)
edf_file.close()

n_samples = len(stim_raw)

# Convert annotation times (sec) to absolute EDF sample indices
event_times = [ev[0] for ev in filtered_events]
event_samples_all = [int(round(t * fs)) for t in event_times]
event_samples_all = [s for s in event_samples_all if 0 <= s < n_samples]

if len(event_samples_all) == 0:
    raise Exception("Decoded events are out of EDF bounds after conversion to samples.")

# Define 10 s visualization window around the FIRST event
first_event_sample = event_samples_all[0]

win_start = max(0, first_event_sample - int(window_pre_s * fs))
win_end = min(n_samples, win_start + int(window_len_s * fs))

# Events that fall inside the 10 s visualization window
event_samples_in_win = [s for s in event_samples_all if win_start <= s < win_end]


# Apply filters to full signals
print("\nApplying filters...")

if en_band_pass_filter:
    stim_full = filter_data(
        stim_raw.astype(np.float64),
        fs,
        l_freq=bpf_low_freq,
        h_freq=bpf_high_freq,
        method='iir',
        verbose=False
    )
    evok_full = filter_data(
        evok_raw.astype(np.float64),
        fs,
        l_freq=bpf_low_freq,
        h_freq=bpf_high_freq,
        method='iir',
        verbose=False
    )
else:
    stim_full = stim_raw.astype(np.float64)
    evok_full = evok_raw.astype(np.float64)

if en_notch:
    freqs_notch = np.arange(60, fs/2, 60)
    stim_full = notch_filter(
        stim_full,
        fs,
        freqs=[60],
        method='iir',
        verbose=False
    )
    evok_full = notch_filter(
        evok_full,
        fs,
        freqs=[60],
        method='iir',
        verbose=False
    )

    
# Fallback: if annotations only give one timestamp per train,
# derive individual pulses from the stimulated channel within this window
if len(event_samples_in_win) < 2:
    stim_win = stim_full[win_start:win_end]
    stim_mean = np.nanmean(stim_win)
    stim_std = np.nanstd(stim_win)
    stim_height = stim_mean + stim_std

    # distance ~0.9 * fs → assumes pulses are at least ~0.9 s apart;
    # adjust if your train frequency is different
    stim_pks, _ = find_peaks(
        stim_win,
        height=stim_height,
        distance=int(0.9 * fs)
    )

    # convert peak indices in the window back to absolute EDF sample indices
    event_samples_in_win = [win_start + int(pk) for pk in stim_pks]
    event_samples_all = event_samples_in_win

print(f"\nFound {len(event_samples_all)} stimulation events")

# Filter events based on valid pulse indices if provided
if valid_pulse_indices is not None:
    event_samples_valid =[]
    for smp in event_samples_all:
        found = False
        for ref in valid_pulse_indices:
            if found == True:
                continue
            if abs(smp-ref) < 5e-3*fs:
                found = True
        if found == True:
            event_samples_valid.append(smp)            
                
                
    # event_samples_valid = [s for s in event_samples_all if s in valid_pulse_indices]
    print(f"Using {len(event_samples_valid)} valid pulses out of {len(event_samples_all)} total pulses")
    event_samples_filtered = event_samples_valid
    # Create mask for plotting
    is_valid_pulse = [s in valid_pulse_indices for s in event_samples_all]
else:
    event_samples_filtered = event_samples_all
    is_valid_pulse = [True] * len(event_samples_all)
    print("Using all pulses (no filtering)")

if len(event_samples_filtered) == 0:
    raise Exception("No valid events found!")

# Define 10 s visualization window around the FIRST event
first_event_sample = event_samples_filtered[0]
# win_start = max(0, first_event_sample - int(window_pre_s * fs))
# win_end = min(n_samples, win_start + int(window_len_s * fs))

win_start = int(max(min(event_samples_valid)-fs,0))
win_end   = int(min(max(event_samples_valid)+fs,n_samples))


# Events that fall inside the 10 s visualization window
event_samples_in_win = [s for s in event_samples_filtered if win_start <= s < win_end]

print(f"\nEvents in visualization window: {len(event_samples_in_win)}")

# Adaptive threshold
if use_adaptive_threshold:
    signal_scale = np.std(evok_full)
    baseline_minimum_std_used = max(baseline_minimum_std, signal_scale * 0.1)
    print(f"Adaptive threshold: signal_scale={signal_scale:.2f}, using min_std={baseline_minimum_std_used:.2f}")
else:
    baseline_minimum_std_used = baseline_minimum_std

# ER Detection
print("\nERDetect-style detection:")
er_detections = []
baseline_stds_collected = []

for i, ev_samp in enumerate(event_samples_in_win):
    # Baseline window
    bl_start_samp = ev_samp + int(baseline_epoch[0] * fs)
    bl_end_samp = ev_samp + int(baseline_epoch[1] * fs)
    
    if bl_start_samp >= 0 and bl_end_samp <= len(evok_full):
        baseline_seg = evok_full[bl_start_samp:bl_end_samp]
        baseline_std = np.std(baseline_seg)
        baseline_stds_collected.append(baseline_std)
        baseline_std = max(baseline_std, baseline_minimum_std_used)
        baseline_str = f"{baseline_std:.2f}"
    else:
        baseline_std = None
        baseline_str = "N/A"
    
    # Response window
    resp_start_samp = ev_samp + int(response_search_epoch[0] * fs)
    resp_end_samp = ev_samp + int(response_search_epoch[1] * fs)
    
    if resp_start_samp >= 0 and resp_end_samp <= len(evok_full):
        response_seg = evok_full[resp_start_samp:resp_end_samp]
        response_seg_abs = np.abs(response_seg)
        er_peak_idx_rel = np.argmax(response_seg_abs)
        er_peak_idx = resp_start_samp + er_peak_idx_rel
        er_peak_amp = response_seg[er_peak_idx_rel]
        latency_ms = (er_peak_idx - ev_samp) / fs * 1000
    else:
        er_peak_idx = None
        er_peak_amp = 0
        latency_ms = 0
    
    # Detection decision
    if baseline_std is not None and er_peak_idx is not None:
        detection_threshold = baseline_std * baseline_threshold_factor
        if abs(er_peak_amp) > detection_threshold:
            print(
                f"  Stim {i+1}: ER detected at {latency_ms:.1f} ms, "
                f"amplitude={er_peak_amp:.2f}, baseline_std={baseline_str}, "
                f"threshold={detection_threshold:.2f}"
            )
            er_detections.append((er_peak_idx, er_peak_amp, latency_ms))
        else:
            print(
                f"  Stim {i+1}: No ER detected "
                f"(baseline_std={baseline_std:.2f}, threshold={detection_threshold:.2f})"
            )
    else:
        print(f"  Stim {i+1}: No ER detected (baseline window out of bounds)")

# Detection summary
if len(baseline_stds_collected) > 0:
    print("\nDetection summary:")
    print(
        f"  Actual baseline stds: "
        f"min={np.min(baseline_stds_collected):.2f}, "
        f"max={np.max(baseline_stds_collected):.2f}, "
        f"mean={np.mean(baseline_stds_collected):.2f}"
    )
    print(
        "  Detection threshold range: "
        f"{np.min(baseline_stds_collected)*baseline_threshold_factor:.2f} "
        f"to {np.max(baseline_stds_collected)*baseline_threshold_factor:.2f}"
    )
    print(f"  Total detections: {len(er_detections)} out of {len(event_samples_in_win)} stimulations")

# -----------------------
# Epoching / averaging (absolute indices) with orientation-based flip
# -----------------------
signal_to_epoch = evok_full

# Use filtered events for epoching
event_samples_for_epoch = list(event_samples_filtered)
if skip_first_stim and len(event_samples_for_epoch) > 0:
    event_samples_for_epoch = event_samples_for_epoch[1:]

if enable_polarity_flip:
    epochs, t_ms, ttfp_list, flips, flip_diagnostics = epochs_with_orientation_flip(
        args,
        stim_full,
        evok_full,
        event_samples_for_epoch,
        orientation_window_ms=orientation_window_ms,
        zscore_baseline=True
    )
    
    # Print flip diagnostics
    print("\n" + "="*80)
    print("ORIENTATION-BASED FLIP DIAGNOSTICS (per-pulse)")
    print("="*80)
    
    n_flipped = sum(flips)
    n_kept = len(flips) - n_flipped
    n_ambiguous = sum(1 for d in flip_diagnostics if 'AMBIGUOUS' in str(d.get('reason', '')))
    
    print(f"\nSummary: {n_flipped} flipped, {n_kept} kept, {n_ambiguous} ambiguous (defaulted to no flip)")
    print(f"Window size: {orientation_window_ms} ms ({int(orientation_window_ms * fs / 1000)} samples)")
    print()
    
    for diag in flip_diagnostics:
        event_idx = diag.get('event_idx', '?')
        event_sample = diag.get('event_sample', '?')
        flip_decision = diag.get('flip_decision', None)
        reason = diag.get('reason', 'unknown')
        
        stim_orient = diag.get('stim_orientation', 'N/A')
        evoked_orient = diag.get('evoked_orientation', 'N/A')
        stim_pre = diag.get('stim_pre_deriv', None)
        stim_post = diag.get('stim_post_deriv', None)
        evoked_pre = diag.get('evoked_pre_deriv', None)
        evoked_post = diag.get('evoked_post_deriv', None)
        
        flip_str = "FLIP" if flip_decision else "KEEP" if flip_decision is not None else "SKIP"
        
        print(f"Pulse {event_idx+1} (sample {event_sample}):")
        print(f"  Decision: {flip_str}")
        print(f"  Reason: {reason}")
        
        if stim_pre is not None:
            print(f"  Stim channel:   pre_deriv={stim_pre:+.4f}, post_deriv={stim_post:+.4f} -> {stim_orient}")
        if evoked_pre is not None:
            print(f"  Evoked channel: pre_deriv={evoked_pre:+.4f}, post_deriv={evoked_post:+.4f} -> {evoked_orient}")
        print()
    
    print("="*80)

else:
    epochs, t_ms, ttfp_list, flips = epochs_no_flip(
        args,
        signal_to_epoch,
        event_samples_for_epoch,
        zscore_baseline=True
    )
    flip_diagnostics = []

if len(epochs) == 0:
    raise Exception("No epochs produced")

avg = np.nanmean(np.vstack(epochs), axis=0)
t = t_ms.copy()

# -----------------------
# Plotting with valid pulse highlighting
# -----------------------
stim_len = len(stim_full[win_start:win_end]) / fs
evok_len = len(evok_full[win_start:win_end]) / fs

plt.figure(figsize=(14, 10))

# Apply Hamming window (optional)
if apply_hamming:
    window = np.power(np.hamming(len(avg)), hamming_power) + 0.5
    window = window / np.nanmax(window)
    epochs = [ep * window for ep in epochs]
    avg = avg * window

# Plot 1: Individual segments
plt.subplot(2, 2, 1)
for i_seg, seg in enumerate(epochs):
    color = 'red' if flips[i_seg] else 'gray'
    plt.plot(t, seg, color=color, alpha=0.5)
plt.title(f"Individual Segments (Red=Flipped, avg_source={avg_source}), Seg count = {len(epochs)}")
plt.xlabel("Time (ms)")
plt.ylabel("Z")
plt.grid(True)

# Plot 2: Average
plt.subplot(2, 2, 2)
plt.plot(t, final_polarity * avg, linewidth=2)
plt.axvline(0, color="black", linestyle="--", linewidth=1)

plt.axvspan(
    N1_search_window[0]*1000,
    N1_search_window[1]*1000,
    alpha=0.2,
    color='green',
    label='N1 Range'
)
plt.axvspan(
    N2_search_window[0]*1000,
    N2_search_window[1]*1000,
    alpha=0.1,
    color='blue',
    label='N2 Range'
)

mask = (t >= -10) & (t <= 10)
if np.any(mask):
    plt.plot(t[mask], final_polarity * avg[mask], linewidth=2.5, color="red")

plt.title(f"Average ({avg_source}) | sync={sync_source} | flip={'on' if enable_polarity_flip else 'off'}")
plt.xlabel("Time (ms)")
plt.ylabel("Z")
plt.grid(True)
plt.legend()

flip_count = sum(flips)

# Windowed signals for plotting
stim_win = stim_full[win_start:win_end]
evok_win = evok_full[win_start:win_end]

# Plot 3: Full window with valid pulse highlighting
ax = plt.subplot(2, 1, 2)
time_axis = np.linspace(0, evok_len, len(evok_win))

ev_ch_str = f"evoked channel = {evoked_channel} (µV)"
st_ch_str = f"stim channels = {stimulated_channel_a}-{stimulated_channel_b} (µV)"

# Primary y-axis: evoked response
line_evok, = ax.plot(time_axis, evok_win, 'b-', alpha=0.7, linewidth=0.5,
                     label=ev_ch_str)

# Mark ALL pulses in red (regular)
event_times_all = [(s - win_start) / fs for s in event_samples_all if win_start <= s < win_end]
event_samples_in_win_idx = [i for i, s in enumerate(event_samples_all) if win_start <= s < win_end]

# Convert to window-relative indices for signal values
stim_pks_all = []
for s in event_samples_all:
    if win_start <= s < win_end:
        rel_idx = s - win_start
        if 0 <= rel_idx < len(evok_win):
            stim_pks_all.append(rel_idx)

if len(stim_pks_all) > 0:
    ax.plot(time_axis[stim_pks_all], evok_win[stim_pks_all], 'ro',
            alpha=0.7, markersize=6, label="All pulses")

# Mark VALID pulses in magenta (overlay on top)
if event_samples_valid is not None:
    indices = np.int32(event_samples_valid)-win_start
    ax.plot(time_axis[indices], evok_win[indices], 'mo',
            alpha=0.9, markersize=8, label="Valid pulses (used in avg)")

evok_min = np.nanmin(evok_win)
evok_max = np.nanmax(evok_win)
evok_dyn = evok_max - evok_min

ax.set_ylim(evok_min - 0.5*evok_dyn,
            evok_max + 1.5*evok_dyn)

ax.set_xlabel("Time [sec]")
ax.set_ylabel(ev_ch_str, color="blue")
ax.grid(True, linestyle=':', alpha=0.6)

# Secondary y-axis: stim channel
ax2 = ax.twinx()
line_stim, = ax2.plot(time_axis, stim_win, 'g-', alpha=0.7, linewidth=0.5,
                      label=st_ch_str)

stim_min = np.nanmin(stim_win)
stim_max = np.nanmax(stim_win)
stim_dyn = stim_max - stim_min

ax2.set_ylim(stim_min - 1.5*stim_dyn,
             stim_max + 0.5*stim_dyn)

ax2.set_ylabel(st_ch_str, color="green")

# Combined legend
lines = [line_evok, line_stim]
labels = [l.get_label() for l in lines]

# Add pulse markers to legend
if len(stim_pks_all) > 0:
    all_pulse_line = plt.Line2D([0], [0], marker='o', color='w', 
                                markerfacecolor='r', markersize=6, label='All pulses')
    lines.append(all_pulse_line)
    labels.append('All pulses')

if event_samples_valid is not None:
    valid_pulse_line = plt.Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='m', markersize=8, label='Valid pulses (used in avg)')
    lines.append(valid_pulse_line)
    labels.append('Valid pulses (used in avg)')

ax.legend(lines, labels, loc='upper right')

# Title with valid pulse info
title_text = (
    f"Subject {subject_id}, {session_id} | Current = {current} mA | "
    f"BandPass = {'En' if en_band_pass_filter else 'Dis'} | "
    f"Notch = {'En' if en_notch else 'Dis'} | "
    f"ERDetect: bl_factor={baseline_threshold_factor}, "
    f"min_std={baseline_minimum_std_used:.2f} | "
    f"Flipped: {flip_count}/{len(flips)}"
)

if valid_pulse_indices is not None:
    title_text += f" | Valid pulses: {len(event_samples_filtered)}/{len(event_samples_all)}"

plt.suptitle(title_text, fontsize=14, y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

out_png = (
    f"./Outputs/"
    f"{subject_id}_S{session_id[4:]}_ccep_{stimulated_channel_a}_{stimulated_channel_b}_"
    f"evoked_{evoked_channel}_{current}mA_sync-{sync_source}_"
    f"avg-{avg_source}_flip-{'on' if enable_polarity_flip else 'off'}_ERDetect_valid_pulses.png"
)
plt.savefig(out_png, dpi=300)
plt.show()
print(f"\nSaved: {out_png}")
