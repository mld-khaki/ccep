# -*- coding: utf-8 -*-
"""
CCEP Processing Pipeline with ERDetect-style Detection and JSON Configuration

Modified on Sun Oct 5 12:59:16 2025

Original Author:    Alaina Mahalanobis
Modified by:        Milad Khaki
Reviewed by:        (pending review)

Enhanced with ERDetect integration

Current update (non-breaking structural improvements):
- Organize code into .py modules (no changes to notebook logic)
- Enable annotation-based EDF synchronization (existing behavior preserved)
- Optional: support JSON file for pulse selection
- NEW: Support for JSON-based configuration for all user parameters

NOTE:
    This module-based (.py) structure is used to improve debugging,
    reproducibility, and version control. No changes were made to the
    original notebook logic; this only reorganizes the code for maintainability.
"""

import os
import json
from types import SimpleNamespace

import numpy as np
import pyedflib
import matplotlib.pyplot as plt
import pandas as pd
from mne.filter import filter_data, notch_filter

from ccep_lib import (
    load_valid_pulses,
    decode_events_rev1,
    decode_events_rev2,
    epochs_no_flip,
    find_peaks,
    epochs_with_orientation_flip,
)

# -------------------------------------------------------------------------
# Configuration Loader
# -------------------------------------------------------------------------


def _dict_to_namespace(d):
    """Recursively convert dicts to SimpleNamespace for attribute-style access."""
    if isinstance(d, dict):
        return SimpleNamespace(**{k: _dict_to_namespace(v) for k, v in d.items()})
    return d


def load_user_config(path):
    """
    Load configuration file containing all user/session/filter parameters.

    The JSON is expected to define keys such as:
        subject, session, channels, stim, paths, options, filter, windows,
        erdetect, epoch, etc.
    """
    with open(path, "r") as f:
        cfg = json.load(f)
    return _dict_to_namespace(cfg)


# -------------------------------------------------------------------------
# Core EDF / Event Handling
# -------------------------------------------------------------------------


def build_paths(cfg, subject_id, session_id, session_fold, run, session_name_trailer):
    """Construct EDF path and (optional) valid-pulses JSON path."""
    main_folder = cfg.paths.main_folder
    edf_filename = (
        f"sub-{subject_id}_{session_id}_task-ccep_run-{run:02.0f}_ieeg"
        f"{session_name_trailer}.edf"
    )
    edf_path = os.path.join(main_folder, edf_filename)

    if cfg.paths.valid_pulses_json:
        review_path = os.path.join(main_folder, cfg.paths.valid_pulses_json)
    else:
        review_path = None

    return edf_path, review_path


def read_events(edf_file, cfg, subject_id, session_id, run, session_name_trailer):
    """
    Read annotations from EDF or TSV events file, depending on cfg.options.read_events_edf0_tsv1.

    Returns:
        annot_timestamps (np.ndarray)
        annot_labels (np.ndarray or list)
    """
    main_folder = cfg.paths.main_folder
    read_events_edf0_tsv1 = cfg.options.read_events_edf0_tsv1

    if not read_events_edf0_tsv1:
        ann = edf_file.readAnnotations()
        annot_timestamps = np.array(ann[0])  # seconds from recording start
        annot_labels = np.array(ann[2])
    else:
        events_filename = (
            f"sub-{subject_id}_{session_id}_task-ccep_run-{run:02.0f}_events"
            f"{session_name_trailer}.tsv"
        )
        events_path = os.path.join(main_folder, events_filename)
        df = pd.read_csv(events_path, sep="\t")
        print(df.columns)
        annot_timestamps = df["onset"].values
        # column might be named "event" or "trial_type" etc.; here assume "event"
        if "event" in df.columns:
            annot_labels = df["event"].values
        else:
            # fallback if schema differs
            annot_labels = df.iloc[:, 2].values

    return annot_timestamps, annot_labels


def decode_stim_events(annot_timestamps, annot_labels, stim_pair_key, current):
    """
    Decode stimulation events for a given stim pair and current.

    Returns:
        filtered_events: list of (timestamp_sec, current_str)
    """
    events_by_channel = decode_events_rev1(annot_timestamps, annot_labels)
    if not events_by_channel:
        events_by_channel = decode_events_rev2(annot_timestamps, annot_labels)

    if not events_by_channel:
        raise RuntimeError("No stimulation events found in annotations.")

    if stim_pair_key not in events_by_channel:
        raise RuntimeError(f"No stimulation events found for pair {stim_pair_key}.")

    filtered_events = [
        (t, c) for t, c in events_by_channel[stim_pair_key] if c == current
    ]

    if not filtered_events:
        raise RuntimeError(
            f"No stimulation events found for pair {stim_pair_key} with current {current} mA."
        )

    return filtered_events


def read_signals(edf_path, stim_ch_a, stim_ch_b, ev_ch):
    """
    Read EDF, return:
        stim_raw (difference of stim_a - stim_b),
        evok_raw,
        fs (sampling frequency),
        n_samples,
        labels (channel labels)
    """
    edf_file = pyedflib.EdfReader(edf_path)
    labels = edf_file.getSignalLabels()

    try:
        stim_idx_a = labels.index(stim_ch_a)
        stim_idx_b = labels.index(stim_ch_b)
        ev_idx = labels.index(ev_ch)
    except ValueError as e:
        edf_file.close()
        raise RuntimeError(f"Channel not found in EDF: {e}")

    fs = edf_file.getSampleFrequency(stim_idx_a)
    stim_raw = edf_file.readSignal(stim_idx_a) - edf_file.readSignal(stim_idx_b)
    evok_raw = edf_file.readSignal(ev_idx)
    n_samples = len(stim_raw)

    edf_file.close()

    return stim_raw, evok_raw, fs, n_samples, labels


def convert_events_to_samples(filtered_events, fs, n_samples):
    """
    Convert event timestamps (seconds) to absolute sample indices,
    clipped to [0, n_samples).
    """
    event_times = [ev[0] for ev in filtered_events]
    event_samples_all = [int(round(t * fs)) for t in event_times]
    event_samples_all = [s for s in event_samples_all if 0 <= s < n_samples]

    if len(event_samples_all) == 0:
        raise RuntimeError("Decoded events are out of EDF bounds after conversion.")
    return event_samples_all


# -------------------------------------------------------------------------
# Filtering / Pulse Detection
# -------------------------------------------------------------------------


def apply_filters(stim_raw, evok_raw, fs, cfg):
    """Apply band-pass and notch filters (if enabled) to full-length signals."""
    en_band_pass_filter = cfg.filter.enable_bandpass
    en_notch = cfg.filter.enable_notch
    bpf_low_freq = cfg.filter.bandpass_low
    bpf_high_freq = cfg.filter.bandpass_high

    if en_band_pass_filter:
        stim_full = filter_data(
            stim_raw.astype(np.float64),
            fs,
            l_freq=bpf_low_freq,
            h_freq=bpf_high_freq,
            method="iir",
            verbose=False,
        )
        evok_full = filter_data(
            evok_raw.astype(np.float64),
            fs,
            l_freq=bpf_low_freq,
            h_freq=bpf_high_freq,
            method="iir",
            verbose=False,
        )
    else:
        stim_full = stim_raw.astype(np.float64)
        evok_full = evok_raw.astype(np.float64)

    if en_notch:
        # example: 60 Hz; can be extended if needed
        stim_full = notch_filter(
            stim_full,
            fs,
            freqs=[60],
            method="iir",
            verbose=False,
        )
        evok_full = notch_filter(
            evok_full,
            fs,
            freqs=[60],
            method="iir",
            verbose=False,
        )

    return stim_full, evok_full


def detect_pulses_if_needed(
    event_samples_all, stim_full, fs, win_start, win_end, min_separation_sec=0.9
):
    """
    If annotations only give one timestamp per train, derive individual pulses
    from the stimulated channel within the visualization window.
    """
    if len(event_samples_all) >= 2:
        return event_samples_all

    stim_win = stim_full[win_start:win_end]
    stim_mean = np.nanmean(stim_win)
    stim_std = np.nanstd(stim_win)
    stim_height = stim_mean + stim_std

    # distance ~0.9 * fs → assumes pulses are at least ~0.9 s apart
    stim_pks, _ = find_peaks(
        stim_win,
        height=stim_height,
        distance=int(min_separation_sec * fs),
    )

    # convert window-relative indices to absolute EDF sample indices
    event_samples_all = [win_start + int(pk) for pk in stim_pks]

    return event_samples_all


def filter_valid_pulses(event_samples_all, valid_pulse_indices, fs, tolerance_ms=5.0):
    """
    Match decoded events to a list of valid pulse sample indices (from JSON),
    within a tolerance in milliseconds.

    Returns:
        event_samples_filtered
        event_samples_valid (for plotting / logging)
    """
    if valid_pulse_indices is None:
        return list(event_samples_all), None

    tol_samples = int((tolerance_ms / 1000.0) * fs)
    event_samples_valid = []

    for smp in event_samples_all:
        found = False
        for ref in valid_pulse_indices:
            if found:
                break
            if abs(smp - ref) <= tol_samples:
                found = True
        if found:
            event_samples_valid.append(smp)

    return event_samples_valid, event_samples_valid


# -------------------------------------------------------------------------
# ERDetect-style Detection
# -------------------------------------------------------------------------


def compute_adaptive_baseline_min_std(evok_full, baseline_minimum_std, use_adaptive):
    """Compute minimum baseline std based on global signal scale, if enabled."""
    if not use_adaptive:
        return baseline_minimum_std
    signal_scale = np.std(evok_full)
    baseline_minimum_std_used = max(baseline_minimum_std, signal_scale * 0.1)
    print(
        f"Adaptive threshold: signal_scale={signal_scale:.2f}, "
        f"using min_std={baseline_minimum_std_used:.2f}"
    )
    return baseline_minimum_std_used


def erdetect_loop(
    evok_full,
    event_samples_in_win,
    fs,
    baseline_epoch,
    response_search_epoch,
    baseline_threshold_factor,
    baseline_minimum_std_used,
):
    """
    Run ERDetect-style detection across events in visualization window.

    Returns:
        er_detections: list of (peak_idx_abs, peak_amp, latency_ms)
        baseline_stds_collected: list of baseline stds
    """
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
            er_peak_idx_rel = int(np.argmax(response_seg_abs))
            er_peak_idx = resp_start_samp + er_peak_idx_rel
            er_peak_amp = response_seg[er_peak_idx_rel]
            latency_ms = (er_peak_idx - ev_samp) / fs * 1000.0
        else:
            er_peak_idx = None
            er_peak_amp = 0.0
            latency_ms = 0.0

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

    return er_detections, baseline_stds_collected


def print_detection_summary(baseline_stds_collected, baseline_threshold_factor, n_events):
    """Print summary of baseline stds and detection thresholds."""
    if len(baseline_stds_collected) == 0:
        return

    print("\nDetection summary:")
    b = np.array(baseline_stds_collected)
    print(f"  Actual baseline stds: min={b.min():.2f}, max={b.max():.2f}, mean={b.mean():.2f}")
    print(
        "  Detection threshold range: "
        f"{b.min() * baseline_threshold_factor:.2f} to "
        f"{b.max() * baseline_threshold_factor:.2f}"
    )
    print(f"  Total detections: {len(baseline_stds_collected)} baseline segments")
    print(f"  Events considered: {n_events}")


# -------------------------------------------------------------------------
# Epoching / Averaging
# -------------------------------------------------------------------------


def epoch_and_flip(
    args,
    stim_full,
    evok_full,
    event_samples_for_epoch,
    fs,
    enable_polarity_flip,
    orientation_window_ms,
):
    """
    Epoch evoked responses and optionally perform orientation-based polarity flipping.

    Returns:
        epochs (list of np.ndarray),
        t_ms (time vector),
        flips (list of bool),
        flip_diagnostics (list of dict)
    """
    if enable_polarity_flip:
        epochs, t_ms, ttfp_list, flips, flip_diagnostics = epochs_with_orientation_flip(
            args,
            stim_full,
            evok_full,
            event_samples_for_epoch,
            orientation_window_ms=orientation_window_ms,
            zscore_baseline=True,
        )

        # Print flip diagnostics
        print("\n" + "=" * 80)
        print("ORIENTATION-BASED FLIP DIAGNOSTICS (per-pulse)")
        print("=" * 80)

        n_flipped = sum(flips)
        n_kept = len(flips) - n_flipped
        n_ambiguous = sum(
            1
            for d in flip_diagnostics
            if "AMBIGUOUS" in str(d.get("reason", "")).upper()
        )

        print(
            f"\nSummary: {n_flipped} flipped, {n_kept} kept, "
            f"{n_ambiguous} ambiguous (defaulted to no flip)"
        )
        print(
            f"Window size: {orientation_window_ms} ms "
            f"({int(orientation_window_ms * fs / 1000)} samples)"
        )
        print()

        for diag in flip_diagnostics:
            event_idx = diag.get("event_idx", "?")
            event_sample = diag.get("event_sample", "?")
            flip_decision = diag.get("flip_decision", None)
            reason = diag.get("reason", "unknown")

            stim_orient = diag.get("stim_orientation", "N/A")
            evoked_orient = diag.get("evoked_orientation", "N/A")
            stim_pre = diag.get("stim_pre_deriv", None)
            stim_post = diag.get("stim_post_deriv", None)
            evoked_pre = diag.get("evoked_pre_deriv", None)
            evoked_post = diag.get("evoked_post_deriv", None)

            flip_str = (
                "FLIP" if flip_decision else "KEEP" if flip_decision is not None else "SKIP"
            )

            print(f"Pulse {event_idx + 1} (sample {event_sample}):")
            print(f"  Decision: {flip_str}")
            print(f"  Reason: {reason}")

            if stim_pre is not None:
                print(
                    f"  Stim channel:   pre_deriv={stim_pre:+.4f}, "
                    f"post_deriv={stim_post:+.4f} -> {stim_orient}"
                )
            if evoked_pre is not None:
                print(
                    f"  Evoked channel: pre_deriv={evoked_pre:+.4f}, "
                    f"post_deriv={evoked_post:+.4f} -> {evoked_orient}"
                )
            print()

        print("=" * 80)

    else:
        epochs, t_ms, ttfp_list, flips = epochs_no_flip(
            args,
            evok_full,
            event_samples_for_epoch,
            zscore_baseline=True,
        )
        flip_diagnostics = []

    if len(epochs) == 0:
        raise RuntimeError("No epochs produced.")

    return epochs, t_ms, flips, flip_diagnostics


# -------------------------------------------------------------------------
# Plotting
# -------------------------------------------------------------------------


def plot_results(
    cfg,
    subject_id,
    session_id,
    stim_full,
    evok_full,
    fs,
    win_start,
    win_end,
    epochs,
    t_ms,
    flips,
    final_polarity,
    sync_source,
    avg_source,
    stimulated_channel_a,
    stimulated_channel_b,
    evoked_channel,
    current,
    N1_search_window,
    N2_search_window,
    event_samples_all,
    event_samples_filtered,
    event_samples_valid,
    baseline_minimum_std_used,
):
    """
    Create summary plots:
      - Individual epochs (flipped vs non-flipped)
      - Average evoked response with N1/N2 windows
      - Full window signals with pulse markers (all vs valid)
    """
    apply_hamming = cfg.options.apply_hamming
    hamming_power = cfg.options.hamming_power

    stim_win = stim_full[win_start:win_end]
    evok_win = evok_full[win_start:win_end]

    stim_len = len(stim_win) / fs
    evok_len = len(evok_win) / fs

    avg = np.nanmean(np.vstack(epochs), axis=0)
    t = t_ms.copy()

    plt.figure(figsize=(14, 10))

    # Optional Hamming window
    if apply_hamming:
        window = np.power(np.hamming(len(avg)), hamming_power) + 0.5
        window = window / np.nanmax(window)
        epochs = [ep * window for ep in epochs]
        avg = avg * window

    # Plot 1: Individual segments
    plt.subplot(2, 2, 1)
    for i_seg, seg in enumerate(epochs):
        color = "red" if flips[i_seg] else "gray"
        plt.plot(t, seg, color=color, alpha=0.5)
    plt.title(
        f"Individual Segments (Red=Flipped, avg_source={avg_source}), "
        f"Seg count = {len(epochs)}"
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Z")
    plt.grid(True)

    # Plot 2: Average
    plt.subplot(2, 2, 2)
    plt.plot(t, final_polarity * avg, linewidth=2)
    plt.axvline(0, color="black", linestyle="--", linewidth=1)

    plt.axvspan(
        N1_search_window[0] * 1000.0,
        N1_search_window[1] * 1000.0,
        alpha=0.2,
        color="green",
        label="N1 Range",
    )
    plt.axvspan(
        N2_search_window[0] * 1000.0,
        N2_search_window[1] * 1000.0,
        alpha=0.1,
        color="blue",
        label="N2 Range",
    )

    mask = (t >= -10.0) & (t <= 10.0)
    if np.any(mask):
        plt.plot(t[mask], final_polarity * avg[mask], linewidth=2.5, color="red")

    plt.title(
        f"Average ({avg_source}) | sync={sync_source} | "
        f"flip={'on' if cfg.options.enable_polarity_flip else 'off'}"
    )
    plt.xlabel("Time (ms)")
    plt.ylabel("Z")
    plt.grid(True)
    plt.legend()

    flip_count = sum(flips)

    # Plot 3: Full window with valid pulse highlighting
    ax = plt.subplot(2, 1, 2)
    time_axis = np.linspace(0, evok_len, len(evok_win))

    ev_ch_str = f"evoked channel = {evoked_channel} (µV)"
    st_ch_str = f"stim channels = {stimulated_channel_a}-{stimulated_channel_b} (µV)"

    # Primary y-axis: evoked response
    line_evok, = ax.plot(
        time_axis,
        evok_win,
        "b-",
        alpha=0.7,
        linewidth=0.5,
        label=ev_ch_str,
    )

    # Mark ALL pulses in red
    stim_pks_all = []
    for s in event_samples_all:
        if win_start <= s < win_end:
            rel_idx = s - win_start
            if 0 <= rel_idx < len(evok_win):
                stim_pks_all.append(rel_idx)

    if len(stim_pks_all) > 0:
        ax.plot(
            time_axis[stim_pks_all],
            evok_win[stim_pks_all],
            "ro",
            alpha=0.7,
            markersize=6,
            label="All pulses",
        )

    # Mark VALID pulses in magenta (overlay on top)
    if event_samples_valid is not None and len(event_samples_valid) > 0:
        indices = np.int32(event_samples_valid) - win_start
        valid_mask = (indices >= 0) & (indices < len(evok_win))
        indices = indices[valid_mask]
        if len(indices) > 0:
            ax.plot(
                time_axis[indices],
                evok_win[indices],
                "mo",
                alpha=0.9,
                markersize=8,
                label="Valid pulses (used in avg)",
            )

    evok_min = np.nanmin(evok_win)
    evok_max = np.nanmax(evok_win)
    evok_dyn = evok_max - evok_min

    ax.set_ylim(evok_min - 0.5 * evok_dyn, evok_max + 1.5 * evok_dyn)

    ax.set_xlabel("Time [sec]")
    ax.set_ylabel(ev_ch_str, color="blue")
    ax.grid(True, linestyle=":", alpha=0.6)

    # Secondary y-axis: stim channel
    ax2 = ax.twinx()
    line_stim, = ax2.plot(
        time_axis,
        stim_win,
        "g-",
        alpha=0.7,
        linewidth=0.5,
        label=st_ch_str,
    )

    stim_min = np.nanmin(stim_win)
    stim_max = np.nanmax(stim_win)
    stim_dyn = stim_max - stim_min

    ax2.set_ylim(stim_min - 1.5 * stim_dyn, stim_max + 0.5 * stim_dyn)
    ax2.set_ylabel(st_ch_str, color="green")

    # Combined legend
    lines = [line_evok, line_stim]
    labels = [l.get_label() for l in lines]

    if len(stim_pks_all) > 0:
        all_pulse_line = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="r",
            markersize=6,
            label="All pulses",
        )
        lines.append(all_pulse_line)
        labels.append("All pulses")

    if event_samples_valid is not None and len(event_samples_valid) > 0:
        valid_pulse_line = plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor="m",
            markersize=8,
            label="Valid pulses (used in avg)",
        )
        lines.append(valid_pulse_line)
        labels.append("Valid pulses (used in avg)")

    ax.legend(lines, labels, loc="upper right")

    # Title with valid pulse info
    en_band_pass_filter = cfg.filter.enable_bandpass
    en_notch = cfg.filter.enable_notch

    title_text = (
        f"Subject {subject_id}, {session_id} | Current = {current} mA | "
        f"BandPass = {'En' if en_band_pass_filter else 'Dis'} | "
        f"Notch = {'En' if en_notch else 'Dis'} | "
        f"ERDetect: bl_factor={cfg.erdetect.baseline_factor}, "
        f"min_std={baseline_minimum_std_used:.2f} | "
        f"Flipped: {flip_count}/{len(flips)}"
    )

    if event_samples_filtered is not None:
        title_text += (
            f" | Valid pulses: "
            f"{len(event_samples_filtered)}/{len(event_samples_all)}"
        )

    plt.suptitle(title_text, fontsize=14, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    output_folder = getattr(cfg.paths, "output_folder", "./Outputs")
    os.makedirs(output_folder, exist_ok=True)

    out_png = os.path.join(
        output_folder,
        (
            f"{subject_id}_S{session_id[4:]}_ccep_"
            f"{stimulated_channel_a}_{stimulated_channel_b}_"
            f"evoked_{evoked_channel}_{current}mA_sync-{cfg.options.sync_source}_"
            f"avg-{cfg.options.avg_source}_flip-"
            f"{'on' if cfg.options.enable_polarity_flip else 'off'}_"
            f"ERDetect_valid_pulses.png"
        ),
    )

    plt.savefig(out_png, dpi=300)
    plt.show()
    print(f"\nSaved: {out_png}")


# -------------------------------------------------------------------------
# Main Orchestration
# -------------------------------------------------------------------------


def process_ccep(config_path="ccep_config.json"):
    """
    Main entry point for CCEP processing:
      - Load JSON config
      - Load EDF and events
      - Apply filters
      - Run ER detection
      - Epoch and (optionally) flip
      - Plot and save results
    """
    # Load config
    cfg = load_user_config(config_path)

    # User/session parameters
    subject_id = cfg.subject.id
    session_id = cfg.session.id
    session_fold = cfg.session.folder
    run = cfg.session.run
    session_name_trailer = cfg.session.name_suffix

    stimulated_channel_a = cfg.channels.stim_a
    stimulated_channel_b = cfg.channels.stim_b
    evoked_channel = cfg.channels.evoked
    current = cfg.stim.current

    sync_source = cfg.options.sync_source
    avg_source = cfg.options.avg_source
    enable_polarity_flip = cfg.options.enable_polarity_flip
    final_polarity = cfg.options.final_polarity

    window_pre_s = cfg.windows.pre_seconds
    window_len_s = cfg.windows.length_seconds
    skip_first_stim = getattr(cfg.options, "skip_first_stim", False)

    peak_search_epoch = tuple(cfg.erdetect.peak_search)  # currently unused
    response_search_epoch = tuple(cfg.erdetect.response_search)
    N1_search_window = tuple(cfg.erdetect.N1_window)
    N2_search_window = tuple(cfg.erdetect.N2_window)
    baseline_epoch = tuple(cfg.erdetect.baseline)
    baseline_threshold_factor = cfg.erdetect.baseline_factor
    baseline_minimum_std = cfg.erdetect.baseline_minstd
    use_adaptive_threshold = cfg.erdetect.use_adaptive_threshold
    orientation_window_ms = cfg.erdetect.orientation_window_ms

    # Epoching / polarity args
    args = SimpleNamespace()
    args.sampling_freq = None
    args.SEG_MS = tuple(cfg.epoch.seg_ms)
    args.BASELINE_MS = tuple(cfg.epoch.baseline_ms)
    args.PEAK_SEARCH_MS = tuple(cfg.epoch.peak_search_ms)
    args.PEAK_PROMINENCE_Z = cfg.epoch.peak_prominence_z
    args.enable_detrend = cfg.filter.enable_detrend
    args.DETREND_MAX_CURV = cfg.epoch.detrend_max_curv
    args.enable_median = cfg.filter.enable_median
    args.MED_FILT_K = int(cfg.filter.median_kernel)
    args.enable_moving_avg = cfg.filter.enable_movavg
    args.MOV_AVG_K = int(cfg.filter.movavg_kernel)
    args.enable_polarity_flip = bool(enable_polarity_flip)

    # Stim key (for pulses JSON)
    stim_pair_key = f"{stimulated_channel_a}-{stimulated_channel_b}"

    # Build file paths
    edf_path, review_path = build_paths(
        cfg,
        subject_id,
        session_id,
        session_fold,
        run,
        session_name_trailer,
    )

    # Load valid pulse indices if JSON is provided
    if review_path is not None:
        valid_pulse_indices = load_valid_pulses(review_path, stim_pair_key)
    else:
        valid_pulse_indices = None

    print("\n" + "=" * 80)
    print(f"Processing: {edf_path}")
    print(f"Stim pair: {stim_pair_key}")
    print(f"Evoked channel: {evoked_channel}")
    print("=" * 80 + "\n")

    # Read EDF signals and annotations
    stim_raw, evok_raw, fs, n_samples, labels = read_signals(
        edf_path, stimulated_channel_a, stimulated_channel_b, evoked_channel
    )
    args.sampling_freq = fs

    print(f"Sampling frequency: {fs} Hz")
    print(f"Total channels: {len(labels)}")

    # Reopen EDF for annotations/events (pyedflib annotations need open handle)
    edf_file = pyedflib.EdfReader(edf_path)
    annot_timestamps, annot_labels = read_events(
        edf_file,
        cfg,
        subject_id,
        session_id,
        run,
        session_name_trailer,
    )
    edf_file.close()

    # Decode stim events
    filtered_events = decode_stim_events(
        annot_timestamps, annot_labels, stim_pair_key, current
    )
    event_samples_all = convert_events_to_samples(filtered_events, fs, n_samples)

    # Initial 10 s window around FIRST event (for fallback detection)
    first_event_sample = event_samples_all[0]
    win_start = max(0, first_event_sample - int(window_pre_s * fs))
    win_end = min(n_samples, win_start + int(window_len_s * fs))
    event_samples_in_win = [
        s for s in event_samples_all if win_start <= s < win_end
    ]

    # Apply filters to full signals
    print("\nApplying filters...")
    stim_full, evok_full = apply_filters(stim_raw, evok_raw, fs, cfg)

    # Fallback: detect pulses from stim channel if few events found
    event_samples_all = detect_pulses_if_needed(
        event_samples_all, stim_full, fs, win_start, win_end, min_separation_sec=0.9
    )
    print(f"\nFound {len(event_samples_all)} stimulation events (after fallback check)")

    # Filter events based on valid pulse indices (if provided)
    event_samples_filtered, event_samples_valid = filter_valid_pulses(
        event_samples_all,
        valid_pulse_indices,
        fs,
        tolerance_ms=5.0,
    )

    if event_samples_filtered is None or len(event_samples_filtered) == 0:
        raise RuntimeError("No valid events found after pulse filtering.")

    print(
        f"Using {len(event_samples_filtered)} valid pulses out of "
        f"{len(event_samples_all)} total pulses"
        if valid_pulse_indices is not None
        else f"Using all {len(event_samples_filtered)} pulses (no pulse filtering JSON)."
    )

    # Visualization window around valid pulses (±1 s from min/max)
    min_ev = min(event_samples_filtered)
    max_ev = max(event_samples_filtered)
    win_start = int(max(min_ev - fs, 0))
    win_end = int(min(max_ev + fs, n_samples))

    # Events that fall inside the visualization window
    event_samples_in_win = [
        s for s in event_samples_filtered if win_start <= s < win_end
    ]
    print(f"\nEvents in visualization window: {len(event_samples_in_win)}")

    # Adaptive threshold for ERDetect
    baseline_minimum_std_used = compute_adaptive_baseline_min_std(
        evok_full, baseline_minimum_std, use_adaptive_threshold
    )

    # ERDetect loop
    er_detections, baseline_stds_collected = erdetect_loop(
        evok_full,
        event_samples_in_win,
        fs,
        baseline_epoch,
        response_search_epoch,
        baseline_threshold_factor,
        baseline_minimum_std_used,
    )
    print_detection_summary(
        baseline_stds_collected,
        baseline_threshold_factor,
        len(event_samples_in_win),
    )
    print(f"  Total detections: {len(er_detections)} out of {len(event_samples_in_win)} stimulations")

    # Epoching / averaging
    event_samples_for_epoch = list(event_samples_filtered)
    if skip_first_stim and len(event_samples_for_epoch) > 0:
        event_samples_for_epoch = event_samples_for_epoch[1:]

    epochs, t_ms, flips, flip_diagnostics = epoch_and_flip(
        args,
        stim_full,
        evok_full,
        event_samples_for_epoch,
        fs,
        enable_polarity_flip,
        orientation_window_ms,
    )

    # Plot and save results
    plot_results(
        cfg,
        subject_id,
        session_id,
        stim_full,
        evok_full,
        fs,
        win_start,
        win_end,
        epochs,
        t_ms,
        flips,
        final_polarity,
        sync_source,
        avg_source,
        stimulated_channel_a,
        stimulated_channel_b,
        evoked_channel,
        current,
        N1_search_window,
        N2_search_window,
        event_samples_all,
        event_samples_filtered,
        event_samples_valid,
        baseline_minimum_std_used,
    )


def main():
    process_ccep("ccep_session_parameters.json")


if __name__ == "__main__":
    main()
