# ccep_onepass/pipeline.py
# Orchestrates the run, preserves key variables for Spyder inspection.
# Crash-safe: exports locals + checkpoints on exceptions.

import os
import sys
import glob
import time
import argparse
from pathlib import Path
from typing import List, Optional, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal

from scipy.signal import medfilt




# # sub-153
# # --------------------------------------------------------------------------------------
# # subject specs loading
# sub_str = "sub-153"
# ses_str = "ses-001"
# ses_fld_str = "ses-001"
# sub_dir = f"o:/Other_Datasets_phis/CCEP-DB/{sub_str}/"
# cache_dir = "c:/_Code/ccep_cache/"
# enable_cache_setting = True # above folder should be valid and on a fast ssd drive if this option is set to True

# if sub_dir not in sys.path:
#     sys.path.insert(0, sub_dir)
   
# from sub_153_ses_001_ccep_specifications import update_args
# # --------------------------------------------------------------------------------------

# sub-160
# --------------------------------------------------------------------------------------
# subject specs loading
sub_str = "sub-160"
ses_str = "ses-001"
ses_fld_str = "ses_2024_11_01"
sub_dir = f"o:/Other_Datasets_phis/CCEP-DB/{sub_str}/"
subses_dir = f"o:/Other_Datasets_phis/CCEP-DB/{sub_str}/{ses_fld_str}/"
cache_dir = "c:/_Code/ccep_cache/"
enable_cache_setting = True # above folder should be valid and on a fast ssd drive if this option is set to True

if subses_dir not in sys.path:
    sys.path.insert(0, subses_dir)
   
from sub_160_ses_001_ccep_specifications import update_args

# --------------------------------------------------------------------------------------

# # sub-161
# # --------------------------------------------------------------------------------------
# # subject specs loading
# sub_str = "sub-161"
# ses_str = "ses-001"
# ses_fld_str = "ses_2024_11_15"
# sub_dir = f"o:/Other_Datasets_phis/CCEP-DB/{sub_str}/"
# cache_dir = "c:/_Code/ccep_cache/"
# enable_cache_setting = True # above folder should be valid and on a fast ssd drive if this option is set to True

# if sub_dir not in sys.path:
#     sys.path.insert(0, sub_dir)
   
# # --------------------------------------------------------------------------------------






# --------------------------------------------------------------------------------------
# Import bootstrap: allow both
#  - package mode: "from .config import ..."
#  - script mode (Spyder runfile in ccep_onepass/): absolute "from ccep_onepass.config ..."
# --------------------------------------------------------------------------------------
try:
    from .io import open_edf, get_labels_and_fs, read_events, parse_all_stim_candidates, decode_events_rev1, decode_events_rev2
    from .preproc import (
        prefilter_all_channels, apply_bipolar, same_shaft_as_stim,
        resolve_cache_dir, save_prefilter_cache, try_load_prefilter_cache,
    )
    from .analysis import (
        segments_from_evoked, baseline_normalize, first_peak_with_polarity, epochs_with_polarity,
        OnlineBucket, compute_stats, detrend_cubic,
    )

except ImportError:
    THIS_DIR = os.path.dirname(os.path.abspath(__file__))
    PKG_PARENT = os.path.dirname(THIS_DIR)
    if PKG_PARENT not in sys.path:
        sys.path.insert(0, PKG_PARENT)
    from ccep_onepass.io import open_edf, get_labels_and_fs, read_events, parse_all_stim_candidates, decode_events_rev1, decode_events_rev2
    from ccep_onepass.preproc import (
        prefilter_all_channels, apply_bipolar, same_shaft_as_stim,
        resolve_cache_dir, save_prefilter_cache, try_load_prefilter_cache, prefilter_and_cache_channels,
    )
    from ccep_onepass.analysis import (
        segments_from_evoked, baseline_normalize, first_peak_with_polarity, epochs_with_polarity,
        OnlineBucket, compute_stats, detrend_cubic,
    )
    


# ---- Crash-safe debug/export helpers ----
DBG_STATE: Dict[str, object] = {}
DBG_CHECKPOINTS: Dict[str, Dict[str, object]] = {}

def state_update(**kwargs):
    """Continuously-available state for Spyder inspection even if we crash."""
    DBG_STATE.update(kwargs)

def checkpoint(name: str, **kwargs):
    """Named snapshots of important milestones."""
    DBG_CHECKPOINTS[name] = dict(kwargs)

def _export(name: str, value):
    globals()[name] = value

def _export_crash(exc: BaseException):
    """Publish traceback + deepest-frame locals + debug state into globals."""
    import traceback as _tb_mod
    etype, evalue, tb = sys.exc_info()
    # Walk to deepest frame
    last_tb = tb
    while last_tb and last_tb.tb_next:
        last_tb = last_tb.tb_next
    last_frame_locals = {}
    if last_tb:
        try:
            last_frame_locals = dict(last_tb.tb_frame.f_locals)
        except Exception as e:
            if args.raise_errors == True:
                raise(e)
        
            last_frame_locals = {}
    _export("CRASH_TRACEBACK", "".join(_tb_mod.format_exception(etype, evalue, tb)))
    _export("CRASH_LOCALS", last_frame_locals)
    _export("DBG_STATE", dict(DBG_STATE))
    _export("DBG_CHECKPOINTS", dict(DBG_CHECKPOINTS))
    print("\n[CRASH] Exception captured. See variables: CRASH_TRACEBACK, CRASH_LOCALS, DBG_STATE, DBG_CHECKPOINTS")

# ---- Minimal memory/progress utils ----
try:
    import psutil
    _HAS_PSUTIL = True
except Exception:
    _HAS_PSUTIL = False

def _proc_mem():
    if _HAS_PSUTIL:
        p = psutil.Process(os.getpid())
        try: 
            rss = p.memory_info().rss
        except Exception as e:
            if args.raise_errors == True:
                raise(e)
            rss = p.memory_full_info().rss
            
        try: 
            tot = psutil.virtual_memory().total
        except Exception as e:
            if args.raise_errors == True:
                raise(e)
            tot = 0
        return int(rss), int(tot)
    # fallback
    try:
        import resource as _resource
        r = _resource.getrusage(_resource.RUSAGE_SELF).ru_maxrss
        rss = int(r if sys.platform == "darwin" else r * 1024)
    except Exception as e:
        if args.raise_errors == True:
            raise(e)
        rss = 0
    try:
        tot = int(os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES'))
    except Exception as e:
        if args.raise_errors == True:
            raise(e)
        tot = 0
    return rss, tot

def _bytes_h(n: float) -> str:
    units = ["B","KB","MB","GB","TB"]; i = 0
    while n >= 1024 and i < len(units)-1:
        n /= 1024.0; i += 1
    return f"{n:.2f} {units[i]}"

def _fmt_eta(elapsed_s: float, done: int, total: int) -> str:
    if done <= 0 or total <= 0: return "ETA: --:--:--"
    rate = done / max(elapsed_s, 1e-9)
    rem = max(total - done, 0)
    eta_s = rem / max(rate, 1e-9)
    h = int(eta_s // 3600); m = int((eta_s % 3600) // 60); s = int(eta_s % 60)
    return f"ETA: {h:02d}:{m:02d}:{s:02d}"

def _bucket_lists_bytes(bucket_traces: Dict[str, List[np.ndarray]]) -> int:
    return sum(getattr(arr, "nbytes", 0) for lst in bucket_traces.values() for arr in lst)

def _agg_bytes(aggs: Dict[str, OnlineBucket]) -> int:
    return sum(a.mean.nbytes + a.M2.nbytes + 8 for a in aggs.values())

def _print_status(prefix: str, start_t: float, done: int, total: int,
                  bucket_counts: Dict[str,int], mode: str,
                  bucket_mem_bytes: int, agg_mem_bytes: int,
                  warn_frac: float):
    elapsed = time.time() - start_t
    eta = _fmt_eta(elapsed, done, total)
    rss, tot = _proc_mem()
    pct = (done / total * 100.0) if total else 0.0
    b0 = bucket_counts.get("0-50", 0); b1 = bucket_counts.get("50-100", 0); b2 = bucket_counts.get("100-200", 0)
    msg = (f"{prefix} {done}/{total} ({pct:5.1f}%) | elapsed {elapsed:7.1f}s | {eta} | "
           f"RSS {_bytes_h(rss)}{f'/{_bytes_h(tot)}' if tot else ''} "
           f"| lists {_bytes_h(bucket_mem_bytes)} | aggs {_bytes_h(agg_mem_bytes)} | "
           f"mode={mode} | n[0-50]={b0} n[50-100]={b1} n[100-200]={b2}")
    if tot and rss / tot >= warn_frac:
        msg += "  [WARN: high memory]"
    print("\r" + msg, end="", flush=True)

# ---- Planning helpers ----
def resolve_stim_channel_idx(stim_name: str, ch_labels: List[str]):
    from ccep_onepass.preproc import normalize_label  # safe either mode due to bootstrap
    if not stim_name: return None, None
    stim_norm = normalize_label(stim_name)
    norm_to_idx, norm_to_original = {}, {}
    for i, lbl in enumerate(ch_labels):
        n = normalize_label(lbl)
        if n not in norm_to_idx:
            norm_to_idx[n] = i
            norm_to_original[n] = lbl
    if stim_norm in norm_to_idx:
        idx = norm_to_idx[stim_norm]
        return idx, norm_to_original[stim_norm]
    candidates = []
    for i, lbl in enumerate(ch_labels):
        n = normalize_label(lbl)
        if n.startswith(stim_norm) or stim_norm.startswith(n):
            candidates.append((i, lbl, n))
    if candidates:
        candidates.sort(key=lambda x: abs(len(x[2]) - len(stim_norm)))
        i, lbl, _ = candidates[0]
        return i, lbl
    return None, None



# ---- CLI ----
def _build_argparser():
    ap = argparse.ArgumentParser(description="One-pass CCEP aggregator from EDF (Spyder-friendly, cached prefilter, crash-safe).")

    # Inputs
    ap.add_argument("--edf", type=str, default=None, help="Path to a single EDF file.")
    ap.add_argument("--root", type=str, default=None, help="Root folder to search for EDF files (recursive).")
    ap.add_argument("--tsv",  type=str, default=None, help="Path to events TSV. Default: <edf_dir>/<edf_stem>_events_full.tsv")
    ap.add_argument("--currents", nargs="*", default=["5"], help='Currents to analyze, e.g., 1 3 5. Default: 5')


    # Montage / eligibility
    ap.add_argument("--disable-evoked", nargs="*", default=[], help="Exclude evoked bipolar labels (substring, case-insensitive).")
    ap.add_argument("--include-same-shaft", action="store_true", default=False, help="Include evoked on stim shaft.")
    # derived
    ap.add_argument("--exclude-same-shaft", action="store_true", default=True, help=argparse.SUPPRESS)

    # Aggregation / plotting
    ap.add_argument("--median-kernel", type=int, default=11)
    ap.add_argument("--detrend", action="store_true", default=True)
    ap.add_argument("--no-plot", action="store_true", default=False)
    ap.add_argument("--save-fig", action="store_true", default=True)

    # Output / status
    ap.add_argument("--outdir", type=Path, default=Path("output"))
    ap.add_argument("--quiet", action="store_true", default=False)
    
    ap.add_argument(
        "--prefilter-compression",
        choices=["none", "npz", "gz"],
        default="npz",   # set npz as the default if you want
        help="Per-channel cache format (none: .npy, npz: compressed, gz: .npy.gz)."
    )
    
    return ap

# ---- Spyder quick-start ----
if __name__ == "__main__":
    ap = _build_argparser()
    args = ap.parse_args(args=[] if "spyder" in sys.modules else None)

    # args.root = "d:/CCEP_CodeReview/CheckedCases/sub-P153/ses-001/"
    # # Example quick settings (edit as needed in Spyder):
    # args.sub_signature = "sub-153_ses-001"
    # args.edf = args.root + "sub-153_ses-001_task-ccep_run-01_ieeg_reduced.edf"
    # args.tsv = args.root + "sub-153_ses-001_task-ccep_run-01_events.tsv"
    # args.currents = ["3"]
    # args.outdir = Path(r"D:\ccep_codereview\output")
    # args.disable_evoked = ["EKG", "Pleth","SpO2","TRIG","PR","PatientEvent"]
    
        
    # args.disable_evoked 
    temp_list = ["EKG", "Pleth","SpO2","TRIG","PR","PatientEvent", "Photic", "ECG", "EMG", "EOG"]
    
    #add unnamed, unused channels to disable them
    for qctr in range(1,257):   
        el_name = f'C{qctr:.0f}'
        temp_list.append(el_name)
        
    #add DC channels to disable    
    for qctr in range(1,17):   
        el_name = f'DC{qctr:.0f}'
        temp_list.append(el_name)        
        
    args.disable_evoked = temp_list



    # make sure outdir is set and writable
    args.sub_str = sub_str
    args.sub_dir = sub_dir
    args.ses_str = ses_str
    args.ses_fld_str = ses_fld_str
    args.cache_dir = cache_dir
    args = update_args(sub_str,ses_str,args)
    args.prefilter_cache = enable_cache_setting
    
    
    args.outdir = Path(f"d:/CCEP_CodeReview/ccep_cache/{args.sub_signature}/")
    if os.path.exists(args.outdir) == False:
        os.makedirs(args.outdir)

    args.exclude_same_shaft = not args.include_same_shaft
    state_update(main_args=args)

    verbose = not args.quiet
    state_update(args=args)

    try:
        # Collect EDF files
        edf_files: List[Path] = []
        if args.edf is not None:
            edf_files = [Path(args.edf)]
        elif args.root is not None:
            edf_files = [Path(p) for p in glob.glob(str(Path(args.sub_dir) / "**" / "*.edf"), recursive=True)]
            edf_files.sort()

        state_update(edf_files=edf_files)
        if not edf_files:
            raise(Exception("[INFO] No EDF files found."))

        last_results = None

        for edf_path in edf_files:
            state_update(edf_path=str(edf_path))
            edf = open_edf(edf_path)
            try:
                ch_labels, sfreq = get_labels_and_fs(args,edf)
                state_update(fs=sfreq, ch_labels=ch_labels)

                # Events (TSV preferred)
                # ---- Events (TSV handling: follows args.tsv rules) ----
                #   1. If args.tsv is a file → use it for all EDFs.
                #   2. If args.tsv is a folder → look for matching *_events*.tsv inside it.
                #   3. If args.tsv is None → look next to EDF for *_events_full.tsv or *_events.tsv.
                #   4. If not found → fall back to EDF annotations.
                
                tsv_path = None
                if args.tsv is not None:
                    tsv_candidate = Path(args.tsv)
                    if tsv_candidate.is_file():
                        # explicit TSV file
                        tsv_path = tsv_candidate
                    elif tsv_candidate.is_dir():
                        # directory containing per-file TSVs
                        stem = edf_path.stem
                        candidates = sorted(tsv_candidate.glob(f"{stem}_events*.tsv"))
                        if candidates:
                            tsv_path = candidates[0]
                        else:
                            if verbose:
                                print(f"[WARN] No TSV found in {tsv_candidate} for {stem}, will use EDF annotations.")
                    else:
                        if verbose:
                            print(f"[WARN] Specified --tsv path not found: {tsv_candidate}")
                else:
                    # default fallback near EDF
                    local_candidates = sorted(edf_path.parent.glob(f"{edf_path.stem}_events*.tsv"))
                    if local_candidates:
                        tsv_path = local_candidates[0]
                
                state_update(tsv_path=str(tsv_path) if tsv_path else None)
                
                try:
                    annot_ts, annot_lb, src = read_events(
                        args, edf, tsv_path, prefer="tsv", fs=sfreq, verbose=verbose
                    )
                except Exception as e:
                    if args.raise_errors:
                        raise
                    print(f"[WARN] Event reading failed ({edf_path.name}): {e}")
                    annot_ts, annot_lb, src = np.array([]), [], "error"
                
                checkpoint("events", source=src, n=len(annot_ts))
                if verbose:
                    print(f"[INFO] {edf_path.name}: events source={src}, count={len(annot_ts)}, fs={sfreq:.0f} Hz")
                checkpoint("events", source=src, n=len(annot_ts))
                if verbose:
                    print(f"[INFO] {edf_path.name}: events={src}, fs={sfreq:.2f} Hz")

                stim_candidates = parse_all_stim_candidates(annot_lb)
                state_update(stim_candidates=stim_candidates)

                # Prefilter: load-or-compute
                filt_all = None
                if args.prefilter_cache:
                    cache_dir = resolve_cache_dir(args)
                    state_update(cache_dir=str(cache_dir))
                    
                    filt_all = try_load_prefilter_cache(args,cache_dir, mmap=args.prefilter_mmap)
                
                if filt_all is None:
                    if verbose:
                        print(f"[INFO] Prefiltering all channels (band {args.BANDPASS_HZ}, notch {args.LINE_FREQS_HZ})...")
                    if args.prefilter_cache:
                        filt_all = prefilter_and_cache_channels(args,
                            edf_reader=edf, sfreq=sfreq,
                            bandpass=args.BANDPASS_HZ, line_freqs=args.LINE_FREQS_HZ,
                            cache_dir=cache_dir,
                            ch_labels=ch_labels, verbose=verbose,
                            dtype=args.prefilter_dtype,
                            mmap=args.prefilter_mmap,                 # only applies if compression == "none"
                            compression=args.prefilter_compression    # <<<<< npz here
                        )
                    else:
                        filt_all = prefilter_all_channels(
                            args, edf, sfreq, args.BANDPASS_HZ, args.LINE_FREQS_HZ,
                            ch_labels=ch_labels, verbose=verbose, out_dtype=args.prefilter_dtype
                        )

                    checkpoint("prefilter_done", n_channels=len(filt_all))

                    if args.prefilter_cache:
                        cache_dir = resolve_cache_dir(args)
                        save_prefilter_cache(cache_dir, filt_all, ch_labels, sfreq, dtype=args.prefilter_dtype)


                ch_labels_sel = []
                for istr in args.grey_matter_selects:
                    # if istr in ch_labels and istr not in args.disable_evoked:
                    if istr not in args.disable_evoked:
                        ch_labels_sel.append(istr)

                # Bipolar montage
                filt_all_bp, bp_labels, bp_meta = apply_bipolar(filt_all, ch_labels_sel)
                state_update(bp_labels=bp_labels)
                bp_index = {lbl: i for i, lbl in enumerate(bp_labels)}
                


                # Evoked-eligible labels + disable patterns
                evoked_labels_all = bp_labels
                disabled_patterns = [s.lower() for s in (args.disable_evoked or [])]
                def _is_disabled(lbl: str) -> bool:
                    low = lbl.lower()
                    return any(pat in low for pat in disabled_patterns)
                disabled_evoked = [lbl for lbl in evoked_labels_all if _is_disabled(lbl)]
                evoked_labels   = [lbl for lbl in evoked_labels_all if not _is_disabled(lbl)]
                state_update(disabled_patterns=disabled_patterns,
                             disabled_evoked=disabled_evoked,
                             evoked_labels=evoked_labels)
                if verbose:
                    if disabled_patterns:
                        print(f"[INFO] Disable patterns: {disabled_patterns}")
                    if disabled_evoked:
                        shown = ", ".join(disabled_evoked[:20])
                        more  = f" … (+{len(disabled_evoked)-20} more)" if len(disabled_evoked) > 20 else ""
                        print(f"[INFO] Disabled evoked: {len(disabled_evoked)} → {shown}{more}")
                    print(f"[INFO] Evoked-eligible after disable filter: {len(evoked_labels)}")

                # Build work plan
                currents = [str(c) for c in (args.currents or [])]
                state_update(currents=currents)
                # if len(currents) == 0:
                #     raise(Exception("[INFO] No currents specified; nothing to do."))

                plan = []
                for curr in currents:
                    for stim in stim_candidates:
                        # decode events for (curr, stim)
                        events = []
                        try:
                            events = decode_events_rev1(annot_ts, annot_lb, curr, stim)
                        except Exception as e:
                            if args.raise_errors == True:
                                raise(e)
                            
                            events = []
                        if not events:
                            try:
                                events = decode_events_rev2(annot_ts, annot_lb, curr, stim)
                            except Exception as e:
                                if args.raise_errors == True:
                                    raise(e)
                                events = []
                        if not events:
                            continue
                        event_samples_abs = [int(e[0] * sfreq) for e in events]
                        stim_idx, matched_stim_lbl = resolve_stim_channel_idx(stim, ch_labels)

                        # candidate evoked after optional same-shaft exclusion
                        if args.exclude_same_shaft and matched_stim_lbl:
                            candidate_evoked = []
                            for lbl in evoked_labels:
                                i_bp = bp_index.get(lbl, None)
                                if i_bp is None: continue
                                pair_stem = bp_meta[i_bp]["stem_norm"]
                                if same_shaft_as_stim(pair_stem, matched_stim_lbl):
                                    continue
                                candidate_evoked.append(lbl)
                        else:
                            candidate_evoked = evoked_labels.copy()

                        if not candidate_evoked:
                            continue

                        plan.append({
                            "curr": curr,
                            "stim": stim,
                            "event_samples_abs": event_samples_abs,
                            "candidate_evoked": candidate_evoked,
                            "matched_stim_lbl": matched_stim_lbl
                        })

                checkpoint("plan", n_entries=len(plan))
                state_update(plan=plan)

                total_units = sum(len(p["candidate_evoked"]) for p in plan)
                if total_units == 0:
                    print("[INFO] No usable (stim,current,evoked) combinations found.")
                    continue

                # Accumulation structures
                canonical_t_ms: Optional[np.ndarray] = None
                bucket_traces: Dict[str, List[np.ndarray]] = {"0-50": [], "50-100": [], "100-200": []}
                bucket_aggs: Dict[str, OnlineBucket] = {}
                def ensure_agg(bucket_key: str, length: int):
                    if bucket_key not in bucket_aggs:
                        bucket_aggs[bucket_key] = OnlineBucket(length)

                # progress state
                start_time = time.time()
                last_status = 0.0
                done_units = 0
                bucket_counts = {"0-50": 0, "50-100": 0, "100-200": 0}

                # median kernel (odd)
                mk = int(args.median_kernel) if args.median_kernel is not None else 0
                if mk < 1: mk = 1
                if mk % 2 == 0: mk += 1
                enable_median = (args.median_kernel is not None) and (args.median_kernel >= 3)
                state_update(median_kernel=mk, enable_median=enable_median)

                # ---- EXECUTION ----
                print(f"[INFO] Planned units: {total_units}. Starting processing…")
                for item in plan:
                    curr = item["curr"]
                    stim = item["stim"]
                    event_samples_abs = item["event_samples_abs"]
                    candidate_evoked = item["candidate_evoked"]

                    for ev_label in candidate_evoked:
                        ev_idx = bp_index.get(ev_label, None)
                        if ev_idx is None or ev_idx >= len(filt_all_bp):
                            done_units += 1
                            continue

                        # Stim-aligned segments
                        # Per-trial, polarity-aligned, baseline-z epochs (stim-centered)
                        epochs_stim, canonical_t_ms, ttfp_list, flips = epochs_with_polarity(
                            args,
                            evoked_signal=filt_all_bp[ev_idx],        # your evoked channel signal
                            event_samples=event_samples_abs,
                            zscore_baseline=True,
                            primary_ms=(10.0, 50.0),
                            fallback_ms=(50.0, 200.0),
                        )

                        if not epochs_stim:
                            done_units += 1
                            continue

                        # Mean of *already aligned* epochs (no extra flip needed later)
                        y_stim_norm = np.mean(np.vstack(epochs_stim), axis=0).astype(np.float32)

                        valid_stim = [s for s in epochs_stim if s is not None and not np.all(np.isnan(s))]
                        if not valid_stim:
                            done_units += 1; continue

                        avg_stim = np.nanmean(np.vstack(valid_stim), axis=0).astype(float)

                        y_stim = avg_stim.copy()
                        if enable_median and len(y_stim) >= mk:
                            y_stim = medfilt(y_stim, kernel_size=mk)
                        if args.detrend:
                            try:
                                y_stim, _, _ = detrend_cubic(y_stim, max_curvature=1e-3)
                            except Exception as e:
                                if args.raise_errors == True:
                                    raise(e)

                        if canonical_t_ms is None:
                            n = len(y_stim)
                            canonical_t_ms = np.linspace(args.SEG_MS[0], args.SEG_MS[1], n)
                            state_update(t_ms=canonical_t_ms)

                        y_stim_norm = baseline_normalize(y_stim, canonical_t_ms, args.BASELINE_MS)
                        if y_stim_norm is None:
                            done_units += 1; continue

                        ttfp_for_bucket = first_peak_with_polarity(canonical_t_ms, y_stim_norm,
                                                                   search_ms=args.PEAK_SEARCH_MS,
                                                                   prominence=args.PEAK_PROMINENCE_Z)
                        bucket = None
                        if ttfp_for_bucket is not None and np.isfinite(ttfp_for_bucket):
                            if 0.0 < ttfp_for_bucket < 50.0: bucket = "0-50"
                            elif 50.0 <= ttfp_for_bucket < 100.0: bucket = "50-100"
                            elif 100.0 <= ttfp_for_bucket < 200.0: bucket = "100-200"

                        if bucket is None:
                            done_units += 1; continue

                        # Sync choice
                        if args.DEFAULT_SYNC_TO == "stim":
                            to_bucket = y_stim_norm
                        else:
                            # evoked-centered re-cut
                            ev_aligned = []
                            pre_samp   = int(args.SEG_MS[0] * sfreq / 1000.0)
                            post_samp  = int(args.SEG_MS[1] * sfreq / 1000.0)
                            b0_off = int(args.BASELINE_MS[0]*sfreq/1000.0)
                            b1_off = int(args.BASELINE_MS[1]*sfreq/1000.0)
                            zero_idx = -pre_samp

                            for stim_idx_abs, tt, flip in zip(event_samples_abs, ttfp_list, flips):
                                if tt is None or not np.isfinite(tt): 
                                    continue
                                ev_center = stim_idx_abs + int(round(float(tt) * sfreq / 1000.0))
                                start = ev_center + pre_samp-1
                                end   = ev_center + post_samp
                                if start < 0 or end > len(filt_all_bp[ev_idx]): 
                                    continue
                                seg = filt_all_bp[ev_idx][start:end].astype(np.float32)

                                # baseline z (local to this evoked-centered cut)
                                b0 = max(0, zero_idx + b0_off)
                                b1 = min(len(seg), zero_idx + b1_off)
                                base = seg[b0:b1]
                                m = float(np.mean(base)) if base.size else 0.0
                                s = float(np.std(base))  if base.size else 1.0
                                seg = (seg - m) / (s + 1e-9)

                                # **Apply the same trial's polarity decision**
                                if flip:
                                    seg = -seg

                                ev_aligned.append(seg)

                            if ev_aligned:
                                avg_evoked = np.nanmean(np.vstack(ev_aligned), axis=0).astype(float)
                                y_e = avg_evoked.copy()
                                if enable_median and len(y_e) >= mk:
                                    y_e = medfilt(y_e, kernel_size=mk)
                                if args.detrend:
                                    try:
                                        y_e, _, _ = detrend_cubic(y_e, max_curvature=1e-3)
                                    except Exception as e:
                                        if args.raise_errors == True:
                                            raise(e)
                                y_e_norm = baseline_normalize(y_e, canonical_t_ms, args.BASELINE_MS)
                                to_bucket = y_e_norm if y_e_norm is not None else y_stim_norm
                            else:
                                to_bucket = y_stim_norm

                        # Accumulate
                        if args.DEFAULT_ACCUM_MODE == "keep":
                            bucket_traces[bucket].append(to_bucket)
                        else:
                            if bucket not in bucket_aggs: ensure_agg(bucket, len(to_bucket))
                            bucket_aggs[bucket].add(to_bucket)
                        bucket_counts[bucket] += 1

                        # Status
                        done_units += 1
                        now = time.time()
                        if (now - last_status) >= args.DEFAULT_STATUS_INTERVAL_S or done_units == total_units:
                            lists_bytes = _bucket_lists_bytes(bucket_traces) if args.DEFAULT_ACCUM_MODE == "keep" else 0
                            agg_bytes = _agg_bytes(bucket_aggs) if bucket_aggs else 0
                            _print_status("[RUN]", start_time, done_units, total_units,
                                          bucket_counts, args.DEFAULT_ACCUM_MODE, lists_bytes, agg_bytes, args.DEFAULT_WARN_MEM_FRAC)
                            last_status = now

                print()  # newline

                if canonical_t_ms is None or (bucket_counts["0-50"] + bucket_counts["50-100"] + bucket_counts["100-200"]) == 0:
                    print(f"[INFO] {edf_path.name}: no usable traces were found.")
                    continue

                # Final stats
                groups = {}
                if args.DEFAULT_ACCUM_MODE == "keep":
                    for key in ["0-50","50-100","100-200"]:
                        mean, sem, std = compute_stats(bucket_traces[key])
                        n = len(bucket_traces[key])
                        groups[key] = {"n": n, "mean": mean, "sem": sem, "std": std}
                else:
                    for key in ["0-50","50-100","100-200"]:
                        if key in bucket_aggs:
                            mean, sem, std, n = bucket_aggs[key].finalize()
                            groups[key] = {"n": n, "mean": mean, "sem": sem, "std": std}
                        else:
                            groups[key] = {"n": 0, "mean": None, "sem": None, "std": None}

                # Save CSVs
                if args.outdir is not None:
                    Path(args.outdir).mkdir(parents=True, exist_ok=True)
                    for key in ["0-50","50-100","100-200"]:
                        g = groups[key]
                        if g["mean"] is None: continue
                        df = pd.DataFrame({
                            "time_ms": canonical_t_ms,
                            "mean": g["mean"],
                            "sem": g["sem"],
                            "std": g["std"],
                        })
                        df["n"] = g["n"]
                        csv_path = Path(args.outdir) / f"grand_average_{key.replace('-', '_')}__sync-{args.DEFAULT_SYNC_TO}.csv"
                        df.to_csv(csv_path, index=False)
                        if verbose: print(f"[SAVE] {csv_path}")

                # Plot
                if not args.no_plot:
                    plt.figure(figsize=(10, 6))
               
                    def _plot_one(k: str, label: str, plot_range, extension):
                        g = groups[k]
                        if g["mean"] is None:
                            return
                        mean = g["mean"]
                        disp = g["std"] if args.SHADE_ALPHA == "std" else (
                            g["sem"] if args.SHADE_ALPHA == "sem" else None
                        )
                        if args.SMOOTH_WIN_SAMPLES and args.SMOOTH_WIN_SAMPLES > 1:
                            mean_p = np.convolve(mean, np.ones(args.SMOOTH_WIN_SAMPLES) / args.SMOOTH_WIN_SAMPLES, mode="same")
                        else:
                            mean_p = mean
                
                        # Extend plot range by a safety margin
                        pr0, pr1 = plot_range
                        range_width = pr1 - pr0
                        pr0_ext = pr0 - extension * range_width
                        pr1_ext = pr1 + extension * range_width
                
                        # Create boolean mask for the extended range
                        focus_mask = (canonical_t_ms >= pr0_ext) & (canonical_t_ms <= pr1_ext)
                        early_mask = (canonical_t_ms >= 0) & (canonical_t_ms <= 100)
                        
                        
                        if not np.any(focus_mask):
                            return  # nothing in range
                
                        # Apply window (Hamming or Hanning)
                        window_len = np.count_nonzero(focus_mask)
                        if window_len < 5:
                            return
                        # choose type
                        window = signal.windows.hamming(window_len) if args.window_type == "hamming" else signal.windows.hann(window_len)
                
                        # Apply window to the focused signal segment
                        mean_p_win = mean_p.copy()
                        mean_p_win[focus_mask] = mean_p_win[focus_mask] * window
                
                        # Normalize (baseline and amplitude)
                        # mean_p_win = mean_p_win - np.nanmean(mean_p_win)
                        # mean_p_win = np.divide(mean_p_win,np.nanmax(np.abs(mean_p_win)))

                        # if np.abs(np.nanmax(mean_p_win[early_mask]) ) < np.abs(np.abs(np.nanmin(mean_p_win[early_mask]) )):
                            # mean_p_win *= -1
                
                
                        # Plot
                        plt.plot(canonical_t_ms, mean_p_win, label=f"{label} (n={g['n']})", linewidth=2)
                
                    # Example calls
                    _plot_one("0-50", "0–50 ms", [0, 200], extension=0.1)
                    _plot_one("50-100", "50–100 ms", [0, 200], extension=0.1)
                    _plot_one("100-200", "100–200 ms", [0, 200], extension=0.1)
                
                    plt.title(f"{edf_path.name}  |  sync={args.DEFAULT_SYNC_TO} | current = {args.currents}")
                    plt.xlabel("Time (ms)")
                    plt.ylabel("Normalized amplitude (z)")
                    plt.grid(True)
                    plt.xlim(*args.XLIM)
                    # plt.ylim(*args.YLIM)
                    plt.legend(loc="upper right")
                    plt.tight_layout()
                    if args.save_fig and args.outdir is not None:
                        fig_path = Path(args.outdir) / f"grand_averages_{edf_path.stem}__sync-{args.DEFAULT_SYNC_TO}.png"
                        plt.savefig(fig_path, dpi=300)
                        if verbose:
                            print(f"[SAVE] {fig_path}")
                    plt.show()


                # --- Pack results for Spyder workspace ---
                last_results = {
                    "edf_path": edf_path,
                    "fs": sfreq,
                    "ch_labels": ch_labels,
                    "bp_labels": bp_labels,
                    "disabled_evoked": disabled_evoked,
                    "evoked_labels": evoked_labels,
                    "plan": plan,
                    "groups": groups,
                    "t_ms": canonical_t_ms,
                    "bucket_counts": bucket_counts,
                    "filt_all_bp": filt_all_bp,  # comment if memory is tight
                }
                checkpoint("final", edf=str(edf_path), counts=bucket_counts)

            finally:
                edf.close()

    except Exception as e:
        if args.raise_errors == True:
            raise(e)
            
        # In Spyder, keep the console alive (don't re-raise)


