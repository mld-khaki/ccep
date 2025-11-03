# ccep_onepass/io.py
import os
import re
from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import pandas as pd
import pyedflib

def open_edf(edf_path: Path) -> pyedflib.EdfReader:
    return pyedflib.EdfReader(str(edf_path))

def get_labels_and_fs(args,edf: pyedflib.EdfReader) -> Tuple[List[str], float]:
    ch_labels = edf.getSignalLabels()
    # pick first "valid" channel for fs (most EDFs have uniform fs)
    for i in range(len(ch_labels)):
        try:
            fs = float(edf.getSampleFrequency(i))
            if fs > 0:
                return ch_labels, fs
        except Exception as e:
            if args.raise_errors == True:
                raise(e)
            pass
    raise RuntimeError("Could not determine sampling frequency from EDF.")

# -------------------- Events I/O --------------------
def _read_tsv_any(args,tsv_file: str) -> Optional[pd.DataFrame]:
    try:
        return pd.read_csv(tsv_file, sep="\t")
    except Exception as e:
        if args.raise_errors == True:
            raise(e)
        try:
            return pd.read_csv(tsv_file, sep=None, engine="python")
        except Exception as e:
            if args.raise_errors == True:
                raise(e)
            return None

def _lower_map(cols: List[str]):
    return {c.lower().strip(): c for c in cols}

def _pick(df: pd.DataFrame, names: List[str]) -> Optional[str]:
    lm = _lower_map(list(df.columns))
    for n in names:
        if n in lm:
            return lm[n]
    return None

def read_events(args,edf_reader: pyedflib.EdfReader,
                tsv_path: Path,
                prefer: str = "tsv",
                fs: Optional[float] = None,
                verbose: bool = False) -> Tuple[List[float], List[str], str]:
    """Return (timestamps_sec, labels, source_str). TSV preferred, EDF as fallback."""
    def _parse_tsv(tsv_file: str, fs_for_samples: Optional[float]):
        df = _read_tsv_any(args,tsv_file)
        if df is None or df.empty:
            if verbose: print(f"[EVENTS] TSV '{tsv_file}' not found or empty.")
            return None, None

        onset_col    = _pick(df, ["onset","time","t","latency","latency_s","latency_sec"])
        onset_ms_col = _pick(df, ["onset_ms","time_ms","latency_ms"])
        sample_col   = _pick(df, ["sample","sample_index","idx","index"])

        ts_sec: List[float] = []
        if onset_col is not None:
            ts_raw = df[onset_col].astype(str).str.replace("+","", regex=False).str.strip()
            ts_sec = [float(x) if x not in ("", "nan", "None") else np.nan for x in ts_raw]
            units = "seconds"
        elif onset_ms_col is not None:
            ts_raw = df[onset_ms_col].astype(str).str.replace("+","", regex=False).str.strip()
            ts_ms  = [float(x) if x not in ("", "nan", "None") else np.nan for x in ts_raw]
            ts_sec = [x/1000.0 if np.isfinite(x) else np.nan for x in ts_ms]
            units = "milliseconds→seconds"
        elif sample_col is not None and fs_for_samples and fs_for_samples > 0:
            ts_raw  = df[sample_col].astype(str).str.strip()
            samples = [float(x) if x not in ("", "nan", "None") else np.nan for x in ts_raw]
            ts_sec  = [x/float(fs_for_samples) if np.isfinite(x) else np.nan for x in samples]
            units = f"samples@{fs_for_samples:.1f}Hz→seconds"
        else:
            if verbose: print("[EVENTS] TSV has no recognizable onset column.")
            return None, None

        label_col = _pick(df, ["annotation","label","event","description","text","note"])
        stimA_col = _pick(df, ["stima","stim_a","stimsource","stim_src","stimsourcea"])
        stimB_col = _pick(df, ["stimb","stim_b","stimreturn","stim_dst","stimsink","stimreturnb"])

        if label_col is not None:
            labels = df[label_col].astype(str).tolist()
        elif (stimA_col is not None) and (stimB_col is not None):
            A = df[stimA_col].astype(str).str.strip().tolist()
            B = df[stimB_col].astype(str).str.strip().tolist()
            labels = [f"Start Stimulation from {a} to {b}" for a, b in zip(A, B)]
        else:
            labels = df.iloc[:, -1].astype(str).tolist()

        m = min(len(ts_sec), len(labels))
        ts, lb = [], []
        for i in range(m):
            v = ts_sec[i]
            if np.isfinite(v):
                ts.append(float(v))
                lb.append(str(labels[i]))

        if verbose:
            print(f"[EVENTS] TSV parsed: kept={len(ts)} units={units}")
        return ts, lb

    def _parse_edf(edf_reader: pyedflib.EdfReader):
        a = edf_reader.readAnnotations()
        return list(map(float, a[0])), list(map(str, a[2]))

    prefer = (prefer or "tsv").lower()
    tsv_exists = os.path.exists(str(tsv_path))

    if prefer == "tsv":
        if tsv_exists:
            ts, lb = _parse_tsv(str(tsv_path), fs)
            if ts is not None and lb is not None:
                if verbose: print("[EVENTS] Source selected: TSV")
                return ts, lb, "tsv"
            if verbose: print("[EVENTS] TSV unusable → falling back to EDF annotations.")
        ts, lb = _parse_edf(edf_reader)
        return ts, lb, "edf"

    if prefer == "edf":
        ts, lb = _parse_edf(edf_reader)
        return ts, lb, "edf"

    # auto
    if tsv_exists:
        ts, lb = _parse_tsv(str(tsv_path), fs)
        if ts is not None and lb is not None:
            if verbose: print("[EVENTS] Source selected: TSV (auto)")
            return ts, lb, "tsv"
    if verbose: print("[EVENTS] Source selected: EDF (auto fallback)")
    ts, lb = _parse_edf(edf_reader)
    return ts, lb, "edf"

def parse_all_stim_candidates(annot_labels: List[str]) -> List[str]:
    """Get stim source names from annotation strings."""
    stims = set()
    for s in map(str.strip, map(str, annot_labels)):
        m1 = re.match(r"Closed relay to (\S+) and (\S+)", s)
        if m1:
            stims.add(m1.group(1)); continue
        m2 = re.match(r"Start Stimulation from (\S+) to (\S+)", s)
        if m2:
            stims.add(m2.group(1)); continue
    return sorted(stims)

# Thin wrappers (prefer user’s ccep_lib if available)
def decode_events_rev1(annot_ts, annot_lb, curr, stim):
    """Parse 'Closed relay to A and B' blocks followed by numeric current labels.

    Returns list of tuples (timestamp_sec, label_str, (src,dst)).
    Filters to events where src==stim and label==curr (or curr==-1 for all).
    """
    events = []
    current_pair = None
    in_block = False
    curr_str = str(curr).strip()

    for t, lab in zip(annot_ts, annot_lb):
        lab = str(lab).strip()

        if lab.startswith("Closed relay"):
            import re as _re
            m = _re.match(r"Closed relay to (\S+) and (\S+)", lab)
            if m:
                current_pair = (m.group(1), m.group(2))
                in_block = True
            continue

        if ("Opened relay" in lab) or ("De-block" in lab):
            in_block = False
            current_pair = None
            continue

        if in_block and lab.isdigit():
            if curr_str == lab or curr_str == "-1":
                if current_pair is not None:
                    events.append((float(t), lab, current_pair))

    # keep only those from desired stim source
    return [e for e in events if e[2][0] == stim]


def decode_events_rev2(annot_ts, annot_lb, curr, stim):
    #%%
    """Parse 'Start Stimulation from A to B' blocks followed by numeric current labels.
    Accepts until 'Stop Stimulation', 'Opened relay', or 'De-block'.
    """
    import re as _re
    
    events = []
    current_pair = None
    in_block = False
    curr_str = str(curr).strip()
    
    for t, lab in zip(annot_ts, annot_lb):
        s = str(lab).strip()
    
        # Robust start match (allow extra text after the pair)
        m = _re.search(r"Start Stimulation from\s+(\S+)\s+to\s+(\S+)", s)
        if m:
            current_pair = (m.group(1), m.group(2))
            in_block = True
    
            # If capturing all currents (-1), record the start as an event too
            # so you don't miss timestamps when there are no numeric labels.
            if curr_str == "-1" and current_pair[0] == stim:
                events.append((float(t), "start", current_pair))
            continue
    
        # Only stop on explicit stop (don't close on De-block lines)
        if "Stop Stimulation" in s:
            in_block = False
            current_pair = None
            continue
    
        # Ignore De-block/Open transitions for rev2 (do NOT close the block)
        if "De-block" in s or "Opened relay" in s:
            continue
    
        if in_block:
            # Accept either standalone digits OR "I=1.0 mA" style lines
            mI = _re.search(r"I\s*=\s*([0-9]+(?:\.[0-9]+)?)\s*mA", s)
            if mI:
                label = mI.group(1)  # e.g., "1.0"
                # Normalize label so "1.0" and "1" match
                norm_label = str(int(float(label))) if float(label).is_integer() else label
            elif s.isdigit():
                norm_label = s
            else:
                continue
    
            # Filter by requested current
            if curr_str == "-1" or curr_str == norm_label or curr_str == f"{float(norm_label):.1f}":
                if current_pair and current_pair[0] == stim:
                    events.append((float(t), norm_label, current_pair))

#%%
    return [e for e in events if e[2][0] == stim]
