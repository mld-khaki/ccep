# ccep_onepass/preproc.py
"""
Preprocessing utilities for CCEP One-Pass pipeline.

Includes:
- Channel-wise prefilter (band-pass + multi-notch) with progress prints
- Robust label parsing / normalization helpers
- Bipolar montage creation (adjacent contacts)
- Evoked-channel eligibility helpers
- Per-channel cache save/load (one file per channel + meta.json)
  * Filenames now include the actual channel label:
      ch_002__LIns1.npy
    (sanitized; index always present for uniqueness)

Compression modes:
  - "none" (default):   .npy (fastest, supports np.memmap on load)
  - "gz":               .npy.gz (smaller on disk, no memmap)
  - "npz":              .npz with a single array inside (smaller, no memmap)
"""

from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from mne.filter import filter_data, notch_filter


# ----------------------------
# Label normalization & helpers
# ----------------------------

_REF_SUFFIXES = ("-ref", "_ref", " ref", "-gnd", "_gnd", " gnd", "-avg", "_avg", " avg")

def normalize_label(s: str) -> str:
    """
    Normalize a channel label: strip ref/gnd/avg suffixes, remove punctuation,
    remove zero-padding before trailing digits, and lowercase.

    Examples:
        "LA01-ref"  -> "la1"
        "RB_010 avg"-> "rb10"
    """
    s = (s or "").strip().lower()
    for suf in _REF_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    s = s.replace("_", "").replace("-", "").replace(".", "")
    s = re.sub(r'([a-z]+)0+(\d+)$', r'\1\2', s)
    return s


def parse_contact(label: str) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """
    Parse a label with trailing digits into (stem_norm, stem_raw_prefix, number).
    Returns (None, None, None) if unparsable.
    """
    raw = (label or "").strip()
    s = raw.lower().strip()
    for suf in _REF_SUFFIXES:
        if s.endswith(suf):
            s = s[: -len(suf)]
            break
    s = s.replace("_", "").replace("-", "").replace(".", "")
    m = re.match(r"^(.*?)(\d+)$", s)
    if not m:
        return None, None, None
    stem_norm = m.group(1)
    num = int(m.group(2))
    m_raw = re.match(r"^(.*?)(\d+)$", raw.replace("_", "").replace("-", "").replace(".", ""))
    stem_raw = m_raw.group(1) if m_raw else raw
    return stem_norm, stem_raw, num


def build_bipolar_pairs(ch_labels: List[str]) -> Tuple[List[Tuple[int,int,str,str]], List[str]]:
    """
    Build adjacent-contact (n, n+1) pairs per shaft (same normalized stem).
    Returns:
        pairs: list of (idx_lo, idx_hi, pair_label, stem_norm)
        pair_labels: list[str]
    """
    by_stem: Dict[str, List[Tuple[int,int,str,str]]] = {}
    for i, lbl in enumerate(ch_labels):
        stem_norm, stem_raw, num = parse_contact(lbl)
        if stem_norm is None or num is None:
            continue
        by_stem.setdefault(stem_norm, []).append((num, i, lbl, stem_raw))

    pairs = []
    pair_labels: List[str] = []
    for stem_norm, items in by_stem.items():
        items.sort(key=lambda x: x[0])  # sort by numeric suffix
        for k in range(len(items) - 1):
            n0, i0, lbl0, raw0 = items[k]
            n1, i1, lbl1, raw1 = items[k + 1]
            pair_label = f"{raw0}{n0}-{n1}"
            pairs.append((i0, i1, pair_label, stem_norm))
            pair_labels.append(pair_label)
    return pairs, pair_labels


def apply_bipolar(signals_per_channel: List[np.ndarray],
                  ch_labels: List[str]) -> Tuple[List[np.ndarray], List[str], List[Dict[str, object]]]:
    """
    Create adjacent-contact bipolars from single-ended channels.

    Returns:
        bp_signals: list[np.ndarray]
        bp_labels: list[str] (e.g., "LA1-2")
        bp_meta:   list[dict] with keys: idx_lo, idx_hi, stem_norm, pair_label
    """
    pairs, pair_labels = build_bipolar_pairs(ch_labels)
    bp_signals: List[np.ndarray] = []
    bp_meta: List[Dict[str, object]] = []

    for (i_lo, i_hi, pair_label, stem_norm) in pairs:
        sig_hi = signals_per_channel[i_hi]
        sig_lo = signals_per_channel[i_lo]
        L = min(len(sig_hi), len(sig_lo))
        bp = (sig_hi[:L] - sig_lo[:L]).astype(np.float64, copy=False)
        bp_signals.append(bp)
        bp_meta.append({
            "idx_lo": i_lo,
            "idx_hi": i_hi,
            "stem_norm": stem_norm,
            "pair_label": pair_label
        })
    return bp_signals, pair_labels, bp_meta


def same_shaft_as_stim(pair_stem_norm: str, stim_label: str) -> bool:
    """
    Whether bipolar pair stem matches stim contact stem (exclude if True).
    """
    stim_stem_norm, _, _ = parse_contact(stim_label or "")
    return (stim_stem_norm is not None) and (pair_stem_norm == stim_stem_norm)


# ----------------------------
# Filename helpers
# ----------------------------

def _sanitize_for_filename(label: str) -> str:
    """
    Make a label filesystem-friendly: ASCII only, remove spaces, keep [A-Za-z0-9-_].
    Dots and other punctuation removed to avoid confusion.
    """
    s = unicodedata.normalize("NFKD", str(label or "")).encode("ascii", "ignore").decode("ascii")
    s = s.replace(" ", "")
    s = re.sub(r"[^A-Za-z0-9\-_]+", "", s)
    if not s:
        s = "unnamed"
    # cap length to avoid Windows path issues
    return s[:80]


def _channel_filename(idx: int, label: Optional[str], compression: str) -> str:
    """
    Build file name for one channel given index, label, and compression mode.
    """
    lab = _sanitize_for_filename(label or f"ch{idx}")
    if compression == "gz":
        return f"ch_{idx:03d}__{lab}.npy.gz"
    elif compression == "npz":
        return f"ch_{idx:03d}__{lab}.npz"
    else:
        return f"ch_{idx:03d}__{lab}.npy"


# ----------------------------
# Prefilter (channel-by-channel)
# ----------------------------

def prefilter_all_channels(args,edf_reader,
                           sfreq: float,
                           bandpass: Tuple[float,float],
                           line_freqs: List[float],
                           ch_labels: Optional[List[str]] = None,
                           verbose: bool = False,
                           out_dtype: str = "float32") -> List[np.ndarray]:
    """
    Stream + filter each channel, return list of arrays with desired dtype.
    Prints progress per channel for Spyder.

    Returns list[np.ndarray]
    """
    n_ch = int(edf_reader.signals_in_file)
    out: List[np.ndarray] = []

    for idx in range(n_ch):
        label = None
        try:
            if ch_labels is not None and idx < len(ch_labels):
                label = str(ch_labels[idx])
        except Exception as e:
            if args.raise_errors == True:
                raise(e)
            label = None
        label = label or f"ch{idx}"

        if verbose:
            print(f"[PREF] {idx+1:03d}/{n_ch:03d} | {label} | band-pass + notch …", flush=True)

        # read one channel
        x = edf_reader.readSignal(idx).astype(np.float64, copy=False)

        # band-pass
        x = filter_data(
            x, sfreq=sfreq,
            l_freq=bandpass[0], h_freq=bandpass[1],
            fir_design='firwin', fir_window='hamming',
            copy=False, verbose=False
        )
        # multi-notch
        x = notch_filter(
            x, Fs=sfreq, freqs=line_freqs,
            notch_widths=6.0, fir_design='firwin',
            copy=False, verbose=False
        )

        if out_dtype == "float32":
            x = x.astype(np.float32, copy=False)
        else:
            x = x.astype(np.float64, copy=False)

        out.append(x)

    return out


# ----------------------------
# Prefilter cache (per channel)
# ----------------------------

def resolve_cache_dir(args):
    ret_dir = args.cache_dir + f"/{args.sub_str}_{args.ses_str}/prefilter_cache/cache__fs{args.sampling_freq:.0f}_{args.prefilter_dtype}_{args.prefilter_compression}/"
    return ret_dir



def save_prefilter_cache(cache_dir: Path | str,
                         filt_all: List[np.ndarray],
                         ch_labels: Optional[List[str]],
                         sfreq: float,
                         dtype: str = "float32",
                         compression: str = "none") -> None:
    """
    Save per-channel files + meta.json into cache_dir.
    compression: "none" -> .npy, "gz" -> .npy.gz, "npz" -> .npz
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    labels_out = list(map(str, ch_labels or []))
    n_ch = len(filt_all)

    meta = {
        "fs": float(sfreq),
        "dtype": dtype,
        "n_channels": int(n_ch),
        "labels": labels_out,
        "format": "per-channel-v2",
        "compression": compression,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    for i, x in enumerate(filt_all):
        arr = np.asarray(x, dtype=np.float32 if dtype == "float32" else np.float64)
        fname = _channel_filename(i, labels_out[i] if i < len(labels_out) else None, compression)
        fpath = cache_dir / fname
        if compression == "gz":
            import gzip
            with gzip.open(fpath, "wb") as f:
                np.save(f, arr)
        elif compression == "npz":
            # store a single array with key "x"
            np.savez_compressed(fpath, x=arr)
        else:
            np.save(fpath, arr)

    print(f"[CACHE SAVE] Wrote {n_ch} channels to {cache_dir} (compression={compression})")


def try_load_prefilter_cache(args,cache_dir: Path | str,
                             mmap: bool = False) -> Optional[List[np.ndarray]]:
    """
    Load per-channel files + meta.json from cache_dir.

    Returns:
        list[np.ndarray] (memmap only if compression=="none") or None if missing.
    Supports both new filenames (ch_###__LABEL.ext) and legacy ch_###.npy.
    """
    cache_dir = Path(cache_dir)
    meta_path = cache_dir / "meta.json"
    if not meta_path.exists():
        return None

    try:
        meta = json.loads(meta_path.read_text())
        n = int(meta.get("n_channels", 0))
        labels = list(map(str, meta.get("labels", [])))
        compression = str(meta.get("compression", "none")).lower()

        if n <= 0:
            return None

        files = []
        for i in range(n):
            lab = labels[i] if i < len(labels) else None
            # preferred (new) filename with label
            fname_new = _channel_filename(i, lab, compression)
            f_new = cache_dir / fname_new

            if f_new.exists():
                files.append(f_new)
                continue

            # legacy fallback (no label, .npy only)
            f_legacy = cache_dir / f"ch_{i:03d}.npy"
            if f_legacy.exists():
                files.append(f_legacy)
                continue

            # As a last resort, try any known extension without label
            try_exts = {
                "none": cache_dir / f"ch_{i:03d}.npy",
                "gz":   cache_dir / f"ch_{i:03d}.npy.gz",
                "npz":  cache_dir / f"ch_{i:03d}.npz",
            }
            f_try = try_exts.get(compression)
            if f_try and f_try.exists():
                
                files.append(f_try)
                continue

            # missing file
            return None

        out: List[np.ndarray] = []
        for cnt,f in enumerate(files):
            print(f"[CACHE LOAD, ({cnt*100/len(labels):.2f}%, ({cnt} of {len(labels)} channels))] Loading channel {labels[cnt]} from cache",end="")
            if f.suffix == ".npz":
                with np.load(f) as z:
                    print(" (npz)")
                    out.append(z["x"])
            elif f.suffixes[-2:] == [".npy", ".gz"] or f.suffix == ".gz":
                import gzip
                print(f"Reading the npz file <{f}>...",end="")
                with gzip.open(f, "rb") as gf:
                    out.append(np.load(gf, allow_pickle=False))
                print("done!")
            else:
                # plain .npy (only case where mmap is supported)
                print(" (npy)")
                out.append(np.load(f, mmap_mode=("r" if mmap else None)))

        return out

    except Exception as e:
        if args.raise_errors == True:
            raise(e)
        return None


def prefilter_and_cache_channels(args,edf_reader,
                                 sfreq: float,
                                 bandpass: Tuple[float,float],
                                 line_freqs: List[float],
                                 cache_dir: Path | str,
                                 ch_labels: Optional[List[str]] = None,
                                 verbose: bool = False,
                                 dtype: str = "float32",
                                 mmap: bool = False,
                                 compression: str = "none") -> List[np.ndarray]:
    """
    Stream-filter each channel and save immediately to per-channel files in cache_dir.
    Also writes meta.json up-front. Returns list of arrays (np.memmap only if compression=='none' and mmap=True).
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n_ch = int(edf_reader.signals_in_file)
    labels_out = list(map(str, ch_labels or []))

    meta = {
        "fs": float(sfreq),
        "dtype": dtype,
        "n_channels": int(n_ch),
        "labels": labels_out,
        "format": "per-channel-v2",
        "compression": compression,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    out: List[np.ndarray] = []

    for idx in range(n_ch):
        label = None
        try:
            if ch_labels is not None and idx < len(ch_labels):
                label = str(ch_labels[idx])
        except Exception as e:
            if args.raise_errors == True:
                raise(e)

            label = None
        label = label or f"ch{idx}"

        if verbose:
            print(f"[PREF] {idx+1:03d}/{n_ch:03d} | {label} | band-pass + notch …", flush=True)

        # read one channel
        x = edf_reader.readSignal(idx).astype(np.float64, copy=False)

        # band-pass
        x = filter_data(
            x, sfreq=sfreq,
            l_freq=bandpass[0], h_freq=bandpass[1],
            fir_design='firwin', fir_window='hamming',
            copy=False, verbose=False
        )
        # multi-notch
        x = notch_filter(
            x, Fs=sfreq, freqs=line_freqs,
            notch_widths=6.0, fir_design='firwin',
            copy=False, verbose=False
        )

        x = x.astype(np.float32 if dtype == "float32" else np.float64, copy=False)

        # save immediately
        fname = _channel_filename(idx, label, compression)
        fpath = cache_dir / fname
        if compression == "gz":
            import gzip
            with gzip.open(fpath, "wb") as f:
                np.save(f, x)
        elif compression == "npz":
            np.savez_compressed(fpath, x=x)
        else:
            np.save(fpath, x)

        if verbose:
            print(f"[CACHE SAVE] {fpath}", flush=True)

        # load back for downstream use
        if compression == "npz":
            with np.load(fpath) as z:
                x_loaded = z["x"]
        elif compression == "gz":
            import gzip
            with gzip.open(fpath, "rb") as gf:
                x_loaded = np.load(gf, allow_pickle=False)
        else:
            x_loaded = np.load(fpath, mmap_mode=("r" if mmap else None))

        out.append(x_loaded)

    print(f"[CACHE SAVE] Completed {n_ch} channels → {cache_dir} (compression={compression})")
    return out


# ----------------------------
# Public API
# ----------------------------

__all__ = [
    "normalize_label",
    "parse_contact",
    "build_bipolar_pairs",
    "apply_bipolar",
    "same_shaft_as_stim",
    "prefilter_all_channels",
    "resolve_cache_dir",
    "save_prefilter_cache",
    "try_load_prefilter_cache",
    "prefilter_and_cache_channels",
]
