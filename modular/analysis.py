# ccep_onepass/analysis.py
from typing import List, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks
# ---- Polarity alignment (per-trial, before averaging) ----
from typing import List, Tuple, Optional
import numpy as np
from scipy.signal import find_peaks

def _decide_flip(yz: np.ndarray, t_ms: np.ndarray,
                 win: Tuple[float, float],
                 prom: float,
                 min_area_z: float = 0.0) -> Optional[bool]:
    """
    Decide whether to flip the epoch so the *earliest significant* deflection
    in `win` is positive.
      returns True  -> flip (epoch * -1)
              False -> keep
              None  -> no decision (not enough evidence)
    """
    m = (t_ms >= win[0]) & (t_ms <= win[1])
    if not np.any(m): 
        return None
    seg = yz[m]
    if seg.size < 3:
        return None

    # Prominence-based peaks first
    p_pos, prop_pos = find_peaks(seg, prominence=prom)
    p_neg, prop_neg = find_peaks(-seg, prominence=prom)

    candidates = []
    for p, prop, sign in ((p_pos, prop_pos, +1), (p_neg, prop_neg, -1)):
        if len(p):
            idx0 = int(p[0])
            prom0 = float(prop["prominences"][0]) if "prominences" in prop else 0.0
            candidates.append((idx0, sign, prom0))
    if candidates:
        candidates.sort(key=lambda x: (x[0], -x[2]))  # earliest, then higher prominence
        _, sign, _ = candidates[0]
        return True if sign < 0 else False  # flip if earliest is negative

    # Fallback: signed area
    area = float(np.nansum(seg))
    if abs(area) >= min_area_z:
        return True if area < 0 else False

    # Last fallback: |min| vs |max|
    mx = float(np.nanmax(seg))
    mn = float(np.nanmin(seg))
    if not np.isfinite(mx) or not np.isfinite(mn):
        return None
    if max(abs(mn), abs(mx)) == 0.0:
        return None
    return True if abs(mn) > abs(mx) else False


def epochs_with_polarity(args,
                         evoked_signal: np.ndarray,
                         event_samples: List[int],
                         zscore_baseline: bool = True,
                         primary_ms: Tuple[float,float] = (10.0, 50.0),
                         fallback_ms: Tuple[float,float] = (50.0, 200.0),
                         prominence_z: Optional[float] = None):
    """
    Cut epochs per event, baseline-z-score each, decide polarity in N1 window
    with N2 fallback, flip BEFORE averaging, and compute TTFP on the aligned epochs.

    Returns
    -------
    epochs : list[np.ndarray]       # flipped-aligned, z-scored epochs
    t_ms   : np.ndarray             # canonical epoch timebase (ms)
    ttfp_list : list[Optional[float]]   # first-peak latencies for bucketing
    flips : list[bool]              # True if that epoch was flipped
    """
    fs = args.sampling_freq
    pre_samp   = int(args.SEG_MS[0]    * fs / 1000.0)  # negative
    post_samp  = int(args.SEG_MS[1]    * fs / 1000.0)
    base_start = int(args.BASELINE_MS[0]* fs / 1000.0)
    base_end   = int(args.BASELINE_MS[1]* fs / 1000.0)

    L = post_samp - pre_samp + 1
    t_samples = np.arange(pre_samp, post_samp + 1, dtype=int)
    t_ms = t_samples * (1000.0 / fs)

    epochs, flips, ttfp_list = [], [], []
    prom = float(args.PEAK_PROMINENCE_Z) if (prominence_z is None) else float(prominence_z)

    N = len(evoked_signal)
    zero_idx = -pre_samp
    b0 = zero_idx + base_start
    b1 = zero_idx + base_end

    for ev_abs in event_samples:
        start = ev_abs + pre_samp
        stop  = ev_abs + post_samp + 1
        if start < 0 or stop > N:
            continue
        y = evoked_signal[start:stop].astype(np.float64, copy=False)
        if y.shape[0] != L:
            continue

        # Baseline z
        if zscore_baseline:
            if not (0 <= b0 < L and 0 <= b1 < L and b1 > b0):
                continue
            bseg = y[b0:b1+1]
            bmean = np.nanmean(bseg); bstd = np.nanstd(bseg)
            if not np.isfinite(bmean) or not np.isfinite(bstd) or bstd == 0:
                continue
            yz = (y - bmean) / (bstd + 1e-9)
        else:
            yz = y.copy()

        flip = _decide_flip(yz, t_ms, primary_ms, prom)
        if flip is None:
            flip = _decide_flip(yz, t_ms, fallback_ms, prom)
        if flip is None:
            flip = False

        y_aligned = (-yz if flip else yz)
        epochs.append(y_aligned)
        flips.append(bool(flip))

        # TTFP for bucketing (on aligned epoch)
        ttfp = first_peak_with_polarity(t_ms, y_aligned,
                                        search_ms=args.PEAK_SEARCH_MS,
                                        prominence=prom)
        ttfp_list.append(ttfp)

    return epochs, t_ms, ttfp_list, flips


# ---- Segments & measures ----
def segments_from_evoked(args,evoked_signal: np.ndarray,
                         event_samples: List[int],
                         zscore_baseline=True):
    fs = args.sampling_freq
    pre_samp   = int(args.SEG_MS[0]    * fs / 1000.0)  # negative
    post_samp  = int(args.SEG_MS[1]    * fs / 1000.0)
    base_start = int(args.BASELINE_MS[0]* fs / 1000.0)
    base_end   = int(args.BASELINE_MS[1]* fs / 1000.0)
    peak_start = int(args.PEAK_SEARCH_MS[0]*fs / 1000.0)
    peak_end   = int(args.PEAK_SEARCH_MS[1]*fs / 1000.0)

    segments, ttpf_list = [], []
    zero_idx = -pre_samp

    for stim_idx in event_samples:
        start = stim_idx + pre_samp
        end   = stim_idx + post_samp
        if start < 0 or end > len(evoked_signal):
            continue
        seg = evoked_signal[start:end].astype(np.float32)

        if zscore_baseline:
            b0 = max(0, zero_idx + base_start)
            b1 = min(len(seg), zero_idx + base_end)
            base = seg[b0:b1]
            m = float(np.mean(base))  if base.size else 0.0
            s = float(np.std(base))   if base.size else 1.0
            seg = (seg - m) / (s + 1e-9)

        pw0 = max(0, zero_idx + peak_start)
        pw1 = min(len(seg), zero_idx + peak_end)
        search = seg[pw0:pw1]
        if search.size == 0:
            segments.append(seg); ttpf_list.append(np.nan); continue
        pos_peaks, pos_props = find_peaks(search, prominence=args.PEAK_PROMINENCE_Z)
        neg_peaks, neg_props = find_peaks(-search, prominence=args.PEAK_PROMINENCE_Z)
        candidates = []
        if len(pos_peaks): candidates.append((pos_peaks[0], pos_props["prominences"][0]))
        if len(neg_peaks): candidates.append((neg_peaks[0], neg_props["prominences"][0]))
        if candidates:
            candidates.sort(key=lambda x: x[0])
            first_t = candidates[0][0]
            tt_ms = (peak_start + first_t) * 1000.0 / fs
        else:
            tt_ms = np.nan
        segments.append(seg)
        ttpf_list.append(tt_ms)
    return segments, ttpf_list

def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    w = max(1, int(w))
    if w == 1: return x
    return np.convolve(x, np.ones(w)/w, mode="same")

def baseline_normalize(y: np.ndarray, t_ms: np.ndarray, baseline_ms: Tuple[float,float]) -> Optional[np.ndarray]:
    mask = (t_ms >= baseline_ms[0]) & (t_ms <= baseline_ms[1])
    if not np.any(mask): return None
    b = y[mask]
    bmean = np.nanmean(b); bstd = np.nanstd(b)
    if not np.isfinite(bmean) or not np.isfinite(bstd) or bstd == 0: return None
    return (y - bmean) / (bstd + 1e-9)

def first_peak_with_polarity(t_ms: np.ndarray, y: np.ndarray,
                             search_ms: Tuple[float,float], prominence: float) -> Optional[float]:
    mask = (t_ms >= search_ms[0]) & (t_ms <= search_ms[1])
    if not np.any(mask): return None
    ym = y[mask]
    p_pos, prop_pos = find_peaks(ym, prominence=prominence)
    max_pos = prop_pos["prominences"].max() if len(p_pos) else -np.inf
    p_neg, prop_neg = find_peaks(-ym, prominence=prominence)
    max_neg = prop_neg["prominences"].max() if len(p_neg) else -np.inf
    if max_pos == -np.inf and max_neg == -np.inf: return None
    p_use = p_neg if max_neg > max_pos else p_pos
    first_local_idx = int(p_use[0])
    first_global_idx = np.flatnonzero(mask)[0] + first_local_idx
    return float(t_ms[first_global_idx])

def bucket_from_ttfp(ttfp_ms: float) -> Optional[str]:
    if ttfp_ms is None or not np.isfinite(ttfp_ms): return None
    if 0.0 < ttfp_ms < 50.0:  return "0-50"
    if 50.0 <= ttfp_ms < 100.0: return "50-100"
    if 100.0 <= ttfp_ms < 200.0: return "100-200"
    return None

# ---- Stats (Welford) ----
class OnlineBucket:
    def __init__(self, length: int):
        self.n = 0
        self.mean = np.zeros(length, dtype=np.float64)
        self.M2 = np.zeros(length, dtype=np.float64)
    def add(self, x: np.ndarray):
        x = x.astype(np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2
    def finalize(self):
        if self.n == 0:
            return None, None, None, 0
        if self.n == 1:
            std = np.zeros_like(self.mean)
        else:
            std = np.sqrt(self.M2 / (self.n - 1))
        sem = std / np.sqrt(self.n) if self.n > 1 else np.zeros_like(std)
        return self.mean.copy(), sem, std, self.n

def compute_stats(traces: List[np.ndarray]):
    if not traces:
        return None, None, None
    G = np.vstack(traces)
    mean = np.nanmean(G, axis=0)
    std  = np.nanstd(G, axis=0, ddof=1) if G.shape[0] > 1 else np.zeros_like(mean)
    sem  = std / np.sqrt(G.shape[0]) if G.shape[0] > 1 else np.zeros_like(mean)
    return mean, sem, std

def detrend_cubic(signal, max_curvature=1e-3):
    """
    Remove a cubic trend from a signal with optional curvature constraint.

    Parameters
    ----------
    signal : array-like
        Input signal.
    max_curvature : float
        Maximum allowed absolute cubic coefficient (convexity constraint).
        Smaller values => flatter fit.

    Returns
    -------
    detrended : ndarray
        Signal after cubic detrend.
    poly_fit : ndarray
        The fitted polynomial values (trend removed).
    coeffs : ndarray
        Polynomial coefficients [a3, a2, a1, a0].
    """
    import numpy as _np
    x = _np.arange(len(signal))
    coeffs = _np.polyfit(x, signal, 3)
    if abs(coeffs[0]) > max_curvature:
        coeffs[0] = _np.sign(coeffs[0]) * max_curvature
    poly_fit = _np.polyval(coeffs, x)
    detrended = signal - poly_fit
    return detrended, poly_fit, coeffs
