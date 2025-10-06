# -*- coding: utf-8 -*-
"""
Created on Wed Aug 27 12:59:16 2025

@author: Milad Khaki
"""

# CORRECT EVOKED POTENTIAL PLOTTING CODE

import numpy as np
import re


#%%

def get_channel_summary(ch_labels):
    electrodes = set([])
    for ind, ch_lbl in enumerate(ch_labels):
        
        tmp2 = [tmp if not tmp.isdigit() else "" for ind2,tmp in enumerate(ch_lbl)]
        tmp2 = "".join(tmp2)
        
        drop_str = list(map(lambda str:str.lower(),["C","DC","SpO","TRIG","PR","Pleth","Patient Event","DC."]))
        drop_set = set(drop_str)
        if tmp2.lower() in drop_set:
            continue
            
        electrodes.add(tmp2)
    return electrodes
    
# print("Electrodes :",get_channel_summary(channel_labels))

#%%
def decode_events_rev1(annot_timestamps, annot_labels, current, stimulated_channel):
    all_events = []
    current_pair = None
    in_block = False

    for time_stamp, label in zip(annot_timestamps, annot_labels):
        label = str(label).strip()

        # Relay closes -> define new pair, expect currents afterwards
        if label.startswith("Closed relay"):
            match = re.match(r"Closed relay to (\S+) and (\S+)", label)
            if match:
                current_pair = (match.group(1), match.group(2))
                in_block = True
            continue

        # Relay opened or de-block -> end the block
        if "Opened relay" in label or "De-block" in label:
            in_block = False
            current_pair = None
            continue

        # Numeric label inside an active block = current amplitude
        if in_block and label.isdigit():
            if label == str(current) and current_pair is not None:
                all_events.append((time_stamp, label, current_pair))

    # Filter for stimulated channel
    filtered = [(t, l, p) for (t, l, p) in all_events if p[0] == stimulated_channel]
    return filtered



# filtered_events = decode_events_rev1(annot_timestamps, annot_labels, current, stimulated_channel)
# print(filtered_events)
#%%
# -----------------------------------
# Revision 2 decoding (Start Stimulation)
# -----------------------------------
def decode_events_rev2(annot_timestamps, annot_labels, current, stimulated_channel):
    all_events = []
    for time_stamp, label in zip(annot_timestamps, annot_labels):

        current_pair = None

        if label.startswith("Start Stimulation"):
            match = re.match(r"Start Stimulation from (\S+) to (\S+)", label)
            if match:
                current_pair = (match.group(1), match.group(2))

        if current in label and current_pair is not None:
            all_events.append((time_stamp, label, current_pair))

    # Filter for desired stimulated channel
    filtered = [
        (t, l, p) for (t, l, p) in all_events if p[0] == stimulated_channel
    ]
    return filtered


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
    x = np.arange(len(signal))
    
    # Fit cubic polynomial
    coeffs = np.polyfit(x, signal, 3)
    
    # Constrain convexity (cubic coefficient)
    if abs(coeffs[0]) > max_curvature:
        coeffs[0] = np.sign(coeffs[0]) * max_curvature
    
    poly_fit = np.polyval(coeffs, x)
    detrended = signal - poly_fit
    return detrended, poly_fit, coeffs


