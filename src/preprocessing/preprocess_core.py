"""
Optimized EEG preprocessing pipeline for graph-based epilepsy detection.
Based on Neuro-GPT (arXiv 2311.03764) with modifications for connectivity analysis.

Key features:
- Spherical spline interpolation for missing channels (not zero-padding)
- Linear detrending for MVAR model stationarity
- No ICA to preserve epileptiform activity
- 98th percentile artifact rejection
"""

import re
import pickle
import numpy as np
import mne
from pathlib import Path
from typing import Dict, Optional, Tuple, List

# Fixed 10-20 core channel layout (extended international system)
# Same 22 channels as Neuro-GPT paper
CORE_CHS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
    "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"
]


def clean_channel_names(raw: mne.io.BaseRaw) -> None:
    """
    Standardize channel names to match CORE_CHS format.
    
    Removes prefixes like 'EEG ', suffixes like '-LE'/'-REF',
    and corrects capitalization (e.g., 'FP1' -> 'Fp1').
    
    Args:
        raw: MNE Raw object (modified in-place)
    """
    # Step 1: Remove common prefixes and suffixes
    mapping = {
        orig: re.sub(r'^(?:EEG\s*)', '', orig)
        .replace('-LE', '')
        .replace('-REF', '')
        .strip()
        for orig in raw.ch_names
    }
    raw.rename_channels(mapping)
    
    # Step 2: Fix capitalization to match CORE_CHS
    core_ch_map = {ch.lower(): ch for ch in CORE_CHS}
    case_mapping = {}
    for ch in raw.ch_names:
        if ch.lower() in core_ch_map:
            case_mapping[ch] = core_ch_map[ch.lower()]
    
    if case_mapping:
        raw.rename_channels(case_mapping)


def set_montage_for_corechs(raw: mne.io.BaseRaw) -> None:
    """
    Assign 10-20 electrode positions to channels.
    
    Uses standard_1020 montage with custom positions for T1/T2
    (mastoid electrodes, approximated as FT9/FT10).
    
    Args:
        raw: MNE Raw object (modified in-place)
    """
    # Load standard 10-20 montage
    std = mne.channels.make_standard_montage('standard_1020')
    pos = std.get_positions()['ch_pos'].copy()
    
    ch_pos = {}
    
    # Add coordinates for channels present in raw
    for ch in raw.ch_names:
        if ch in pos:
            ch_pos[ch] = pos[ch]
    
    # Add approximate mastoid positions (T1/T2)
    # These are based on FT9/FT10 positions
    if 'T1' in raw.ch_names:
        ch_pos.setdefault('T1', (-0.0840759, 0.0145673, -0.050429))
    if 'T2' in raw.ch_names:
        ch_pos.setdefault('T2', (0.0841131, 0.0143647, -0.050538))
    
    # Create and set montage
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, match_case=False, on_missing='ignore')


def detect_dead_channels(raw: mne.io.BaseRaw, std_threshold: float = 0.1) -> List[str]:
    """
    Detect channels with abnormally low variance (likely disconnected).
    
    Args:
        raw: MNE Raw object
        std_threshold: Minimum standard deviation in microvolts
        
    Returns:
        List of channel names with near-zero variance
    """
    dead_chs = []
    
    for ch in raw.ch_names:
        data = raw.get_data(picks=[ch])
        std_uv = np.std(data) * 1e6  # Convert to microvolts
        
        if std_uv < std_threshold:
            dead_chs.append(ch)
    
    return dead_chs


def infer_label_from_path(p: Path) -> int:
    """
    Infer binary label from file path.
    
    Args:
        p: Path to EDF file
        
    Returns:
        1 if epilepsy (path contains '00_epilepsy'), 0 otherwise
    """
    s = str(p).replace('\\', '/').lower()
    return 1 if "/00_epilepsy/" in s else 0


def preprocess_single(
    edf_path: Path,
    notch: float = 60.0,
    band: Tuple[float, float] = (0.5, 100.0),
    resample_hz: float = 250.0,
    epoch_len: float = 2.0,
    epoch_overlap: float = 0.0,
    reject_percentile: float = 98.0,
    reject_cap_uv: float = None,
    return_psd: bool = False,
    verbose: bool = True
) -> Optional[Dict]:
    """
    End-to-end preprocessing for epilepsy detection with connectivity analysis.
    
    Pipeline (optimized for DTF/PDC):
    1. Load EDF, pick EEG channels, clean names
    2. Identify present/missing CORE_CHS channels
    3. Add missing channels as flat signals (temporary)
    4. Set montage (electrode positions)
    5. Mark bad channels (missing + dead)
    6. INTERPOLATE bad channels (spherical spline)
    7. Notch filter (power line noise)
    8. Bandpass filter (0.5-100 Hz)
    9. Resample (250 Hz)
    10. Common average reference (CAR)
    11. Linear detrend (essential for MVAR stationarity)
    12. Epoch (2-second windows)
    13. Artifact rejection (98th percentile + cap)
    14. Z-score normalization (per-epoch, per-channel)
    
    Args:
        edf_path: Path to EDF file
        notch: Notch filter frequency (Hz) for power line noise
        band: (low, high) bandpass filter frequencies (Hz)
        resample_hz: Target sampling rate (Hz)
        epoch_len: Epoch duration (seconds)
        epoch_overlap: Overlap between epochs (seconds)
        reject_percentile: Percentile for adaptive rejection threshold
        reject_cap_uv: Hard cap for rejection threshold (microvolts)
        return_psd: Whether to compute and return PSD
        verbose: Whether to print progress messages
        
    Returns:
        Dictionary with keys:
        - 'raw_after': Preprocessed Raw object
        - 'epochs': Preprocessed Epochs object
        - 'labels': Per-epoch labels (0=control, 1=epilepsy)
        - 'threshold_uv': Artifact rejection threshold used
        - 'present_channels': List of 22 channel names (always CORE_CHS)
        - 'present_mask': Boolean array (True=originally present, False=interpolated)
        - 'psd_before': PSD before preprocessing (if return_psd=True)
        - 'psd_after': PSD after preprocessing (if return_psd=True)
        
        Returns None if preprocessing fails.
    """
    edf_path = Path(edf_path)
    pid = edf_path.stem
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Processing: {pid}")
        print(f"{'='*60}")
    
    # ========== 1. LOAD DATA ==========
    try:
        raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    except Exception as e:
        if verbose:
            print(f"✗ Failed to load {pid}: {e}")
        return None
    
    # Pick only EEG channels (exclude EOG, ECG, EMG, etc.)
    raw.pick_types(eeg=True, exclude=['EOG', 'ECG', 'EMG', 'MISC', 'STIM'])
    
    if len(raw.ch_names) == 0:
        if verbose:
            print(f"✗ No EEG channels found in {pid}")
        return None
    
    # Keep a copy for PSD comparison (before preprocessing)
    raw_before = raw.copy() if return_psd else None
    
    # ========== 2. CLEAN CHANNEL NAMES ==========
    clean_channel_names(raw)
    if return_psd:
        clean_channel_names(raw_before)
    
    if verbose:
        print(f"✓ Loaded {len(raw.ch_names)} EEG channels")
    
    # ========== 3. IDENTIFY PRESENT/MISSING CHANNELS ==========
    present_chs = [ch for ch in CORE_CHS if ch in raw.ch_names]
    missing_chs = [ch for ch in CORE_CHS if ch not in raw.ch_names]
    
    if not present_chs:
        if verbose:
            print(f"✗ No CORE_CHS channels found in {pid}")
        return None
    
    if verbose:
        print(f"✓ Found {len(present_chs)}/{len(CORE_CHS)} CORE_CHS channels")
        if missing_chs:
            print(f"  Missing: {', '.join(missing_chs)}")
    
    # ========== 4. ADD MISSING CHANNELS (TEMPORARY) ==========
    # These will be interpolated later
    if missing_chs:
        data = np.zeros((len(missing_chs), raw.n_times))
        info = mne.create_info(missing_chs, sfreq=raw.info["sfreq"], ch_types="eeg")
        raw.add_channels([mne.io.RawArray(data, info)], force_update_info=True)
    
    # Reorder to match CORE_CHS
    raw.reorder_channels(CORE_CHS)
    
    # ========== 5. SET MONTAGE ==========
    set_montage_for_corechs(raw)
    
    # ========== 6. DETECT DEAD CHANNELS ==========
    dead_chs = detect_dead_channels(raw)
    # Only consider dead channels among originally present ones
    dead_chs = [ch for ch in dead_chs if ch in present_chs]
    
    # ========== 7. MARK BAD CHANNELS ==========
    bad_chs = list(set(missing_chs + dead_chs))
    raw.info['bads'] = bad_chs
    
    if verbose and bad_chs:
        print(f"✓ Marked {len(bad_chs)} bad channels:")
        print(f"  - {len(missing_chs)} missing")
        print(f"  - {len(dead_chs)} dead")
    
    # ========== 8. INTERPOLATE BAD CHANNELS ==========
    # KEY STEP: Spherical spline interpolation
    if bad_chs:
        try:
            raw.interpolate_bads(reset_bads=True, mode='accurate')
            if verbose:
                print(f"✓ Interpolated {len(bad_chs)} channels")
        except Exception as e:
            if verbose:
                print(f"✗ Interpolation failed: {e}")
            return None
    
    # ========== 9. NOTCH FILTER ==========
    raw.notch_filter(freqs=[notch], verbose="ERROR")
    if verbose:
        print(f"✓ Notch filter at {notch} Hz")
    
    # ========== 10. BANDPASS FILTER ==========
    raw.filter(l_freq=band[0], h_freq=band[1], fir_design='firwin', 
               filter_length='auto', verbose="ERROR")
    if verbose:
        print(f"✓ Bandpass filter {band[0]}-{band[1]} Hz")
    
    # ========== 11. RESAMPLE ==========
    raw.resample(resample_hz, npad="auto", verbose="ERROR")
    if verbose:
        print(f"✓ Resampled to {resample_hz} Hz")
    
    # ========== 12. COMMON AVERAGE REFERENCE ==========
    raw.set_eeg_reference('average', projection=False, verbose="ERROR")
    if verbose:
        print(f"✓ Common average reference")
    
    # ========== 13. LINEAR DETREND ==========
    # CRITICAL for DTF/PDC: ensures stationarity for MVAR models
    raw.apply_function(lambda x: x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x))), 
                       channel_wise=True)
    if verbose:
        print(f"✓ Linear detrend")
    
    # ========== 14. EPOCHING ==========
    epochs = mne.make_fixed_length_epochs(
        raw, duration=epoch_len, overlap=epoch_overlap, 
        preload=True, verbose="ERROR"
    )
    
    if len(epochs) == 0:
        if verbose:
            print(f"✗ No epochs created for {pid}")
        return None
    
    if verbose:
        print(f"✓ Created {len(epochs)} epochs ({epoch_len}s each)")
    
    # ========== 15. ARTIFACT REJECTION ==========
    X = epochs.get_data()  # Shape: (n_epochs, n_channels, n_times)
    
    # Compute peak-to-peak amplitude per epoch (max across channels)
    max_ptp_uv = np.ptp(X, axis=2).max(axis=1) * 1e6
    
    # Adaptive threshold (98th percentile)
    adaptive_thr_uv = float(np.percentile(max_ptp_uv, reject_percentile))
    
    # Apply hard cap if specified
    if reject_cap_uv is not None:
        final_thr_uv = min(adaptive_thr_uv, reject_cap_uv)
    else:
        final_thr_uv = adaptive_thr_uv
    
    if verbose:
        print(f"  Adaptive threshold ({reject_percentile}th %ile): {adaptive_thr_uv:.1f} µV")
        if reject_cap_uv is not None:
            print(f"  Final threshold (capped at {reject_cap_uv:.1f} µV): {final_thr_uv:.1f} µV")
        else:
            print(f"  Final threshold (no cap): {final_thr_uv:.1f} µV")
    
    # Reject bad epochs
    epochs_clean = epochs.copy().drop_bad(
        reject=dict(eeg=final_thr_uv * 1e-6), 
        verbose="ERROR"
    )
    
    if len(epochs_clean) == 0:
        if verbose:
            print(f"✗ All epochs rejected for {pid}")
        return None
    
    if verbose:
        n_rejected = len(epochs) - len(epochs_clean)
        print(f"✓ Kept {len(epochs_clean)}/{len(epochs)} epochs ({n_rejected} rejected)")
    
    # ========== 16. Z-SCORE NORMALIZATION ==========
    # Per-epoch, per-channel normalization across time
    Xc = epochs_clean.get_data()
    m = Xc.mean(axis=2, keepdims=True)
    s = Xc.std(axis=2, keepdims=True)
    s[s == 0] = 1.0  # Avoid division by zero
    Xz = (Xc - m) / s
    
    epochs_final = mne.EpochsArray(
        Xz, epochs_clean.info, events=epochs_clean.events,
        tmin=epochs_clean.tmin, event_id=epochs_clean.event_id, 
        on_missing='ignore', verbose="ERROR"
    )
    
    if verbose:
        print(f"✓ Z-score normalization")
    
    # ========== 17. LABELS ==========
    label = infer_label_from_path(edf_path)
    y = np.full(len(epochs_final), label, dtype=int)
    
    if verbose:
        label_str = "EPILEPSY" if label == 1 else "CONTROL"
        print(f"✓ Label: {label_str}")
    
    # ========== 18. COLLECT RESULTS ==========
    out = {
        "raw_after": raw,
        "epochs": epochs_final,
        "labels": y,
        "threshold_uv": final_thr_uv,
        "present_channels": CORE_CHS,  # Always all 22 channels
        "present_mask": np.array([ch not in missing_chs for ch in CORE_CHS], dtype=bool)
    }
    
    # ========== 19. OPTIONAL: COMPUTE PSD ==========
    if return_psd:
        try:
            # Before: only originally present channels
            if raw_before is not None:
                raw_before.pick_channels(present_chs, ordered=False)
                out["psd_before"] = raw_before.compute_psd(
                    fmax=band[1], average='mean', verbose="ERROR"
                )
            
            # After: all 22 channels (including interpolated)
            out["psd_after"] = raw.compute_psd(
                fmax=band[1], average='mean', verbose="ERROR"
            )
        except Exception as e:
            if verbose:
                print(f"⚠ Could not compute PSD: {e}")
    
    if verbose:
        print(f"{'='*60}\n")
    
    return out