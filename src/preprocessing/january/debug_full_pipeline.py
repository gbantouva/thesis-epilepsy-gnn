"""
Full Pipeline Diagnostic - Track T1/T2 through every preprocessing step.
This will show us exactly WHERE the channels become flat.

Usage:
    python debug_full_pipeline.py --edf path/to/your/file.edf
"""

import argparse
import numpy as np
import mne
import re
from pathlib import Path

CORE_CHS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
    "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"
]


def clean_channel_names(raw):
    mapping = {
        orig: re.sub(r'^(?:EEG\s*)', '', orig)
        .replace('-LE', '')
        .replace('-REF', '')
        .strip()
        for orig in raw.ch_names
    }
    raw.rename_channels(mapping)
    
    core_ch_map = {ch.lower(): ch for ch in CORE_CHS}
    case_mapping = {}
    for ch in raw.ch_names:
        if ch.lower() in core_ch_map:
            case_mapping[ch] = core_ch_map[ch.lower()]
    
    if case_mapping:
        raw.rename_channels(case_mapping)


def set_montage_for_corechs(raw):
    std = mne.channels.make_standard_montage('standard_1020')
    pos = std.get_positions()['ch_pos'].copy()
    
    ch_pos = {}
    for ch in raw.ch_names:
        if ch in pos:
            ch_pos[ch] = pos[ch]
    
    if 'T1' in raw.ch_names:
        ch_pos.setdefault('T1', (-0.0840759, 0.0145673, -0.050429))
    if 'T2' in raw.ch_names:
        ch_pos.setdefault('T2', (0.0841131, 0.0143647, -0.050538))
    
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, match_case=False, on_missing='ignore')


def check_channels(raw, step_name, channels=['T1', 'T2', 'Fp1', 'Cz']):
    """Print stats for specified channels."""
    print(f"\n    --- {step_name} ---")
    for ch in channels:
        if ch in raw.ch_names:
            data = raw.get_data(picks=[ch])
            std_uv = np.std(data) * 1e6
            mean_uv = np.mean(data) * 1e6
            status = "✓ OK" if std_uv > 0.1 else "⚠️ FLAT!"
            print(f"    {ch}: std={std_uv:8.2f} µV, mean={mean_uv:8.2f} µV  {status}")


def check_epochs(epochs, step_name, channels=['T1', 'T2', 'Fp1', 'Cz']):
    """Print stats for specified channels in epochs."""
    print(f"\n    --- {step_name} ---")
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ch_names = epochs.ch_names
    
    for ch in channels:
        if ch in ch_names:
            idx = ch_names.index(ch)
            ch_data = data[:, idx, :]  # All epochs, this channel
            std_uv = np.std(ch_data) * 1e6
            mean_uv = np.mean(ch_data) * 1e6
            status = "✓ OK" if std_uv > 0.1 else "⚠️ FLAT!"
            print(f"    {ch}: std={std_uv:8.2f} µV, mean={mean_uv:8.2f} µV  {status}")


def main():
    parser = argparse.ArgumentParser(description="Full pipeline diagnostic")
    parser.add_argument("--edf", required=True, help="Path to EDF file")
    args = parser.parse_args()
    
    edf_path = Path(args.edf)
    
    print("=" * 70)
    print("FULL PIPELINE DIAGNOSTIC - Tracking T1/T2")
    print("=" * 70)
    
    # ========== STEP 1: Load ==========
    print(f"\n[STEP 1] Loading: {edf_path.name}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, exclude=['EOG', 'ECG', 'EMG', 'MISC', 'STIM'])
    clean_channel_names(raw)
    
    present_chs = [ch for ch in CORE_CHS if ch in raw.ch_names]
    missing_chs = [ch for ch in CORE_CHS if ch not in raw.ch_names]
    print(f"    Missing channels: {missing_chs}")
    
    if not missing_chs:
        print("    No missing channels - nothing to interpolate!")
        return
    
    # ========== STEP 2: Add missing channels ==========
    print(f"\n[STEP 2] Adding missing channels as zeros")
    data = np.zeros((len(missing_chs), raw.n_times))
    info = mne.create_info(missing_chs, sfreq=raw.info["sfreq"], ch_types="eeg")
    raw.add_channels([mne.io.RawArray(data, info, verbose="ERROR")], force_update_info=True)
    raw.reorder_channels(CORE_CHS)
    
    check_channels(raw, "After adding zeros (before interpolation)")
    
    # ========== STEP 3: Set montage ==========
    print(f"\n[STEP 3] Setting montage")
    set_montage_for_corechs(raw)
    
    # ========== STEP 4: Mark bads and interpolate ==========
    print(f"\n[STEP 4] Interpolating bad channels")
    raw.info['bads'] = missing_chs
    raw.interpolate_bads(reset_bads=True, mode='accurate', verbose="ERROR")
    
    check_channels(raw, "After INTERPOLATION")
    
    # ========== STEP 5: Notch filter ==========
    print(f"\n[STEP 5] Notch filter @ 60 Hz")
    raw.notch_filter(freqs=[60.0], phase='zero', verbose="ERROR")
    
    check_channels(raw, "After NOTCH FILTER")
    
    # ========== STEP 6: Bandpass filter ==========
    print(f"\n[STEP 6] Bandpass filter 0.5-80 Hz")
    raw.filter(l_freq=0.5, h_freq=80.0, fir_design='firwin', phase='zero', verbose="ERROR")
    
    check_channels(raw, "After BANDPASS FILTER")
    
    # ========== STEP 7: Resample ==========
    print(f"\n[STEP 7] Resample to 250 Hz")
    raw.resample(250.0, npad="auto", verbose="ERROR")
    
    check_channels(raw, "After RESAMPLE")
    
    # ========== STEP 8: Common Average Reference ==========
    print(f"\n[STEP 8] Common Average Reference")
    raw.set_eeg_reference('average', projection=False, verbose="ERROR")
    
    check_channels(raw, "After CAR")
    
    # ========== STEP 9: Epoching ==========
    print(f"\n[STEP 9] Creating epochs (4s)")
    epochs = mne.make_fixed_length_epochs(raw, duration=4.0, overlap=0.0, 
                                          preload=True, verbose="ERROR")
    
    check_epochs(epochs, "After EPOCHING")
    
    # ========== STEP 10: Get final data ==========
    print(f"\n[STEP 10] Final epoch data")
    X = epochs.get_data()
    print(f"    Epochs shape: {X.shape}")
    
    # Check T1/T2 in final numpy array
    t1_idx = CORE_CHS.index('T1')
    t2_idx = CORE_CHS.index('T2')
    fp1_idx = CORE_CHS.index('Fp1')
    
    print(f"\n    --- Final numpy array ---")
    print(f"    T1 (idx={t1_idx}): std={np.std(X[:, t1_idx, :])*1e6:.2f} µV")
    print(f"    T2 (idx={t2_idx}): std={np.std(X[:, t2_idx, :])*1e6:.2f} µV")
    print(f"    Fp1 (idx={fp1_idx}): std={np.std(X[:, fp1_idx, :])*1e6:.2f} µV")
    
    # ========== SUMMARY ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    t1_std = np.std(X[:, t1_idx, :]) * 1e6
    t2_std = np.std(X[:, t2_idx, :]) * 1e6
    
    if t1_std < 0.1 or t2_std < 0.1:
        print("\n⚠️  T1/T2 ARE FLAT in the final output!")
        print("    Look above to see which step caused the problem.")
    else:
        print("\n✓ T1/T2 have non-zero variance in final output.")
        print("    The issue might be in how you SAVE or LOAD the epochs.")
        print("\n    Check your preprocessing script:")
        print("    1. How are you saving the epochs? (np.save?)")
        print("    2. Is the channel order preserved when saving?")
    
    print("=" * 70)


if __name__ == "__main__":
    main()