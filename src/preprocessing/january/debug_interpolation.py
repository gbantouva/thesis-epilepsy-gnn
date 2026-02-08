"""
Diagnostic script to check why T1/T2 interpolation is failing.
Run this on one of your EDF files to see what's happening.

Usage:
    python debug_interpolation.py --edf path/to/your/file.edf
"""

import argparse
import numpy as np
import mne
import re
from pathlib import Path

# Same CORE_CHS as your preprocessing script
CORE_CHS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
    "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"
]


def clean_channel_names(raw):
    """Same as your preprocess_core.py"""
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
    """Same as your preprocess_core.py"""
    std = mne.channels.make_standard_montage('standard_1020')
    pos = std.get_positions()['ch_pos'].copy()
    
    ch_pos = {}
    
    for ch in raw.ch_names:
        if ch in pos:
            ch_pos[ch] = pos[ch]
    
    # Add approximate mastoid positions (T1/T2)
    if 'T1' in raw.ch_names:
        ch_pos.setdefault('T1', (-0.0840759, 0.0145673, -0.050429))
    if 'T2' in raw.ch_names:
        ch_pos.setdefault('T2', (0.0841131, 0.0143647, -0.050538))
    
    montage = mne.channels.make_dig_montage(ch_pos=ch_pos, coord_frame='head')
    raw.set_montage(montage, match_case=False, on_missing='ignore')


def main():
    parser = argparse.ArgumentParser(description="Debug interpolation issues")
    parser.add_argument("--edf", required=True, help="Path to EDF file")
    args = parser.parse_args()
    
    edf_path = Path(args.edf)
    
    print("=" * 70)
    print("INTERPOLATION DIAGNOSTIC")
    print("=" * 70)
    
    # 1. Load the file
    print(f"\n[1] Loading: {edf_path.name}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, exclude=['EOG', 'ECG', 'EMG', 'MISC', 'STIM'])
    
    print(f"    Original channels: {len(raw.ch_names)}")
    print(f"    Original names: {raw.ch_names[:10]}...")  # First 10
    
    # 2. Clean channel names
    print(f"\n[2] Cleaning channel names...")
    clean_channel_names(raw)
    print(f"    After cleaning: {raw.ch_names[:10]}...")
    
    # 3. Check which CORE_CHS are present/missing
    print(f"\n[3] Checking CORE_CHS presence...")
    present_chs = [ch for ch in CORE_CHS if ch in raw.ch_names]
    missing_chs = [ch for ch in CORE_CHS if ch not in raw.ch_names]
    
    print(f"    Present ({len(present_chs)}): {present_chs}")
    print(f"    Missing ({len(missing_chs)}): {missing_chs}")
    
    if not missing_chs:
        print("\n    ✓ No missing channels - interpolation not needed!")
        return
    
    # 4. Add missing channels as zeros
    print(f"\n[4] Adding missing channels as zeros...")
    if missing_chs:
        data = np.zeros((len(missing_chs), raw.n_times))
        info = mne.create_info(missing_chs, sfreq=raw.info["sfreq"], ch_types="eeg")
        raw.add_channels([mne.io.RawArray(data, info, verbose="ERROR")], force_update_info=True)
    
    # Reorder to match CORE_CHS
    raw.reorder_channels(CORE_CHS)
    print(f"    Channels after adding: {raw.ch_names}")
    
    # 5. Check data BEFORE montage/interpolation
    print(f"\n[5] Data BEFORE interpolation:")
    for ch in missing_chs:
        idx = raw.ch_names.index(ch)
        data = raw.get_data(picks=[ch])
        print(f"    {ch}: mean={np.mean(data)*1e6:.4f} µV, std={np.std(data)*1e6:.4f} µV, "
              f"min={np.min(data)*1e6:.4f} µV, max={np.max(data)*1e6:.4f} µV")
    
    # 6. Set montage
    print(f"\n[6] Setting montage...")
    set_montage_for_corechs(raw)
    
    # 7. CHECK: What positions were actually set?
    print(f"\n[7] Checking montage positions:")
    montage = raw.get_montage()
    if montage is None:
        print("    ✗ ERROR: No montage set!")
    else:
        positions = montage.get_positions()['ch_pos']
        print(f"    Channels with positions: {len(positions)}")
        
        for ch in missing_chs:
            if ch in positions:
                pos = positions[ch]
                print(f"    ✓ {ch}: position = {pos}")
            else:
                print(f"    ✗ {ch}: NO POSITION FOUND!")
        
        # Also check a few known channels
        for ch in ['Fp1', 'Cz', 'O1']:
            if ch in positions:
                print(f"    ✓ {ch}: position = {positions[ch]}")
    
    # 8. Mark bad channels
    print(f"\n[8] Marking bad channels...")
    raw.info['bads'] = missing_chs
    print(f"    Bads: {raw.info['bads']}")
    
    # 9. Try interpolation with VERBOSE output
    print(f"\n[9] Attempting interpolation (VERBOSE)...")
    print("-" * 50)
    try:
        raw.interpolate_bads(reset_bads=True, mode='accurate', verbose=True)
        print("-" * 50)
        print("    ✓ Interpolation completed without error")
    except Exception as e:
        print("-" * 50)
        print(f"    ✗ Interpolation FAILED: {e}")
        return
    
    # 10. Check data AFTER interpolation
    print(f"\n[10] Data AFTER interpolation:")
    for ch in missing_chs:
        data = raw.get_data(picks=[ch])
        std_val = np.std(data) * 1e6
        print(f"    {ch}: mean={np.mean(data)*1e6:.4f} µV, std={std_val:.4f} µV, "
              f"min={np.min(data)*1e6:.4f} µV, max={np.max(data)*1e6:.4f} µV")
        
        if std_val < 0.1:
            print(f"        ⚠️  WARNING: {ch} appears to be FLAT (std < 0.1 µV)!")
        else:
            print(f"        ✓ {ch} has non-zero variance - interpolation worked!")
    
    # 11. Compare with a real channel
    print(f"\n[11] Comparison with real channels:")
    real_chs = ['Fp1', 'Cz', 'O1']
    for ch in real_chs:
        if ch in raw.ch_names:
            data = raw.get_data(picks=[ch])
            print(f"    {ch} (real): std={np.std(data)*1e6:.2f} µV")
    
    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()