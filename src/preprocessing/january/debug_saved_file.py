"""
Check Saved Files Diagnostic
Inspects the actual saved .npy files to see if T1/T2 are flat there.

Usage:
    python debug_saved_files.py --preprocessed_dir path/to/preprocessed --pid patient_id

Example:
    python debug_saved_files.py --preprocessed_dir data_pp_balanced/01_no_epilepsy/aaaaaebo/s001_2006/02_tcp_le --pid aaaaaebo_s001_t000
"""

import argparse
import numpy as np
from pathlib import Path

CORE_CHS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
    "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"
]


def main():
    parser = argparse.ArgumentParser(description="Check saved preprocessed files")
    parser.add_argument("--preprocessed_dir", required=True, help="Directory with preprocessed files")
    parser.add_argument("--pid", required=True, help="Patient ID (filename stem)")
    args = parser.parse_args()
    
    prep_dir = Path(args.preprocessed_dir)
    pid = args.pid
    
    print("=" * 70)
    print("SAVED FILES DIAGNOSTIC")
    print("=" * 70)
    
    # Check what files exist
    print(f"\n[1] Checking files in: {prep_dir}")
    
    epochs_file = prep_dir / f"{pid}_epochs.npy"
    mask_file = prep_dir / f"{pid}_present_mask.npy"
    labels_file = prep_dir / f"{pid}_labels.npy"
    
    print(f"    Epochs file: {epochs_file.name} - {'EXISTS' if epochs_file.exists() else 'NOT FOUND'}")
    print(f"    Mask file: {mask_file.name} - {'EXISTS' if mask_file.exists() else 'NOT FOUND'}")
    print(f"    Labels file: {labels_file.name} - {'EXISTS' if labels_file.exists() else 'NOT FOUND'}")
    
    if not epochs_file.exists():
        print("\n    ✗ Cannot continue - epochs file not found!")
        return
    
    # Load epochs
    print(f"\n[2] Loading epochs...")
    epochs = np.load(epochs_file)
    print(f"    Shape: {epochs.shape}")
    print(f"    Expected: (n_epochs, 22, 1000) for 4s @ 250Hz")
    
    if epochs.shape[1] != 22:
        print(f"    ⚠️  WARNING: Expected 22 channels, got {epochs.shape[1]}!")
    
    # Check each channel
    print(f"\n[3] Channel statistics in SAVED file:")
    print(f"    {'Channel':<8} {'Index':<6} {'Std (µV)':<12} {'Mean (µV)':<12} {'Status'}")
    print(f"    {'-'*50}")
    
    flat_channels = []
    for i, ch in enumerate(CORE_CHS):
        if i < epochs.shape[1]:
            ch_data = epochs[:, i, :]
            std_uv = np.std(ch_data) * 1e6
            mean_uv = np.mean(ch_data) * 1e6
            
            if std_uv < 0.1:
                status = "⚠️ FLAT!"
                flat_channels.append(ch)
            else:
                status = "✓ OK"
            
            print(f"    {ch:<8} {i:<6} {std_uv:<12.2f} {mean_uv:<12.2f} {status}")
    
    # Load and check present_mask
    print(f"\n[4] Checking present_mask...")
    if mask_file.exists():
        present_mask = np.load(mask_file)
        print(f"    Shape: {present_mask.shape}")
        print(f"    Dtype: {present_mask.dtype}")
        print(f"    Values: {present_mask}")
        
        print(f"\n    Channel mask interpretation:")
        for i, ch in enumerate(CORE_CHS):
            if i < len(present_mask):
                status = "REAL (originally present)" if present_mask[i] else "INTERPOLATED (was missing)"
                print(f"    {ch:<8}: {present_mask[i]} = {status}")
        
        # Check if flat channels match interpolated channels
        interpolated_chs = [CORE_CHS[i] for i in range(len(present_mask)) if not present_mask[i]]
        print(f"\n    Interpolated channels according to mask: {interpolated_chs}")
        print(f"    Flat channels in saved data: {flat_channels}")
        
        if set(flat_channels) == set(interpolated_chs):
            print(f"\n    ⚠️  CONFIRMED: Flat channels = Interpolated channels")
            print(f"       This means interpolation worked during preprocessing,")
            print(f"       but the SAVING step saved zeros instead of interpolated data!")
    else:
        print(f"    Mask file not found - cannot check")
    
    # Summary
    print(f"\n" + "=" * 70)
    print("DIAGNOSIS")
    print("=" * 70)
    
    if flat_channels:
        print(f"""
⚠️  PROBLEM FOUND: Channels {flat_channels} are FLAT in the saved file.

This confirms the bug is in your SAVE code, not the preprocessing.

Look in your preprocessing runner script for code like:
    
    # WRONG - This saves the raw data before processing:
    np.save(f"{{pid}}_epochs.npy", raw_epochs)
    
    # Or WRONG - This might be saving from wrong variable:
    np.save(f"{{pid}}_epochs.npy", something_else)

The fix depends on how your runner script is structured.
Please share your preprocessing runner script (the one that calls 
preprocess_single() and saves the files) so I can identify the bug.
""")
    else:
        print(f"""
✓ All channels have non-zero variance in saved file.

The issue might be in:
1. The inspection script loading the wrong file
2. A different preprocessing run that had issues
3. Something else in the visualization code
""")
    
    print("=" * 70)


if __name__ == "__main__":
    main()