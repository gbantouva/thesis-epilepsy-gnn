"""
Calculate grand average PSD for ONE patient (all their epoch files combined).

Usage:
  python src\grand_av_patient.py --patient_id aaaaaanr --data_dir F:\October-Thesis\thesis-epilepsy-gnn\test\data_pp --output_dir F:\October-Thesis\thesis-epilepsy-gnn\test\figures\grand_average\single_patient
"""

import mne
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import argparse


def calculate_single_patient_grand_average(patient_id: str, data_dir: Path, output_dir: Path):
    """
    Find all epoch files for ONE patient, concatenate them, and compute PSD grand average.
    
    Args:
        patient_id: Patient identifier (e.g., "aaaaaanr")
        data_dir: Root preprocessed data directory
        output_dir: Where to save results
    """
    print(f"\n{'='*70}")
    print(f"GRAND AVERAGE PSD COMPUTATION: Patient {patient_id}")
    print(f"{'='*70}\n")
    
    # ========== STEP 1: Find all epoch files for this patient ==========
    print("Step 1: Finding epoch files...")
    epoch_files = list(data_dir.rglob(f"**/{patient_id}/**/*_epochs.npy"))
    
    if not epoch_files:
        print(f"❌ Error: No epoch files found for patient {patient_id}")
        return
    
    print(f"✓ Found {len(epoch_files)} epoch files:")
    for f in epoch_files:
        print(f"  - {f.name}")
    
    # ========== STEP 2: Load and concatenate all epochs ==========
    print("\nStep 2: Loading and concatenating epochs...")
    all_epochs = []
    info = None
    
    for i, epoch_file in enumerate(epoch_files):
        try:
            # Load epochs
            data = np.load(epoch_file)
            all_epochs.append(data)
            print(f"  ✓ Loaded {epoch_file.name}: {data.shape[0]} epochs")
            
            # Load info (only need once)
            if i == 0:
                pid = epoch_file.stem.replace("_epochs", "")
                info_file = epoch_file.parent / f"{pid}_info.pkl"
                with open(info_file, 'rb') as f:
                    info = pickle.load(f)
                    
        except Exception as e:
            print(f"  ✗ Error loading {epoch_file.name}: {e}")
    
    if not all_epochs:
        print("❌ No valid data loaded")
        return
    
    # Concatenate: (n_epochs_total, n_channels, n_times)
    all_data = np.concatenate(all_epochs, axis=0)
    print(f"\n✓ Concatenated: {all_data.shape[0]} total epochs")
    print(f"  Shape: {all_data.shape} (epochs × channels × times)")
    
    # ========== STEP 3: Create MNE Epochs object ==========
    print("\nStep 3: Creating MNE Epochs object...")
    patient_epochs = mne.EpochsArray(all_data, info, verbose=False)
    print(f"✓ Created Epochs object")
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # ========== STEP 4: FREQUENCY-DOMAIN GRAND AVERAGE (PSD) ==========
    print("\nStep 4: Computing FREQUENCY-DOMAIN grand average (PSD)...")
    print("ℹ️  NOTE: Time-domain averaging is skipped because data is z-scored (unitless).")
    
    # Compute PSD
    patient_psd = patient_epochs.compute_psd(fmin=0.5, fmax=100.0, verbose=False)
    patient_avg_psd = patient_psd.average()
    
    # Get data
    psd_data = patient_avg_psd.get_data()  # (n_channels, n_freqs)
    freqs = patient_avg_psd.freqs
    print(f"✓ PSD grand average shape: {psd_data.shape}")
    
    # Save as .npy
    np.save(output_dir / f"{patient_id}_psd_grand_avg.npy", psd_data)
    np.save(output_dir / f"{patient_id}_psd_freqs.npy", freqs)
    print(f"✓ Saved: {patient_id}_psd_grand_avg.npy")
    print(f"✓ Saved: {patient_id}_psd_freqs.npy")
    
    # Prepare plotting
    ch_names = info['ch_names']
    plot_channels = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2"]
    available = [ch for ch in plot_channels if ch in ch_names]
    
    # Plot with frequency bands
    fig, ax = plt.subplots(figsize=(12, 6))
    for ch in available:
        idx = ch_names.index(ch)
        ax.semilogy(freqs, psd_data[idx], label=ch, alpha=0.8, linewidth=1.5)
    
    # Mark frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, 100)
    }
    
    y_min, y_max = ax.get_ylim()
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.1)
        ax.text((f_low + f_high)/2, y_max * 0.7, band_name,
               ha='center', fontsize=10, fontweight='bold')
    
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Power Spectral Density", fontsize=12)
    ax.set_title(f"Patient {patient_id} - PSD Grand Average", 
                fontsize=14, fontweight='bold')
    ax.set_xlim([0, 100])
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / f"{patient_id}_psd_grand_avg.png", dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"✓ Saved: {patient_id}_psd_grand_avg.png")
    
    # ========== SUMMARY ==========
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"Patient ID:         {patient_id}")
    print(f"Files processed:    {len(epoch_files)}")
    print(f"Total epochs:       {all_data.shape[0]}")
    print(f"Channels:           {all_data.shape[1]}")
    print(f"Timepoints/epoch:   {all_data.shape[2]}")
    print(f"Sampling rate:      {info['sfreq']} Hz")
    print(f"Output directory:   {output_dir}")
    print(f"{'='*70}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Calculate grand average PSD for a single patient"
    )
    parser.add_argument("--patient_id", required=True, help="Patient ID (e.g., 'aaaaaanr')")
    parser.add_argument("--data_dir", required=True, help="Root data directory (e.g., 'data_pp')")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    
    args = parser.parse_args()
    
    calculate_single_patient_grand_average(
        patient_id=args.patient_id,
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir)
    )