"""
Batch Grand Average PSD Analysis: All Patients (Epilepsy vs Control)

CORRECTED VERSION:
- Fixed frequency range (45 Hz instead of 100 Hz for clinical bands)
- Improved path detection
- Added more robust error handling

This script:
1. Discovers all patients from directory structure
2. Computes PSD grand average for each patient
3. Groups patients by Epilepsy vs Control
4. Creates group-level averages
5. Performs statistical comparisons
6. Generates publication-quality figures
7. Extracts band power features for GNN

Usage:
  python grand_average_batch_psd_corrected.py --data_dir data_pp --output_dir figures/grand_average_analysis

Example:
  python grand_average_batch_psd_corrected.py \
    --data_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\data_pp \
    --output_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\figures\\grand_average_analysis \
    --max_patients 50
"""

import mne
import numpy as np
import pickle
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import argparse
from collections import defaultdict
from tqdm import tqdm
from scipy import stats
import seaborn as sns

# Set plot style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# IMPORTANT: Set frequency range (matching Neuro-GPT and preprocessing)
FMAX = 80.0  # Maximum frequency (Hz) - full preprocessed range


def discover_patients(data_dir: Path):
    """
    Discover all unique patients from directory structure.
    
    Returns:
        Dictionary mapping patient_id -> {
            "label": 0 or 1,
            "files": [list of epoch file paths]
        }
    """
    patients = defaultdict(lambda: {"label": None, "files": []})
    
    print("Discovering patients from directory structure...")
    all_epoch_files = list(data_dir.rglob("*_epochs.npy")) + list(data_dir.rglob("*_epochs_balanced.npy"))
    print(f"Found {len(all_epoch_files)} total epoch files")
    
    for epoch_file in all_epoch_files:
        try:
            # Get the full path as string for easier checking
            full_path_str = str(epoch_file).replace('\\', '/').lower()
            
            # Determine label from path (check multiple possible naming schemes)
            if any(x in full_path_str for x in ['/00_epilepsy/', '\\00_epilepsy\\', '/epilepsy/', '\\epilepsy\\']):
                label = 1  # Epilepsy
            elif any(x in full_path_str for x in ['/01_no_epilepsy/', '\\01_no_epilepsy\\', '/01_control/', '\\01_control\\', '/control/', '\\control\\']):
                label = 0  # Control
            else:
                # Try to infer from labels.npy if path doesn't help
                print(f"  ⚠️ Cannot determine label from path: {epoch_file}")
                print(f"     Checking labels.npy...")
                
                pid = epoch_file.stem.replace("_epochs_balanced", "").replace("_epochs", "")
                labels_file = epoch_file.parent / f"{pid}_labels.npy"
                
                if labels_file.exists():
                    labels = np.load(labels_file)
                    label = int(labels[0])  # Assume all epochs have same label
                    print(f"     Found label from file: {label}")
                else:
                    print(f"     Skipping file (cannot determine label)")
                    continue
            
            # Extract patient ID from filename or path
            try:
                # Try to get relative path structure
                rel_path = epoch_file.relative_to(data_dir)
                parts = rel_path.parts
                
                # Structure: data_pp/00_epilepsy/PATIENT_ID/session/recording/file
                # or:        data_pp/01_no_epilepsy/PATIENT_ID/session/recording/file
                if len(parts) >= 2:
                    patient_id = parts[1] if len(parts) > 1 else parts[0]
                else:
                    # Fallback: extract from filename
                    filename = epoch_file.stem.replace("_epochs_balanced", "").replace("_epochs", "")
                    patient_id = filename.split('_')[0]
            except (ValueError, IndexError):
                # File not under data_dir or path issues
                filename = epoch_file.stem.replace("_epochs_balanced", "").replace("_epochs", "")
                patient_id = filename.split('_')[0]
            
            # Add to patient's file list
            patients[patient_id]["files"].append(epoch_file)
            
            # Set or verify label consistency
            if patients[patient_id]["label"] is None:
                patients[patient_id]["label"] = label
            elif patients[patient_id]["label"] != label:
                print(f"  ⚠️ WARNING: Patient {patient_id} has inconsistent labels!")
                print(f"     Previous: {patients[patient_id]['label']}, Current: {label}")
                print(f"     File: {epoch_file}")
                
        except Exception as e:
            print(f"  ⚠️ Could not process {epoch_file}: {e}")
            continue
    
    # Debug: print first few patients from each group to verify
    print(f"\nDiscovered {len(patients)} unique patients")
    
    epilepsy_list = [(pid, data) for pid, data in patients.items() if data["label"] == 1]
    control_list = [(pid, data) for pid, data in patients.items() if data["label"] == 0]
    
    print(f"\nEpilepsy patients: {len(epilepsy_list)}")
    if epilepsy_list:
        print(f"First 3 examples:")
        for pid, data in epilepsy_list[:3]:
            print(f"  {pid}: {len(data['files'])} files, label={data['label']}")
    
    print(f"\nControl patients: {len(control_list)}")
    if control_list:
        print(f"First 3 examples:")
        for pid, data in control_list[:3]:
            print(f"  {pid}: {len(data['files'])} files, label={data['label']}")
    
    return dict(patients)


def compute_patient_psd(patient_id: str, epoch_files: list, label: int):
    """
    Compute PSD grand average for a single patient.
    
    Args:
        patient_id: Patient identifier
        epoch_files: List of epoch file paths
        label: 0 (control) or 1 (epilepsy)
        
    Returns:
        Dictionary with psd_data, freqs, info, n_epochs, or None if failed
    """
    try:
        # Load info from first file
        first_file = epoch_files[0]
        pid = first_file.stem.replace("_epochs_balanced", "").replace("_epochs", "")
        info_file = first_file.parent / f"{pid}_info.pkl"
        with open(info_file, 'rb') as f:
            info = pickle.load(f)
        
        # First pass: count total epochs to check memory requirements
        total_epochs = 0
        for epoch_file in epoch_files:
            data = np.load(epoch_file, mmap_mode='r')  # Memory-map to check shape without loading
            total_epochs += data.shape[0]
        
        # Memory check: if > 50,000 epochs, use chunked processing
        MAX_EPOCHS_IN_MEMORY = 50000
        
        if total_epochs > MAX_EPOCHS_IN_MEMORY:
            print(f"\n  ⚠️  {patient_id}: Large dataset ({total_epochs} epochs), using chunked processing...")
            return compute_patient_psd_chunked(patient_id, epoch_files, label, info, total_epochs)
        
        # Normal processing for reasonable-sized datasets
        all_epochs = []
        for epoch_file in epoch_files:
            data = np.load(epoch_file)
            all_epochs.append(data)
        
        if not all_epochs:
            return None
        
        # Concatenate all epochs
        all_data = np.concatenate(all_epochs, axis=0)
        
        # Create MNE Epochs object
        patient_epochs = mne.EpochsArray(all_data, info, verbose=False)
        
        # Compute PSD (frequency-domain averaging)
        patient_psd = patient_epochs.compute_psd(fmin=0.5, fmax=FMAX, verbose=False)
        patient_avg_psd = patient_psd.average()
        
        # Get data
        psd_data = patient_avg_psd.get_data()  # (n_channels, n_freqs)
        freqs = patient_avg_psd.freqs
        
        return {
            "psd_data": psd_data,
            "freqs": freqs,
            "info": info,
            "n_epochs": all_data.shape[0],
            "label": label,
            "patient_id": patient_id
        }
        
    except Exception as e:
        print(f"  ✗ Error processing {patient_id}: {e}")
        return None


def compute_patient_psd_chunked(patient_id: str, epoch_files: list, label: int, info, total_epochs: int):
    """
    Compute PSD for patients with very large numbers of epochs using chunked processing.
    
    This avoids memory errors by processing files in chunks and averaging PSDs.
    """
    try:
        # Process files in chunks and compute PSDs
        all_psds = []
        all_epoch_counts = []
        
        CHUNK_SIZE = 10000  # Process 10,000 epochs at a time
        
        current_chunk = []
        current_chunk_size = 0
        
        for epoch_file in epoch_files:
            data = np.load(epoch_file)
            
            # Add to current chunk
            current_chunk.append(data)
            current_chunk_size += data.shape[0]
            
            # If chunk is large enough, process it
            if current_chunk_size >= CHUNK_SIZE:
                chunk_data = np.concatenate(current_chunk, axis=0)
                chunk_epochs = mne.EpochsArray(chunk_data, info, verbose=False)
                chunk_psd = chunk_epochs.compute_psd(fmin=0.5, fmax=FMAX, verbose=False)
                chunk_avg_psd = chunk_psd.average()
                
                all_psds.append(chunk_avg_psd.get_data())
                all_epoch_counts.append(chunk_data.shape[0])
                
                # Reset chunk
                current_chunk = []
                current_chunk_size = 0
        
        # Process remaining data
        if current_chunk:
            chunk_data = np.concatenate(current_chunk, axis=0)
            chunk_epochs = mne.EpochsArray(chunk_data, info, verbose=False)
            chunk_psd = chunk_epochs.compute_psd(fmin=0.5, fmax=FMAX, verbose=False)
            chunk_avg_psd = chunk_psd.average()
            
            all_psds.append(chunk_avg_psd.get_data())
            all_epoch_counts.append(chunk_data.shape[0])
        
        # Weighted average of PSDs (weight by number of epochs in each chunk)
        all_psds = np.array(all_psds)  # (n_chunks, n_channels, n_freqs)
        weights = np.array(all_epoch_counts) / total_epochs  # Normalize weights
        
        # Compute weighted average
        psd_data = np.average(all_psds, axis=0, weights=weights)
        
        # Get frequencies from last chunk
        freqs = chunk_avg_psd.freqs
        
        print(f"  ✓ {patient_id}: Processed {len(all_psds)} chunks, {total_epochs} total epochs")
        
        return {
            "psd_data": psd_data,
            "freqs": freqs,
            "info": info,
            "n_epochs": total_epochs,
            "label": label,
            "patient_id": patient_id
        }
        
    except Exception as e:
        print(f"  ✗ Error in chunked processing for {patient_id}: {e}")
        return None


def extract_band_powers(psd_data, freqs):
    """
    Extract band powers from PSD.
    
    Args:
        psd_data: (n_channels, n_freqs) PSD array
        freqs: Frequency array
        
    Returns:
        Dictionary with band powers per channel
    """
    bands = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 80)  # Full range matching preprocessing and Neuro-GPT
    }
    
    band_powers = {}
    
    for band_name, (f_low, f_high) in bands.items():
        # Find frequency indices
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        
        if len(idx) == 0:
            # Band not available in frequency range
            band_powers[band_name] = np.zeros(psd_data.shape[0])
        else:
            # Integrate power in band (mean across frequencies)
            band_power = np.mean(psd_data[:, idx], axis=1)  # (n_channels,)
            band_powers[band_name] = band_power
    
    return band_powers


def plot_group_comparison_psd(epilepsy_results, control_results, output_dir):
    """
    Create publication-quality PSD comparison figure (Epilepsy vs Control).
    """
    print("\nGenerating group comparison plots...")
    
    # Extract data
    epilepsy_psds = np.array([r["psd_data"] for r in epilepsy_results])  # (n_patients, n_channels, n_freqs)
    control_psds = np.array([r["psd_data"] for r in control_results])
    freqs = epilepsy_results[0]["freqs"]
    ch_names = epilepsy_results[0]["info"]["ch_names"]
    
    # Compute group means and SEM
    epilepsy_mean = np.mean(epilepsy_psds, axis=0)  # (n_channels, n_freqs)
    epilepsy_sem = np.std(epilepsy_psds, axis=0) / np.sqrt(len(epilepsy_psds))
    
    control_mean = np.mean(control_psds, axis=0)
    control_sem = np.std(control_psds, axis=0) / np.sqrt(len(control_psds))
    
    # Select representative channels for plotting
    plot_channels = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2"]
    available = [ch for ch in plot_channels if ch in ch_names]
    
    # Create figure with two subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
    
    # ========== PANEL A: Epilepsy Group ==========
    ax = axes[0]
    for ch in available:
        idx = ch_names.index(ch)
        
        # Plot mean
        ax.semilogy(freqs, epilepsy_mean[idx], label=ch, alpha=0.8, linewidth=2)
        
        # Plot SEM as shaded area
        ax.fill_between(
            freqs,
            epilepsy_mean[idx] - epilepsy_sem[idx],
            epilepsy_mean[idx] + epilepsy_sem[idx],
            alpha=0.2
        )
    
    ax.set_ylabel("Power Spectral Density (V²/Hz)", fontsize=12, fontweight='bold')
    ax.set_title(f"EPILEPSY GROUP (n={len(epilepsy_results)} patients)", 
                fontsize=14, fontweight='bold', color='red')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, FMAX])
    
    # Mark frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, FMAX)
    }
    
    y_min, y_max = ax.get_ylim()
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.05, color='gray')
        ax.text((f_low + f_high)/2, y_max * 0.8, band_name,
               ha='center', fontsize=9, fontweight='bold', color='gray')
    
    # ========== PANEL B: Control Group ==========
    ax = axes[1]
    for ch in available:
        idx = ch_names.index(ch)
        
        # Plot mean
        ax.semilogy(freqs, control_mean[idx], label=ch, alpha=0.8, linewidth=2)
        
        # Plot SEM as shaded area
        ax.fill_between(
            freqs,
            control_mean[idx] - control_sem[idx],
            control_mean[idx] + control_sem[idx],
            alpha=0.2
        )
    
    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight='bold')
    ax.set_ylabel("Power Spectral Density (V²/Hz)", fontsize=12, fontweight='bold')
    ax.set_title(f"CONTROL GROUP (n={len(control_results)} patients)",
                fontsize=14, fontweight='bold', color='blue')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, FMAX])
    
    # Mark frequency bands
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.05, color='gray')
    
    plt.suptitle("Grand Average PSD: Group Comparison (Frequency-Domain Averaging)", 
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / "group_comparison_psd.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: group_comparison_psd.png")
    
    # Save group data
    np.save(output_dir / "epilepsy_psd_mean.npy", epilepsy_mean)
    np.save(output_dir / "epilepsy_psd_sem.npy", epilepsy_sem)
    np.save(output_dir / "control_psd_mean.npy", control_mean)
    np.save(output_dir / "control_psd_sem.npy", control_sem)
    np.save(output_dir / "psd_freqs.npy", freqs)
    print(f"  ✓ Saved: group PSD data arrays")


def plot_psd_difference(epilepsy_results, control_results, output_dir):
    """
    Plot the difference in PSD (Epilepsy - Control) with statistical significance.
    """
    print("\nGenerating PSD difference plot...")
    
    # Extract data
    epilepsy_psds = np.array([r["psd_data"] for r in epilepsy_results])
    control_psds = np.array([r["psd_data"] for r in control_results])
    freqs = epilepsy_results[0]["freqs"]
    ch_names = epilepsy_results[0]["info"]["ch_names"]
    
    # Compute means
    epilepsy_mean = np.mean(epilepsy_psds, axis=0)
    control_mean = np.mean(control_psds, axis=0)
    
    # Compute difference
    diff_mean = epilepsy_mean - control_mean  # (n_channels, n_freqs)
    
    # Statistical test at each frequency
    p_values = np.zeros((len(ch_names), len(freqs)))
    for ch_idx in range(len(ch_names)):
        for f_idx in range(len(freqs)):
            epi_vals = epilepsy_psds[:, ch_idx, f_idx]
            ctrl_vals = control_psds[:, ch_idx, f_idx]
            _, p_values[ch_idx, f_idx] = stats.ttest_ind(epi_vals, ctrl_vals)
    
    # Select representative channels
    plot_channels = ["Fp1", "Fz", "Cz", "Pz", "O1", "O2"]
    available = [ch for ch in plot_channels if ch in ch_names]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))
    
    for ch in available:
        idx = ch_names.index(ch)
        
        # Plot difference
        ax.plot(freqs, diff_mean[idx], label=ch, alpha=0.8, linewidth=2)
        
        # Mark significant frequencies (p < 0.05)
        sig_mask = p_values[idx] < 0.05
        sig_freqs = freqs[sig_mask]
        sig_vals = diff_mean[idx][sig_mask]
        ax.scatter(sig_freqs, sig_vals, s=10, alpha=0.5)
    
    # Mark frequency bands
    bands = {
        'Delta': (0.5, 4),
        'Theta': (4, 8),
        'Alpha': (8, 13),
        'Beta': (13, 30),
        'Gamma': (30, FMAX)
    }
    
    y_min, y_max = ax.get_ylim()
    for band_name, (f_low, f_high) in bands.items():
        ax.axvspan(f_low, f_high, alpha=0.05, color='gray')
        ax.text((f_low + f_high)/2, y_max * 0.9, band_name,
               ha='center', fontsize=10, fontweight='bold', color='gray')
    
    ax.axhline(0, color='k', linestyle='--', linewidth=1)
    ax.set_xlabel("Frequency (Hz)", fontsize=12, fontweight='bold')
    ax.set_ylabel("PSD Difference (Epilepsy - Control) [V²/Hz]", fontsize=12, fontweight='bold')
    ax.set_title("PSD Difference with Statistical Significance (p < 0.05)", 
                fontsize=14, fontweight='bold')
    ax.set_xlim([0, FMAX])
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    fig.savefig(output_dir / "psd_difference.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: psd_difference.png")
    
    # Save difference data
    np.save(output_dir / "psd_difference.npy", diff_mean)
    np.save(output_dir / "psd_pvalues.npy", p_values)


def plot_band_power_comparison(epilepsy_results, control_results, output_dir):
    """
    Create bar plots comparing band powers between groups.
    """
    print("\nGenerating band power comparison...")
    
    # Extract band powers for all patients
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    epilepsy_band_powers = {band: [] for band in bands}
    control_band_powers = {band: [] for band in bands}
    
    for result in epilepsy_results:
        band_powers = extract_band_powers(result["psd_data"], result["freqs"])
        for band in bands:
            # Average across channels for this patient
            epilepsy_band_powers[band].append(np.mean(band_powers[band]))
    
    for result in control_results:
        band_powers = extract_band_powers(result["psd_data"], result["freqs"])
        for band in bands:
            control_band_powers[band].append(np.mean(band_powers[band]))
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(bands))
    width = 0.35
    
    epilepsy_means = [np.mean(epilepsy_band_powers[b]) for b in bands]
    epilepsy_sems = [np.std(epilepsy_band_powers[b]) / np.sqrt(len(epilepsy_band_powers[b])) for b in bands]
    
    control_means = [np.mean(control_band_powers[b]) for b in bands]
    control_sems = [np.std(control_band_powers[b]) / np.sqrt(len(control_band_powers[b])) for b in bands]
    
    bars1 = ax.bar(x - width/2, epilepsy_means, width, label='Epilepsy', 
                   color='red', alpha=0.7, yerr=epilepsy_sems, capsize=5)
    bars2 = ax.bar(x + width/2, control_means, width, label='Control',
                   color='blue', alpha=0.7, yerr=control_sems, capsize=5)
    
    # Add statistical significance markers
    for i, band in enumerate(bands):
        t_stat, p_val = stats.ttest_ind(epilepsy_band_powers[band], control_band_powers[band])
        if p_val < 0.001:
            marker = '***'
        elif p_val < 0.01:
            marker = '**'
        elif p_val < 0.05:
            marker = '*'
        else:
            marker = 'ns'
        
        y_pos = max(epilepsy_means[i], control_means[i]) + max(epilepsy_sems[i], control_sems[i])
        ax.text(i, y_pos * 1.1, marker, ha='center', fontsize=12, fontweight='bold')
    
    ax.set_xlabel("Frequency Band", fontsize=12, fontweight='bold')
    ax.set_ylabel("Mean Power (averaged across channels) [V²/Hz]", fontsize=12, fontweight='bold')
    ax.set_title("Band Power Comparison: Epilepsy vs Control\n(* p<0.05, ** p<0.01, *** p<0.001)",
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([b.capitalize() for b in bands])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    fig.savefig(output_dir / "band_power_comparison.png", dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f"  ✓ Saved: band_power_comparison.png")


def save_band_powers_csv(all_results, output_dir):
    """
    Save band powers for all patients as CSV (for GNN features).
    """
    print("\nExtracting band power features for GNN...")
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    ch_names = all_results[0]["info"]["ch_names"]
    
    # Create dataframe
    rows = []
    
    for result in all_results:
        patient_id = result["patient_id"]
        label = result["label"]
        
        band_powers = extract_band_powers(result["psd_data"], result["freqs"])
        
        row = {
            'patient_id': patient_id,
            'label': label,
            'label_name': 'epilepsy' if label == 1 else 'control',
            'n_epochs': result["n_epochs"]
        }
        
        # Add band powers per channel
        for band in bands:
            for ch_idx, ch_name in enumerate(ch_names):
                row[f'{band}_{ch_name}'] = band_powers[band][ch_idx]
        
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Save full dataset
    csv_path = output_dir / "band_powers_all_patients.csv"
    df.to_csv(csv_path, index=False)
    print(f"  ✓ Saved: band_powers_all_patients.csv")
    print(f"    Shape: {df.shape} (patients × features)")
    print(f"    Columns: patient_id, label, label_name, n_epochs, + {len(bands) * len(ch_names)} band power features")
    
    # Also save separate files for each group
    df[df['label'] == 1].to_csv(output_dir / "band_powers_epilepsy.csv", index=False)
    df[df['label'] == 0].to_csv(output_dir / "band_powers_control.csv", index=False)
    print(f"  ✓ Saved: band_powers_epilepsy.csv and band_powers_control.csv")
    
    return df


def create_summary_report(epilepsy_results, control_results, output_dir):
    """
    Create a text summary report.
    """
    print("\nGenerating summary report...")
    
    report_path = output_dir / "analysis_summary.txt"
    
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("="*70 + "\n")
        f.write("GRAND AVERAGE PSD ANALYSIS - SUMMARY REPORT\n")
        f.write("(Frequency-Domain Averaging: PSD per epoch - average PSDs)\n")
        f.write("="*70 + "\n\n")
        
        f.write(f"FREQUENCY RANGE: 0.5 - {FMAX} Hz (clinical bands)\n\n")
        
        f.write("SAMPLE SIZES:\n")
        f.write(f"  Epilepsy patients: {len(epilepsy_results)}\n")
        f.write(f"  Control patients: {len(control_results)}\n")
        f.write(f"  Total patients: {len(epilepsy_results) + len(control_results)}\n\n")
        
        f.write("AVERAGE EPOCHS PER PATIENT:\n")
        epi_epochs = [r["n_epochs"] for r in epilepsy_results]
        ctrl_epochs = [r["n_epochs"] for r in control_results]
        f.write(f"  Epilepsy: {np.mean(epi_epochs):.1f} ± {np.std(epi_epochs):.1f}\n")
        f.write(f"  Control: {np.mean(ctrl_epochs):.1f} ± {np.std(ctrl_epochs):.1f}\n\n")
        
        f.write("BAND POWER COMPARISONS (averaged across channels):\n")
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        
        for band in bands:
            epi_powers = []
            ctrl_powers = []
            
            for result in epilepsy_results:
                bp = extract_band_powers(result["psd_data"], result["freqs"])
                epi_powers.append(np.mean(bp[band]))
            
            for result in control_results:
                bp = extract_band_powers(result["psd_data"], result["freqs"])
                ctrl_powers.append(np.mean(bp[band]))
            
            t_stat, p_val = stats.ttest_ind(epi_powers, ctrl_powers)
            cohen_d = (np.mean(epi_powers) - np.mean(ctrl_powers)) / np.sqrt(
                (np.std(epi_powers)**2 + np.std(ctrl_powers)**2) / 2
            )
            
            f.write(f"\n  {band.upper()}:\n")
            f.write(f"    Epilepsy: {np.mean(epi_powers):.6e} ± {np.std(epi_powers):.6e} V²/Hz\n")
            f.write(f"    Control:  {np.mean(ctrl_powers):.6e} ± {np.std(ctrl_powers):.6e} V²/Hz\n")
            f.write(f"    t-statistic: {t_stat:.3f}\n")
            f.write(f"    p-value: {p_val:.6f}\n")
            f.write(f"    Cohen's d: {cohen_d:.3f}\n")
            
            if p_val < 0.001:
                f.write(f"    Significance: *** (p < 0.001)\n")
            elif p_val < 0.01:
                f.write(f"    Significance: ** (p < 0.01)\n")
            elif p_val < 0.05:
                f.write(f"    Significance: * (p < 0.05)\n")
            else:
                f.write(f"    Significance: ns (not significant)\n")
        
        f.write("\n" + "="*70 + "\n")
        f.write("OUTPUT FILES:\n")
        f.write("  - group_comparison_psd.png: Main comparison figure\n")
        f.write("  - psd_difference.png: Difference plot with significance\n")
        f.write("  - band_power_comparison.png: Bar plot by frequency band\n")
        f.write("  - band_powers_all_patients.csv: Features for GNN\n")
        if Path(output_dir / "individual_patients").exists():
            f.write("  - individual_patients/: PSD data for each patient\n")
        f.write("="*70 + "\n")
    
    print(f"  ✓ Saved: analysis_summary.txt")


def main():
    import time
    
    parser = argparse.ArgumentParser(
        description="Batch grand average PSD analysis with group comparisons",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data_dir",
        required=True,
        help="Root preprocessed data directory"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for analysis results"
    )
    parser.add_argument(
        "--max_patients",
        type=int,
        default=None,
        help="Maximum number of patients to process (default: all)"
    )
    parser.add_argument(
        "--save_individual",
        action="store_true",
        help="Save individual patient PSDs"
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        return
    
    # Create output directories
    output_dir.mkdir(parents=True, exist_ok=True)
    if args.save_individual:
        (output_dir / "individual_patients").mkdir(exist_ok=True)
    
    # Start timing
    start_time = time.time()
    
    print("\n" + "="*70)
    print("BATCH GRAND AVERAGE PSD ANALYSIS")
    print(f"(Frequency-Domain Averaging: 0.5-{FMAX} Hz)")
    print("="*70)
    
    # Discover patients
    patients = discover_patients(data_dir)
    
    if not patients:
        print("Error: No patients found!")
        return
    
    # Limit if requested
    if args.max_patients:
        patients = dict(list(patients.items())[:args.max_patients])
        print(f"Limited to {len(patients)} patients")
    
    # Separate by group
    epilepsy_patients = {pid: data for pid, data in patients.items() if data["label"] == 1}
    control_patients = {pid: data for pid, data in patients.items() if data["label"] == 0}
    
    print(f"\nPatients discovered:")
    print(f"  - Epilepsy: {len(epilepsy_patients)}")
    print(f"  - Control: {len(control_patients)}")
    print(f"  - Total: {len(patients)}")
    
    # Process all patients
    print("\n" + "="*70)
    print("PROCESSING PATIENTS")
    print("="*70)
    
    all_results = []
    epilepsy_results = []
    control_results = []
    
    for patient_id, patient_data in tqdm(patients.items(), desc="Computing PSDs"):
        result = compute_patient_psd(
            patient_id=patient_id,
            epoch_files=patient_data["files"],
            label=patient_data["label"]
        )
        
        if result is not None:
            all_results.append(result)
            
            if result["label"] == 1:
                epilepsy_results.append(result)
            else:
                control_results.append(result)
            
            # Save individual patient data if requested
            if args.save_individual:
                ind_dir = output_dir / "individual_patients"
                np.save(ind_dir / f"{patient_id}_psd.npy", result["psd_data"])
                np.save(ind_dir / f"{patient_id}_freqs.npy", result["freqs"])
    
    print(f"\n✓ Successfully processed: {len(all_results)}/{len(patients)} patients")
    print(f"  - Epilepsy: {len(epilepsy_results)}")
    print(f"  - Control: {len(control_results)}")
    
    if len(epilepsy_results) == 0 or len(control_results) == 0:
        print("\n⚠️ WARNING: Need both epilepsy and control patients for comparison!")
        return
    
    # Generate analysis outputs
    print("\n" + "="*70)
    print("GENERATING ANALYSIS OUTPUTS")
    print("="*70)
    
    # Main comparison figure
    plot_group_comparison_psd(epilepsy_results, control_results, output_dir)
    
    # Difference plot
    plot_psd_difference(epilepsy_results, control_results, output_dir)
    
    # Band power comparison
    plot_band_power_comparison(epilepsy_results, control_results, output_dir)
    
    # Save band powers as CSV (for GNN)
    df = save_band_powers_csv(all_results, output_dir)
    
    # Summary report
    create_summary_report(epilepsy_results, control_results, output_dir)
    
    # Calculate total elapsed time
    total_time = time.time() - start_time
    minutes = int(total_time // 60)
    seconds = int(total_time % 60)
    
    # Final summary
    print("\n" + "="*70)
    print("ANALYSIS COMPLETE")
    print("="*70)
    print(f"Output directory: {output_dir}")
    print(f"Total time: {minutes} minutes {seconds} seconds ({total_time:.1f}s)")
    print(f"Average time per patient: {total_time/len(all_results):.2f} seconds")
    print(f"\nKey outputs:")
    print(f"  📊 group_comparison_psd.png - Main figure for thesis")
    print(f"  📊 psd_difference.png - Statistical comparison")
    print(f"  📊 band_power_comparison.png - Band power bar plot")
    print(f"  📄 band_powers_all_patients.csv - Features for GNN ({df.shape[0]} patients × {df.shape[1]-4} features)")
    print(f"  📄 analysis_summary.txt - Statistical results")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()