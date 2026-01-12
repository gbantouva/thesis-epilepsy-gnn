"""
BATCH CONNECTIVITY COMPUTATION - PRODUCTION VERSION
===================================================
Processes all patients in data_pp_balanced/
- BIC-adaptive order selection per epoch
- Saves DTF/PDC matrices + metadata
- Optional: Save diagnostic plots for subset of patients
- Resume capability (skips existing files)
"""

import argparse
from pathlib import Path
import numpy as np
import warnings
from tqdm import tqdm
from scipy import linalg
import sys

from statsmodels.tsa.vector_ar.var_model import VAR

# Optional plotting imports
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==============================================================================
# CORE CONNECTIVITY FUNCTIONS
# ==============================================================================

def compute_dtf_pdc_from_var(coefs, fs=250.0, nfft=512):
    """Compute DTF and PDC from VAR coefficients."""
    p, K, _ = coefs.shape
    n_freqs = nfft // 2 + 1
    freqs = np.linspace(0, fs/2, n_freqs)
    
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I = np.eye(K)
    
    # Transfer Function
    for f_idx, f in enumerate(freqs):
        A_sum = np.zeros((K, K), dtype=complex)
        for k in range(p):
            phase = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
            A_sum += coefs[k] * phase
        
        A_f[f_idx] = I - A_sum
        
        try:
            H_f[f_idx] = linalg.inv(A_f[f_idx])
        except linalg.LinAlgError:
            H_f[f_idx] = linalg.pinv(A_f[f_idx])
    
    # PDC (Normalized by Source/Column)
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af = A_f[f_idx]
        col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]
    
    # DTF (Normalized by Sink/Row)
    dtf = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Hf = H_f[f_idx]
        row_norms = np.sqrt(np.sum(np.abs(Hf)**2, axis=1))
        row_norms[row_norms == 0] = 1e-10
        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]
    
    return dtf, pdc, freqs


def compute_epoch_connectivity(data, fs=250.0, min_order=2, max_order=20,
                               freq_range=(0.5, 80), nfft=512):
    """
    Compute DTF and PDC for one epoch with BIC order selection.
    
    Returns
    -------
    dict or None
        Dictionary with 'dtf', 'pdc', 'order', 'bic' or None if failed
    """
    # Scale data
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    data_scaled = data / data_std
    
    # BIC order selection
    try:
        model = VAR(data_scaled.T)
        
        bic_values = []
        for p in range(min_order, max_order + 1):
            try:
                result = model.fit(maxlags=p, trend='c', verbose=False)
                bic_values.append((p, result.bic))
            except:
                continue
        
        if len(bic_values) == 0:
            return None
        
        # Select best order
        best_order, best_bic = min(bic_values, key=lambda x: x[1])
        
        # Fit final model
        results = model.fit(maxlags=best_order, trend='c', verbose=False)
        
        if results.k_ar == 0:
            return None
        
        # Check stability (lenient)
        try:
            stable = results.is_stable()
            if not stable:
                from statsmodels.tsa.vector_ar.util import comp_matrix
                try:
                    A_comp = comp_matrix(results.coefs)
                    eigenvalues = np.linalg.eigvals(A_comp)
                    max_eig = np.max(np.abs(eigenvalues))
                    if max_eig > 1.2:
                        return None
                except:
                    return None
        except:
            pass
        
        coefs = results.coefs
        selected_order = results.k_ar
        
    except Exception:
        return None
    
    # Compute DTF/PDC
    try:
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(coefs, fs, nfft)
    except Exception:
        return None
    
    # Integrate over frequency band
    idx_band = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
    if len(idx_band) == 0:
        return None
    
    dtf_integrated = np.mean(dtf_spectrum[:, :, idx_band], axis=2)
    pdc_integrated = np.mean(pdc_spectrum[:, :, idx_band], axis=2)
    
    return {
        'dtf': dtf_integrated,
        'pdc': pdc_integrated,
        'order': selected_order,
        'bic': best_bic
    }


# ==============================================================================
# OPTIONAL: DIAGNOSTIC PLOTTING
# ==============================================================================

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
    'T3', 'C3', 'Cz', 'C4', 'T4',
    'T5', 'P3', 'Pz', 'P4', 'T6',
    'O1', 'Oz', 'O2', 'A1', 'A2'
]


def save_diagnostic_plot(dtf_matrices, pdc_matrices, orders, patient_id, output_dir):
    """
    Save diagnostic plot for one patient (first epoch only).
    
    Parameters
    ----------
    dtf_matrices : ndarray, shape (n_epochs, 22, 22)
    pdc_matrices : ndarray, shape (n_epochs, 22, 22)
    orders : ndarray, shape (n_epochs,)
    patient_id : str
    output_dir : Path
    """
    if len(dtf_matrices) == 0:
        return
    
    # Use first epoch
    dtf = dtf_matrices[0]
    pdc = pdc_matrices[0]
    order = orders[0]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    global_max = max(dtf.max(), pdc.max())
    
    # DTF
    sns.heatmap(dtf, ax=axes[0], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=global_max,
               cbar_kws={'label': 'Strength'})
    axes[0].set_title(f'DTF - {patient_id} (p={order})', fontsize=12, fontweight='bold')
    axes[0].tick_params(axis='x', rotation=45)
    
    # PDC
    sns.heatmap(pdc, ax=axes[1], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=global_max,
               cbar_kws={'label': 'Strength'})
    axes[1].set_title(f'PDC - {patient_id} (p={order})', fontsize=12, fontweight='bold')
    axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    
    plot_dir = output_dir / 'diagnostic_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{patient_id}_connectivity.png', dpi=150, bbox_inches='tight')
    plt.close()


# ==============================================================================
# MAIN BATCH PROCESSING
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Batch connectivity computation with BIC order selection"
    )
    parser.add_argument("--input_dir", required=True, help="Path to data_pp_balanced/")
    parser.add_argument("--output_dir", required=True, help="Where to save results")
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--freq_low", type=float, default=0.5)
    parser.add_argument("--freq_high", type=float, default=80.0)
    parser.add_argument("--min_order", type=int, default=2)
    parser.add_argument("--max_order", type=int, default=20)
    parser.add_argument("--save_plots", type=int, default=5,
                       help="Save diagnostic plots for first N patients (0=none)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all epoch files
    epoch_files = list(input_dir.rglob("*_epochs.npy"))
    
    if len(epoch_files) == 0:
        print(f"❌ Error: No *_epochs.npy files found in {input_dir}")
        sys.exit(1)
    
    print("="*80)
    print("BATCH CONNECTIVITY COMPUTATION")
    print("="*80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(epoch_files)} patient files")
    print(f"Frequency range:  {args.freq_low}-{args.freq_high} Hz")
    print(f"FFT points:       {args.nfft}")
    print(f"Order range:      {args.min_order}-{args.max_order} (BIC-selected)")
    print(f"Save plots:       {'First ' + str(args.save_plots) + ' patients' if args.save_plots > 0 else 'No'}")
    print("="*80)
    print()
    
    stats = {
        'success': 0,
        'failed': 0,
        'total_epochs': 0,
        'valid_epochs': 0,
        'orders': [],
        'plots_saved': 0
    }
    
    for file_idx, f in enumerate(tqdm(epoch_files, desc="Processing files")):
        rel_path = f.relative_to(input_dir)
        out_file = output_dir / rel_path.parent / f"{f.stem.replace('_epochs', '_graphs')}.npz"
        
        # Resume capability
        if out_file.exists():
            continue
        
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load data
        try:
            epochs = np.load(f)
            labels_file = f.parent / f"{f.stem.replace('_epochs', '_labels')}.npy"
            labels = np.load(labels_file)
        except Exception as e:
            tqdm.write(f"⚠️  Failed to load {f.name}: {e}")
            stats['failed'] += 1
            continue
        
        stats['total_epochs'] += len(epochs)
        
        # Process all epochs for this patient
        dtf_matrices = []
        pdc_matrices = []
        valid_indices = []
        selected_orders = []
        bic_values = []
        
        for i in range(len(epochs)):
            result = compute_epoch_connectivity(
                epochs[i],
                fs=250.0,
                min_order=args.min_order,
                max_order=args.max_order,
                freq_range=(args.freq_low, args.freq_high),
                nfft=args.nfft
            )
            
            if result is not None:
                dtf_matrices.append(result['dtf'])
                pdc_matrices.append(result['pdc'])
                valid_indices.append(i)
                selected_orders.append(result['order'])
                bic_values.append(result['bic'])
        
        if len(dtf_matrices) > 0:
            # Save results
            save_data = {
                'dtf': np.array(dtf_matrices),
                'pdc': np.array(pdc_matrices),
                'labels': labels[valid_indices],
                'indices': np.array(valid_indices),
                'orders': np.array(selected_orders),
                'bic_values': np.array(bic_values)
            }
            
            np.savez_compressed(out_file, **save_data)
            
            stats['success'] += 1
            stats['valid_epochs'] += len(dtf_matrices)
            stats['orders'].extend(selected_orders)
            
            # Optional: Save diagnostic plot
            if args.save_plots > 0 and stats['plots_saved'] < args.save_plots:
                try:
                    patient_id = f.stem.replace('_epochs', '')
                    save_diagnostic_plot(
                        np.array(dtf_matrices),
                        np.array(pdc_matrices),
                        np.array(selected_orders),
                        patient_id,
                        output_dir
                    )
                    stats['plots_saved'] += 1
                except Exception as e:
                    tqdm.write(f"⚠️  Failed to save plot for {f.name}: {e}")
        else:
            tqdm.write(f"⚠️  No valid epochs for {f.name}")
            stats['failed'] += 1
    
    # Print summary
    print("\n" + "="*80)
    print("PROCESSING COMPLETE")
    print("="*80)
    print(f"Successful files:       {stats['success']}")
    print(f"Failed files:           {stats['failed']}")
    print(f"Total epochs processed: {stats['total_epochs']}")
    print(f"Valid epochs:           {stats['valid_epochs']}")
    
    if stats['total_epochs'] > 0:
        success_rate = 100 * stats['valid_epochs'] / stats['total_epochs']
        print(f"Success rate:           {success_rate:.1f}%")
    
    if stats['orders']:
        orders_array = np.array(stats['orders'])
        print(f"\nMVAR Order Statistics (BIC-selected):")
        print(f"  Mean:   {orders_array.mean():.2f}")
        print(f"  Median: {np.median(orders_array):.0f}")
        print(f"  Std:    {orders_array.std():.2f}")
        print(f"  Range:  {orders_array.min()}-{orders_array.max()}")
        
        # Top 5 most common orders
        unique_orders, counts = np.unique(orders_array, return_counts=True)
        print(f"\n  Top 5 Most Common Orders:")
        for i in range(min(5, len(unique_orders))):
            idx = np.argsort(-counts)[i]
            order = unique_orders[idx]
            count = counts[idx]
            percentage = 100 * count / len(orders_array)
            print(f"    Order {order:2d}: {count:6d} epochs ({percentage:5.1f}%)")
    
    if stats['plots_saved'] > 0:
        print(f"\nDiagnostic plots saved: {stats['plots_saved']} (in diagnostic_plots/)")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()
