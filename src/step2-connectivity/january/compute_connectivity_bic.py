"""
Effective Connectivity with BIC Order Selection
================================================
- Adaptive order selection per epoch using BIC
- Orders evaluated from 2 to 20
- Data scaling for numerical stability
- Lenient stability criteria (max_eig < 1.2)
"""

import argparse
from pathlib import Path
import numpy as np
import warnings
from tqdm import tqdm
from scipy import linalg
import sys

from statsmodels.tsa.vector_ar.var_model import VAR

warnings.filterwarnings("ignore")


def compute_dtf_pdc_from_var(coefs, fs=250.0, nfft=512):
    """Compute DTF and PDC from VAR coefficients."""
    p, K, _ = coefs.shape
    n_freqs = nfft // 2 + 1
    freqs = np.linspace(0, fs/2, n_freqs)
    
    A_f = np.zeros((n_freqs, K, K), dtype=complex)
    H_f = np.zeros((n_freqs, K, K), dtype=complex)
    I = np.eye(K)
    
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
    
    # PDC
    pdc = np.zeros((K, K, n_freqs))
    for f_idx in range(n_freqs):
        Af = A_f[f_idx]
        col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
        col_norms[col_norms == 0] = 1e-10
        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]
    
    # DTF
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
    
    KEY FEATURES:
    - Scales data for numerical stability (critical for BIC!)
    - BIC-based order selection (2-20 range)
    - Adaptive order per epoch
    - Lenient stability check (max_eig < 1.2)
    """
    # 1. Scale data for numerical stability
    data_std = np.std(data)
    if data_std < 1e-10:
        return None
    
    data_scaled = data / data_std
    
    # 2. BIC-based order selection
    try:
        model = VAR(data_scaled.T)
        
        # Evaluate different orders
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
        
        # Fit final model with best order
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
    
    # 3. Compute DTF/PDC
    try:
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(coefs, fs, nfft)
    except Exception:
        return None
    
    # 4. Integrate over frequency band
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


def main():
    parser = argparse.ArgumentParser(
        description="Compute effective connectivity (DTF/PDC) with BIC order selection"
    )
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--nfft", type=int, default=512)
    parser.add_argument("--freq_low", type=float, default=0.5)
    parser.add_argument("--freq_high", type=float, default=80.0)
    parser.add_argument("--min_order", type=int, default=2,
                       help="Minimum MVAR order for BIC search (default: 2)")
    parser.add_argument("--max_order", type=int, default=20,
                       help="Maximum MVAR order for BIC search (default: 20)")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    
    if not input_dir.exists():
        print(f"❌ Error: Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    epoch_files = list(input_dir.rglob("*_epochs.npy"))
    
    if len(epoch_files) == 0:
        print(f"❌ Error: No *_epochs.npy files found in {input_dir}")
        sys.exit(1)
    
    print("="*80)
    print("EFFECTIVE CONNECTIVITY COMPUTATION - BIC ORDER SELECTION")
    print("="*80)
    print(f"Input directory:  {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Found {len(epoch_files)} patient files")
    print(f"Frequency range:  {args.freq_low}-{args.freq_high} Hz")
    print(f"FFT points:       {args.nfft}")
    print(f"Order range:      {args.min_order}-{args.max_order} (BIC-selected)")
    print(f"Method:           DTF & PDC (magnitude)")
    print(f"Trend:            With constant term")
    print(f"Stability:        Lenient (accept max_eig < 1.2)")
    print(f"Data scaling:     Yes (for numerical stability)")
    print("="*80)
    print()
    
    stats = {
        'success': 0,
        'failed': 0,
        'total_epochs': 0,
        'valid_epochs': 0,
        'orders': [],
        'bic_values': []
    }
    
    for f in tqdm(epoch_files, desc="Processing files"):
        rel_path = f.relative_to(input_dir)
        out_file = output_dir / rel_path.parent / f"{f.stem.replace('_epochs', '_graphs')}.npz"
        
        if out_file.exists():
            continue
        
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            epochs = np.load(f)
            labels_file = f.parent / f"{f.stem.replace('_epochs', '_labels')}.npy"
            labels = np.load(labels_file)
        except Exception as e:
            tqdm.write(f"⚠️  Failed to load {f.name}: {e}")
            stats['failed'] += 1
            continue
        
        stats['total_epochs'] += len(epochs)
        
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
            stats['bic_values'].extend(bic_values)
        else:
            tqdm.write(f"⚠️  No valid epochs for {f.name}")
            stats['failed'] += 1
    
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
        print(f"  Mean:   {orders_array.mean():.1f}")
        print(f"  Median: {np.median(orders_array):.0f}")
        print(f"  Std:    {orders_array.std():.1f}")
        print(f"  Range:  {orders_array.min()}-{orders_array.max()}")
        
        # Order distribution
        unique_orders, counts = np.unique(orders_array, return_counts=True)
        print(f"\n  Order Distribution:")
        for order, count in zip(unique_orders, counts):
            percentage = 100 * count / len(orders_array)
            print(f"    Order {order:2d}: {count:5d} epochs ({percentage:5.1f}%)")
    
    print(f"\nOutput directory: {output_dir}")
    print("="*80)


if __name__ == "__main__":
    main()