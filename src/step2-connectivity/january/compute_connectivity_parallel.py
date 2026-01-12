"""
BATCH CONNECTIVITY - PARALLEL VERSION (FAST) WITH MULTI-BAND SUPPORT
============================================
- Computes connectivity for multiple frequency bands
- Uses MULTIPROCESSING to run on all CPU cores
- Integrated (0.5-80 Hz) + 6 specific bands
"""

import argparse
from pathlib import Path
import numpy as np
import warnings
from tqdm import tqdm
from scipy import linalg
import sys
import concurrent.futures
import time

from statsmodels.tsa.vector_ar.var_model import VAR

# Plotting setup for the main process only
import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ==============================================================================
# CORE FUNCTIONS (Must be strictly defined for pickling)
# ==============================================================================

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

def process_single_epoch(data, fs, min_order, max_order, freq_range, nfft):
    """Helper to process one epoch inside the worker. NOW COMPUTES MULTIPLE FREQUENCY BANDS!"""
    # Scale data
    data_std = np.std(data)
    if data_std < 1e-10: return None
    data_scaled = data / data_std
    
    try:
        model = VAR(data_scaled.T)
        bic_values = []
        for p in range(min_order, max_order + 1):
            try:
                result = model.fit(maxlags=p, trend='c', verbose=False)
                bic_values.append((p, result.bic))
            except: continue
        
        if not bic_values: return None
        
        best_order, best_bic = min(bic_values, key=lambda x: x[1])
        results = model.fit(maxlags=best_order, trend='c', verbose=False)
        
        if results.k_ar == 0: return None
        
        # Stability check
        try:
            if not results.is_stable():
                from statsmodels.tsa.vector_ar.util import comp_matrix
                max_eig = np.max(np.abs(np.linalg.eigvals(comp_matrix(results.coefs))))
                if max_eig > 1.2: return None
        except: pass
        
        # Compute DTF/PDC
        #dtf_s, pdc_s, freqs = compute_dtf_pdc_from_var(results.coefs, fs, nfft)
        #new
        # Compute DTF/PDC full spectrum
        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(
            results.coefs, fs, nfft
        )
        #end of new
        # === MULTI-BAND INTEGRATION ===
        #new
        # Define frequency bands (as requested by professor)
        bands = {
            'integrated': (0.5, 80.0),   # Overall
            'delta': (0.5, 4.0),         # δ
            'theta': (4.0, 8.0),         # θ
            'alpha': (8.0, 15.0),        # α
            'beta': (15.0, 30.0),        # β
            'gamma1': (30.0, 55.0),      # γ1
            'gamma2': (65.0, 80.0)       # γ2
        }
        #end of new
        # Integrate
        #idx_band = np.where((freqs >= freq_range[0]) & (freqs <= freq_range[1]))[0]
        #if len(idx_band) == 0: return None
        #new
        # Integrate each band
        dtf_bands = {}
        pdc_bands = {}
        
        for band_name, (f_low, f_high) in bands.items():
            idx_band = np.where((freqs >= f_low) & (freqs <= f_high))[0]
            
            if len(idx_band) == 0:
                # Band too narrow or outside range, skip this epoch
                return None
            
            # Average over frequency band
            dtf_bands[band_name] = np.mean(dtf_spectrum[:, :, idx_band], axis=2)
            pdc_bands[band_name] = np.mean(pdc_spectrum[:, :, idx_band], axis=2)
        #end of new
        
        #return {
        #    'dtf': np.mean(dtf_s[:, :, idx_band], axis=2),
        #    'pdc': np.mean(pdc_s[:, :, idx_band], axis=2),
        #    'order': results.k_ar,
        #    'bic': best_bic
        #}
        #new
        return {
            'dtf_bands': dtf_bands,  # Dict with 7 matrices (20×20 each)
            'pdc_bands': pdc_bands,  # Dict with 7 matrices (20×20 each)
            'order': results.k_ar,
            'bic': best_bic
        }
        # end of new
    except:
        return None

# ==============================================================================
# FILE WORKER (Runs on 1 CPU Core)
# ==============================================================================

def process_patient_file(args_bundle):
    """
    Processes one entire patient file.
    Unpacks args_bundle to avoid pickling issues.
    """
    f, input_dir, output_dir, fs, min_order, max_order, freq_low, freq_high, nfft = args_bundle
    
    try:
        # Setup paths
        rel_path = f.relative_to(input_dir)
        out_file = output_dir / rel_path.parent / f"{f.stem.replace('_epochs', '_graphs')}.npz"
        
        # Skip if done
        if out_file.exists():
            return ('skipped', 0)
        
        out_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Load data
        epochs = np.load(f)
        labels_file = f.parent / f"{f.stem.replace('_epochs', '_labels')}.npy"
        labels = np.load(labels_file)
        

        #NEW TO SAVE ALSO BANDS
        # dtf_list, pdc_list = [], []
        # Initialize lists for each band
        dtf_integrated, pdc_integrated = [], []
        dtf_delta, pdc_delta = [], []
        dtf_theta, pdc_theta = [], []
        dtf_alpha, pdc_alpha = [], []
        dtf_beta, pdc_beta = [], []
        dtf_gamma1, pdc_gamma1 = [], []
        dtf_gamma2, pdc_gamma2 = [], []
        #END OF NEW
        valid_idx, orders, bics = [], [], []
        
        # Process epochs
        for i in range(len(epochs)):
            res = process_single_epoch(epochs[i], fs, min_order, max_order, (freq_low, freq_high), nfft)
            if res:
                #dtf_list.append(res['dtf'])
                #pdc_list.append(res['pdc'])
                # Extract all bands
                dtf_integrated.append(res['dtf_bands']['integrated'])
                pdc_integrated.append(res['pdc_bands']['integrated'])
                
                dtf_delta.append(res['dtf_bands']['delta'])
                pdc_delta.append(res['pdc_bands']['delta'])
                
                dtf_theta.append(res['dtf_bands']['theta'])
                pdc_theta.append(res['pdc_bands']['theta'])
                
                dtf_alpha.append(res['dtf_bands']['alpha'])
                pdc_alpha.append(res['pdc_bands']['alpha'])
                
                dtf_beta.append(res['dtf_bands']['beta'])
                pdc_beta.append(res['pdc_bands']['beta'])
                
                dtf_gamma1.append(res['dtf_bands']['gamma1'])
                pdc_gamma1.append(res['pdc_bands']['gamma1'])
                
                dtf_gamma2.append(res['dtf_bands']['gamma2'])
                pdc_gamma2.append(res['pdc_bands']['gamma2'])
                #END OF NEW
                valid_idx.append(i)
                orders.append(res['order'])
                bics.append(res['bic'])
        
        # Save if valid
        #if dtf_list:
        #NEW
        if dtf_integrated:
            np.savez_compressed(out_file, 
                #dtf=np.array(dtf_list), 
                #pdc=np.array(pdc_list),
                # Integrated (0.5-80 Hz)
                dtf_integrated=np.array(dtf_integrated),
                pdc_integrated=np.array(pdc_integrated),
                
                # Per-band
                dtf_delta=np.array(dtf_delta),
                pdc_delta=np.array(pdc_delta),
                
                dtf_theta=np.array(dtf_theta),
                pdc_theta=np.array(pdc_theta),
                
                dtf_alpha=np.array(dtf_alpha),
                pdc_alpha=np.array(pdc_alpha),
                
                dtf_beta=np.array(dtf_beta),
                pdc_beta=np.array(pdc_beta),
                
                dtf_gamma1=np.array(dtf_gamma1),
                pdc_gamma1=np.array(pdc_gamma1),
                
                dtf_gamma2=np.array(dtf_gamma2),
                pdc_gamma2=np.array(pdc_gamma2),
                
                # Metadata
                #END OF NEW
                labels=labels[valid_idx], 
                indices=np.array(valid_idx),
                orders=np.array(orders), 
                bic_values=np.array(bics)
            )
            #return ('success', len(dtf_list), f, dtf_list[0], pdc_list[0], orders[0])
            #new
            # Return for diagnostic plotting (use integrated band)
            return ('success', len(dtf_integrated), f, 
                   dtf_integrated[0], pdc_integrated[0], orders[0])
            #end of new
        else:
            return ('failed', 0)
            
    except Exception as e:
        return ('error', str(e))

# ==============================================================================
# MAIN (Orchestrator)
# ==============================================================================

def save_diagnostic_plot(dtf, pdc, order, patient_id, output_dir):
    """Quick plot saver for the main process."""
    #CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 
    #                 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'A1', 'A2']
    CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                     'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                     'T5', 'P3', 'Pz', 'P4', 'T6', 
                     'O1', 'Oz', 'O2']  # ← T1, T2 (NOT A1, A2!)

    fig, ax = plt.subplots(1, 2, figsize=(14, 6))
    vmax = max(dtf.max(), pdc.max())
    
    sns.heatmap(dtf, ax=ax[0], cmap='viridis', square=True, vmin=0, vmax=vmax,
                xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES)
    ax[0].set_title(f'DTF (Integrated) - {patient_id} (p={order})')
    
    sns.heatmap(pdc, ax=ax[1], cmap='viridis', square=True, vmin=0, vmax=vmax,
                xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES)
    ax[1].set_title(f'PDC (Integrated) - {patient_id} (p={order})')
    
    plot_dir = output_dir / 'diagnostic_plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    plt.savefig(plot_dir / f'{patient_id}_connectivity.png', dpi=100, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description="Parallel Connectivity Computation - Multi-Band")
    parser.add_argument("--input_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--workers", type=int, default=None, help="Number of CPU cores (default: all)")
    parser.add_argument("--save_plots", type=int, default=5)
    # Add other args as defaults for simplicity
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    epoch_files = list(input_dir.rglob("*_epochs.npy"))
    
    print(f"STARTING PARALLEL PROCESSING (MULTI-BAND)")
    print(f"Files to process: {len(epoch_files)}")
    #NEW
    print(f"Bands: integrated, delta, theta, alpha, beta, gamma1, gamma2")
    #END OF NEW
    print(f"Output: {output_dir}")
    
    # Prepare arguments for workers
    # (file, input_dir, output_dir, fs, min_order, max_order, freq_low, freq_high, nfft)
    tasks = [
        (f, input_dir, output_dir, 250.0, 2, 15, 0.5, 80.0, 512)
        for f in epoch_files
    ]
    
    plots_saved = 0
    stats = {'success': 0, 'skipped': 0, 'failed': 0}
    
    # Run Parallel
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all tasks
        futures = {executor.submit(process_patient_file, task): task[0] for task in tasks}
        
        # Watch progress
        for future in tqdm(concurrent.futures.as_completed(futures), total=len(epoch_files), desc="Multi-Band Parallel Progress"):
            result = future.result()
            status = result[0]
            
            if status == 'success':
                stats['success'] += 1
                # Save plot if needed (Main process handles plotting to avoid threading issues)
                if plots_saved < args.save_plots:
                    _, _, f_path, dtf, pdc, order = result
                    save_diagnostic_plot(dtf, pdc, order, f_path.stem.replace('_epochs', ''), output_dir)
                    plots_saved += 1
            elif status == 'skipped':
                stats['skipped'] += 1
            elif status == 'failed':
                stats['failed'] += 1
            elif status == 'error':
                print(f"Error: {result[1]}")
                stats['failed'] += 1

    print(f"\nDONE! Success: {stats['success']}, Skipped: {stats['skipped']}, Failed: {stats['failed']}")

    #new
    print(f"\nEach epoch now has 7 frequency bands:")
    print(f"  - integrated (0.5-80 Hz)")
    print(f"  - delta (0.5-4 Hz)")
    print(f"  - theta (4-8 Hz)")
    print(f"  - alpha (8-15 Hz)")
    print(f"  - beta (15-30 Hz)")
    print(f"  - gamma1 (30-55 Hz)")
    print(f"  - gamma2 (65-80 Hz)")
    #end of new
if __name__ == "__main__":
    main()