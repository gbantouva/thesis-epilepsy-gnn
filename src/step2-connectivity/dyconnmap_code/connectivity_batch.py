"""
Batch Effective Connectivity (PDC/DTF) - FINAL HIERARCHY VERSION
Features:
1. Mirrors EXACT folder structure from input to output.
2. Groups by Patient ID (aaaaaanr, etc.).
3. --max_patients N: Limits processing to first N patients (alphabetical).
4. Skips existing files (safe to restart).
5. Robust Math: Stability Check, N_FFT=512.
"""

import argparse
import numpy as np
from pathlib import Path
import warnings
from scipy import linalg
import sys
from tqdm import tqdm
import gc
from collections import defaultdict

# Try importing statsmodels
try:
    from statsmodels.tsa.api import VAR
except ImportError:
    print("❌ statsmodels not installed: pip install statsmodels")
    sys.exit(1)

warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLING_RATE = 250.0
MVAR_ORDER = 15
N_FFT = 512 

BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 80)
}
BAND_NAMES = list(BANDS.keys())


def get_freq_indices(freqs, bands):
    indices = {}
    for name, (f_low, f_high) in bands.items():
        indices[name] = np.where((freqs >= f_low) & (freqs <= f_high))[0]
    return indices


def compute_connectivity_epoch(data, method='PDC', order=15, n_fft=512, fs=250.0, squared=True):
    """Compute connectivity for a single epoch array (channels, times)."""
    n_channels, n_samples = data.shape
    
    # 1. Fit VAR
    try:
        model = VAR(data.T)
        results = model.fit(maxlags=order, trend='c')
    except:
        return None, None
    
    # 2. Stability Check
    if not results.is_stable():
        return None, None

    coefs = results.coefs
    
    # 3. Spectral Matrix A(f)
    freqs = np.linspace(0, fs/2, n_fft)
    A_f = np.zeros((n_fft, n_channels, n_channels), dtype=np.complex128)
    I = np.eye(n_channels)
    
    for f_idx, f in enumerate(freqs):
        sum_Ak = np.zeros((n_channels, n_channels), dtype=np.complex128)
        for k in range(order):
            phasor = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
            sum_Ak += coefs[k] * phasor
        A_f[f_idx] = I - sum_Ak

    # 4. Invert for DTF if needed
    H_f = None
    if method == 'DTF':
        H_f = np.zeros((n_fft, n_channels, n_channels), dtype=np.complex128)
        for f in range(n_fft):
            try:
                H_f[f] = linalg.inv(A_f[f])
            except:
                H_f[f] = linalg.pinv(A_f[f])

    # 5. Calculate Measure
    connectivity = np.zeros((n_channels, n_channels, n_fft))
    
    if method == 'PDC':
        for f in range(n_fft):
            Af = A_f[f]
            col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
            col_norms[col_norms == 0] = 1e-10
            connectivity[:, :, f] = np.abs(Af) / col_norms[None, :]
            
    elif method == 'DTF':
        for f in range(n_fft):
            Hf = H_f[f]
            row_norms = np.sqrt(np.sum(np.abs(Hf)**2, axis=1))
            row_norms[row_norms == 0] = 1e-10
            connectivity[:, :, f] = np.abs(Hf) / row_norms[:, None]

    if squared:
        connectivity = connectivity ** 2
        
    return connectivity, freqs


def process_file(input_file: Path, output_file: Path, method: str):
    """Process one file containing multiple epochs."""
    
    # Load Data
    try:
        epochs = np.load(input_file)
    except Exception as e:
        print(f"\n❌ Error loading {input_file.name}: {e}")
        return

    n_epochs, n_channels, n_times = epochs.shape
    
    # Prepare Output
    adj_matrices = np.zeros((n_epochs, len(BANDS), n_channels, n_channels), dtype=np.float32)
    
    valid_count = 0
    band_indices = None
    
    for i in range(n_epochs):
        conn, freqs = compute_connectivity_epoch(
            epochs[i], method=method, order=MVAR_ORDER, 
            n_fft=N_FFT, fs=SAMPLING_RATE, squared=True
        )
        
        if conn is None:
            continue # Skip unstable epoch
            
        if band_indices is None:
            band_indices = get_freq_indices(freqs, BANDS)
            
        for b_idx, band in enumerate(BAND_NAMES):
            f_idx = band_indices[band]
            if len(f_idx) > 0:
                adj_matrices[i, b_idx, :, :] = np.mean(conn[:, :, f_idx], axis=2)
        
        valid_count += 1

    # Save result
    if valid_count > 0:
        np.save(output_file, adj_matrices)
        
        # Save valid mask
        is_valid = np.any(adj_matrices.reshape(n_epochs, -1), axis=1)
        mask_file = output_file.parent / f"{input_file.stem}_valid_mask.npy"
        np.save(mask_file, is_valid)
        
        tqdm.write(f"  ✓ {input_file.stem}: {valid_count}/{n_epochs} valid")
    else:
        tqdm.write(f"  ⚠️ {input_file.stem}: 0 valid epochs (skipping save)")
        
    # Force memory cleanup
    del epochs
    del adj_matrices
    gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Batch Connectivity Processing")
    parser.add_argument("--data_dir", required=True, help="Root preprocessed data folder")
    parser.add_argument("--output_dir", required=True, help="Where to save connectivity files")
    parser.add_argument("--method", default="PDC", choices=["PDC", "DTF"])
    parser.add_argument("--max_patients", type=int, default=None, 
                        help="Limit processing to the first N patients (useful for testing/avoiding crash)")
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    
    if not data_dir.exists():
        print(f"❌ Error: Input directory not found: {data_dir}")
        return

    # 1. Find and Group Files by Patient
    print(f"Searching for files in {data_dir}...")
    all_files = list(data_dir.rglob("*_epochs.npy"))
    
    if not all_files:
        print("❌ No _epochs.npy files found.")
        return

    patient_map = defaultdict(list)
    for f in all_files:
        # Extract PID (e.g. "aaaaaanr" from "aaaaaanr_s001_...")
        pid = f.stem.split('_')[0]
        patient_map[pid].append(f)
        
    all_patients = sorted(patient_map.keys())
    print(f"Found {len(all_files)} files across {len(all_patients)} patients.")
    
    # 2. Apply Patient Limit
    if args.max_patients is not None:
        selected_patients = all_patients[:args.max_patients]
        print(f"⚠️  Limiting to first {args.max_patients} patients (Alphabetical order)")
    else:
        selected_patients = all_patients

    # Flatten list back to files
    files_to_process = []
    for pid in selected_patients:
        files_to_process.extend(patient_map[pid])

    print(f"\n{'='*60}")
    print(f"STARTING BATCH PROCESSING")
    print(f"Patients: {len(selected_patients)}")
    print(f"Files:    {len(files_to_process)}")
    print(f"Method:   {args.method} (Squared)")
    print(f"{'='*60}")

    # 3. Loop and Process
    for f in tqdm(files_to_process, desc="Processing"):
        # --- MIRRORING LOGIC ---
        # Calculate relative path from input root
        # e.g. version3/00_epilepsy/aaaaaanr/s001/02_tcp_le/file.npy
        rel_path = f.relative_to(data_dir)
        
        # Create target folder structure
        target_folder = output_dir / rel_path.parent
        target_folder.mkdir(parents=True, exist_ok=True)
        
        # Define output filename: {pid}_{method}_connectivity.npy
        pid = f.stem.replace("_epochs", "")
        out_name = f"{pid}_{args.method}_connectivity.npy"
        out_file = target_folder / out_name
        
        # SKIP LOGIC: If result exists, skip calculation
        if out_file.exists():
            continue
            
        process_file(f, out_file, args.method)

    print(f"\n✅ Batch processing complete.")

if __name__ == "__main__":
    main()