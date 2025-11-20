"""
Batch Effective Connectivity (PDC/DTF) for GNNs.

This script:
1. Loads preprocessed epochs (n_epochs, 22, n_times)
2. Fits a Multivariate Autoregressive (MVAR) model PER EPOCH
3. Computes Partial Directed Coherence (PDC)
4. Integrates PDC into frequency bands (Delta, Theta, Alpha, Beta, Gamma)
5. Saves adjacency matrices: (n_epochs, n_bands, 22, 22)

Usage:
  python src/connectivity_batch.py --data_dir data_pp
  python src/connectivity_batch.py --data_dir F:\October-Thesis\thesis-epilepsy-gnn\data_pp\version3 --method PDC
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import pickle
from tqdm import tqdm
import warnings

# Import dyconnmap (from your professor's link)
try:
    from dyconnmap.fc import PDC, DTF
    from dyconnmap.ts import VAR
except ImportError:
    print("❌ Error: dyconnmap not installed.")
    print("   Run: pip install dyconnmap statsmodels")
    sys.exit(1)

# Suppress statsmodels warnings for cleaner output
warnings.filterwarnings("ignore")

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLING_RATE = 250.0
MVAR_ORDER = 15         # Model order p (15 @ 250Hz = 60ms lag). Optimal for EEG.
METHOD = 'PDC'          # 'PDC' or 'DTF'
N_FFT = 512             # Frequency resolution for the spectral conversion

# Frequency Bands for Graph Edges
BANDS = {
    'Delta': (0.5, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta':  (13, 30),
    'Gamma': (30, 80)
}
BAND_NAMES = list(BANDS.keys())

def get_freq_indices(freqs, bands):
    """Map frequency bands to FFT indices."""
    indices = {}
    for name, (f_low, f_high) in bands.items():
        # Find indices in the frequency array
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        indices[name] = idx
    return indices

def process_single_file(epoch_file: Path, out_file: Path):
    """Compute connectivity for a single file."""
    
    # 1. Load Data
    epochs = np.load(epoch_file)  # Shape: (n_epochs, n_channels, n_times)
    n_epochs, n_channels, n_times = epochs.shape
    
    # Load info to ensure channels are correct (optional, for logging)
    # pid = epoch_file.stem.replace("_epochs", "")
    # info_file = epoch_file.parent / f"{pid}_info.pkl"
    
    # 2. Prepare Output Array
    # Shape: (n_epochs, n_bands, n_channels_source, n_channels_target)
    adj_matrices = np.zeros((n_epochs, len(BANDS), n_channels, n_channels), dtype=np.float32)
    
    # 3. Initialize Estimators
    # dyconnmap VAR wrapper uses statsmodels internally
    # We do this per epoch
    
    for i in tqdm(range(n_epochs), desc=f"  Computing {METHOD}", leave=False):
        epoch_data = epochs[i]  # (n_channels, n_times)
        
        # --- A. Fit MVAR Model ---
        # Input to dyconnmap VAR: (n_channels, n_samples)
        # Note: dyconnmap expects (channels, samples)
        try:
            # Fit VAR model
            # method='yw' (Yule-Walker) is fast and stable for EEG
            var_model = VAR(order=MVAR_ORDER, estimator='yw')
            var_fit = var_model.fit(epoch_data)
            
            # Get coefficients and residuals
            coefs = var_fit.coefs  # (p, n_channels, n_channels)
            cov = var_fit.cov      # Noise covariance
            
        except Exception as e:
            # If fit fails (rare), zero out this epoch
            continue

        # --- B. Compute Connectivity (Spectral) ---
        # Compute PDC/DTF from coefficients
        if METHOD == 'PDC':
            # pdc output: (n_channels, n_channels, n_freqs)
            con_estimator = PDC(coefs, cov, n_fft=N_FFT, sampling_frequency=SAMPLING_RATE)
            connectivity_spectrum = con_estimator.estimate()
        elif METHOD == 'DTF':
            con_estimator = DTF(coefs, cov, n_fft=N_FFT, sampling_frequency=SAMPLING_RATE)
            connectivity_spectrum = con_estimator.estimate()
        
        freqs = con_estimator.freqs
        
        # --- C. Integrate into Bands ---
        # Get indices for bands (only calculate once really, but freqs constant)
        if i == 0:
            band_indices = get_freq_indices(freqs, BANDS)
            
        for b_idx, band_name in enumerate(BAND_NAMES):
            f_idx = band_indices[band_name]
            
            if len(f_idx) > 0:
                # Average across the frequencies in this band
                # connectivity_spectrum shape: (n_targets, n_sources, n_freqs) -> Check dyconnmap docs
                # dyconnmap usually returns (n_channels, n_channels, n_freqs)
                # We take mean along last axis (frequency)
                band_con = np.mean(connectivity_spectrum[:, :, f_idx], axis=2)
                
                # Store in output
                adj_matrices[i, b_idx, :, :] = band_con

    # 4. Save Result
    np.save(out_file, adj_matrices)
    
    # Save band names for reference
    band_file = out_file.parent / f"{out_file.stem}_bands.json"
    import json
    with open(band_file, 'w') as f:
        json.dump(BAND_NAMES, f)

def main():
    parser = argparse.ArgumentParser(description="Compute Graph Connectivity (PDC/DTF)")
    parser.add_argument("--data_dir", required=True, help="Root preprocessed data directory")
    parser.add_argument("--method", default="PDC", choices=["PDC", "DTF"], help="Connectivity measure")
    args = parser.parse_args()
    
    global METHOD
    METHOD = args.method
    
    data_dir = Path(args.data_dir)
    
    # Find all epoch files recursively
    print(f"Searching for epoch files in {data_dir}...")
    epoch_files = list(data_dir.rglob("*_epochs.npy"))
    print(f"Found {len(epoch_files)} files.")
    
    for f in tqdm(epoch_files, desc="Processing Files"):
        # Define output filename
        # Save as {pid}_connectivity.npy in the same folder
        pid = f.stem.replace("_epochs", "")
        out_file = f.parent / f"{pid}_connectivity.npy"
        
        # Skip if already exists
        if out_file.exists():
            continue
            
        try:
            process_single_file(f, out_file)
        except Exception as e:
            print(f"\n❌ Failed {f.name}: {e}")

if __name__ == "__main__":
    main()