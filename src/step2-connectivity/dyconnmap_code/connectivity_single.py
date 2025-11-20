"""
Single-File Connectivity (PDC/DTF) - Manual Implementation with Validation Plots.
Uses statsmodels for VAR fitting and custom NumPy math for PDC/DTF.

Usage:
  python src/step2-connectivity/dyconnmap_code/connectivity_single_manual_vis.py --input "path/to/file_epochs.npy" --method PDC --vis
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import seaborn as sns
import sys
from scipy import linalg
from statsmodels.tsa.api import VAR

# --- NUMPY PATCH ---
if not hasattr(np, 'int'): np.int = int
if not hasattr(np, 'float'): np.float = float
if not hasattr(np, 'bool'): np.bool = bool

warnings.filterwarnings("ignore")

# ==========================================
# 🧠 CUSTOM CONNECTIVITY CALCULATOR
# ==========================================
def compute_connectivity(data, method='PDC', order=15, n_fft=512, fs=250.0):
    """Manually computes PDC or DTF for a single epoch."""
    n_channels, n_samples = data.shape
    
    # 1. Fit VAR Model
    try:
        model = VAR(data.T)
        results = model.fit(maxlags=order, trend='c')
    except:
        return None, None
    
    coefs = results.coefs 
    
    # 2. Compute A(f)
    freqs = np.linspace(0, fs/2, n_fft)
    A_f = np.zeros((n_fft, n_channels, n_channels), dtype=np.complex128)
    I = np.eye(n_channels)
    
    for f_idx, f in enumerate(freqs):
        sum_Ak = np.zeros((n_channels, n_channels), dtype=np.complex128)
        for k in range(order):
            phasor = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
            sum_Ak += coefs[k] * phasor
        A_f[f_idx] = I - sum_Ak

    # 3. Compute H(f) if needed
    H_f = None
    if method == 'DTF':
        H_f = np.zeros((n_fft, n_channels, n_channels), dtype=np.complex128)
        for f in range(n_fft):
            try:
                H_f[f] = linalg.inv(A_f[f])
            except linalg.LinAlgError:
                H_f[f] = linalg.pinv(A_f[f])

    # 4. Compute Metric
    result = np.zeros((n_channels, n_channels, n_fft))
    
    if method == 'PDC':
        for f in range(n_fft):
            Af = A_f[f]
            col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))
            col_norms[col_norms == 0] = 1e-10
            result[:, :, f] = np.abs(Af) / col_norms[None, :]
            
    elif method == 'DTF':
        for f in range(n_fft):
            Hf = H_f[f]
            row_norms = np.sqrt(np.sum(np.abs(Hf)**2, axis=1))
            row_norms[row_norms == 0] = 1e-10
            result[:, :, f] = np.abs(Hf) / row_norms[:, None]
            
    return result, freqs

# ==========================================
# CONFIGURATION
# ==========================================
SAMPLING_RATE = 250.0
MVAR_ORDER = 15 
N_FFT = 128 

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
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        indices[name] = idx
    return indices

def save_connectivity_plot(matrix, title, output_path, channel_names=None):
    """Generates and SAVES the heatmap."""
    fig = plt.figure(figsize=(12, 10))
    sns.heatmap(matrix, cmap='viridis', square=True, 
                xticklabels=channel_names if channel_names else True,
                yticklabels=channel_names if channel_names else True)
    plt.title(title, fontsize=14)
    plt.xlabel("Source Node (From)", fontsize=12)
    plt.ylabel("Target Node (To)", fontsize=12)
    plt.tight_layout()
    
    # Save and close to free memory
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to _epochs.npy file")
    parser.add_argument("--method", default="PDC", choices=["PDC", "DTF"])
    parser.add_argument("--vis", action="store_true", help="Save validation plots")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ File not found: {input_path}")
        return

    print(f"⬇️ Loading {input_path.name}...")
    epochs = np.load(input_path) 
    n_epochs, n_channels, n_times = epochs.shape
    
    # Load channel names
    channel_names = None
    try:
        import pickle
        pid = input_path.stem.replace("_epochs", "")
        info_path = input_path.parent / f"{pid}_info.pkl"
        if info_path.exists():
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
                channel_names = info['ch_names']
    except: pass

    print(f"   Shape: {epochs.shape}")
    print(f"   Method: {args.method}")

    # Output array
    adj_matrices = np.zeros((n_epochs, len(BANDS), n_channels, n_channels), dtype=np.float32)

    print(f"⚙️ Computing connectivity...")
    
    valid_epochs = 0
    for i in range(n_epochs):
        epoch_data = epochs[i]
        
        conn_spectrum, freqs = compute_connectivity(
            epoch_data, 
            method=args.method, 
            order=MVAR_ORDER, 
            n_fft=N_FFT, 
            fs=SAMPLING_RATE
        )
        
        if conn_spectrum is None:
            continue

        if i == 0:
            band_indices = get_freq_indices(freqs, BANDS)
        
        for b_idx, band in enumerate(BAND_NAMES):
            f_idx = band_indices[band]
            if len(f_idx) > 0:
                adj_matrices[i, b_idx, :, :] = np.mean(conn_spectrum[:, :, f_idx], axis=2)
        
        valid_epochs += 1
        if i % 10 == 0:
            print(f"   Processed {i}/{n_epochs}...", end='\r')
            
    print(f"\n✅ Processed {valid_epochs}/{n_epochs} epochs successfully.")

    # Save Data
    output_path = input_path.parent / f"{input_path.stem.replace('_epochs', '')}_connectivity.npy"
    np.save(output_path, adj_matrices)
    print(f"💾 Saved data to: {output_path.name}")

    # Save Plots (if --vis)
    if args.vis:
        print("\n📊 Saving validation plots for all bands...")
        
        # Average across all epochs for plotting
        avg_matrix = np.mean(adj_matrices, axis=0) # (n_bands, 22, 22)
        pid = input_path.stem.replace("_epochs", "")
        
        for b_idx, band in enumerate(BAND_NAMES):
            # Construct filename: {PID}_PDC_Gamma.png
            plot_filename = input_path.parent / f"{pid}_{args.method}_{band}.png"
            
            save_connectivity_plot(
                avg_matrix[b_idx], 
                f"Average {args.method} - {band} Band ({pid})", 
                plot_filename, 
                channel_names
            )
            print(f"   Saved: {plot_filename.name}")

if __name__ == "__main__":
    main()