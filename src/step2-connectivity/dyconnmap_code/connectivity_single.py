"""
Single-File Connectivity (PDC/DTF) - FINAL THESIS VERSION
Features:
1. Input: Accepts FILE path OR FOLDER path (auto-finds _epochs.npy)
2. Output: Saves as {PID}_{METHOD}_connectivity.npy (prevents overwriting)
3. Math: Correct N_FFT=512, Stability Check, Squared Magnitude
4. Vis: Plots with channel names
"""

import argparse
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
import seaborn as sns
from scipy import linalg
import pickle
import sys

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
N_FFT = 512  # Matches batch processing

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
        indices[name] = np.where((freqs >= f_low) & (freqs <= f_high))[0]
    return indices


def compute_connectivity(data, method='PDC', order=15, n_fft=512, fs=250.0, squared=True):
    """Compute PDC or DTF connectivity for a single epoch."""
    n_channels, n_samples = data.shape
    
    # 1. Fit VAR Model
    try:
        model = VAR(data.T)
        results = model.fit(maxlags=order, trend='c')
    except Exception:
        return None, None
    
    # 2. CHECK STABILITY (Critical)
    if not results.is_stable():
        return None, None
    
    coefs = results.coefs
    
    # 3. Compute A(f)
    freqs = np.linspace(0, fs/2, n_fft)
    A_f = np.zeros((n_fft, n_channels, n_channels), dtype=np.complex128)
    I = np.eye(n_channels)
    
    for f_idx, f in enumerate(freqs):
        sum_Ak = np.zeros((n_channels, n_channels), dtype=np.complex128)
        for k in range(order):
            phasor = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)
            sum_Ak += coefs[k] * phasor
        A_f[f_idx] = I - sum_Ak
    
    # 4. Compute H(f) if needed for DTF
    H_f = None
    if method == 'DTF':
        H_f = np.zeros((n_fft, n_channels, n_channels), dtype=np.complex128)
        for f in range(n_fft):
            try:
                H_f[f] = linalg.inv(A_f[f])
            except linalg.LinAlgError:
                H_f[f] = linalg.pinv(A_f[f])
    
    # 5. Compute Connectivity Measure
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
    
    # 6. Square for power interpretation
    if squared:
        connectivity = connectivity ** 2
    
    return connectivity, freqs


def save_connectivity_plot(matrix, title, output_path, channel_names=None):
    """Generate and save connectivity heatmap."""
    fig = plt.figure(figsize=(12, 10))
    labels = channel_names if channel_names else True
    
    sns.heatmap(matrix, cmap='viridis', square=True,
                xticklabels=labels, yticklabels=labels,
                vmin=0, vmax=1, cbar_kws={'label': 'Strength'})
    
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel("Source Channel (From)", fontsize=12)
    plt.ylabel("Target Channel (To)", fontsize=12)
    plt.tight_layout()
    
    fig.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Compute EEG Connectivity (PDC/DTF)")
    parser.add_argument("--input", required=True, 
                        help="Path to _epochs.npy file OR the FOLDER containing it")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: same as input)")
    parser.add_argument("--method", default="PDC", choices=["PDC", "DTF"],
                        help="Connectivity measure")
    parser.add_argument("--vis", action="store_true", help="Save validation plots")
    parser.add_argument("--squared", action="store_true", default=True, help="Return squared values")
    
    args = parser.parse_args()
    
    # --- 1. SMART INPUT HANDLING ---
    input_path = Path(args.input)
    target_file = None
    
    if not input_path.exists():
        print(f"❌ Error: Input not found: {input_path}")
        return

    if input_path.is_file():
        target_file = input_path
    elif input_path.is_dir():
        # Auto-detect _epochs.npy inside the folder
        candidates = list(input_path.glob("*_epochs.npy"))
        if len(candidates) == 0:
            print(f"❌ Error: No '*_epochs.npy' file found in {input_path}")
            return
        elif len(candidates) > 1:
            print(f"⚠️  Warning: Multiple epoch files found. Using the first one:")
            print(f"   -> {candidates[0].name}")
            target_file = candidates[0]
        else:
            target_file = candidates[0]
            print(f"✓ Found epoch file: {target_file.name}")

    # --- 2. SETUP OUTPUT ---
    if args.output_dir:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir = target_file.parent
        
    # --- 3. LOAD DATA ---
    print(f"\n{'='*60}")
    print(f"PROCESSING: {target_file.name}")
    print(f"METHOD:     {args.method}{'²' if args.squared else ''}")
    print(f"{'='*60}")
    
    epochs = np.load(target_file)
    n_epochs, n_channels, n_times = epochs.shape
    
    # Try to load info for channel names
    channel_names = None
    try:
        pid = target_file.stem.replace("_epochs", "")
        info_path = target_file.parent / f"{pid}_info.pkl"
        if info_path.exists():
            with open(info_path, 'rb') as f:
                info = pickle.load(f)
                channel_names = info['ch_names']
    except: pass

    # --- 4. COMPUTE ---
    adj_matrices = np.zeros((n_epochs, len(BANDS), n_channels, n_channels), dtype=np.float32)
    valid_epochs = 0
    
    print(f"Computing connectivity for {n_epochs} epochs...")
    
    for i in range(n_epochs):
        conn, freqs = compute_connectivity(
            epochs[i], method=args.method, order=MVAR_ORDER, 
            n_fft=N_FFT, fs=SAMPLING_RATE, squared=args.squared
        )
        
        if conn is None: continue
        
        if i == 0: band_indices = get_freq_indices(freqs, BANDS)
        
        for b_idx, band in enumerate(BAND_NAMES):
            f_idx = band_indices[band]
            if len(f_idx) > 0:
                adj_matrices[i, b_idx, :, :] = np.mean(conn[:, :, f_idx], axis=2)
        
        valid_epochs += 1
        if i % 10 == 0: print(f"  Processed {i}/{n_epochs}...", end='\r')

    print(f"\n✓ Finished. Valid epochs: {valid_epochs}/{n_epochs}")
    
    if valid_epochs == 0:
        print("❌ All epochs failed stability check.")
        return

    # --- 5. SAVE RESULTS (WITH METHOD NAME) ---
    # Construct filename: e.g. "aaaaaanr_s001_t000_PDC_connectivity.npy"
    out_name = f"{pid}_{args.method}_connectivity.npy"
    out_path = out_dir / out_name
    
    np.save(out_path, adj_matrices)
    print(f"💾 Saved: {out_path}")
    
    # Save bands info
    with open(out_dir / f"{pid}_{args.method}_bands.txt", 'w') as f:
        f.write('\n'.join(BAND_NAMES))

    # --- 6. VISUALIZATION ---
    if args.vis:
        print("📊 Generating plots...")
        avg_matrix = np.mean(adj_matrices[:valid_epochs], axis=0)
        
        for b_idx, band in enumerate(BAND_NAMES):
            # e.g. "aaaaaanr_s001_t000_PDC_Alpha.png"
            plot_name = f"{pid}_{args.method}_{band}.png"
            save_connectivity_plot(
                avg_matrix[b_idx], 
                f"Average {args.method} - {band} Band", 
                out_dir / plot_name, 
                channel_names
            )

if __name__ == "__main__":
    main()