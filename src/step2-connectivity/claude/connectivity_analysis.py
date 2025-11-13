"""
Effective Connectivity Analysis: DTF and PDC
Based on dyconnmap library and MVAR models.

This script:
1. Loads preprocessed epochs from .npy files
2. Fits MVAR model per epoch
3. Computes DTF and PDC connectivity matrices
4. Extracts frequency-band connectivity (delta, theta, alpha, beta, gamma)
5. Saves results for graph construction

Usage:
  python connectivity_analysis.py --epoch_file data_pp/patient_epochs.npy --output_dir connectivity_results
  
Example:
  python connectivity_analysis.py \
    --epoch_file F:\\October-Thesis\\thesis-epilepsy-gnn\\test\\data_pp\\00_epilepsy\\aaaaaanr\\s001_2003\\02_tcp_le\\aaaaaanr_s001_t001_epochs.npy \
    --output_dir F:\\October-Thesis\\thesis-epilepsy-gnn\\connectivity_results
"""

import numpy as np
import pickle
from pathlib import Path
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')



# We'll also need this for MVAR
from scipy import signal
from scipy.linalg import solve


# ============================================================================
# MVAR MODEL FITTING
# ============================================================================

def estimate_mvar_order(data, max_order=30, ic='aic'):
    """
    Estimate optimal MVAR model order using information criteria.
    
    Args:
        data: (n_channels, n_times) - single epoch
        max_order: Maximum model order to test
        ic: Information criterion ('aic', 'bic', or 'hq')
    
    Returns:
        Optimal model order
    """
    n_channels, n_times = data.shape
    
    ic_values = []
    
    for p in range(1, min(max_order + 1, n_times // 10)):
        try:
            # Fit MVAR model of order p
            A, sigma = fit_mvar(data, p)
            
            # Calculate residuals
            residuals = compute_mvar_residuals(data, A, p)
            
            # Calculate information criterion
            n_params = n_channels * n_channels * p
            log_likelihood = -0.5 * n_times * np.log(np.linalg.det(sigma))
            
            if ic == 'aic':
                ic_val = -2 * log_likelihood + 2 * n_params
            elif ic == 'bic':
                ic_val = -2 * log_likelihood + n_params * np.log(n_times)
            elif ic == 'hq':
                ic_val = -2 * log_likelihood + 2 * n_params * np.log(np.log(n_times))
            
            ic_values.append(ic_val)
        except:
            ic_values.append(np.inf)
    
    optimal_order = np.argmin(ic_values) + 1
    
    return optimal_order


def fit_mvar(data, order):
    """
    Fit multivariate autoregressive (MVAR) model using least squares.
    
    The model is:
        X(t) = A1*X(t-1) + A2*X(t-2) + ... + Ap*X(t-p) + E(t)
    
    Where:
        X(t) is the vector of channel values at time t
        Ai are coefficient matrices
        E(t) is white noise (residuals)
    
    Args:
        data: (n_channels, n_times) array
        order: Model order (number of past time points to use)
    
    Returns:
        A: (order, n_channels, n_channels) coefficient matrices
        sigma: (n_channels, n_channels) residual covariance matrix
    """
    n_channels, n_times = data.shape
    
    # Build design matrix X and target matrix Y
    # X contains lagged values: [X(t-1), X(t-2), ..., X(t-p)]
    # Y contains current values: X(t)
    
    n_samples = n_times - order
    
    # Initialize
    X = np.zeros((n_channels * order, n_samples))
    Y = data[:, order:]
    
    # Fill design matrix with lagged values
    for i in range(order):
        X[i*n_channels:(i+1)*n_channels, :] = data[:, order-i-1:n_times-i-1]
    
    # Solve least squares: Y = A @ X + E
    # We want: A = Y @ X.T @ inv(X @ X.T)
    A_flat = Y @ X.T @ np.linalg.pinv(X @ X.T)
    
    # Reshape into (order, n_channels, n_channels)
    A = A_flat.reshape(n_channels, order, n_channels).transpose(1, 0, 2)
    
    # Compute residuals
    predictions = A_flat @ X
    residuals = Y - predictions
    
    # Residual covariance
    sigma = (residuals @ residuals.T) / n_samples
    
    return A, sigma


def compute_mvar_residuals(data, A, order):
    """
    Compute residuals from fitted MVAR model.
    
    Args:
        data: (n_channels, n_times)
        A: (order, n_channels, n_channels) MVAR coefficients
        order: Model order
    
    Returns:
        residuals: (n_channels, n_times - order)
    """
    n_channels, n_times = data.shape
    n_samples = n_times - order
    
    # Predictions
    predictions = np.zeros((n_channels, n_samples))
    
    for t in range(n_samples):
        for lag in range(order):
            predictions[:, t] += A[lag] @ data[:, order - lag - 1 + t]
    
    # Residuals
    residuals = data[:, order:] - predictions
    
    return residuals


# ============================================================================
# DTF AND PDC COMPUTATION (Frequency Domain)
# ============================================================================

def mvar_to_transfer_function(A, sigma, freqs, sfreq):
    """
    Convert MVAR coefficients to transfer function H(f).
    
    H(f) = [I - sum(A(k) * exp(-2πifk))]^(-1)
    
    Args:
        A: (order, n_channels, n_channels) MVAR coefficients
        sigma: (n_channels, n_channels) noise covariance
        freqs: Frequency bins to evaluate
        sfreq: Sampling frequency (Hz)
    
    Returns:
        H: (n_freqs, n_channels, n_channels) transfer function
    """
    order, n_channels, _ = A.shape
    n_freqs = len(freqs)
    
    # Initialize transfer function
    H = np.zeros((n_freqs, n_channels, n_channels), dtype=complex)
    
    for f_idx, f in enumerate(freqs):
        # Compute A(f) = I - sum(A(k) * exp(-2πifk/fs))
        A_f = np.eye(n_channels, dtype=complex)
        
        for k in range(order):
            # Frequency in normalized form
            phase = -2j * np.pi * f * (k + 1) / sfreq
            A_f -= A[k] * np.exp(phase)
        
        # H(f) = A(f)^(-1)
        H[f_idx] = np.linalg.inv(A_f)
    
    return H


def compute_dtf(H):
    """
    Compute Directed Transfer Function (DTF) from transfer function.
    
    DTF_ij(f) = |H_ij(f)| / sqrt(sum_m |H_mj(f)|^2)
    
    Interpretation: DTF_ij measures the influence from channel j → channel i
    (normalized by total inflow to i)
    
    Args:
        H: (n_freqs, n_channels, n_channels) complex transfer function
    
    Returns:
        DTF: (n_freqs, n_channels, n_channels) real-valued DTF
    """
    n_freqs, n_channels, _ = H.shape
    
    DTF = np.zeros((n_freqs, n_channels, n_channels))
    
    for f in range(n_freqs):
        H_abs = np.abs(H[f])  # |H_ij(f)|
        
        # Normalization: sum over sources for each target
        for i in range(n_channels):
            norm = np.sqrt(np.sum(H_abs[i, :]**2))
            if norm > 0:
                DTF[f, i, :] = H_abs[i, :] / norm
    
    return DTF


def compute_pdc(A, sigma, freqs, sfreq):
    """
    Compute Partial Directed Coherence (PDC).
    
    PDC is based on the inverse transfer function A(f), which represents
    direct causal influences (unlike DTF which includes indirect paths).
    
    PDC_ij(f) = |A_ij(f)| / sqrt(sum_m |A_im(f)|^2)
    
    Args:
        A: (order, n_channels, n_channels) MVAR coefficients
        sigma: (n_channels, n_channels) noise covariance
        freqs: Frequency bins
        sfreq: Sampling frequency (Hz)
    
    Returns:
        PDC: (n_freqs, n_channels, n_channels) real-valued PDC
    """
    order, n_channels, _ = A.shape
    n_freqs = len(freqs)
    
    PDC = np.zeros((n_freqs, n_channels, n_channels))
    
    for f_idx, f in enumerate(freqs):
        # Compute A(f) = I - sum(A(k) * exp(-2πifk/fs))
        A_f = np.eye(n_channels, dtype=complex)
        
        for k in range(order):
            phase = -2j * np.pi * f * (k + 1) / sfreq
            A_f -= A[k] * np.exp(phase)
        
        A_abs = np.abs(A_f)
        
        # Normalization: sum over targets for each source
        for j in range(n_channels):
            norm = np.sqrt(np.sum(A_abs[:, j]**2))
            if norm > 0:
                PDC[f_idx, :, j] = A_abs[:, j] / norm
    
    return PDC


# ============================================================================
# FREQUENCY BAND EXTRACTION
# ============================================================================

def extract_band_connectivity(connectivity, freqs, band_ranges):
    """
    Average connectivity over frequency bands.
    
    Args:
        connectivity: (n_freqs, n_channels, n_channels) DTF or PDC
        freqs: Frequency array
        band_ranges: Dict like {'delta': (0.5, 4), 'theta': (4, 8), ...}
    
    Returns:
        Dict mapping band_name → (n_channels, n_channels) connectivity matrix
    """
    band_connectivity = {}
    
    for band_name, (f_low, f_high) in band_ranges.items():
        # Find frequency indices in range
        idx = np.where((freqs >= f_low) & (freqs <= f_high))[0]
        
        if len(idx) == 0:
            print(f"Warning: No frequencies found for {band_name} band ({f_low}-{f_high} Hz)")
            band_connectivity[band_name] = np.zeros((connectivity.shape[1], connectivity.shape[2]))
            continue
        
        # Average over frequency band
        band_connectivity[band_name] = np.mean(connectivity[idx], axis=0)
    
    return band_connectivity


# ============================================================================
# MAIN PROCESSING FUNCTION
# ============================================================================

def analyze_epoch_connectivity(epoch_data, sfreq=250.0, model_order=None, max_order=30):
    """
    Complete connectivity analysis pipeline for a single epoch.
    
    Args:
        epoch_data: (n_channels, n_times) single epoch
        sfreq: Sampling frequency in Hz
        model_order: MVAR model order (if None, auto-select using AIC)
        max_order: Maximum order for auto-selection
    
    Returns:
        Dictionary with:
            - 'dtf': (n_freqs, n_channels, n_channels)
            - 'pdc': (n_freqs, n_channels, n_channels)
            - 'dtf_bands': dict of band-averaged DTF
            - 'pdc_bands': dict of band-averaged PDC
            - 'freqs': frequency array
            - 'model_order': order used
    """
    n_channels, n_times = epoch_data.shape
    
    # Step 1: Estimate or use provided model order
    if model_order is None:
        model_order = estimate_mvar_order(epoch_data, max_order=max_order, ic='aic')
    
    # Step 2: Fit MVAR model
    try:
        A, sigma = fit_mvar(epoch_data, model_order)
    except Exception as e:
        print(f"Warning: MVAR fitting failed: {e}")
        return None
    
    # Step 3: Define frequency range
    # We'll compute connectivity from 0.5 to 100 Hz (your bandpass range)
    freqs = np.linspace(0.5, 100, 200)  # 200 frequency bins
    
    # Step 4: Compute transfer function H(f)
    H = mvar_to_transfer_function(A, sigma, freqs, sfreq)
    
    # Step 5: Compute DTF
    dtf = compute_dtf(H)
    
    # Step 6: Compute PDC
    pdc = compute_pdc(A, sigma, freqs, sfreq)
    
    # Step 7: Extract band-averaged connectivity
    band_ranges = {
        'delta': (0.5, 4),
        'theta': (4, 8),
        'alpha': (8, 13),
        'beta': (13, 30),
        'gamma': (30, 100)
    }
    
    dtf_bands = extract_band_connectivity(dtf, freqs, band_ranges)
    pdc_bands = extract_band_connectivity(pdc, freqs, band_ranges)
    
    return {
        'dtf': dtf,
        'pdc': pdc,
        'dtf_bands': dtf_bands,
        'pdc_bands': pdc_bands,
        'freqs': freqs,
        'model_order': model_order
    }


# ============================================================================
# BATCH PROCESSING
# ============================================================================

def process_file(epoch_file, output_dir, model_order=None):
    """
    Process a single patient's epoch file.
    
    Args:
        epoch_file: Path to *_epochs.npy file
        output_dir: Where to save results
        model_order: Fixed MVAR order (None = auto)
    
    Returns:
        True if successful, False otherwise
    """
    epoch_file = Path(epoch_file)
    output_dir = Path(output_dir)
    
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract patient ID from filename
    pid = epoch_file.stem.replace("_epochs", "")
    
    print(f"\nProcessing: {pid}")
    print("=" * 60)
    
    # Load epochs
    try:
        epochs = np.load(epoch_file)  # Shape: (n_epochs, n_channels, n_times)
        print(f"Loaded: {epochs.shape[0]} epochs, {epochs.shape[1]} channels, {epochs.shape[2]} timepoints")
    except Exception as e:
        print(f"✗ Failed to load {epoch_file}: {e}")
        return False
    
    # Load metadata (sampling frequency)
    info_file = epoch_file.parent / f"{pid}_info.pkl"
    try:
        with open(info_file, 'rb') as f:
            info = pickle.load(f)
        sfreq = info['sfreq']
        ch_names = info['ch_names']
        print(f"Sampling frequency: {sfreq} Hz")
        print(f"Channels: {ch_names}")
    except Exception as e:
        print(f"Warning: Could not load info file, using default sfreq=250 Hz")
        sfreq = 250.0
        ch_names = [f"Ch{i}" for i in range(epochs.shape[1])]
    
    # Process each epoch
    n_epochs = epochs.shape[0]
    
    # Storage for results
    all_dtf_bands = {band: [] for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']}
    all_pdc_bands = {band: [] for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']}
    model_orders = []
    
    print(f"\nComputing connectivity for {n_epochs} epochs...")
    
    for epoch_idx in tqdm(range(n_epochs), desc="Epochs"):
        epoch_data = epochs[epoch_idx]  # Shape: (n_channels, n_times)
        
        # Analyze connectivity
        result = analyze_epoch_connectivity(
            epoch_data, 
            sfreq=sfreq, 
            model_order=model_order,
            max_order=30
        )
        
        if result is None:
            # Skip failed epochs
            continue
        
        # Store band-averaged results
        for band in all_dtf_bands.keys():
            all_dtf_bands[band].append(result['dtf_bands'][band])
            all_pdc_bands[band].append(result['pdc_bands'][band])
        
        model_orders.append(result['model_order'])
    
    if len(model_orders) == 0:
        print("✗ All epochs failed")
        return False
    
    # Convert to numpy arrays
    for band in all_dtf_bands.keys():
        all_dtf_bands[band] = np.array(all_dtf_bands[band])  # (n_epochs, n_channels, n_channels)
        all_pdc_bands[band] = np.array(all_pdc_bands[band])
    
    print(f"\n✓ Successfully processed {len(model_orders)}/{n_epochs} epochs")
    print(f"Average MVAR order used: {np.mean(model_orders):.1f} ± {np.std(model_orders):.1f}")
    
    # Save results
    print("\nSaving results...")
    
    # Save connectivity matrices (per-band, all epochs)
    for band in all_dtf_bands.keys():
        np.save(output_dir / f"{pid}_dtf_{band}.npy", all_dtf_bands[band])
        np.save(output_dir / f"{pid}_pdc_{band}.npy", all_pdc_bands[band])
        print(f"  ✓ Saved {band} band: DTF and PDC")
    
    # Save metadata
    metadata = {
        'patient_id': pid,
        'n_epochs': len(model_orders),
        'n_channels': epochs.shape[1],
        'channel_names': ch_names,
        'sfreq': sfreq,
        'avg_model_order': float(np.mean(model_orders)),
        'std_model_order': float(np.std(model_orders)),
        'frequency_bands': {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 100)
        }
    }
    
    import json
    with open(output_dir / f"{pid}_connectivity_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"  ✓ Saved metadata")
    
    # Visualize average connectivity (across all epochs)
    visualize_connectivity(all_dtf_bands, all_pdc_bands, ch_names, output_dir, pid)
    
    print("=" * 60)
    print(f"✓ Complete! Results saved to: {output_dir}")
    
    return True


# ============================================================================
# VISUALIZATION
# ============================================================================

def visualize_connectivity(dtf_bands, pdc_bands, ch_names, output_dir, pid):
    """
    Create visualization of average connectivity matrices.
    """
    print("\nGenerating visualizations...")
    
    bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
    
    # Create figure with 2 rows (DTF, PDC) × 5 columns (bands)
    fig, axes = plt.subplots(2, 5, figsize=(20, 8))
    
    # Plot DTF
    for band_idx, band in enumerate(bands):
        ax = axes[0, band_idx]
        
        # Average over epochs
        avg_dtf = np.mean(dtf_bands[band], axis=0)
        
        # Plot heatmap
        im = ax.imshow(avg_dtf, cmap='hot', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f"DTF - {band.upper()}", fontweight='bold')
        ax.set_xlabel("Source (from)")
        ax.set_ylabel("Target (to)")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Set tick labels (show every 5th channel)
        tick_indices = np.arange(0, len(ch_names), 5)
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([ch_names[i] for i in tick_indices], fontsize=8)
    
    # Plot PDC
    for band_idx, band in enumerate(bands):
        ax = axes[1, band_idx]
        
        # Average over epochs
        avg_pdc = np.mean(pdc_bands[band], axis=0)
        
        # Plot heatmap
        im = ax.imshow(avg_pdc, cmap='hot', vmin=0, vmax=1, aspect='auto')
        ax.set_title(f"PDC - {band.upper()}", fontweight='bold')
        ax.set_xlabel("Source (from)")
        ax.set_ylabel("Target (to)")
        
        # Add colorbar
        plt.colorbar(im, ax=ax, fraction=0.046)
        
        # Set tick labels
        tick_indices = np.arange(0, len(ch_names), 5)
        ax.set_xticks(tick_indices)
        ax.set_yticks(tick_indices)
        ax.set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels([ch_names[i] for i in tick_indices], fontsize=8)
    
    fig.suptitle(f"Patient {pid} - Average Connectivity (All Epochs)", 
                 fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    
    # Save figure
    fig.savefig(output_dir / f"{pid}_connectivity_visualization.png", 
                dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print(f"  ✓ Saved visualization: {pid}_connectivity_visualization.png")


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Compute effective connectivity (DTF and PDC) from preprocessed EEG epochs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process single file
  python connectivity_analysis.py --epoch_file data_pp/patient_epochs.npy --output_dir connectivity_results
  
  # Process with fixed MVAR order
  python connectivity_analysis.py --epoch_file data.npy --output_dir results --model_order 15
        """
    )
    
    parser.add_argument(
        "--epoch_file",
        required=True,
        help="Path to *_epochs.npy file"
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Output directory for connectivity matrices"
    )
    parser.add_argument(
        "--model_order",
        type=int,
        default=None,
        help="MVAR model order (default: auto-select using AIC)"
    )
    
    args = parser.parse_args()
    
    # Process file
    success = process_file(
        epoch_file=args.epoch_file,
        output_dir=args.output_dir,
        model_order=args.model_order
    )
    
    if not success:
        exit(1)


if __name__ == "__main__":
    main()
