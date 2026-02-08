"""
STEP 2: FEATURE EXTRACTION (FROM PRE-COMPUTED CONNECTIVITY)
===========================================================
Extract node features for GNN from PRE-COMPUTED connectivity files.

Since you already have PDC computed, this script:
1. Loads PDC matrices from your .npz files
2. Extracts connectivity features (in/out strength & degree)
3. Optionally adds spectral/statistical features from epoch files (if available)

Output: control_epilepsy_features.npz containing:
- node_features: (N, 22, F) - F features per channel
- adjacency: (N, 22, 22) - PDC connectivity matrices
- labels: (N,) - 0=control, 1=epilepsy
- file_ids: (N,) - source file for each epoch
"""

import numpy as np
from pathlib import Path
from scipy.stats import skew, kurtosis
from scipy import signal
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION - UPDATE THESE PATHS
# =============================================================================

CONNECTIVITY_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_fixed_15")
EPOCHS_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced")  # Set to None if not available
OUTPUT_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\gnn\control_vs_epilepsy")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Parameters
FS = 250  # Sampling frequency
N_CHANNELS = 22

# Class identification
EPILEPSY_DIR_PATTERN = "00_epilepsy"
CONTROL_DIR_PATTERN = "01_no_epilepsy"

# Feature options
USE_SPECTRAL_FEATURES = True   # Requires epoch files
USE_CONNECTIVITY_FEATURES = True  # From PDC (always available)

# Which PDC band to use for connectivity
PDC_BAND = 'pdc_integrated'  # Options: pdc_integrated, pdc_alpha, pdc_beta, etc.

# =============================================================================
# FEATURE EXTRACTION
# =============================================================================

def extract_connectivity_features(pdc, threshold=0.1):
    """
    Extract connectivity features from PDC matrix.
    
    Returns: (n_channels, 4) array with:
    - out_strength: sum of outgoing connections
    - in_strength: sum of incoming connections
    - out_degree: number of strong outgoing connections
    - in_degree: number of strong incoming connections
    """
    n_channels = pdc.shape[0]
    features = []
    
    for ch in range(n_channels):
        out_strength = np.sum(pdc[:, ch]) - pdc[ch, ch]  # Column sum (to others)
        in_strength = np.sum(pdc[ch, :]) - pdc[ch, ch]   # Row sum (from others)
        out_degree = np.sum(pdc[:, ch] > threshold) - 1   # Count strong connections
        in_degree = np.sum(pdc[ch, :] > threshold) - 1
        
        features.append([out_strength, in_strength, out_degree, in_degree])
    
    return np.array(features)  # (22, 4)


def extract_spectral_features(epoch, fs=250):
    """
    Extract spectral features from raw epoch.
    
    Returns: (n_channels, 6) array with:
    - total_power
    - relative delta, theta, alpha, beta, gamma power
    """
    bands = {
        #'delta': (0.5, 4),
        #'theta': (4, 8),
        #'alpha': (8, 13),
        #'beta': (13, 30),
        #'gamma': (30, 45)
        'integrated': (0.5, 80.0),
        'delta': (0.5, 4.0),
        'theta': (4.0, 8.0),
        'alpha': (8.0, 15.0),
        'beta': (15.0, 30.0),
        'gamma1': (30.0, 55.0),
        'gamma2': (65.0, 80.0)
    }
    
    n_channels = epoch.shape[0]
    features = []
    
    for ch in range(n_channels):
        x = epoch[ch]
        
        # Welch PSD
        freqs, psd = signal.welch(x, fs, nperseg=min(256, len(x)))
        
        # Band powers
        powers = {}
        for band_name, (fmin, fmax) in bands.items():
            idx = np.where((freqs >= fmin) & (freqs <= fmax))[0]
            powers[band_name] = np.trapz(psd[idx], freqs[idx]) if len(idx) > 0 else 0.0
        
        total_power = sum(powers.values()) + 1e-10
        
        ch_feat = [
            total_power,
            powers['integrated'] / total_power,
            powers['delta'] / total_power,
            powers['theta'] / total_power,
            powers['alpha'] / total_power,
            powers['beta'] / total_power,
            powers['gamma1'] / total_power,
            powers['gamma2'] / total_power
        ]
        
        features.append(ch_feat)
    
    return np.array(features)  # (22, 6)


def extract_statistical_features(epoch):
    """
    Extract statistical features from raw epoch.
    
    Returns: (n_channels, 5) array with:
    - mean, std, skewness, kurtosis, line_length
    """
    n_channels = epoch.shape[0]
    features = []
    
    for ch in range(n_channels):
        x = epoch[ch]
        
        ch_feat = [
            np.mean(x),
            np.std(x),
            skew(x),
            kurtosis(x),
            np.sum(np.abs(np.diff(x)))  # Line length
        ]
        
        features.append(ch_feat)
    
    return np.array(features)  # (22, 5)


def extract_all_features(pdc, epoch=None, fs=250):
    """
    Extract all features for one epoch.
    
    Parameters
    ----------
    pdc : ndarray (22, 22)
        PDC connectivity matrix
    epoch : ndarray (22, n_samples), optional
        Raw epoch data
    
    Returns
    -------
    features : ndarray (22, F)
        Combined features (F = 4, 10, or 15 depending on what's available)
    """
    all_features = []
    
    # Connectivity features (always available)
    if USE_CONNECTIVITY_FEATURES:
        conn_feat = extract_connectivity_features(pdc)
        all_features.append(conn_feat)
    
    # Spectral and statistical features (require epoch data)
    if epoch is not None and USE_SPECTRAL_FEATURES:
        spec_feat = extract_spectral_features(epoch, fs)
        stat_feat = extract_statistical_features(epoch)
        all_features.extend([spec_feat, stat_feat])
    
    # Combine all features
    if len(all_features) > 0:
        combined = np.concatenate(all_features, axis=1)
        return combined
    else:
        raise ValueError("No features extracted! Check configuration.")


# =============================================================================
# FILE PROCESSING
# =============================================================================

def find_matching_epoch_file(conn_file, epochs_dir):
    """
    Try to find the corresponding epoch file for a connectivity file.
    
    Example mapping:
      connectivity/.../aaaaaanr_s004_t000_graphs.npz
      → epochs/.../aaaaaanr_s004_t000_epochs.npy
    """
    if epochs_dir is None or not epochs_dir.exists():
        return None
    
    # Extract base name
    base_name = conn_file.stem.replace('_graphs', '_epochs')
    
    # Search for matching file
    matches = list(epochs_dir.rglob(f"{base_name}.npy"))
    
    return matches[0] if matches else None


def process_connectivity_file(conn_file, label, file_id, epochs_dir=None):
    """
    Process one connectivity file.
    
    Returns lists of: features, adjacencies, labels, file_ids
    """
    try:
        # Load connectivity
        conn_data = np.load(conn_file)
        
        # Get PDC matrices
        if PDC_BAND not in conn_data.files:
            print(f"\n⚠️  {PDC_BAND} not found in {conn_file.name}")
            print(f"   Available: {conn_data.files}")
            return None
        
        pdc_matrices = conn_data[PDC_BAND]  # (n_epochs, 22, 22)
        n_epochs = len(pdc_matrices)
        
        # Try to load corresponding epochs
        epoch_file = find_matching_epoch_file(conn_file, epochs_dir)
        epochs = None
        
        if epoch_file is not None:
            try:
                epochs = np.load(epoch_file)  # (n_epochs, 22, n_samples)
                
                # Verify alignment
                if len(epochs) != n_epochs:
                    # Check if indices are available
                    if 'indices' in conn_data:
                        valid_indices = conn_data['indices']
                        epochs = epochs[valid_indices]
                    else:
                        print(f"\n⚠️  Epoch/connectivity mismatch for {conn_file.name}")
                        epochs = None
            except Exception as e:
                print(f"\n⚠️  Could not load epochs for {conn_file.name}: {e}")
                epochs = None
        
        # Extract features for each epoch
        features_list = []
        adjacency_list = []
        labels_list = []
        file_ids_list = []
        
        for i in range(n_epochs):
            pdc_matrix = pdc_matrices[i]
            epoch_data = epochs[i] if epochs is not None else None
            
            # Extract features
            try:
                node_features = extract_all_features(pdc_matrix, epoch_data, FS)
                
                features_list.append(node_features)
                adjacency_list.append(pdc_matrix)
                labels_list.append(label)
                file_ids_list.append(file_id)
            except Exception as e:
                # Skip problematic epochs
                continue
        
        return features_list, adjacency_list, labels_list, file_ids_list
        
    except Exception as e:
        print(f"\n❌ Error processing {conn_file.name}: {e}")
        return None


# =============================================================================
# MAIN PROCESSING
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("STEP 2: FEATURE EXTRACTION (FROM PRE-COMPUTED CONNECTIVITY)")
    print("="*80)
    print(f"\nConfiguration:")
    print(f"  Connectivity dir: {CONNECTIVITY_DIR}")
    print(f"  Epochs dir:       {EPOCHS_DIR}")
    print(f"  Output dir:       {OUTPUT_DIR}")
    print(f"  PDC band:         {PDC_BAND}")
    print(f"  Use connectivity: {USE_CONNECTIVITY_FEATURES}")
    print(f"  Use spectral:     {USE_SPECTRAL_FEATURES}")
    
    # Find connectivity files
    print(f"\n[1] Finding connectivity files...")
    npz_files = list(CONNECTIVITY_DIR.rglob("*_graphs.npz"))
    print(f"  ✓ Found {len(npz_files)} files")
    
    # Classify by label
    epilepsy_files = [f for f in npz_files if EPILEPSY_DIR_PATTERN in str(f)]
    control_files = [f for f in npz_files if CONTROL_DIR_PATTERN in str(f)]
    
    print(f"  ✓ Epilepsy: {len(epilepsy_files)} files")
    print(f"  ✓ Control:  {len(control_files)} files")
    
    # Initialize collectors
    all_features = []
    all_adjacencies = []
    all_labels = []
    all_file_ids = []
    
    # Process epilepsy files
    print(f"\n[2] Processing EPILEPSY files...")
    for file_id, conn_file in enumerate(tqdm(epilepsy_files, desc="Epilepsy")):
        result = process_connectivity_file(conn_file, label=1, file_id=file_id, epochs_dir=EPOCHS_DIR)
        
        if result is not None:
            features, adjacencies, labels, file_ids = result
            all_features.extend(features)
            all_adjacencies.extend(adjacencies)
            all_labels.extend(labels)
            all_file_ids.extend([f"epi_{fid}" for fid in file_ids])
    
    
    epilepsy_count = sum(1 for l in all_labels if l == 1)
    
    # Process control files
    print(f"\n[3] Processing CONTROL files...")
    for file_id, conn_file in enumerate(tqdm(control_files, desc="Control")):
        result = process_connectivity_file(conn_file, label=0, file_id=file_id, epochs_dir=EPOCHS_DIR)
        
        if result is not None:
            features, adjacencies, labels, file_ids = result
            all_features.extend(features)
            all_adjacencies.extend(adjacencies)
            all_labels.extend(labels)
            all_file_ids.extend([f"ctrl_{fid}" for fid in file_ids])
    
    control_count = sum(1 for l in all_labels if l == 0)
    
    # Convert to arrays
    all_features = np.array(all_features)        # (N, 22, F)
    all_adjacencies = np.array(all_adjacencies)  # (N, 22, 22)
    all_labels = np.array(all_labels)            # (N,)
    
    print(f"\n[4] Feature extraction complete!")
    print(f"  Total epochs: {len(all_labels):,}")
    print(f"  Control (0):  {control_count:,} ({100*control_count/len(all_labels):.1f}%)")
    print(f"  Epilepsy (1): {epilepsy_count:,} ({100*epilepsy_count/len(all_labels):.1f}%)")
    
    # Normalize features
    print(f"\n[5] Normalizing features...")
    n_samples, n_nodes, n_features = all_features.shape
    features_flat = all_features.reshape(-1, n_features)
    
    feat_mean = features_flat.mean(axis=0)
    feat_std = features_flat.std(axis=0) + 1e-10
    features_normalized = (features_flat - feat_mean) / feat_std
    all_features = features_normalized.reshape(n_samples, n_nodes, n_features)
    
    # Save normalization parameters
    np.savez(
        OUTPUT_DIR / 'normalization_params.npz',
        mean=feat_mean,
        std=feat_std
    )
    
    # Save features
    print(f"\n[6] Saving features...")
    np.savez_compressed(
        OUTPUT_DIR / 'control_epilepsy_features.npz',
        node_features=all_features,
        adjacency=all_adjacencies,
        labels=all_labels,
        file_ids=all_file_ids
    )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ FEATURE EXTRACTION COMPLETE!")
    print(f"{'='*80}")
    print(f"\nShapes:")
    print(f"  Node features: {all_features.shape} (epochs, channels, features)")
    print(f"  Adjacency:     {all_adjacencies.shape}")
    print(f"  Labels:        {all_labels.shape}")
    
    print(f"\nFeatures per channel: {n_features}")
    
    feature_breakdown = []
    if USE_CONNECTIVITY_FEATURES:
        feature_breakdown.append("Connectivity: 4 (in/out strength & degree)")
    if USE_SPECTRAL_FEATURES and epochs is not None:
        feature_breakdown.append("Spectral: 6 (power bands)")
        feature_breakdown.append("Statistical: 5 (mean, std, skew, kurt, ll)")
    
    for item in feature_breakdown:
        print(f"  - {item}")
    
    print(f"\nSaved to: {OUTPUT_DIR / 'control_epilepsy_features.npz'}")
    print(f"\nNext step: Run step3_create_graphs.py")
