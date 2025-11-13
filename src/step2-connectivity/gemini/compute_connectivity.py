import numpy as np
import pickle
import mne
from pathlib import Path
import mvar_connectivity 
from tqdm import tqdm
import matplotlib.pyplot as plt

# Configuration
data_dir = Path("data_pp/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le")
output_dir = Path("data_connectivity") 
output_dir.mkdir(parents=True, exist_ok=True)

pid = "aaaaaanr_s001_t001" 
epoch_file = data_dir / f"{pid}_epochs.npy"
info_file = data_dir / f"{pid}_info.pkl"

# Load data
X = np.load(epoch_file)
with open(info_file, 'rb') as f:
    info = pickle.load(f)

n_epochs, n_channels, n_times = X.shape
sfreq = info['sfreq']
ch_names = info['ch_names']
print(f"Loaded: {n_epochs} epochs, {n_channels} channels, {n_times} timepoints")
print(f"Sampling frequency: {sfreq} Hz")

# Model order
model_order = 15

# Frequency bands
bands = {
    'delta': (0.5, 4),
    'theta': (4, 8),
    'alpha': (8, 13),
    'beta': (13, 30),
    'gamma': (30, 100)
}

n_fft = 512

# Storage
all_dtf_graphs = {band: [] for band in bands}
all_pdc_graphs = {band: [] for band in bands}

print(f"\nProcessing {n_epochs} epochs (model_order={model_order})...")

for epoch_data in tqdm(X):
    # Fit MVAR
    A, sigma_cov = mvar_connectivity.mvar_fit(epoch_data, model_order)
    sigma_diag = np.diag(sigma_cov)


    # FIX: Take absolute value to handle numerical errors
    sigma_diag = np.abs(sigma_diag)

    # FIX: Add small epsilon to avoid zero
    sigma_diag = np.maximum(sigma_diag, 1e-10)
    
    # Compute DTF and PDC
    dtf_spectrum, freqs_norm = mvar_connectivity.DTF(A, sigma_diag, n_fft=n_fft)
    pdc_spectrum, _ = mvar_connectivity.PDC(A, sigma_diag, n_fft=n_fft)
    
    # ========== FIX: Convert normalized frequencies to Hz ==========
    freqs_hz = freqs_norm * sfreq
    # ===============================================================
    
    # Average over frequency bands
    for band_name, (fmin, fmax) in bands.items():
        # Now comparing Hz to Hz!
        freq_indices = np.where((freqs_hz >= fmin) & (freqs_hz <= fmax))[0]
        
        if len(freq_indices) > 0:
            dtf_avg = np.mean(dtf_spectrum[freq_indices, :, :], axis=0)
            pdc_avg = np.mean(pdc_spectrum[freq_indices, :, :], axis=0)
        else:
            print(f"Warning: No frequencies found for {band_name} band!")
            dtf_avg = np.zeros((n_channels, n_channels))
            pdc_avg = np.zeros((n_channels, n_channels))
        
        all_dtf_graphs[band_name].append(dtf_avg)
        all_pdc_graphs[band_name].append(pdc_avg)

# Save results
print("\nSaving results...")
for band_name in bands:
    dtf_file = output_dir / f"{pid}_dtf_{band_name}.npy"
    pdc_file = output_dir / f"{pid}_pdc_{band_name}.npy"
    
    np.save(dtf_file, np.array(all_dtf_graphs[band_name]))
    np.save(pdc_file, np.array(all_pdc_graphs[band_name]))
    
    print(f"  ✓ Saved {band_name}: DTF and PDC")

print(f"\n✅ All done! Results saved to: {output_dir}")
print(f"   Each file shape: ({n_epochs}, {n_channels}, {n_channels})")

# Visualization (same as before)
print("\n📊 Creating visualization...")

fig, axes = plt.subplots(2, 5, figsize=(20, 8))
band_names = list(bands.keys())

# Plot DTF
for idx, band in enumerate(band_names):
    ax = axes[0, idx]
    dtf_data = np.array(all_dtf_graphs[band])
    dtf_mean = dtf_data.mean(axis=0)
    
    im = ax.imshow(dtf_mean, cmap='hot', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f"DTF - {band.upper()}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Source", fontsize=9)
    ax.set_ylabel("Target", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    tick_indices = np.arange(0, len(ch_names), 5)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([ch_names[i] for i in tick_indices], fontsize=7)

# Plot PDC
for idx, band in enumerate(band_names):
    ax = axes[1, idx]
    pdc_data = np.array(all_pdc_graphs[band])
    pdc_mean = pdc_data.mean(axis=0)
    
    im = ax.imshow(pdc_mean, cmap='hot', vmin=0, vmax=1, aspect='auto')
    ax.set_title(f"PDC - {band.upper()}", fontweight='bold', fontsize=12)
    ax.set_xlabel("Source", fontsize=9)
    ax.set_ylabel("Target", fontsize=9)
    plt.colorbar(im, ax=ax, fraction=0.046)
    
    tick_indices = np.arange(0, len(ch_names), 5)
    ax.set_xticks(tick_indices)
    ax.set_yticks(tick_indices)
    ax.set_xticklabels([ch_names[i] for i in tick_indices], rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels([ch_names[i] for i in tick_indices], fontsize=7)

fig.suptitle(f"Patient {pid} - Connectivity (Gramfort Method, Order={model_order})", 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout()

viz_file = output_dir / f"{pid}_connectivity_visualization.png"
fig.savefig(viz_file, dpi=150, bbox_inches='tight')
plt.close(fig)

print(f"  ✓ Saved visualization: {viz_file.name}")

# Statistics
print("\n" + "="*70)
print("CONNECTIVITY STATISTICS")
print("="*70)

for band in band_names:
    dtf_data = np.array(all_dtf_graphs[band])
    pdc_data = np.array(all_pdc_graphs[band])
    
    print(f"\n{band.upper()} Band:")
    print(f"  DTF: mean={dtf_data.mean():.4f}, std={dtf_data.std():.4f}, "
          f"range=[{dtf_data.min():.4f}, {dtf_data.max():.4f}]")
    print(f"  PDC: mean={pdc_data.mean():.4f}, std={pdc_data.std():.4f}, "
          f"range=[{pdc_data.min():.4f}, {pdc_data.max():.4f}]")

print("\n" + "="*70)
print(f"✅ Processing complete!")
print(f"   Results: {output_dir}")
print(f"   Visualization: {viz_file}")
print("="*70 + "\n")