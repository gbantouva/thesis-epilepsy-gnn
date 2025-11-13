import numpy as np
import pickle
import mne
from pathlib import Path
import mvar_connectivity

# Load ONE epoch for testing
data_dir = Path("data_pp/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le")
epoch_file = data_dir / "aaaaaanr_s001_t001_epochs.npy"
info_file = data_dir / "aaaaaanr_s001_t001_info.pkl"

X = np.load(epoch_file)
with open(info_file, 'rb') as f:
    info = pickle.load(f)

print("="*70)
print("DEBUGGING NaN ISSUE")
print("="*70)

# Test with FIRST epoch only
epoch_data = X[0]  # Shape: (22, 500)
print(f"\n1. Input data shape: {epoch_data.shape}")
print(f"   Data range: [{epoch_data.min():.4f}, {epoch_data.max():.4f}]")
print(f"   Data mean: {epoch_data.mean():.4f}")
print(f"   Data std: {epoch_data.std():.4f}")
print(f"   Has NaN in input: {np.isnan(epoch_data).any()}")
print(f"   Has Inf in input: {np.isinf(epoch_data).any()}")

# Check variance per channel
print(f"\n2. Channel variances:")
for i, ch_name in enumerate(info['ch_names'][:5]):  # First 5 channels
    var = epoch_data[i].var()
    print(f"   {ch_name}: {var:.6f}")

# Fit MVAR
print(f"\n3. Fitting MVAR (order=15)...")
try:
    A, sigma_cov = mvar_connectivity.mvar_fit(epoch_data, 15)
    print(f"   ✓ MVAR fit successful")
    print(f"   A shape: {A.shape}")
    print(f"   sigma_cov shape: {sigma_cov.shape}")
    print(f"   sigma_cov:\n{sigma_cov}")
    print(f"   Has NaN in A: {np.isnan(A).any()}")
    print(f"   Has NaN in sigma_cov: {np.isnan(sigma_cov).any()}")
except Exception as e:
    print(f"   ✗ MVAR fit failed: {e}")
    import sys
    sys.exit(1)

# Extract diagonal
print(f"\n4. Extracting sigma diagonal...")
sigma_diag = np.diag(sigma_cov)
print(f"   sigma_diag shape: {sigma_diag.shape}")
print(f"   sigma_diag values: {sigma_diag}")
print(f"   Has zeros: {(sigma_diag == 0).any()}")
print(f"   Has negative: {(sigma_diag < 0).any()}")
print(f"   Has NaN: {np.isnan(sigma_diag).any()}")
print(f"   Min value: {sigma_diag.min()}")

# Compute DTF
print(f"\n5. Computing DTF...")
try:
    dtf_spectrum, freqs = mvar_connectivity.DTF(A, sigma_diag, n_fft=512)
    print(f"   ✓ DTF computed")
    print(f"   DTF shape: {dtf_spectrum.shape}")
    print(f"   Has NaN in DTF: {np.isnan(dtf_spectrum).any()}")
    print(f"   Has Inf in DTF: {np.isinf(dtf_spectrum).any()}")
    print(f"   DTF range: [{np.nanmin(dtf_spectrum):.4f}, {np.nanmax(dtf_spectrum):.4f}]")
except Exception as e:
    print(f"   ✗ DTF failed: {e}")
    import traceback
    traceback.print_exc()

# Compute PDC
print(f"\n6. Computing PDC...")
try:
    pdc_spectrum, _ = mvar_connectivity.PDC(A, sigma_diag, n_fft=512)
    print(f"   ✓ PDC computed")
    print(f"   PDC shape: {pdc_spectrum.shape}")
    print(f"   Has NaN in PDC: {np.isnan(pdc_spectrum).any()}")
    print(f"   Has Inf in PDC: {np.isinf(pdc_spectrum).any()}")
    print(f"   PDC range: [{np.nanmin(pdc_spectrum):.4f}, {np.nanmax(pdc_spectrum):.4f}]")
except Exception as e:
    print(f"   ✗ PDC failed: {e}")
    import traceback
    traceback.print_exc()

# Test frequency conversion
print(f"\n7. Testing frequency conversion...")
sfreq = info['sfreq']
freqs_hz = freqs * sfreq
print(f"   Frequency range (normalized): [{freqs.min():.4f}, {freqs.max():.4f}]")
print(f"   Frequency range (Hz): [{freqs_hz.min():.2f}, {freqs_hz.max():.2f}]")

# Test alpha band extraction
print(f"\n8. Testing alpha band extraction...")
fmin, fmax = 8.0, 13.0
freq_indices = np.where((freqs_hz >= fmin) & (freqs_hz <= fmax))[0]
print(f"   Alpha band indices: {len(freq_indices)} frequencies")
if len(freq_indices) > 0:
    dtf_alpha = dtf_spectrum[freq_indices, :, :]
    dtf_alpha_avg = np.mean(dtf_alpha, axis=0)
    print(f"   Alpha DTF shape: {dtf_alpha_avg.shape}")
    print(f"   Alpha DTF has NaN: {np.isnan(dtf_alpha_avg).any()}")
    print(f"   Alpha DTF mean: {np.nanmean(dtf_alpha_avg):.4f}")

print("\n" + "="*70)