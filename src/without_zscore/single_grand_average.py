import numpy as np
from scipy.signal import welch
import pickle
import matplotlib.pyplot as plt

def load_info(info_pkl):
    with open(info_pkl, "rb") as f:
        info = pickle.load(f)
    chs = list(info["ch_names"])
    fs = float(info["sfreq"])
    return chs, fs

def single_patient_grand_average(epochs_npy, info_pkl):
    # Load epochs (E, C, T)
    X = np.load(epochs_npy)
    chs, fs = load_info(info_pkl)
    
    # Time-domain grand average: mean across epochs
    ga_time = X.mean(axis=0)  # shape (C, T)
    
    # Frequency-domain: mean PSD across epochs
    freqs = None
    psds = []
    for epoch in X:
        epoch_psds = []
        for ch_signal in epoch:
            f, Pxx = welch(ch_signal, fs=fs, nperseg=256)
            epoch_psds.append(Pxx)
        psds.append(epoch_psds)
        if freqs is None:
            freqs = f
    psds = np.array(psds)        # (E, C, F)
    ga_psd = np.mean(psds, axis=0)  # (C, F)
    return ga_time, ga_psd, chs, freqs

# Example usage
ga_time, ga_psd, chs, freqs = single_patient_grand_average(
    r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_epochs.npy",
    r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_info.pkl"
)


# Plot time-domain for first five channels (optional for QC)
plt.figure(figsize=(10,5))
for i, ch in enumerate(chs[:5]):
    plt.plot(np.arange(ga_time.shape[1])/float(freqs[-1]), ga_time[i], label=ch)
plt.xlabel("Time (s)")
plt.ylabel("Amplitude (z-score)")
plt.title("Time-domain grand average (first 5 channels)")
plt.legend()
plt.tight_layout()
plt.show()

# Plot PSD for first five channels (optional for QC)
plt.figure(figsize=(10,5))
for i, ch in enumerate(chs[:5]):
    plt.semilogy(freqs, ga_psd[i], label=ch)
plt.xlabel("Frequency (Hz)")
plt.ylabel("Power (PSD)")
plt.title("Frequency-domain grand average (first 5 channels)")
plt.legend()
plt.tight_layout()
plt.show()

print("Grand average computation and plotting complete.")
