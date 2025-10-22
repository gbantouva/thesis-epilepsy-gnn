#python -c "import sys; sys.path.append(r'F:/October-Thesis/thesis-epilepsy-gnn/src'); from grand_av_single import single_patient_grand_average as run; ga,chs,fs,psd,f = run(r'F:/October-Thesis/thesis-epilepsy-gnn/data_pp/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001_epochs.npy', r'F:/October-Thesis/thesis-epilepsy-gnn/data_pp/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001_info.pkl', out_png_time=r'F:/October-Thesis/thesis-epilepsy-gnn/figures/grand_av/PID_time.png', out_png_psd=r'F:/October-Thesis/thesis-epilepsy-gnn/figures/grand_av/PID_psd.png'); print(ga.shape, fs, psd.shape, f.shape)"
from pathlib import Path
import mne
import numpy as np
import matplotlib.pyplot as plt
import pickle

CANON = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
         "T1","T3","C3","Cz","C4","T4","T2",
         "T5","P3","Pz","P4","T6","O1","Oz","O2"]

def load_info(info_pkl):
    """
    Load channel names and sampling frequency from MNE Raw info pickle file.

    Parameters
    ----------
    info_pkl: str or Path
        Path to the pickle file containing MNE Raw info.

    Returns
    -------
    chs: list[str]
        List of channel names.
    fs: float
        Sampling frequency in Hz.
    """
    with open(info_pkl, "rb") as f:
        info = pickle.load(f)
    chs = list(info["ch_names"])
    fs  = float(info["sfreq"])
    return chs, fs

def compute_psd_grand_average(X, fs):
    """
    Compute grand average power spectral density (PSD) across all epochs.

    Parameters
    ----------
    X : np.ndarray
        Array of shape (epochs, channels, timepoints).
    fs : float
        Sampling rate in Hz.

    Returns
    -------
    mean_psd : np.ndarray
        Mean PSD per channel (channels, freq_bins)
    freqs : np.ndarray
        Frequencies corresponding to the PSD.
    """
    from scipy.signal import welch
    E, C, T = X.shape
    psds = []
    for epoch in X:
        psd_epoch = []
        for ch_signal in epoch:
            f, Pxx = welch(ch_signal, fs=fs, nperseg=256)
            psd_epoch.append(Pxx)
        psds.append(psd_epoch)
    psds = np.array(psds)            # shape = (epochs, channels, freq)
    mean_psd = np.mean(psds, axis=0) # shape = (channels, freq)
    return mean_psd, f

def plot_time_grand_average(ga, chs, fs, out_png=None):
    """
    Plot time-domain grand average for canonical channels.

    Parameters
    ----------
    ga : np.ndarray
        Grand average array with shape (channels, timepoints).
    chs : list[str]
        List of channel names.
    fs : float
        Sampling frequency.
    out_png : str or Path, optional
        If provided, path to save the plot.
    """
    sel = [c for c in ["Fz","Cz","Pz","O1","O2"] if c in chs]
    #sel = chs  # use all available channels #for all channels
    idx = [chs.index(c) for c in sel]
    t = np.arange(ga.shape[1]) / fs

    plt.figure(figsize=(10,5)) 
    #plt.figure(figsize=(12, 6))
    for i, c in zip(idx, sel):
        plt.plot(t, ga[i], label=c, alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude (z-score)")
    plt.title("Single-patient grand average (time domain)")
    plt.legend()
    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
    plt.close()

def plot_topomap_band(mean_psd, freqs, chs, band, out_png=None, sfreq=250.0):
    """
    Plot topographic scalp map for a specific frequency band (e.g., alpha).
    Handles T1/T2 by assigning FT9/FT10-like coordinates.
    """
    import mne
    import numpy as np
    from pathlib import Path
    import matplotlib.pyplot as plt

    fmin, fmax = band
    idx_band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    band_power = mean_psd[:, idx_band].mean(axis=1)  # (C,)

    # Start from standard_1020 positions
    std = mne.channels.make_standard_montage("standard_1020")
    pos = std.get_positions()["ch_pos"].copy()

    # Approximate mastoids for T1/T2 if present in chs
    # (same numbers you used in preprocessing)
    if "T1" in chs and "T1" not in pos:
        pos["T1"] = np.array([-0.0840759, 0.0145673, -0.050429])  # ~FT9
    if "T2" in chs and "T2" not in pos:
        pos["T2"] = np.array([ 0.0841131, 0.0143647, -0.050538])  # ~FT10

    # Keep only channels that we have coordinates for
    chs_with_pos = [c for c in chs if c in pos]
    if not chs_with_pos:
        raise RuntimeError("No channels have positions for topomap.")

    # Map band_power to this subset
    idx_keep = [chs.index(c) for c in chs_with_pos]
    data_keep = band_power[idx_keep]

    # Build a DigMontage for the available channels
    dig = mne.channels.make_dig_montage(
        ch_pos={c: pos[c] for c in chs_with_pos}, coord_frame="head"
    )
    info = mne.create_info(chs_with_pos, sfreq=sfreq, ch_types="eeg")
    info.set_montage(dig, match_case=False)

    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    im, _ = mne.viz.plot_topomap(
        data_keep, info, axes=ax, show=False, names=chs_with_pos,
        contours=0, cmap="RdBu_r", sphere=0.09
    )
    ax.set_title(f"{fmin}-{fmax} Hz band power")
    plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07)
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def plot_psd_grand_average(mean_psd, f, chs, out_png=None):
    """
    Plot frequency-domain grand average (PSD) for canonical channels.

    Parameters
    ----------
    mean_psd : np.ndarray
        Mean PSD per channel (channels, freq_bins).
    f : np.ndarray
        Frequencies corresponding to the PSD.
    chs : list[str]
        List of channel names.
    out_png : str or Path, optional
        If provided, path to save the plot.
    """
    sel = [c for c in ["Fz","Cz","Pz","O1","O2"] if c in chs]
    #sel = chs  # use all available channels #for all channels
    idx = [chs.index(c) for c in sel]

    plt.figure(figsize=(10,5))
    #plt.figure(figsize=(12, 6))
    for i, c in zip(idx, sel):
        plt.semilogy(f, mean_psd[i], label=c) # semilogy for clearer view
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Power (PSD, a.u.)")
    plt.title("Single-patient grand average (PSD domain)")
    plt.legend()
    plt.tight_layout()
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out_png, dpi=150)
    plt.close()

def single_patient_grand_average(epochs_npy, info_pkl, out_png_time=None, out_png_psd=None):
    """
    Compute both time-domain and frequency-domain grand averages for a single patient.

    Parameters
    ----------
    epochs_npy : str or Path
        Path to numpy file with epochs (epochs, channels, timepoints).
    info_pkl : str or Path
        Path to pickle file containing channel names and sample rate.
    out_png_time : str or Path, optional
        Save path for time-domain average figure.
    out_png_psd : str or Path, optional
        Save path for PSD grand average figure.

    Returns
    -------
    ga : np.ndarray
        Time-domain grand average.
    chs : list[str]
        Channel names.
    fs : float
        Sampling frequency.
    mean_psd : np.ndarray
        PSD grand average per channel.
    f : np.ndarray
        Frequency values.
    """
    X = np.load(epochs_npy)          # shape (E, C, T)
    chs, fs = load_info(info_pkl)

    # Time-domain mean
    ga = X.mean(axis=0)              # shape (C, T)
    plot_time_grand_average(ga, chs, fs, out_png_time)

    # Frequency-domain mean (PSD)
    mean_psd, f = compute_psd_grand_average(X, fs)
    plot_psd_grand_average(mean_psd, f, chs, out_png_psd)
    plot_topomap_band(
    mean_psd, f, chs, band=(8, 12),
    out_png=r"F:/October-Thesis/thesis-epilepsy-gnn/figures/grand_av/PID_alpha_topomap.png",
    sfreq=fs
)

    return ga, chs, fs, mean_psd, f

# Example usage:
# ga, chs, fs, mean_psd, f = single_patient_grand_average(
#     "data_pp/..._epochs.npy",
#     "data_pp/..._info.pkl",
#     out_png_time="figures/grand_avg/single_patient_time.png",
#     out_png_psd="figures/grand_avg/single_patient_psd.png"
# )
