# python -c "import sys; sys.path.append(r'C:/Users/georg/Documents/GitHub/thesis-epilepsy-gnn/src'); \
#from grand_av_single import single_patient_grand_average as run; \
#ga,chs,fs,psd,f = run(
#    r'C:/Users/georg/Documents/GitHub/thesis-epilepsy-gnn/data_pp/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001_epochs.npy', 
#    r'C:/Users/georg/Documents/GitHub/thesis-epilepsy-gnn/data_pp/00_epilepsy/aaaaaanr/s001_2003/02_tcp_le/aaaaaanr_s001_t001_info.pkl',
#    out_png_time=r'C:/Users/georg/Documents/GitHub/thesis-epilepsy-gnn/figures/grand_av/PID_time.png',
#    out_png_psd=r'C:/Users/georg/Documents/GitHub/thesis-epilepsy-gnn/figures/grand_av/PID_psd.png'
#); print(ga.shape, fs, psd.shape, f.shape)"

from pathlib import Path
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
    # sel = chs  # use all available channels #for all channels
    idx = [chs.index(c) for c in sel]
    t = np.arange(ga.shape[1]) / fs

    plt.figure(figsize=(10,5)) # plt.figure(figsize=(12, 6))
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
    idx = [chs.index(c) for c in sel]

    plt.figure(figsize=(10,5))
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

    return ga, chs, fs, mean_psd, f

# Example usage:
# ga, chs, fs, mean_psd, f = single_patient_grand_average(
#     "data_pp/..._epochs.npy",
#     "data_pp/..._info.pkl",
#     out_png_time="figures/grand_avg/single_patient_time.png",
#     out_png_psd="figures/grand_avg/single_patient_psd.png"
# )
