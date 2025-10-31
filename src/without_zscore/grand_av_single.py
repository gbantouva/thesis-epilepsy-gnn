# cd to path cd F:\October-Thesis\thesis-epilepsy-gnn                                                                                                                                                                                         
# python
# import sysfrom pathlib import Path# Add the directory containing grand_av_single.py to the pathsys.path.append('src/without_zscore') # Assuming your file is named grand_av_single.pyfrom grand_av_single import single_patient_grand_average # --- CRITICAL CHANGE HERE ---# DATA_DIR must point to the specific folder containing the patient files.# The patient files are named like 'aaaaaanr_s001_t001_epochs.npy'DATA_DIR = r"F:\October-Thesis\thesis-epilepsy-gnn\data_pp\without_zscore\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le" OUTPUT_DIR = r"F:\October-Thesis\thesis-epilepsy-gnn\figures\without_zscore\single_patient_test"# This file prefix must correspond to a file in the DATA_DIR path above.PREFIX = "aaaaaanr_s001_t001" epochs_path = Path(DATA_DIR) / f"{PREFIX}_epochs.npy"info_path = Path(DATA_DIR) / f"{PREFIX}_info.pkl"# Run the functionsingle_patient_grand_average(    epochs_path,    info_path,    out_png_time=Path(OUTPUT_DIR) / f"{PREFIX}_time_average.png",    out_png_psd=Path(OUTPUT_DIR) / f"{PREFIX}_psd_average.png",    out_png_alpha_topo=Path(OUTPUT_DIR) / f"{PREFIX}_alpha_topomap.png")print(f"\nPlots should be saved in: {OUTPUT_DIR}")
import numpy as np
from scipy.signal import welch
from pathlib import Path
import pickle
import matplotlib.pyplot as plt
import mne

CORE_CHS = ["Fp1","Fp2","F7","F3","Fz","F4","F8",
            "T1","T3","C3","Cz","C4","T4","T2",
            "T5","P3","Pz","P4","T6","O1","Oz","O2"]

def load_info(info_pkl):
    with open(info_pkl, "rb") as f:
        info = pickle.load(f)
    return list(info["ch_names"]), float(info["sfreq"])

def _load_present_mask(epochs_npy):
    mask_path = Path(str(epochs_npy).replace("_epochs.npy", "_present_mask.npy"))
    if mask_path.exists():
        return np.load(mask_path).astype(bool)
    # fallback: assume all real
    return None

def compute_psd_grand_average(X, fs):
    E, C, T = X.shape
    psds = []
    freqs = None
    nperseg = min(1024, T)   # safer on various epoch lengths
    noverlap = nperseg // 2
    for e in range(E):
        p_list = []
        for c in range(C):
            f, Pxx = welch(X[e, c], fs=fs, nperseg=nperseg, noverlap=noverlap)
            p_list.append(Pxx)
        psds.append(p_list)
        if freqs is None:
            freqs = f
    psds = np.asarray(psds)         # (E, C, F)
    mean_psd = psds.mean(axis=0)    # (C, F)
    return mean_psd, freqs

def plot_time_grand_average(ga, chs, fs, out_png=None, sel=("Fz","Cz","Pz","O1","O2")):
    t = np.arange(ga.shape[1]) / fs
    idx = [chs.index(c) for c in sel if c in chs]
    plt.figure(figsize=(10,5))
    for i in idx:
        y = ga[i]
        if np.isfinite(y).any():
            plt.plot(t, y, label=chs[i], alpha=0.9)
    plt.xlabel("Time (s)")
    plt.ylabel(r"Amplitude ($\mu V$)") # Added 'r' for raw string to fix escape sequence warning
    plt.title("Single-patient grand average (time domain)")
    plt.legend()
    plt.tight_layout()
    if out_png: Path(out_png).parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_png, dpi=150)
    plt.close()

def plot_psd_grand_average(mean_psd, f, chs, out_png=None, sel=("Fz","Cz","Pz","O1","O2")):
    idx = [chs.index(c) for c in sel if c in chs]
    plt.figure(figsize=(10,5))
    for i in idx:
        y = mean_psd[i]
        if np.isfinite(y).any():
            plt.semilogy(f, y, label=chs[i])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel(r"Power Spectral Density ($\mu V^2/Hz$)") # Added 'r' for raw string
    plt.title("Single-patient grand average (PSD)"); plt.legend(); plt.tight_layout()
    if out_png: Path(out_png).parent.mkdir(parents=True, exist_ok=True); plt.savefig(out_png, dpi=150)
    plt.close()

def plot_topomap_band(mean_psd, freqs, chs, band, out_png=None, sfreq=250.0):
    fmin, fmax = band
    idx_band = np.where((freqs >= fmin) & (freqs <= fmax))[0]
    if idx_band.size == 0:
        return
    band_power = mean_psd[:, idx_band].mean(axis=1)

    std = mne.channels.make_standard_montage("standard_1020")
    pos = std.get_positions()["ch_pos"].copy()
    # Add approximate FT9/FT10 for T1/T2 if needed
    if "T1" in chs and "T1" not in pos: pos["T1"] = np.array([-0.0840759, 0.0145673, -0.050429])
    if "T2" in chs and "T2" not in pos: pos["T2"] = np.array([ 0.0841131, 0.0143647, -0.050538])

    chs_with_pos = [c for c in chs if c in pos]
    if not chs_with_pos: return
    idx_keep = [chs.index(c) for c in chs_with_pos]
    data_keep = band_power[idx_keep]

    dig = mne.channels.make_dig_montage(ch_pos={c: pos[c] for c in chs_with_pos}, coord_frame="head")
    info = mne.create_info(chs_with_pos, sfreq=sfreq, ch_types="eeg")
    info.set_montage(dig, match_case=False)

    fig, ax = plt.subplots(figsize=(5,4))
    im, _ = mne.viz.plot_topomap(data_keep, info, axes=ax, show=False, names=chs_with_pos,
                                 contours=0, cmap="RdBu_r", sphere=0.09)
    ax.set_title(f"{fmin}-{fmax} Hz band power")
    plt.colorbar(im, ax=ax, orientation="horizontal", fraction=0.05, pad=0.07)
    if out_png:
        Path(out_png).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

def single_patient_grand_average(epochs_npy, info_pkl,
                                 out_png_time=None, out_png_psd=None, out_png_alpha_topo=None):
    # Load data
    X = np.load(epochs_npy)   # (E, C, T)
    chs, fs = load_info(info_pkl)

    # OPTIONAL but recommended if you preprocessed with padding=True:
    mask = _load_present_mask(epochs_npy)
    if mask is not None:
        # Ignore padded channels by NaN-ing them before averaging
        X = X.astype(np.float64, copy=False)
        X[:, ~mask, :] = np.nan

    # Time-domain mean
    ga_time = np.nanmean(X, axis=0)        # (C, T)

    # Frequency-domain mean (Welch)
    ga_psd, freqs = compute_psd_grand_average(np.nan_to_num(X, nan=0.0), fs)  # PSD is robust to NaNs after nan->0

    # Plots
    if out_png_time: plot_time_grand_average(ga_time, chs, fs, out_png_time)
    if out_png_psd:  plot_psd_grand_average(ga_psd, freqs, chs, out_png_psd)
    if out_png_alpha_topo:
        plot_topomap_band(ga_psd, freqs, chs, band=(8,12), out_png=out_png_alpha_topo, sfreq=fs)

    return ga_time, ga_psd, chs, freqs
