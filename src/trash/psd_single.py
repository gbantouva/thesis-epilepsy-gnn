# src/psd_single.py
# python src\psd_single.py --epochs data_pp\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_epochs.npy --info data_pp\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_info.pkl --out figures\psd_single --relative 
from pathlib import Path
import argparse, pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

PLOT_CHS = ["Fz","Cz","Pz","O1","O2"]   # shown in overlay plot

def _load(epochs_npy, info_pkl):
    X = np.load(epochs_npy)  # (E, C, T) z-scored in your pipeline
    with open(info_pkl, "rb") as f: info = pickle.load(f)
    return X, float(info["sfreq"]), list(info["ch_names"])

def _relative_psd(f, pxx, lo=0.5, hi=45.0):
    mask = (f >= lo) & (f <= hi)
    area = np.trapz(pxx[..., mask], f[mask], axis=-1)[..., None]  # (...,1)
    area = np.where(area == 0, 1.0, area)
    return pxx / area

def psd_single(epochs_npy, info_pkl, out_dir, nperseg=256, relative=True):
    out = Path(out_dir); out.mkdir(parents=True, exist_ok=True)
    X, fs, chs = _load(epochs_npy, info_pkl)             # (E,C,T)
    E, C, T = X.shape
    # Welch per epoch → average across epochs
    psds = []
    for e in range(E):
        f, pxx = welch(X[e], fs=fs, nperseg=min(nperseg, T), axis=-1, average="median")  # (C,F)
        psds.append(pxx)
    psd_ch = np.mean(np.stack(psds, axis=0), axis=0)      # (C,F)
    if relative:
        psd_ch = _relative_psd(f, psd_ch)                 # (C,F)
    psd_global = psd_ch.mean(axis=0)                      # (F,)

    # save arrays
    np.savez_compressed(out/"psd_single.npz", f=f, psd_ch=psd_ch, psd_global=psd_global, channels=np.array(chs))

    # plots
    # (a) global spectrum
    plt.figure(figsize=(8,4))
    plt.plot(f, psd_global)
    plt.xlabel("Frequency (Hz)"); plt.ylabel("PSD (relative)" if relative else "PSD (a.u.)")
    plt.title("Grand-average PSD (global, mean across channels)")
    plt.tight_layout(); plt.savefig(out/"psd_global.png", dpi=150); plt.close()

    # (b) selected channels overlay
    plt.figure(figsize=(9,5))
    for ch in PLOT_CHS:
        if ch in chs:
            i = chs.index(ch)
            plt.plot(f, psd_ch[i], label=ch)
    plt.legend(); plt.xlabel("Frequency (Hz)")
    plt.ylabel("PSD (relative)" if relative else "PSD (a.u.)")
    plt.title("Grand-average PSD (selected channels)")
    plt.tight_layout(); plt.savefig(out/"psd_selected.png", dpi=150); plt.close()

    print(f"[single] Saved to {out} | fs={fs} Hz | shape psd_ch={psd_ch.shape}")
    return f, psd_ch, psd_global, chs, fs

if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Grand-average PSD for a single patient")
    ap.add_argument("--epochs", required=True)
    ap.add_argument("--info",   required=True)
    ap.add_argument("--out",    required=True)
    ap.add_argument("--nperseg", type=int, default=256)
    ap.add_argument("--relative", action="store_true"); ap.add_argument("--absolute", dest="relative", action="store_false")
    args = ap.parse_args()
    psd_single(args.epochs, args.info, args.out, args.nperseg, args.relative)
