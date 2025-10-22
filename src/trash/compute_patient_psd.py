#!/usr/bin/env python3
"""
Grand-average utilities for TUEP (TUH Epilepsy Corpus) preprocessing outputs.

What this script does
---------------------
1) Crawls a preprocessed directory (e.g., data_pp/) for .pkl or .npy files.
2) Loads each sample, normalizes to a common set of channels and sampling rate.
3) Computes per-subject (or per-file) averages, then GRAND AVERAGES across groups.
4) Optionally computes grand-average Welch PSDs and averaged adjacency matrices
   if connectivity matrices (DTF/PDC) are present.
5) Saves .npz artifacts and simple matplotlib figures under an output directory.

Assumptions (tweak as needed)
-----------------------------
- Each .pkl ideally contains a dict with keys like:
    {
      'X': np.ndarray,  # (n_epochs, n_channels, n_times) OR (n_channels, n_times)
      'fs': float,
      'ch_names': List[str],
      'label': int or str,  # 0/1 or 'no_epilepsy'/'epilepsy'
      'subject_id': str (optional), 'session_id': str (optional)
      # optional if you computed connectivity per epoch or per file:
      'adj': np.ndarray  # (n_epochs, n_channels, n_channels) OR (n_channels, n_channels)
      'adj_type': str    # e.g., 'DTF' or 'PDC'
    }
- If using .npy, we try to load arrays and infer shapes; you can implement a loader shim.
- Channels: we average only across the intersection of channel labels found in all files for each group.
- Sampling rate: everything is resampled to the most common fs (mode) unless overridden.

Usage
-----
python grand_average.py \
    --data_dir data_pp \
    --out_dir figures \
    --group_key label \
    --target_classes 0 1 \
    --class_names no_epilepsy epilepsy \
    --compute_psd \
    --compute_connectivity

Tip: run once without flags to see what it discovers.
"""

from __future__ import annotations
import argparse
import os
import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple, Union

import numpy as np

# Plotting kept very simple
import matplotlib
matplotlib.use('Agg')  # headless
import matplotlib.pyplot as plt

try:
    from scipy.signal import resample_poly, welch
except Exception:
    resample_poly = None
    welch = None

# -----------------------------
# Helpers
# -----------------------------

@dataclass
class Sample:
    X: np.ndarray             # (E, C, T) or (C, T)
    fs: float
    ch_names: List[str]
    label: Union[int, str]
    subject_id: Optional[str] = None
    session_id: Optional[str] = None
    adj: Optional[np.ndarray] = None  # (E, C, C) or (C, C)
    adj_type: Optional[str] = None


def find_files(root: str, exts=(".pkl", ".npy")) -> List[str]:
    paths = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if any(fn.endswith(ext) for ext in exts):
                paths.append(os.path.join(dirpath, fn))
    paths.sort()
    return paths


def safe_load(path: str) -> dict:
    if path.endswith('.pkl'):
        with open(path, 'rb') as f:
            obj = pickle.load(f)
        if isinstance(obj, dict):
            return obj
        raise ValueError(f"Unsupported PKL structure in {path}")
    elif path.endswith('.npy'):
        arr = np.load(path, allow_pickle=True)
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            # could be a dict saved via np.save(object)
            obj = arr.item()
            if isinstance(obj, dict):
                return obj
        # otherwise return an array with minimal keys
        return { 'X': arr }
    else:
        raise ValueError(f"Unsupported file extension: {path}")


def obj_to_sample(obj: dict, fallback_label: Union[int, str] = 'unknown') -> Sample:
    # Try common keys with multiple aliases
    X = obj.get('X') or obj.get('data') or obj.get('eeg')
    if X is None:
        raise ValueError("Object missing 'X'/'data'/'eeg' array")

    fs = obj.get('fs') or obj.get('sfreq') or obj.get('sampling_rate')
    if fs is None:
        raise ValueError("Object missing 'fs'/'sfreq'/'sampling_rate'")

    ch_names = obj.get('ch_names') or obj.get('channels') or obj.get('ch')
    if ch_names is None:
        raise ValueError("Object missing 'ch_names'/'channels'/'ch'")

    label = obj.get('label', fallback_label)

    subject_id = obj.get('subject_id')
    session_id = obj.get('session_id')

    adj = obj.get('adj') or obj.get('A') or obj.get('connectivity')
    adj_type = obj.get('adj_type') or obj.get('conn_type')

    X = np.asarray(X)
    if X.ndim == 2:  # (C, T) -> (1, C, T)
        X = X[None, ...]
    elif X.ndim != 3:
        raise ValueError(f"X must be (E,C,T) or (C,T), got shape {X.shape}")

    if adj is not None:
        adj = np.asarray(adj)
        if adj.ndim == 2:
            adj = adj[None, ...]
        elif adj.ndim != 3:
            raise ValueError(f"adj must be (E,C,C) or (C,C), got shape {adj.shape}")

    return Sample(X=X, fs=float(fs), ch_names=list(ch_names), label=label,
                  subject_id=subject_id, session_id=session_id, adj=adj, adj_type=adj_type)


def mode(items: Iterable) -> Union[int, float, str]:
    c = Counter(items)
    [(val, _cnt)] = c.most_common(1)
    return val


def intersect_ordered(list_of_lists: List[List[str]]) -> List[str]:
    if not list_of_lists:
        return []
    common = set(list_of_lists[0])
    for lst in list_of_lists[1:]:
        common &= set(lst)
    # Keep the order of the first list
    return [ch for ch in list_of_lists[0] if ch in common]


def resample_epochs(X: np.ndarray, fs_in: float, fs_out: float) -> np.ndarray:
    if abs(fs_in - fs_out) < 1e-6:
        return X
    if resample_poly is None:
        raise RuntimeError("scipy is required for resampling (scipy.signal.resample_poly)")
    # Use polyphase resampling with integer-like up/down
    from fractions import Fraction
    ratio = Fraction(fs_out, fs_in).limit_denominator(100)
    up, down = ratio.numerator, ratio.denominator
    E, C, T = X.shape
    Y = np.empty((E, C, int(np.round(T * fs_out / fs_in))), dtype=X.dtype)
    for e in range(E):
        Y[e] = resample_poly(X[e], up, down, axis=-1)
    return Y


def reorder_channels(X: np.ndarray, from_ch: List[str], to_ch: List[str]) -> np.ndarray:
    idx = [from_ch.index(ch) for ch in to_ch]
    return X[:, idx, :]


# -----------------------------
# Incremental aggregators (numerically stable)
# -----------------------------

@dataclass
class RunningMean:
    n: int
    mean: np.ndarray

    @classmethod
    def start(cls, x: np.ndarray) -> 'RunningMean':
        return cls(n=1, mean=x.astype(np.float64))

    def update(self, x: np.ndarray):
        self.n += 1
        self.mean += (x - self.mean) / self.n


@dataclass
class RunningMeanVar:
    n: int
    mean: np.ndarray
    M2: np.ndarray

    @classmethod
    def start(cls, x: np.ndarray) -> 'RunningMeanVar':
        x = x.astype(np.float64)
        return cls(n=1, mean=x.copy(), M2=np.zeros_like(x))

    def update(self, x: np.ndarray):
        x = x.astype(np.float64)
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        self.M2 += delta * (x - self.mean)

    @property
    def var(self) -> np.ndarray:
        return self.M2 / max(self.n - 1, 1)

    @property
    def sem(self) -> np.ndarray:
        return np.sqrt(self.var / max(self.n, 1))


# -----------------------------
# Core computations
# -----------------------------

@dataclass
class GAResult:
    mean: np.ndarray  # (C, T)
    sem: np.ndarray   # (C, T)
    fs: float
    ch_names: List[str]
    times: np.ndarray


def per_subject_average(sample: Sample) -> np.ndarray:
    """Average epochs within a file/subject -> (C, T)."""
    return sample.X.mean(axis=0)


def compute_grand_average(samples: List[Sample], target_fs: Optional[float] = None,
                           common_channels: Optional[List[str]] = None) -> GAResult:
    assert len(samples) > 0

    # Decide fs and channels
    fs_candidates = [s.fs for s in samples]
    fs0 = target_fs if target_fs is not None else mode(fs_candidates)

    if common_channels is None:
        common_channels = intersect_ordered([s.ch_names for s in samples])
    assert len(common_channels) > 0, "No common channels across samples!"

    agg = None
    T_ref = None

    for s in samples:
        Xi = per_subject_average(s)  # (C, T)
        # Align channels
        Xi = Xi[None, ...]  # (1, C, T) for reorder convenience
        Xi = reorder_channels(Xi, s.ch_names, common_channels)[0]
        # Resample to fs0
        Xi = Xi[None, ...]
        Xi = resample_epochs(Xi, s.fs, fs0) if abs(s.fs - fs0) > 1e-6 else Xi
        Xi = Xi[0]

        if T_ref is None:
            T_ref = Xi.shape[-1]
        else:
            # Pad/trim tiny diffs due to rounding
            if Xi.shape[-1] != T_ref:
                T = Xi.shape[-1]
                if T > T_ref:
                    Xi = Xi[:, :T_ref]
                else:
                    pad = np.zeros((Xi.shape[0], T_ref - T), dtype=Xi.dtype)
                    Xi = np.concatenate([Xi, pad], axis=-1)

        if agg is None:
            agg = RunningMeanVar.start(Xi)
        else:
            agg.update(Xi)

    mean = agg.mean  # (C, T)
    sem = agg.sem    # (C, T)
    times = np.arange(mean.shape[-1]) / fs0
    return GAResult(mean=mean, sem=sem, fs=fs0, ch_names=common_channels, times=times)


@dataclass
class PSDResult:
    f: np.ndarray         # frequencies
    psd_mean: np.ndarray  # (C, F)
    psd_sem: np.ndarray   # (C, F)


def compute_grand_psd(samples: List[Sample], fs_target: float, nperseg: int = 1024) -> PSDResult:
    if welch is None:
        raise RuntimeError("scipy is required for PSD (scipy.signal.welch)")

    # Discover common channels
    common_channels = intersect_ordered([s.ch_names for s in samples])
    assert len(common_channels) > 0

    agg_means = None
    f_ref = None

    for s in samples:
        Xi = per_subject_average(s)[None, ...]           # (1, C, T)
        Xi = reorder_channels(Xi, s.ch_names, common_channels)
        Xi = resample_epochs(Xi, s.fs, fs_target) if abs(s.fs - fs_target) > 1e-6 else Xi
        Xi = Xi[0]

        # Welch per channel
        C = Xi.shape[0]
        psds = []
        for c in range(C):
            f, Pxx = welch(Xi[c], fs=fs_target, nperseg=min(nperseg, Xi.shape[-1]))
            psds.append(Pxx)
        psds = np.stack(psds, axis=0)  # (C, F)
        if f_ref is None:
            f_ref = f
        else:
            assert np.allclose(f_ref, f), "Frequency mismatch"

        if agg_means is None:
            agg_means = RunningMeanVar.start(psds)
        else:
            agg_means.update(psds)

    return PSDResult(f=f_ref, psd_mean=agg_means.mean, psd_sem=agg_means.sem)


@dataclass
class ConnResult:
    adj_mean: np.ndarray  # (C, C)
    adj_sem: np.ndarray   # (C, C)
    ch_names: List[str]
    adj_type: str


def compute_grand_connectivity(samples: List[Sample]) -> Optional[ConnResult]:
    # Keep only samples that have adjacency matrices
    samples = [s for s in samples if s.adj is not None]
    if not samples:
        return None

    common_channels = intersect_ordered([s.ch_names for s in samples])
    assert len(common_channels) > 0
    adj_type = samples[0].adj_type or 'connectivity'

    agg = None
    for s in samples:
        A = s.adj
        # Average epochs if necessary
        if A.ndim == 3:
            A = A.mean(axis=0)  # (C, C)
        # Reorder to common channels
        idx = [s.ch_names.index(ch) for ch in common_channels]
        A = A[np.ix_(idx, idx)]
        if agg is None:
            agg = RunningMeanVar.start(A)
        else:
            agg.update(A)

    return ConnResult(adj_mean=agg.mean, adj_sem=agg.sem, ch_names=common_channels, adj_type=adj_type)


# -----------------------------
# Grouping and IO
# -----------------------------

@dataclass
class Grouped:
    groups: Dict[str, List[Sample]]


def normalize_label(label: Union[int, str]) -> str:
    if isinstance(label, (int, np.integer)):
        return 'no_epilepsy' if int(label) == 0 else 'epilepsy'
    s = str(label).strip().lower()
    return s.replace(' ', '_')


def gather_samples(data_dir: str, group_key: str = 'label') -> Grouped:
    files = find_files(data_dir)
    if not files:
        raise SystemExit(f"No .pkl/.npy files found under {data_dir}")

    groups: Dict[str, List[Sample]] = defaultdict(list)
    for path in files:
        obj = safe_load(path)
        # Provide a fallback label if group_key missing
        fallback = obj.get(group_key, obj.get('label', 'unknown'))
        s = obj_to_sample(obj, fallback_label=fallback)
        key = obj.get(group_key, s.label)
        key = normalize_label(key)
        groups[key].append(s)
    return Grouped(groups=groups)


def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)


def save_ga(out_dir: str, name: str, ga: GAResult):
    ensure_dir(out_dir)
    np.savez_compressed(os.path.join(out_dir, f"grand_average_{name}.npz"),
                        mean=ga.mean, sem=ga.sem, fs=ga.fs,
                        ch_names=np.array(ga.ch_names), times=ga.times)
    # Quick figure: per-channel RMS envelope and a global trace
    rms = np.sqrt((ga.mean ** 2).mean(axis=0))
    plt.figure(figsize=(10, 3))
    plt.plot(ga.times, rms)
    plt.xlabel('Time (s)'); plt.ylabel('RMS (a.u.)')
    plt.title(f'Grand-average RMS — {name} (C={len(ga.ch_names)})')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"grand_average_rms_{name}.png"), dpi=150)
    plt.close()


def save_psd(out_dir: str, name: str, psd: PSDResult):
    ensure_dir(out_dir)
    np.savez_compressed(os.path.join(out_dir, f"grand_psd_{name}.npz"),
                        f=psd.f, psd_mean=psd.psd_mean, psd_sem=psd.psd_sem)
    # Figure: channel-mean PSD
    ch_mean = psd.psd_mean.mean(axis=0)
    plt.figure(figsize=(8, 4))
    plt.semilogy(psd.f, ch_mean)
    plt.xlabel('Frequency (Hz)'); plt.ylabel('PSD (a.u./Hz)')
    plt.title(f'Grand-average PSD — {name}')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"grand_psd_{name}.png"), dpi=150)
    plt.close()


def save_conn(out_dir: str, name: str, conn: ConnResult):
    ensure_dir(out_dir)
    np.savez_compressed(os.path.join(out_dir, f"grand_{conn.adj_type}_{name}.npz"),
                        adj_mean=conn.adj_mean, adj_sem=conn.adj_sem,
                        ch_names=np.array(conn.ch_names), adj_type=conn.adj_type)
    # Simple heatmap
    plt.figure(figsize=(6, 5))
    plt.imshow(conn.adj_mean, aspect='equal', origin='lower')
    plt.colorbar(label=conn.adj_type)
    plt.xticks(range(len(conn.ch_names)), conn.ch_names, rotation=90, fontsize=6)
    plt.yticks(range(len(conn.ch_names)), conn.ch_names, fontsize=6)
    plt.title(f'Grand-average {conn.adj_type} — {name}')
    plt.tight_layout(); plt.savefig(os.path.join(out_dir, f"grand_{conn.adj_type}_{name}.png"), dpi=200)
    plt.close()


# -----------------------------
# CLI
# -----------------------------

def main():
    p = argparse.ArgumentParser(description='Grand-average utilities for TUEP outputs')
    p.add_argument('--data_dir', default='data_pp', help='Root of preprocessed outputs')
    p.add_argument('--out_dir', default='figures', help='Where to store figures and .npz')
    p.add_argument('--group_key', default='label', help='Key in each file used for grouping')
    p.add_argument('--target_fs', type=float, default=None, help='Force resampling to this fs')
    p.add_argument('--target_channels', type=str, nargs='*', default=None,
                   help='Optional explicit list of channels to keep (intersection is used otherwise)')
    p.add_argument('--compute_psd', action='store_true')
    p.add_argument('--compute_connectivity', action='store_true')

    args = p.parse_args()

    grouped = gather_samples(args.data_dir, group_key=args.group_key)

    # Decide global fs and channels per group, then compute
    for name, samples in grouped.groups.items():
        print(f"Group '{name}': {len(samples)} files")

        fs_target = args.target_fs if args.target_fs is not None else mode([s.fs for s in samples])
        channels = args.target_channels if args.target_channels else intersect_ordered([s.ch_names for s in samples])

        ga = compute_grand_average(samples, target_fs=fs_target, common_channels=channels)
        save_ga(args.out_dir, name, ga)

        if args.compute_psd:
            psd = compute_grand_psd(samples, fs_target)
            save_psd(args.out_dir, name, psd)

        if args.compute_connectivity:
            conn = compute_grand_connectivity(samples)
            if conn is not None:
                save_conn(args.out_dir, name, conn)
            else:
                print(f"[INFO] No connectivity matrices found for group '{name}'.")

    print("Done.")


if __name__ == '__main__':
    main()
