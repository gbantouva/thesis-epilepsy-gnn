"""
Step 3 — Feature Extraction (TUH Dataset, 22 channels, 250 Hz)
===============================================================
DESIGN PHILOSOPHY:
  Identical philosophy to the TUC (FORTH) pipeline — curated, justified
  feature set — but adapted for TUH differences:

  KEY DIFFERENCES FROM TUC:
  ┌─────────────────────┬──────────────┬──────────────┐
  │ Property            │ TUC (FORTH)  │ TUH          │
  ├─────────────────────┼──────────────┼──────────────┤
  │ Channels            │ 19           │ 22           │
  │ Sampling rate       │ 256 Hz       │ 250 Hz       │
  │ Samples per epoch   │ 1024         │ 1000         │
  │ Frequency bands     │ 6            │ 7 *          │
  │ MVAR order          │ 12           │ 15           │
  │ Classification      │ ictal vs     │ epilepsy vs  │
  │                     │ pre-ictal    │ control      │
  │ Evaluation          │ LOPO (8)     │ Fixed 60-20-20│
  │ Flat features       │ 53           │ 58           │
  │ Node features       │ 16 per ch    │ 17 per ch    │
  └─────────────────────┴──────────────┴──────────────┘

  * TUH splits gamma into gamma1 (30-55 Hz) and gamma2 (65-80 Hz) to
    avoid the notch filter artefact region around 60 Hz.

FEATURE GROUPS:
  A) Spectral (relative band power, region-averaged):
       7 bands × 5 regions = 35 features
  B) Hjorth parameters (region-averaged):
       3 params × 5 regions = 15 features
  C) Graph-level connectivity (integrated band):
       4 features × 2 measures (DTF, PDC) = 8 features
  TOTAL FLAT: 35 + 15 + 8 = 58 features

BRAIN REGIONS (5 groups, 22 channels):
  Frontal   : Fp1 Fp2 F7 F3 Fz F4 F8           (idx 0-6)
  Temporal  : T1 T3 T4 T2 T5 T6               (idx 7-12)
  Central   : C3 Cz C4                          (idx 13-15)
  Parietal  : P3 Pz P4                          (idx 16-18)
  Occipital : O1 Oz O2                          (idx 19-21)

CLASSIFICATION TASK:
  epilepsy=1, control=0
  Label is inherited from the folder structure (00_epilepsy / 01_no_epilepsy)
  and from the patient_split.json produced by step 0.

EVALUATION:
  Fixed 60-20-20 patient-stratified split loaded from patient_split.json.
  The split key stored per epoch allows downstream steps to correctly
  assign epochs to train/val/test without re-reading the split.

CONNECTIVITY FILES:
  Format: <patient>_<session>_<recording>_graphs.npz
  Located at: conn_dir/00_epilepsy/<patient>/<session>/<montage>/
  Keys available: dtf_integrated, pdc_integrated, dtf_delta, pdc_delta,
                  dtf_theta, pdc_theta, dtf_alpha, pdc_alpha,
                  dtf_beta, pdc_beta, dtf_gamma1, pdc_gamma1,
                  dtf_gamma2, pdc_gamma2, labels, indices

Outputs (saved to --output_dir):
  features_all.npz      X (flat 58), node_features (17), raw_epochs,
                        adj_dtf, adj_pdc, y, patient_ids, splits,
                        file_ids, feature_names
  features_all.csv      human-readable flat features
  feature_summary.txt   shapes, counts, descriptions

Usage:
  python step3_extract_features.py \\
      --data_dir   F:\\data_pp_balanced \\
      --conn_dir   F:\\connectivity\\january_fixed_15 \\
      --split_json F:\\patient_split.json \\
      --output_dir F:\\features
"""

import argparse
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from tqdm import tqdm

warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────

FS          = 250           # TUH sampling rate (TUC was 256)
EPOCH_LEN   = 4.0
N_CHANNELS  = 22            # TUH has 22 channels (TUC had 19)
N_SAMPLES   = 1000          # 4 s × 250 Hz  (TUC was 1024)

CHANNEL_NAMES = [
    'Fp1', 'Fp2', 'F7',  'F3',  'Fz',  'F4',  'F8',   # 0-6   frontal
    'T1',  'T3',  'T4',  'T2',  'T5',  'T6',           # 7-12  temporal
    'C3',  'Cz',  'C4',                                 # 13-15 central
    'P3',  'Pz',  'P4',                                 # 16-18 parietal
    'O1',  'Oz',  'O2',                                 # 19-21 occipital
]

# Brain regions
REGIONS = {
    'frontal':   [0,  1,  2,  3,  4,  5,  6],
    'temporal':  [7,  8,  9, 10, 11, 12],
    'central':   [13, 14, 15],
    'parietal':  [16, 17, 18],
    'occipital': [19, 20, 21],
}
REGION_NAMES = list(REGIONS.keys())

# Left / Right hemisphere for asymmetry index (22-channel layout)
LEFT_CH  = [0, 3, 7, 8, 11, 13, 16, 19]   # Fp1 F3 T1 T3 T5 C3 P3 O1
RIGHT_CH = [1, 5, 9, 10, 12, 15, 18, 21]  # Fp2 F4 T4 T2 T6 C4 P4 O2

# TUH frequency bands — 7 bands (gamma split to avoid 60 Hz notch artifact)
BANDS = {
    'delta':  (0.5,   4.0),
    'theta':  (4.0,   8.0),
    'alpha':  (8.0,  15.0),
    'beta':   (15.0, 30.0),
    'gamma1': (30.0, 55.0),   # lower gamma, below 60 Hz notch
    'gamma2': (65.0, 80.0),   # upper gamma, above 60 Hz notch
    'broad':  (0.5,  80.0),   # integrated band
}
BAND_NAMES = list(BANDS.keys())   # 7 bands


# ─────────────────────────────────────────────────────────────
# A) Spectral features — region-averaged relative band power
#    Output: 7 bands × 5 regions = 35 features
# ─────────────────────────────────────────────────────────────

def spectral_region_features(epoch: np.ndarray):
    """
    epoch : (22, 1000)
    Returns (35,) flat vector + 35 names.

    Relative power = band_power / total_power over 0.5-80 Hz.
    Per-channel values are averaged within each brain region.
    """
    per_ch = np.zeros((N_CHANNELS, len(BANDS)), dtype=np.float32)
    for ci in range(N_CHANNELS):
        freqs, psd = welch(epoch[ci], fs=FS, nperseg=250)
        # Denominator: total power over the full usable band
        idx_total = np.where((freqs >= 0.5) & (freqs <= 80.0))[0]
        total     = np.trapz(psd[idx_total], freqs[idx_total]) + 1e-12
        for bi, (_, (flo, fhi)) in enumerate(BANDS.items()):
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            per_ch[ci, bi] = float(np.trapz(psd[idx], freqs[idx]) / total)

    feats, names = [], []
    for reg, ch_idx in REGIONS.items():
        region_power = per_ch[ch_idx, :].mean(axis=0)   # (7,)
        for bi, band in enumerate(BAND_NAMES):
            feats.append(float(region_power[bi]))
            names.append(f'spec_{band}_{reg}')

    return np.array(feats, dtype=np.float32), names   # (35,)


# ─────────────────────────────────────────────────────────────
# B) Hjorth parameters — region-averaged
#    Output: 3 params × 5 regions = 15 features
# ─────────────────────────────────────────────────────────────

def hjorth_region_features(epoch: np.ndarray):
    """
    epoch : (22, 1000)
    Returns (15,) flat vector + 15 names.

    Activity   = variance(x)
    Mobility   = std(dx) / std(x)
    Complexity = mobility(dx) / mobility(x)
    """
    per_ch = np.zeros((N_CHANNELS, 3), dtype=np.float32)
    for ci in range(N_CHANNELS):
        x   = epoch[ci]
        dx  = np.diff(x)
        d2x = np.diff(dx)
        activity   = float(np.var(x))
        mob_x      = float(np.std(dx)  / (np.std(x)  + 1e-12))
        mob_dx     = float(np.std(d2x) / (np.std(dx) + 1e-12))
        complexity = float(mob_dx / (mob_x + 1e-12))
        per_ch[ci] = [activity, mob_x, complexity]

    feats, names = [], []
    for reg, ch_idx in REGIONS.items():
        region_h = per_ch[ch_idx, :].mean(axis=0)   # (3,)
        for pi, param in enumerate(['activity', 'mobility', 'complexity']):
            feats.append(float(region_h[pi]))
            names.append(f'hjorth_{param}_{reg}')

    return np.array(feats, dtype=np.float32), names   # (15,)


# ─────────────────────────────────────────────────────────────
# C) Graph-level connectivity features
#    Output: 8 features (same structure as TUC)
# ─────────────────────────────────────────────────────────────

def graph_level_features(dtf_integrated: np.ndarray,
                          pdc_integrated: np.ndarray):
    """
    dtf_integrated, pdc_integrated : (22, 22) band-averaged, diagonal=0
    Returns (8,) flat vector + 8 names.

    Per measure (DTF, PDC) — 4 features each = 8 total:
      1. Global mean off-diagonal connectivity
      2. Left-hemisphere mean out-degree
      3. Right-hemisphere mean out-degree
      4. Hemispheric asymmetry index (L-R)/(L+R)
    """
    feats, names = [], []
    mask = ~np.eye(N_CHANNELS, dtype=bool)   # off-diagonal only

    for metric_name, mat in [('dtf', dtf_integrated),
                               ('pdc', pdc_integrated)]:
        global_mean = float(mat[mask].mean())
        feats.append(global_mean)
        names.append(f'graph_{metric_name}_global_mean')

        out_deg    = mat.sum(axis=0)   # (22,)
        left_mean  = float(out_deg[LEFT_CH].mean())
        right_mean = float(out_deg[RIGHT_CH].mean())
        feats.append(left_mean)
        feats.append(right_mean)
        names.append(f'graph_{metric_name}_left_outdeg')
        names.append(f'graph_{metric_name}_right_outdeg')

        asym = (left_mean - right_mean) / (left_mean + right_mean + 1e-12)
        feats.append(float(asym))
        names.append(f'graph_{metric_name}_asymmetry')

    assert len(feats) == 8, f'Expected 8 graph features, got {len(feats)}'
    return np.array(feats, dtype=np.float32), names   # (8,)


# ─────────────────────────────────────────────────────────────
# Per-node features for GNN
# Output: (22, 17)
# ─────────────────────────────────────────────────────────────

def node_features_for_gnn(epoch: np.ndarray,
                            dtf_integrated: np.ndarray,
                            pdc_integrated: np.ndarray) -> np.ndarray:
    """
    Returns per-node feature matrix (22, 17) for GNN use.
    No region averaging — GNN operates on individual nodes.

    Features per node (channel):
      [0-6]  : relative band powers (7 bands — TUH has 7 vs TUC's 6)
      [7-9]  : Hjorth (activity, mobility, complexity)
      [10-14]: time-domain (mean, std, skewness, kurtosis, line_length)
      [15]   : DTF out-degree  (source strength in integrated band)
      [16]   : PDC out-degree  (source strength in integrated band)
    Total: 17 features per node
    """
    node_feats = np.zeros((N_CHANNELS, 17), dtype=np.float32)

    for ci in range(N_CHANNELS):
        x = epoch[ci]

        # [0-6] Spectral — relative band powers (7 bands)
        freqs, psd = welch(x, fs=FS, nperseg=250)
        idx_total  = np.where((freqs >= 0.5) & (freqs <= 80.0))[0]
        total      = np.trapz(psd[idx_total], freqs[idx_total]) + 1e-12
        for bi, (_, (flo, fhi)) in enumerate(BANDS.items()):
            idx = np.where((freqs >= flo) & (freqs < fhi))[0]
            node_feats[ci, bi] = float(np.trapz(psd[idx], freqs[idx]) / total)

        # [7-9] Hjorth
        dx  = np.diff(x)
        d2x = np.diff(dx)
        node_feats[ci, 7]  = float(np.var(x))
        node_feats[ci, 8]  = float(np.std(dx) / (np.std(x) + 1e-12))
        mob_dx             = float(np.std(d2x) / (np.std(dx) + 1e-12))
        node_feats[ci, 9]  = float(mob_dx / (node_feats[ci, 8] + 1e-12))

        # [10-14] Time-domain
        node_feats[ci, 10] = float(np.mean(x))
        node_feats[ci, 11] = float(np.std(x))
        node_feats[ci, 12] = float(skew(x))
        node_feats[ci, 13] = float(kurtosis(x))
        node_feats[ci, 14] = float(np.sum(np.abs(np.diff(x))))   # line length

        # [15-16] Connectivity out-degree
        node_feats[ci, 15] = float(dtf_integrated[:, ci].sum())
        node_feats[ci, 16] = float(pdc_integrated[:, ci].sum())

    return node_feats   # (22, 17)


# ─────────────────────────────────────────────────────────────
# Path parsing helpers
# ─────────────────────────────────────────────────────────────

def parse_patient_from_path(epoch_file: Path) -> str | None:
    """
    Extract patient ID from epoch file path.

    Path structure:
      .../00_epilepsy/<patient_id>/<session>/<montage>/<file>_epochs.npy
      .../01_no_epilepsy/<patient_id>/<session>/<montage>/<file>_epochs.npy

    Returns patient_id string (e.g. 'aaaaaanr') or None if not found.
    """
    parts = epoch_file.parts
    for i, part in enumerate(parts):
        if part in ('00_epilepsy', '01_no_epilepsy'):
            if i + 1 < len(parts):
                return parts[i + 1]
    return None


def find_connectivity_file(epoch_file: Path,
                            data_dir:   Path,
                            conn_dir:   Path) -> Path | None:
    """
    Map an epoch file path to its corresponding connectivity .npz file.

    Epoch:   data_dir/.../00_epilepsy/PAT/S/M/PAT_S_T_epochs.npy
    Conn:    conn_dir/.../00_epilepsy/PAT/S/M/PAT_S_T_graphs.npz

    Returns Path or None if the connectivity file does not exist.
    """
    try:
        rel  = epoch_file.relative_to(data_dir)
        stem = epoch_file.stem.replace('_epochs', '_graphs')
        conn = conn_dir / rel.parent / f'{stem}.npz'
        return conn if conn.exists() else None
    except ValueError:
        return None


# ─────────────────────────────────────────────────────────────
# Per-file processing
# ─────────────────────────────────────────────────────────────

def process_epoch_file(epoch_file: Path,
                        data_dir:   Path,
                        conn_dir:   Path,
                        patient_lookup: dict) -> list | None:
    """
    Process one *_epochs.npy file and return a list of record dicts.

    Each record contains:
      flat_features  : (58,)
      node_features  : (22, 17)
      raw_epoch      : (22, 1000)
      adj_dtf        : (22, 22)
      adj_pdc        : (22, 22)
      label          : int  (1=epilepsy, 0=control)
      patient_id     : str
      split          : str  ('train'/'val'/'test')
      file_id        : str  (stem of epoch file without _epochs)

    Returns None if the file cannot be processed.
    """
    # ── Parse patient ID ───────────────────────────────────────
    patient_id = parse_patient_from_path(epoch_file)
    if patient_id is None:
        return None

    # ── Look up split and label ────────────────────────────────
    if patient_id not in patient_lookup:
        return None   # patient not in split (should not happen)

    info  = patient_lookup[patient_id]
    split = info['split']
    label = info['label']

    # ── Find connectivity file ─────────────────────────────────
    conn_file = find_connectivity_file(epoch_file, data_dir, conn_dir)
    if conn_file is None:
        return None

    # ── Load epochs and connectivity ───────────────────────────
    try:
        epochs = np.load(epoch_file).astype(np.float32)  # (N, 22, 1000)
    except Exception:
        return None

    try:
        conn         = np.load(conn_file, allow_pickle=False)
        conn_indices = conn['indices'].astype(np.int64)   # epochs with valid VAR
        dtf_int_all  = conn['dtf_integrated']             # (E, 22, 22)
        pdc_int_all  = conn['pdc_integrated']             # (E, 22, 22)
    except Exception:
        return None

    # Build original-index → connectivity-index mapping
    orig_to_conn = {int(orig): ci for ci, orig in enumerate(conn_indices)}

    file_id = epoch_file.stem.replace('_epochs', '')

    # ── Dummy call to get feature names (computed once) ───────
    _dummy_epoch = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)
    _dummy_mat   = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float32)
    _, sp_names  = spectral_region_features(_dummy_epoch)
    _, hj_names  = hjorth_region_features(_dummy_epoch)
    _, gr_names  = graph_level_features(_dummy_mat, _dummy_mat)
    feature_names = sp_names + hj_names + gr_names   # 35+15+8=58

    records = []
    for orig_idx in range(len(epochs)):
        if orig_idx not in orig_to_conn:
            continue   # this epoch had no valid VAR fit — skip

        ci        = orig_to_conn[orig_idx]
        epoch     = epochs[orig_idx]           # (22, 1000)
        dtf_int   = dtf_int_all[ci]            # (22, 22)
        pdc_int   = pdc_int_all[ci]            # (22, 22)

        # Flat features (58)
        sp_f, _ = spectral_region_features(epoch)
        hj_f, _ = hjorth_region_features(epoch)
        gr_f, _ = graph_level_features(dtf_int, pdc_int)
        flat     = np.concatenate([sp_f, hj_f, gr_f])

        # Per-node features (22, 17)
        nf = node_features_for_gnn(epoch, dtf_int, pdc_int)

        records.append({
            'flat_features': flat,
            'node_features': nf,
            'raw_epoch':     epoch,
            'adj_dtf':       dtf_int,
            'adj_pdc':       pdc_int,
            'label':         int(label),
            'patient_id':    patient_id,
            'split':         split,
            'file_id':       file_id,
        })

    return records if records else None


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='Step 3 — Feature Extraction (TUH dataset)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data_dir',   required=True,
                        help='Root of preprocessed data (data_pp_balanced)')
    parser.add_argument('--conn_dir',   required=True,
                        help='Root of connectivity output (january_fixed_15)')
    parser.add_argument('--split_json', required=True,
                        help='patient_split.json from step 0')
    parser.add_argument('--output_dir', required=True,
                        help='Output directory for features')
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    conn_dir   = Path(args.conn_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print('=' * 70)
    print('STEP 3 — FEATURE EXTRACTION (TUH dataset, 22 ch, 250 Hz)')
    print('=' * 70)
    print(f'Data dir   : {data_dir}')
    print(f'Conn dir   : {conn_dir}')
    print(f'Split JSON : {args.split_json}')
    print(f'Output dir : {output_dir}')
    print()
    print('Feature groups:')
    print(f'  A) Spectral (7 bands × 5 regions): {7*5} features')
    print(f'  B) Hjorth   (3 params × 5 regions): {3*5} features')
    print(f'  C) Graph-level (DTF+PDC integrated): 8 features')
    print(f'  TOTAL flat features: {7*5 + 3*5 + 8}')
    print(f'  GNN node features  : 17 per channel (22 channels)')
    print('=' * 70)

    # ── Load split ─────────────────────────────────────────────────────
    with open(args.split_json, encoding='utf-8') as f:
        split_data = json.load(f)

    patient_lookup = split_data['patient_lookup']
    print(f'\nLoaded split: {len(patient_lookup)} patients')

    # ── Find all epoch files ───────────────────────────────────────────
    epoch_files = sorted(data_dir.rglob('*_epochs.npy'))
    print(f'Found {len(epoch_files)} epoch files\n')

    if len(epoch_files) == 0:
        print('[ERROR] No epoch files found. Check --data_dir.')
        return

    # ── Process all files ──────────────────────────────────────────────
    all_flat        = []
    all_node        = []
    all_raw         = []
    all_adj_dtf     = []
    all_adj_pdc     = []
    all_labels      = []
    all_patient_ids = []
    all_splits      = []
    all_file_ids    = []
    feature_names   = None
    skipped         = 0

    for ep_file in tqdm(epoch_files, desc='Processing files'):
        records = process_epoch_file(
            ep_file, data_dir, conn_dir, patient_lookup
        )

        if not records:
            skipped += 1
            continue

        # Extract feature names on first successful file
        if feature_names is None:
            ep  = np.zeros((N_CHANNELS, N_SAMPLES), dtype=np.float32)
            mat = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float32)
            _, sp_n = spectral_region_features(ep)
            _, hj_n = hjorth_region_features(ep)
            _, gr_n = graph_level_features(mat, mat)
            feature_names = sp_n + hj_n + gr_n

        for rec in records:
            all_flat.append(rec['flat_features'])
            all_node.append(rec['node_features'])
            all_raw.append(rec['raw_epoch'])
            all_adj_dtf.append(rec['adj_dtf'])
            all_adj_pdc.append(rec['adj_pdc'])
            all_labels.append(rec['label'])
            all_patient_ids.append(rec['patient_id'])
            all_splits.append(rec['split'])
            all_file_ids.append(rec['file_id'])

    if len(all_flat) == 0:
        print('\n[ERROR] No features extracted.')
        return

    # ── Stack arrays ───────────────────────────────────────────────────
    X           = np.stack(all_flat)                        # (N, 58)
    node_feats  = np.stack(all_node)                        # (N, 22, 17)
    raw_epochs  = np.stack(all_raw)                         # (N, 22, 1000)
    adj_dtf     = np.stack(all_adj_dtf)                     # (N, 22, 22)
    adj_pdc     = np.stack(all_adj_pdc)                     # (N, 22, 22)
    y           = np.array(all_labels,      dtype=np.int64)
    patient_ids = np.array(all_patient_ids)
    splits      = np.array(all_splits)
    file_ids    = np.array(all_file_ids)

    # ── Sanity checks ──────────────────────────────────────────────────
    n_bad = int(np.sum(~np.isfinite(X)))
    if n_bad > 0:
        print(f'\n[WARN] {n_bad} non-finite values in X — replacing with 0')
        X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    assert len(feature_names) == 58, \
        f'Expected 58 feature names, got {len(feature_names)}'
    assert X.shape[1] == 58, \
        f'Expected X shape (N, 58), got {X.shape}'
    assert node_feats.shape[1:] == (22, 17), \
        f'Expected node_feats shape (N, 22, 17), got {node_feats.shape}'

    # ── Per-split statistics ───────────────────────────────────────────
    for partition in ['train', 'val', 'test']:
        mask  = splits == partition
        n_epi = int(np.sum(y[mask] == 1))
        n_ctr = int(np.sum(y[mask] == 0))
        n_pat = len(np.unique(patient_ids[mask]))
        print(f'  {partition:5s}: {mask.sum():7d} epochs  '
              f'(epilepsy={n_epi}, control={n_ctr}, '
              f'patients={n_pat})')

    # ── Save ───────────────────────────────────────────────────────────
    out_npz = output_dir / 'features_all.npz'
    np.savez_compressed(
        out_npz,
        X=X,                                            # (N, 58)
        node_features=node_feats,                       # (N, 22, 17)
        raw_epochs=raw_epochs,                          # (N, 22, 1000)
        adj_dtf=adj_dtf,                                # (N, 22, 22)
        adj_pdc=adj_pdc,                                # (N, 22, 22)
        y=y,                                            # (N,)
        patient_ids=patient_ids,                        # (N,) string
        splits=splits,                                  # (N,) 'train'/'val'/'test'
        file_ids=file_ids,                              # (N,) string
        feature_names=np.array(feature_names, dtype=object),
    )
    print(f'\n  ✓ Saved: {out_npz}')

    # Human-readable CSV
    out_csv = output_dir / 'features_all.csv'
    df = pd.DataFrame(X, columns=feature_names)
    df.insert(0, 'label',      y)
    df.insert(1, 'patient_id', patient_ids)
    df.insert(2, 'split',      splits)
    df.insert(3, 'file_id',    file_ids)
    df.to_csv(out_csv, index=False)
    print(f'  ✓ Saved: {out_csv}')

    # ── Summary ────────────────────────────────────────────────────────
    n_total    = len(y)
    n_epilepsy = int((y == 1).sum())
    n_control  = int((y == 0).sum())
    majority_b = max(n_epilepsy, n_control) / n_total * 100

    summary = f"""
FEATURE EXTRACTION SUMMARY (TUH)
===================================
Total epochs         : {n_total:,}
  Epilepsy  (1)      : {n_epilepsy:,}   ({100*n_epilepsy/n_total:.1f}%)
  Control   (0)      : {n_control:,}   ({100*n_control/n_total:.1f}%)

Majority-class acc   : {majority_b:.1f}%  ← dummy classifier baseline

Files processed      : {len(epoch_files) - skipped} / {len(epoch_files)}
Files skipped        : {skipped}
Unique patients      : {len(np.unique(patient_ids))}

Per-split breakdown:
  Train : {(splits=='train').sum():,} epochs, {len(np.unique(patient_ids[splits=='train']))} patients
  Val   : {(splits=='val').sum():,} epochs, {len(np.unique(patient_ids[splits=='val']))} patients
  Test  : {(splits=='test').sum():,} epochs, {len(np.unique(patient_ids[splits=='test']))} patients

Flat feature vector  : {X.shape[1]} features (TUC had 53)
  Spectral (A)       : {7*5}  (7 bands × 5 regions)
  Hjorth   (B)       : {3*5}  (3 params × 5 regions)
  Graph    (C)       : 8  (DTF+PDC: mean, L/R outdeg, asymmetry)

GNN node features    : {node_feats.shape}  (N × channels × per-node-feats)
  Per node           : 17 features (TUC had 16; +1 band)
Raw epochs           : {raw_epochs.shape}   (N × channels × samples)
Adjacency (DTF)      : {adj_dtf.shape}
Adjacency (PDC)      : {adj_pdc.shape}

Feature names (58):
{chr(10).join('  ' + n for n in feature_names)}

Saved files:
  features_all.npz   ← main file (X, node_features, raw_epochs,
                        adj_dtf, adj_pdc, y, patient_ids, splits,
                        file_ids, feature_names)
  features_all.csv   ← human-readable

Next steps:
  python step4_baseline_ml.py     --featfile {out_npz}
  python step5_gnn_supervised.py  --featfile {out_npz}
  python step6_ssl_gnn.py         --featfile {out_npz}
"""
    print(summary)

    out_txt = output_dir / 'feature_summary.txt'
    with open(out_txt, 'w', encoding='utf-8') as fh:
        fh.write(summary)
    print(f'  ✓ Saved: {out_txt}')

    print('\n' + '=' * 70)
    print('STEP 3 COMPLETE')
    print('=' * 70)


if __name__ == '__main__':
    main()
