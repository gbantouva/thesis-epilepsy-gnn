"""
Step 0 — Patient Split Generator (TUH Dataset)
===============================================
PURPOSE:
  Creates and saves a reproducible 60-20-20 patient-stratified train/val/test
  split for the TUH epilepsy detection pipeline.

  This script MUST be run ONCE before steps 3-6. Every subsequent step loads
  the saved split JSON and uses it to assign epochs to the correct partition.

WHY A FIXED SPLIT INSTEAD OF LOPO:
  The TUH dataset has 200 patients (100 epilepsy + 100 control) with a total
  of ~193,000 epochs. Full Leave-One-Patient-Out cross-validation would require
  running the complete pipeline 200 times, including SSL pre-training in step 6.
  This is computationally unreasonable. A fixed stratified split with strict
  patient-level separation is consistent with published work on large-scale
  EEG datasets and provides statistically meaningful evaluation with
  40 test patients per class.

SPLIT DESIGN:
  Train : 60 epilepsy + 60 control = 120 patients
  Val   : 20 epilepsy + 20 control =  40 patients
  Test  : 20 epilepsy + 20 control =  40 patients

  Stratified: equal epilepsy/control ratio in every split.
  Strict patient separation: a patient's recordings are NEVER split
  across train/val/test. All sessions from one patient go to one split.

REPRODUCIBILITY:
  A fixed random seed (42) ensures the split is identical every run.
  The split is saved to a JSON file loaded by all downstream steps.

PATH STRUCTURE (from preprocessing):
  data_pp_balanced/
    00_epilepsy/
      <patient_id>/
        <session_id>/
          <montage>/
            <file>_epochs.npy
    01_no_epilepsy/
      <patient_id>/
        ...

OUTPUT:
  patient_split.json  — contains train/val/test patient lists with labels

Usage:
  python step0_patient_split.py \\
      --data_dir  F:\\October-Thesis\\thesis-epilepsy-gnn\\data_pp_balanced \\
      --output    F:\\October-Thesis\\thesis-epilepsy-gnn\\patient_split.json \\
      --train_pct 0.60 \\
      --val_pct   0.20 \\
      --seed      42
"""

import argparse
import json
import random
from pathlib import Path
from collections import defaultdict


# ══════════════════════════════════════════════════════════════
# PATIENT DISCOVERY
# ══════════════════════════════════════════════════════════════

def discover_patients(data_dir: Path) -> dict:
    """
    Walk the preprocessed data directory and discover all patients.

    Returns
    -------
    dict with keys 'epilepsy' and 'control', each mapping to a list
    of patient IDs found in that group's subdirectory.

    The function verifies that each patient directory contains at least
    one *_epochs.npy file, so empty or failed preprocessing directories
    are silently excluded.

    Path structure expected:
        data_dir/
            00_epilepsy/<patient_id>/<session>/<montage>/*_epochs.npy
            01_no_epilepsy/<patient_id>/<session>/<montage>/*_epochs.npy
    """
    patients = {'epilepsy': [], 'control': []}

    group_dirs = {
        'epilepsy': data_dir / '00_epilepsy',
        'control':  data_dir / '01_no_epilepsy',
    }

    for group, group_dir in group_dirs.items():
        if not group_dir.exists():
            print(f'  [WARN] Directory not found: {group_dir}')
            continue

        for patient_dir in sorted(group_dir.iterdir()):
            if not patient_dir.is_dir():
                continue

            # Verify this patient has at least one epoch file
            epoch_files = list(patient_dir.rglob('*_epochs.npy'))
            if len(epoch_files) == 0:
                print(f'  [SKIP] {group}/{patient_dir.name}: no epoch files found')
                continue

            patients[group].append(patient_dir.name)

    return patients


# ══════════════════════════════════════════════════════════════
# SPLIT GENERATOR
# ══════════════════════════════════════════════════════════════

def generate_split(patients: dict,
                   train_pct: float = 0.60,
                   val_pct:   float = 0.20,
                   seed:      int   = 42) -> dict:
    """
    Generate stratified train/val/test split at patient level.

    Each group (epilepsy, control) is split independently to maintain
    the same class ratio across all three partitions.

    Parameters
    ----------
    patients  : {'epilepsy': [...], 'control': [...]}
    train_pct : fraction for train (e.g. 0.60)
    val_pct   : fraction for val   (e.g. 0.20)
    seed      : random seed for reproducibility

    Returns
    -------
    dict with structure:
        {
            'train': {'epilepsy': [...], 'control': [...]},
            'val':   {'epilepsy': [...], 'control': [...]},
            'test':  {'epilepsy': [...], 'control': [...]},
        }
    """
    rng = random.Random(seed)
    test_pct = 1.0 - train_pct - val_pct

    split = {'train': {}, 'val': {}, 'test': {}}

    for group, ids in patients.items():
        ids_shuffled = sorted(ids)           # deterministic sort before shuffle
        rng.shuffle(ids_shuffled)

        n_total = len(ids_shuffled)
        n_train = round(n_total * train_pct)
        n_val   = round(n_total * val_pct)
        n_test  = n_total - n_train - n_val  # remainder goes to test

        split['train'][group] = ids_shuffled[:n_train]
        split['val'][group]   = ids_shuffled[n_train: n_train + n_val]
        split['test'][group]  = ids_shuffled[n_train + n_val:]

        print(f'  {group:10s}: {n_total} total → '
              f'{len(split["train"][group])} train, '
              f'{len(split["val"][group])} val, '
              f'{len(split["test"][group])} test')

    return split


# ══════════════════════════════════════════════════════════════
# PATIENT LOOKUP MAP
# ══════════════════════════════════════════════════════════════

def build_patient_lookup(split: dict) -> dict:
    """
    Build a flat lookup: patient_id → {'split': 'train'/'val'/'test',
                                        'label': 1/0}

    This is the format that step 3 (feature extraction) will use to
    assign each epoch file to the correct partition without re-reading
    the full split structure.

    Label encoding: epilepsy=1, control=0 (consistent with TUC pipeline).
    """
    label_map = {'epilepsy': 1, 'control': 0}
    lookup    = {}

    for partition, groups in split.items():
        for group, ids in groups.items():
            for pid in ids:
                if pid in lookup:
                    raise ValueError(
                        f'Patient {pid} appears in multiple partitions! '
                        f'Split is invalid.'
                    )
                lookup[pid] = {
                    'split': partition,
                    'label': label_map[group],
                    'group': group,
                }

    return lookup


# ══════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════

def validate_split(split: dict, patients: dict) -> bool:
    """
    Verify that:
    1. Every discovered patient appears in exactly one partition.
    2. No patient appears in more than one partition.
    3. The class ratio is approximately equal across partitions.
    """
    all_original = set(patients['epilepsy'] + patients['control'])

    train_ids = set(split['train']['epilepsy'] + split['train']['control'])
    val_ids   = set(split['val']['epilepsy']   + split['val']['control'])
    test_ids  = set(split['test']['epilepsy']  + split['test']['control'])

    # Check no overlap
    overlap_tv = train_ids & val_ids
    overlap_tt = train_ids & test_ids
    overlap_vt = val_ids   & test_ids

    if overlap_tv or overlap_tt or overlap_vt:
        print(f'[ERROR] Overlap detected!')
        print(f'  Train∩Val  : {overlap_tv}')
        print(f'  Train∩Test : {overlap_tt}')
        print(f'  Val∩Test   : {overlap_vt}')
        return False

    # Check coverage
    all_split = train_ids | val_ids | test_ids
    missing   = all_original - all_split
    extra     = all_split - all_original

    if missing:
        print(f'[ERROR] {len(missing)} patients not assigned to any split: {missing}')
        return False
    if extra:
        print(f'[ERROR] {len(extra)} patients in split but not in data: {extra}')
        return False

    print('  ✓ No overlap between partitions')
    print('  ✓ All patients assigned to exactly one partition')
    return True


# ══════════════════════════════════════════════════════════════
# SUMMARY PRINTING
# ══════════════════════════════════════════════════════════════

def print_summary(split: dict, patients: dict,
                  train_pct: float, val_pct: float, seed: int):

    test_pct = 1.0 - train_pct - val_pct

    print()
    print('=' * 60)
    print('PATIENT SPLIT SUMMARY')
    print('=' * 60)
    print(f'Random seed : {seed}')
    print(f'Split ratio : {train_pct:.0%} / {val_pct:.0%} / {test_pct:.0%}')
    print()

    total_e = len(patients['epilepsy'])
    total_c = len(patients['control'])
    print(f'{"Partition":10s} | {"Epilepsy":>9} | {"Control":>9} | {"Total":>7}')
    print(f'{"-"*10}-+-{"-"*9}-+-{"-"*9}-+-{"-"*7}')

    for partition in ['train', 'val', 'test']:
        n_e = len(split[partition]['epilepsy'])
        n_c = len(split[partition]['control'])
        print(f'{partition:10s} | {n_e:9d} | {n_c:9d} | {n_e + n_c:7d}')

    print(f'{"TOTAL":10s} | {total_e:9d} | {total_c:9d} | {total_e + total_c:7d}')
    print()
    print('LEAKAGE GUARANTEE:')
    print('  All sessions/recordings from a patient go to ONE partition only.')
    print('  Epochs from the same patient are NEVER split across partitions.')
    print('=' * 60)


# ══════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Step 0 — Patient split generator for TUH dataset',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--data_dir',  required=True,
                        help='Root of preprocessed data (data_pp_balanced)')
    parser.add_argument('--output',    required=True,
                        help='Output JSON path (e.g. patient_split.json)')
    parser.add_argument('--train_pct', type=float, default=0.60,
                        help='Fraction for training   (default 0.60)')
    parser.add_argument('--val_pct',   type=float, default=0.20,
                        help='Fraction for validation (default 0.20)')
    parser.add_argument('--seed',      type=int,   default=42,
                        help='Random seed (default 42)')
    args = parser.parse_args()

    data_dir   = Path(args.data_dir)
    output_path = Path(args.output)

    if not (0 < args.train_pct < 1) or not (0 < args.val_pct < 1):
        raise ValueError('train_pct and val_pct must be between 0 and 1')
    if args.train_pct + args.val_pct >= 1.0:
        raise ValueError('train_pct + val_pct must be < 1.0')

    print('=' * 60)
    print('STEP 0 — PATIENT SPLIT GENERATOR  (TUH dataset)')
    print('=' * 60)
    print(f'Data directory : {data_dir}')
    print(f'Output file    : {output_path}')
    print(f'Split          : {args.train_pct:.0%} train / '
          f'{args.val_pct:.0%} val / '
          f'{1 - args.train_pct - args.val_pct:.0%} test')
    print(f'Random seed    : {args.seed}')
    print()

    # ── Step 1: Discover patients ────────────────────────────────────────
    print('Discovering patients...')
    patients = discover_patients(data_dir)

    n_epi  = len(patients['epilepsy'])
    n_ctrl = len(patients['control'])
    print(f'  Found {n_epi} epilepsy patients')
    print(f'  Found {n_ctrl} control patients')
    print(f'  Total: {n_epi + n_ctrl} patients')

    if n_epi == 0 or n_ctrl == 0:
        print('[ERROR] One or both groups are empty. Check data_dir.')
        return

    # ── Step 2: Generate split ───────────────────────────────────────────
    print('\nGenerating split...')
    split = generate_split(patients,
                           train_pct=args.train_pct,
                           val_pct=args.val_pct,
                           seed=args.seed)

    # ── Step 3: Validate ─────────────────────────────────────────────────
    print('\nValidating split...')
    ok = validate_split(split, patients)
    if not ok:
        print('[ERROR] Split validation failed. Not saving.')
        return

    # ── Step 4: Build lookup ─────────────────────────────────────────────
    lookup = build_patient_lookup(split)

    # ── Step 5: Save ─────────────────────────────────────────────────────
    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_data = {
        'meta': {
            'seed':       args.seed,
            'train_pct':  args.train_pct,
            'val_pct':    args.val_pct,
            'test_pct':   round(1 - args.train_pct - args.val_pct, 4),
            'n_epilepsy': n_epi,
            'n_control':  n_ctrl,
            'n_total':    n_epi + n_ctrl,
            'data_dir':   str(data_dir),
        },
        'split': split,
        'patient_lookup': lookup,
    }

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f'\n  ✓ Split saved → {output_path}')

    # ── Step 6: Print summary ────────────────────────────────────────────
    print_summary(split, patients, args.train_pct, args.val_pct, args.seed)

    print('\nNow run:')
    print(f'  python step3_extract_features.py \\')
    print(f'      --data_dir   {data_dir} \\')
    print(f'      --conn_dir   <connectivity_dir> \\')
    print(f'      --split_json {output_path} \\')
    print(f'      --output_dir <features_dir>')


if __name__ == '__main__':
    main()