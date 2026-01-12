"""
VISUALIZATION TOOL
==================
- Reads your .npz connectivity files.
- Plots 2x7 Heatmaps (DTF/PDC across all 7 bands).
- Allows you to pick specific epochs (e.g., --epochs 0 10 50).
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================================================================
# CONFIGURATION
# ==============================================================================

# 22 Channels (Matches your data with A1/A2)
#CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8', 'T3', 'C3', 'Cz', 'C4', 
#                 'T4', 'T5', 'P3', 'Pz', 'P4', 'T6', 'O1', 'Oz', 'O2', 'A1', 'A2']

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                     'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                     'T5', 'P3', 'Pz', 'P4', 'T6', 
                     'O1', 'Oz', 'O2']  # ← T1, T2 (NOT A1, A2!)

# The plotting order
BAND_ORDER = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']

# ==============================================================================
# PLOTTING LOGIC
# ==============================================================================

def plot_epoch(data, epoch_idx, patient_id, output_dir):
    """Generates the 2x7 grid for a single epoch."""
    
    # 1. Check if epoch exists
    n_epochs = len(data['orders'])
    if epoch_idx >= n_epochs:
        print(f"❌ Epoch {epoch_idx} out of bounds (File has {n_epochs} epochs). Skipping.")
        return

    # 2. Extract Data & Find Max Value (for consistent colors)
    matrices = {}
    global_max = 0
    
    for band in BAND_ORDER:
        # Load the specific epoch's matrix (22x22)
        d = data[f'dtf_{band}'][epoch_idx]
        p = data[f'pdc_{band}'][epoch_idx]
        
        matrices[f'dtf_{band}'] = d
        matrices[f'pdc_{band}'] = p
        
        global_max = max(global_max, d.max(), p.max())

    # 3. Setup the Grid (2 Rows, 7 Columns)
    fig, axes = plt.subplots(2, 7, figsize=(24, 7), constrained_layout=True)
    
    # Get Label (Seizure vs Non-Seizure)
    label = "SEIZURE" if data['labels'][epoch_idx] == 1 else "Baseline"
    p_order = data['orders'][epoch_idx]

    # 4. Fill the Grid
    for col, band in enumerate(BAND_ORDER):
        # Top Row: DTF
        sns.heatmap(matrices[f'dtf_{band}'], ax=axes[0, col], 
                    cmap='viridis', square=True, vmin=0, vmax=global_max, 
                    cbar=False, xticklabels=[], 
                    yticklabels=CHANNEL_NAMES if col == 0 else [])
        axes[0, col].set_title(f'DTF - {band.capitalize()}')
        
        # Bottom Row: PDC
        # Show colorbar only on the last column
        show_cbar = (col == 6)
        sns.heatmap(matrices[f'pdc_{band}'], ax=axes[1, col], 
                    cmap='viridis', square=True, vmin=0, vmax=global_max, 
                    cbar=show_cbar, xticklabels=CHANNEL_NAMES, 
                    yticklabels=CHANNEL_NAMES if col == 0 else [])
        axes[1, col].set_title(f'PDC - {band.capitalize()}')

    # 5. Titles and Saving
    fig.suptitle(f'{patient_id} | Epoch {epoch_idx} ({label}) | Order p={p_order} | Max={global_max:.2f}', fontsize=16)
    
    save_name = f"{patient_id}_ep{epoch_idx:03d}_{label}.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=100)
    plt.close()
    print(f"✅ Saved: {save_name}")

# ==============================================================================
# MAIN
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Connectivity Matrices")
    parser.add_argument("--file", required=True, help="Path to a single .npz file")
    parser.add_argument("--output_dir", required=True, help="Where to save images")
    parser.add_argument("--epochs", nargs='+', type=int, default=[0, 10, 20], help="List of epoch indices to plot (e.g. 0 5 10)")
    parser.add_argument("--all_seizures", action="store_true", help="If set, plots ALL seizure epochs found in the file")

    args = parser.parse_args()
    
    file_path = Path(args.file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"📂 Loading: {file_path.name}")
    data = np.load(file_path)
    patient_id = file_path.stem.replace('_graphs', '')

    # Determine which epochs to plot
    epochs_to_plot = args.epochs
    
    if args.all_seizures:
        # Find all indices where label == 1
        seizure_indices = np.where(data['labels'] == 1)[0]
        if len(seizure_indices) > 0:
            print(f"Found {len(seizure_indices)} seizure epochs! Adding them to the list.")
            epochs_to_plot = list(set(epochs_to_plot + list(seizure_indices)))
        else:
            print("No seizure epochs found in this file.")

    print(f"🎨 Generating plots for epochs: {epochs_to_plot}")
    
    for ep in epochs_to_plot:
        plot_epoch(data, ep, patient_id, output_dir)

    print(f"\nDone! Images saved in: {output_dir}")

if __name__ == "__main__":
    main()