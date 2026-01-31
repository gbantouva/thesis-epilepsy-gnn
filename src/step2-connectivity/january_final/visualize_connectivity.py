"""
VISUALIZATION TOOL - CORRECTED VERSION
=======================================
- Reads your .npz connectivity files
- Plots 2x7 Heatmaps (DTF/PDC across all 7 bands)
- FIXED: Channel names (T1/T2 instead of A1/A2)
- Allows you to pick specific epochs
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ============================================================================
# CONFIGURATION
# ============================================================================

# CORRECT channel names (matches preprocessing!)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                 'T5', 'P3', 'Pz', 'P4', 'T6', 
                 'O1', 'Oz', 'O2']

# Band order for plotting
BAND_ORDER = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']

BAND_LABELS = {
    'integrated': 'Integrated (0.5-80 Hz)',
    'delta': 'Delta (Î´, 0.5-4 Hz)',
    'theta': 'Theta (Î¸, 4-8 Hz)',
    'alpha': 'Alpha (Î±, 8-15 Hz)',
    'beta': 'Beta (Î², 15-30 Hz)',
    'gamma1': 'Gamma1 (Î³â‚, 30-55 Hz)',
    'gamma2': 'Gamma2 (Î³â‚‚, 65-80 Hz)'
}

# ============================================================================
# PLOTTING LOGIC
# ============================================================================

def plot_epoch(data, epoch_idx, patient_id, output_dir):
    """Generates the 2x7 grid for a single epoch."""
    
    # 1. Check if epoch exists
    n_epochs = len(data['orders'])
    if epoch_idx >= n_epochs:
        print(f"Epoch {epoch_idx} out of bounds (File has {n_epochs} epochs). Skipping.")
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
    fig, axes = plt.subplots(2, 7, figsize=(28, 8), constrained_layout=True)
    
    # Get Label (Epilepsy vs Control)
    label = "EPILEPSY" if data['labels'][epoch_idx] == 1 else "CONTROL"
    p_order = data['orders'][epoch_idx]

    # 4. Fill the Grid
    for col, band in enumerate(BAND_ORDER):
        # Top Row: DTF
        sns.heatmap(matrices[f'dtf_{band}'], ax=axes[0, col], 
                    cmap='viridis', square=True, vmin=0, vmax=global_max, 
                    cbar=False, 
                    xticklabels=[], 
                    yticklabels=CHANNEL_NAMES if col == 0 else [],
                    linewidths=0.5, linecolor='gray')
        axes[0, col].set_title(f'DTF\n{band.capitalize()}', fontsize=10, fontweight='bold')
        
        if col == 0:
            axes[0, col].set_ylabel('Sink (To)', fontsize=10, fontweight='bold')
        
        # Bottom Row: PDC
        # Show colorbar only on the last column
        show_cbar = (col == 6)
        cbar_kws = {'label': 'Connectivity Strength'} if show_cbar else {}
        
        sns.heatmap(matrices[f'pdc_{band}'], ax=axes[1, col], 
                    cmap='viridis', square=True, vmin=0, vmax=global_max, 
                    cbar=show_cbar, cbar_kws=cbar_kws,
                    xticklabels=CHANNEL_NAMES, 
                    yticklabels=CHANNEL_NAMES if col == 0 else [],
                    linewidths=0.5, linecolor='gray')
        axes[1, col].set_title(f'PDC\n{band.capitalize()}', fontsize=10, fontweight='bold')
        axes[1, col].tick_params(axis='x', rotation=90, labelsize=8)
        
        if col == 0:
            axes[1, col].set_ylabel('Sink (To)', fontsize=10, fontweight='bold')
        
        # Add xlabel on bottom row center
        if col == 3:
            axes[1, col].set_xlabel('Source (From)', fontsize=10, fontweight='bold')

    # 5. Title and Saving
    fig.suptitle(f'{patient_id} | Epoch {epoch_idx} ({label}) | Order p={p_order} | Max Connectivity={global_max:.3f}', 
                fontsize=14, fontweight='bold')
    
    save_name = f"{patient_id}_ep{epoch_idx:03d}_{label}.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"âœ… Saved: {save_name}")


def plot_single_band_comparison(data, epoch_idx, patient_id, output_dir, band='integrated'):
    """Plot a single band with better detail for thesis figures."""
    
    n_epochs = len(data['orders'])
    if epoch_idx >= n_epochs:
        print(f"Epoch {epoch_idx} out of bounds")
        return
    
    dtf = data[f'dtf_{band}'][epoch_idx]
    pdc = data[f'pdc_{band}'][epoch_idx]
    label = "EPILEPSY" if data['labels'][epoch_idx] == 1 else "CONTROL"
    p_order = data['orders'][epoch_idx]
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    global_max = max(dtf.max(), pdc.max())
    
    # DTF
    sns.heatmap(dtf, ax=axes[0], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=global_max,
               cbar_kws={'label': 'Connectivity Strength'},
               linewidths=0.5, linecolor='white')
    axes[0].set_title(f'DTF - {BAND_LABELS[band]}', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Source (From)', fontsize=12)
    axes[0].set_ylabel('Sink (To)', fontsize=12)
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].tick_params(axis='y', rotation=0)
    
    # PDC
    sns.heatmap(pdc, ax=axes[1], cmap='viridis', square=True,
               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
               vmin=0, vmax=global_max,
               cbar_kws={'label': 'Connectivity Strength'},
               linewidths=0.5, linecolor='white')
    axes[1].set_title(f'PDC - {BAND_LABELS[band]}', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Source (From)', fontsize=12)
    axes[1].set_ylabel('Sink (To)', fontsize=12)
    axes[1].tick_params(axis='x', rotation=45)
    axes[1].tick_params(axis='y', rotation=0)
    
    fig.suptitle(f'{patient_id} | Epoch {epoch_idx} ({label}) | Order p={p_order}', 
                fontsize=16, fontweight='bold')
    
    plt.tight_layout()
    
    save_name = f"{patient_id}_ep{epoch_idx:03d}_{band}_detailed.png"
    save_path = output_dir / save_name
    plt.savefig(save_path, dpi=200, bbox_inches='tight')
    plt.close()
    print(f"Saved detailed plot: {save_name}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Visualize Connectivity Matrices")
    parser.add_argument("--file", required=True, help="Path to a single .npz file")
    parser.add_argument("--output_dir", required=True, help="Where to save images")
    parser.add_argument("--epochs", nargs='+', type=int, default=[0, 10, 20], 
                       help="List of epoch indices to plot (e.g. 0 5 10)")
    parser.add_argument("--all_epilepsy", action="store_true", 
                       help="If set, plots ALL epilepsy epochs")
    parser.add_argument("--all_control", action="store_true",
                       help="If set, plots ALL control epochs")
    parser.add_argument("--detailed_band", type=str, default=None,
                       choices=['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2'],
                       help="Generate detailed plots for a specific band")
    parser.add_argument("--max_plots", type=int, default=50,
                       help="Maximum number of plots to generate (safety limit)")

    args = parser.parse_args()
    
    file_path = Path(args.file)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if not file_path.exists():
        print(f"Error: File not found: {file_path}")
        return

    print(f"Loading: {file_path.name}")
    data = np.load(file_path)
    patient_id = file_path.stem.replace('_graphs', '')
    
    print(f"   Total epochs: {len(data['orders'])}")
    print(f"   Epilepsy epochs: {np.sum(data['labels'] == 1)}")
    print(f"   Control epochs: {np.sum(data['labels'] == 0)}")

    # Determine which epochs to plot
    epochs_to_plot = list(args.epochs)
    
    if args.all_epilepsy:
        epilepsy_indices = np.where(data['labels'] == 1)[0]
        if len(epilepsy_indices) > 0:
            print(f"Found {len(epilepsy_indices)} epilepsy epochs! Adding them.")
            epochs_to_plot.extend(list(epilepsy_indices))
        else:
            print("No epilepsy epochs found.")
    
    if args.all_control:
        control_indices = np.where(data['labels'] == 0)[0]
        if len(control_indices) > 0:
            print(f"Found {len(control_indices)} control epochs! Adding them.")
            epochs_to_plot.extend(list(control_indices))
        else:
            print("No control epochs found.")
    
    # Remove duplicates and sort
    epochs_to_plot = sorted(list(set(epochs_to_plot)))
    
    # Safety limit
    if len(epochs_to_plot) > args.max_plots:
        print(f"âš ï¸  Too many plots requested ({len(epochs_to_plot)}). Limiting to {args.max_plots}.")
        epochs_to_plot = epochs_to_plot[:args.max_plots]
    
    print(f"Generating plots for {len(epochs_to_plot)} epochs")
    
    # Generate full 2x7 grid plots
    for ep in epochs_to_plot:
        plot_epoch(data, ep, patient_id, output_dir)
    
    # Generate detailed single-band plots if requested
    if args.detailed_band:
        print(f"\nGenerating detailed {args.detailed_band} band plots...")
        for ep in epochs_to_plot:
            plot_single_band_comparison(data, ep, patient_id, output_dir, args.detailed_band)
    
    print(f"\Done! Images saved in: {output_dir}")
    print(f"   Generated {len(epochs_to_plot)} full plots")
    if args.detailed_band:
        print(f"   Generated {len(epochs_to_plot)} detailed {args.detailed_band} plots")


if __name__ == "__main__":
    main()