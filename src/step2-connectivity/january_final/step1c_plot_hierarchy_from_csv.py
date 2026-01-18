"""
STEP 1C: PLOT HIERARCHY FROM EXISTING CSVS
==========================================
Reads step1_*_level_orders.csv files and generates the summary hierarchy plot.
Useful if you already ran the analysis and just want to regenerate the figures.

Usage:
    python step1c_plot_hierarchy_from_csv.py --input_dir "F:\October-Thesis\thesis-epilepsy-gnn\p_order_by_sample_20"
"""

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Settings
MIN_ORDER = 5
MAX_ORDER = 18
RECOMMENDED_ORDER = 13  # Set this to your chosen order (e.g., 13)

def load_csv_safely(file_path):
    if file_path.exists():
        print(f"✅ Loaded: {file_path.name}")
        return pd.read_csv(file_path)
    else:
        print(f"❌ Missing: {file_path.name}")
        return None

def plot_hierarchy(input_dir, output_dir):
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # 1. Load DataFrames
    # Try different naming conventions (step1_ vs step1b_)
    epoch_df = load_csv_safely(input_path / 'step1_epoch_level_orders.csv')
    if epoch_df is None: epoch_df = load_csv_safely(input_path / 'step1b_epoch_level_orders.csv')
    
    file_df = load_csv_safely(input_path / 'step1_file_level_orders.csv')
    session_df = load_csv_safely(input_path / 'step1_session_level_orders.csv')
    patient_df = load_csv_safely(input_path / 'step1_patient_level_orders.csv')

    # 2. Setup Figure
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    plt.suptitle(f'Model Order Distribution Hierarchy (Sample 20 Analysis)', fontsize=16, fontweight='bold')

    # Helper to plot histogram
    def plot_hist(ax, data, title, is_epoch=False):
        if data is None or len(data) == 0:
            ax.text(0.5, 0.5, "Data Missing", ha='center', va='center')
            return
        
        # Determine column name
        col = 'optimal_order' if 'optimal_order' in data.columns else 'mean_order'
        if col not in data.columns and 'order' in data.columns: col = 'order'
        
        values = data[col].dropna()
        
        # Plot
        bins = range(MIN_ORDER, MAX_ORDER + 2)
        ax.hist(values, bins=bins, edgecolor='black', alpha=0.7, color='steelblue', align='left')
        ax.axvline(RECOMMENDED_ORDER, color='red', linestyle='--', linewidth=2, label=f'Rec: {RECOMMENDED_ORDER}')
        
        # Stats
        ax.set_title(f'{title}\n(n={len(values):,}, Mean={values.mean():.2f})', fontweight='bold')
        ax.set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
        ax.grid(True, alpha=0.3)
        ax.set_xlabel('Model Order (p)')
        ax.set_ylabel('Count')

    # 3. Plot Levels
    plot_hist(axes[0, 0], epoch_df, "Epoch Level", is_epoch=True)
    plot_hist(axes[0, 1], file_df, "File Level")
    plot_hist(axes[0, 2], session_df, "Session Level")
    plot_hist(axes[1, 0], patient_df, "Patient Level")

    # 4. Group Comparison (Epochs)
    if epoch_df is not None and 'group' in epoch_df.columns:
        ax = axes[1, 1]
        epilepsy = epoch_df[epoch_df['group'] == 'epilepsy']['optimal_order']
        control = epoch_df[epoch_df['group'] == 'control']['optimal_order']
        
        ax.hist([epilepsy, control], bins=range(MIN_ORDER, MAX_ORDER + 2), 
                label=['Epilepsy', 'Control'], color=['#e74c3c', '#3498db'], 
                alpha=0.7, edgecolor='black', align='left')
        ax.axvline(RECOMMENDED_ORDER, color='red', linestyle='--', linewidth=2)
        ax.set_title("Group Comparison (Epochs)", fontweight='bold')
        ax.legend()
        ax.set_xticks(range(MIN_ORDER, MAX_ORDER + 1))
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 1].text(0.5, 0.5, "Group Data Missing", ha='center')

    # 5. Group Comparison (Patients)
    if patient_df is not None and 'group' in patient_df.columns:
        ax = axes[1, 2]
        epilepsy = patient_df[patient_df['group'] == 'epilepsy']['mean_order']
        control = patient_df[patient_df['group'] == 'control']['mean_order']
        
        ax.hist([epilepsy, control], bins=15, 
                label=['Epilepsy', 'Control'], color=['#e74c3c', '#3498db'], 
                alpha=0.7, edgecolor='black')
        ax.axvline(RECOMMENDED_ORDER, color='red', linestyle='--', linewidth=2)
        ax.set_title("Group Comparison (Patients)", fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
    else:
        axes[1, 2].text(0.5, 0.5, "Group Data Missing", ha='center')

    # Save
    plt.tight_layout()
    save_path = output_path / 'hierarchy_plot_from_csv.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved Plot: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", required=True, help="Folder containing the CSV files")
    parser.add_argument("--output_dir", default=".", help="Where to save the plot")
    args = parser.parse_args()
    
    plot_hierarchy(args.input_dir, args.output_dir)