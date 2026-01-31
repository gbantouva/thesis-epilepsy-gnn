"""
GLOBAL STATISTICS ANALYZER (FIXED ORDER VERSION)
================================================
Generates high-level summary statistics (Mean, Max, Histograms).
Safe for use with 'Fixed Order' output (no BIC requirement).
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

# UPDATE THIS PATH TO YOUR FIXED ORDER RESULTS DIRECTORY
RESULTS_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_fixed_15")

# Output directory
OUTPUT_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\figures\january\connectivity\statistics_of_parallel")
OUTPUT_DIR.mkdir(exist_ok=True)

# Band names matching your saved data
BAND_NAMES = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']

# ============================================================================
# LOAD DATA
# ============================================================================

all_files = list(RESULTS_DIR.rglob('*_graphs.npz'))
print(f"Found {len(all_files)} result files in {RESULTS_DIR}")

# Storage for statistics
stats = {
    'all':      {band: {'pdc': [], 'dtf': []} for band in BAND_NAMES},
    'epilepsy': {band: {'pdc': [], 'dtf': []} for band in BAND_NAMES},
    'control':  {band: {'pdc': [], 'dtf': []} for band in BAND_NAMES}
}

epoch_counts = {'epilepsy': 0, 'control': 0}

print("Loading and calculating global statistics...")

for f in tqdm(all_files):
    try:
        data = np.load(f)
        
        # Determine group based on folder name
        is_epilepsy = '00_epilepsy' in str(f)
        group = 'epilepsy' if is_epilepsy else 'control'
        
        # Count epochs
        n_epochs = len(data['orders'])
        epoch_counts[group] += n_epochs
        
        # Loop through bands and collect MEAN global connectivity
        for band in BAND_NAMES:
            # Load the full matrix (n_epochs, 22, 22)
            pdc_matrix = data[f'pdc_{band}']
            dtf_matrix = data[f'dtf_{band}']
            
            # Collapse (average) the whole brain into ONE number per epoch
            # We use max() or mean() depending on what you want to show. 
            # Mean is safer for "overall strength".
            pdc_mean_vals = np.mean(pdc_matrix, axis=(1, 2))
            dtf_mean_vals = np.mean(dtf_matrix, axis=(1, 2))
            
            # Store
            stats['all'][band]['pdc'].extend(pdc_mean_vals)
            stats['all'][band]['dtf'].extend(dtf_mean_vals)
            stats[group][band]['pdc'].extend(pdc_mean_vals)
            stats[group][band]['dtf'].extend(dtf_mean_vals)
            
    except Exception as e:
        print(f"Error loading {f.name}: {e}")

# ============================================================================
# VISUALIZATION 1: GLOBAL DISTRIBUTION (HISTOGRAMS)
# ============================================================================

print("Generating Global Histograms...")
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

# Integrated Band - PDC
pdc_vals = stats['all']['integrated']['pdc']
axes[0].hist(pdc_vals, bins=50, color='purple', alpha=0.7, edgecolor='black')
axes[0].set_title(f"Global PDC Distribution (Integrated)\nMean: {np.mean(pdc_vals):.4f}", fontweight='bold')
axes[0].set_xlabel("Mean Connectivity Strength")
axes[0].set_ylabel("Count (Epochs)")

# Integrated Band - DTF
dtf_vals = stats['all']['integrated']['dtf']
axes[1].hist(dtf_vals, bins=50, color='teal', alpha=0.7, edgecolor='black')
axes[1].set_title(f"Global DTF Distribution (Integrated)\nMean: {np.mean(dtf_vals):.4f}", fontweight='bold')
axes[1].set_xlabel("Mean Connectivity Strength")

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'global_distribution_histograms.png', dpi=300)
plt.close()

# ============================================================================
# VISUALIZATION 2: EPILEPSY VS CONTROL (BAR CHARTS)
# ============================================================================

print("Generating Group Comparison Bar Charts...")
fig, axes = plt.subplots(2, 1, figsize=(14, 12))

x = np.arange(len(BAND_NAMES))
width = 0.35

# --- PDC Comparison ---
epi_means = [np.mean(stats['epilepsy'][band]['pdc']) for band in BAND_NAMES]
con_means = [np.mean(stats['control'][band]['pdc']) for band in BAND_NAMES]
epi_err   = [np.std(stats['epilepsy'][band]['pdc'])/np.sqrt(len(stats['epilepsy'][band]['pdc'])) for band in BAND_NAMES] # Standard Error
con_err   = [np.std(stats['control'][band]['pdc'])/np.sqrt(len(stats['control'][band]['pdc'])) for band in BAND_NAMES]

rects1 = axes[0].bar(x - width/2, epi_means, width, yerr=epi_err, label='Epilepsy', color='red', alpha=0.7, capsize=5)
rects2 = axes[0].bar(x + width/2, con_means, width, yerr=con_err, label='Control', color='blue', alpha=0.7, capsize=5)

axes[0].set_title('Global Average PDC by Frequency Band', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Mean Connectivity Strength')
axes[0].set_xticks(x)
axes[0].set_xticklabels([b.upper() for b in BAND_NAMES])
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# Add % difference labels
for i in range(len(BAND_NAMES)):
    diff = ((epi_means[i] - con_means[i]) / con_means[i]) * 100
    color = 'red' if diff > 0 else 'blue'
    axes[0].text(x[i], max(epi_means[i], con_means[i]) + 0.005, f"{diff:+.1f}%", 
                 ha='center', color=color, fontweight='bold', fontsize=9)

# --- DTF Comparison ---
epi_means_d = [np.mean(stats['epilepsy'][band]['dtf']) for band in BAND_NAMES]
con_means_d = [np.mean(stats['control'][band]['dtf']) for band in BAND_NAMES]
epi_err_d   = [np.std(stats['epilepsy'][band]['dtf'])/np.sqrt(len(stats['epilepsy'][band]['dtf'])) for band in BAND_NAMES]
con_err_d   = [np.std(stats['control'][band]['dtf'])/np.sqrt(len(stats['control'][band]['dtf'])) for band in BAND_NAMES]

rects3 = axes[1].bar(x - width/2, epi_means_d, width, yerr=epi_err_d, label='Epilepsy', color='red', alpha=0.7, capsize=5)
rects4 = axes[1].bar(x + width/2, con_means_d, width, yerr=con_err_d, label='Control', color='blue', alpha=0.7, capsize=5)

axes[1].set_title('Global Average DTF by Frequency Band', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Mean Connectivity Strength')
axes[1].set_xticks(x)
axes[1].set_xticklabels([b.upper() for b in BAND_NAMES])
axes[1].legend()
axes[1].grid(axis='y', alpha=0.3)

for i in range(len(BAND_NAMES)):
    diff = ((epi_means_d[i] - con_means_d[i]) / con_means_d[i]) * 100
    color = 'red' if diff > 0 else 'blue'
    axes[1].text(x[i], max(epi_means_d[i], con_means_d[i]) + 0.005, f"{diff:+.1f}%", 
                 ha='center', color=color, fontweight='bold', fontsize=9)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'global_group_comparison.png', dpi=300)
plt.close()

# ============================================================================
# SUMMARY PRINT
# ============================================================================
print("\n" + "="*80)
print("GLOBAL STATISTICS SUMMARY")
print("="*80)
print(f"Total Epochs Processed: {epoch_counts['epilepsy'] + epoch_counts['control']}")
print(f"  - Epilepsy: {epoch_counts['epilepsy']}")
print(f"  - Control:  {epoch_counts['control']}")
print("\nPlots Saved:")
print(f"  1. {OUTPUT_DIR / 'global_distribution_histograms.png'}")
print(f"  2. {OUTPUT_DIR / 'global_group_comparison.png'}")
print("="*80)