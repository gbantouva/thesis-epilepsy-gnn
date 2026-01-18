"""
GROUP-LEVEL CONNECTIVITY ANALYSIS
==================================
Compares connectivity between control and epilepsy patient groups.

Since labels are patient-level (0=control, 1=epilepsy), we need to:
1. Load ALL patient files
2. Average connectivity within each patient
3. Compare control group vs epilepsy group
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
from tqdm import tqdm

# ==============================================================================
# SETTINGS
# ==============================================================================

# Your connectivity output directory
CONNECTIVITY_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_final_fixed_order")

# Which band to analyze
BAND_TO_ANALYZE = 'pdc_delta'  # Options: dtf_integrated, pdc_integrated, dtf_delta, etc.

# Output directory
OUTPUT_DIR = Path("group_connectivity_analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Channel names
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                 'T5', 'P3', 'Pz', 'P4', 'T6',
                 'O1', 'Oz', 'O2']

# ==============================================================================
# LOAD ALL PATIENT DATA
# ==============================================================================

print(f"\n{'='*80}")
print(f"GROUP-LEVEL CONNECTIVITY ANALYSIS")
print(f"{'='*80}")
print(f"Connectivity directory: {CONNECTIVITY_DIR}")
print(f"Band to analyze: {BAND_TO_ANALYZE}")
print()

# Find all connectivity files
all_files = list(CONNECTIVITY_DIR.rglob("*_graphs.npz"))
print(f"ðŸ“ Found {len(all_files)} connectivity files")
print()

if len(all_files) == 0:
    print("âŒ No connectivity files found! Make sure Step 2 is complete.")
    exit()

# Separate by group
control_files = [f for f in all_files if '01_no_epilepsy' in str(f)]
epilepsy_files = [f for f in all_files if '00_epilepsy' in str(f)]

print(f"ðŸ“Š Group distribution:")
print(f"  Control patients: {len(control_files)} files")
print(f"  Epilepsy patients: {len(epilepsy_files)} files")
print()

if len(control_files) == 0 or len(epilepsy_files) == 0:
    print("âŒ One group is missing! Check your data.")
    exit()

# ==============================================================================
# LOAD AND AVERAGE WITHIN EACH PATIENT
# ==============================================================================

def load_patient_average(file_path, band_name):
    """
    Load connectivity from one patient file and average across epochs.
    
    Returns
    -------
    avg_connectivity : ndarray (22, 22)
        Average connectivity matrix for this patient
    n_epochs : int
        Number of epochs averaged
    """
    try:
        data = np.load(file_path)
        
        # Check if band exists
        if band_name not in data:
            return None, 0
        
        connectivity = data[band_name]  # Shape: (n_epochs, 22, 22)
        
        # Average across epochs
        avg_connectivity = np.mean(connectivity, axis=0)
        
        return avg_connectivity, len(connectivity)
    
    except Exception as e:
        print(f"âš ï¸  Error loading {file_path.name}: {e}")
        return None, 0


print(f"ðŸ“Š Loading control patients...")
control_matrices = []
control_epoch_counts = []

for f in tqdm(control_files, desc="Control"):
    avg_conn, n_epochs = load_patient_average(f, BAND_TO_ANALYZE)
    if avg_conn is not None:
        control_matrices.append(avg_conn)
        control_epoch_counts.append(n_epochs)

print(f"âœ… Loaded {len(control_matrices)} control patients")
print(f"   Total epochs: {sum(control_epoch_counts)}")
print()

print(f"ðŸ“Š Loading epilepsy patients...")
epilepsy_matrices = []
epilepsy_epoch_counts = []

for f in tqdm(epilepsy_files, desc="Epilepsy"):
    avg_conn, n_epochs = load_patient_average(f, BAND_TO_ANALYZE)
    if avg_conn is not None:
        epilepsy_matrices.append(avg_conn)
        epilepsy_epoch_counts.append(n_epochs)

print(f"âœ… Loaded {len(epilepsy_matrices)} epilepsy patients")
print(f"   Total epochs: {sum(epilepsy_epoch_counts)}")
print()

# Convert to arrays
control_matrices = np.array(control_matrices)  # Shape: (n_control, 22, 22)
epilepsy_matrices = np.array(epilepsy_matrices)  # Shape: (n_epilepsy, 22, 22)

print(f"ðŸ“Š Data shapes:")
print(f"  Control: {control_matrices.shape}")
print(f"  Epilepsy: {epilepsy_matrices.shape}")
print()

# ==============================================================================
# VALIDATE DIAGONAL IS ZERO
# ==============================================================================

print(f"âœ… DIAGONAL VALIDATION:")

# Check control diagonal
control_diag = np.array([np.diag(control_matrices[i]) for i in range(len(control_matrices))])
max_control_diag = np.max(np.abs(control_diag))

# Check epilepsy diagonal
epilepsy_diag = np.array([np.diag(epilepsy_matrices[i]) for i in range(len(epilepsy_matrices))])
max_epilepsy_diag = np.max(np.abs(epilepsy_diag))

print(f"  Control max diagonal: {max_control_diag:.6f}")
print(f"  Epilepsy max diagonal: {max_epilepsy_diag:.6f}")

if max(max_control_diag, max_epilepsy_diag) > 0.01:
    print(f"  âš ï¸  WARNING: Diagonal not zero! Check Step 2.")
else:
    print(f"  âœ… Diagonal correctly set to zero")
print()

# ==============================================================================
# COMPUTE GROUP STATISTICS
# ==============================================================================

print(f"ðŸ“Š Computing group statistics...")

# Group averages
avg_control = np.mean(control_matrices, axis=0)  # Average across patients
avg_epilepsy = np.mean(epilepsy_matrices, axis=0)

# Standard errors (for plotting)
se_control = stats.sem(control_matrices, axis=0)
se_epilepsy = stats.sem(epilepsy_matrices, axis=0)

# Difference
diff = avg_epilepsy - avg_control

# Statistical testing (t-test for each connection)
n_channels = len(CHANNEL_NAMES)
t_stats = np.zeros((n_channels, n_channels))
p_values = np.zeros((n_channels, n_channels))

for i in range(n_channels):
    for j in range(n_channels):
        if i != j:  # Skip diagonal
            control_conn = control_matrices[:, i, j]  # All control patients, connection iâ†’j
            epilepsy_conn = epilepsy_matrices[:, i, j]  # All epilepsy patients, connection iâ†’j
            
            # Independent samples t-test
            t_stat, p_val = stats.ttest_ind(epilepsy_conn, control_conn)
            t_stats[i, j] = t_stat
            p_values[i, j] = p_val

# Effect size (Cohen's d)
cohens_d = np.zeros((n_channels, n_channels))
for i in range(n_channels):
    for j in range(n_channels):
        if i != j:
            control_conn = control_matrices[:, i, j]
            epilepsy_conn = epilepsy_matrices[:, i, j]
            
            pooled_std = np.sqrt((np.var(control_conn) + np.var(epilepsy_conn)) / 2)
            if pooled_std > 0:
                cohens_d[i, j] = (np.mean(epilepsy_conn) - np.mean(control_conn)) / pooled_std

# Multiple comparison correction (Bonferroni)
n_comparisons = n_channels * (n_channels - 1)
alpha_corrected = 0.05 / n_comparisons
significant_mask = (p_values < alpha_corrected) & (p_values > 0)
n_significant = np.sum(significant_mask)

print(f"âœ… Statistics computed:")
print(f"  T-tests performed: {n_comparisons}")
print(f"  Bonferroni-corrected Î±: {alpha_corrected:.6f}")
print(f"  Significant connections: {n_significant} ({n_significant/n_comparisons*100:.2f}%)")
print(f"  Mean |Cohen's d|: {np.mean(np.abs(cohens_d[cohens_d != 0])):.3f}")
print()

# ==============================================================================
# VISUALIZATION
# ==============================================================================

print(f"ðŸ“Š Creating visualizations...")

# Figure 1: Group comparison
fig, axes = plt.subplots(1, 3, figsize=(24, 7))

# Smart scaling
vmax = max(np.percentile(avg_control, 99), np.percentile(avg_epilepsy, 99))

# 1. Control Average
sns.heatmap(avg_control, ax=axes[0], cmap='viridis', square=True, 
            vmin=0, vmax=vmax, xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
            cbar_kws={'label': 'Connectivity Strength'})
axes[0].set_title(f'Control Group Average\n{BAND_TO_ANALYZE}\nn={len(control_matrices)} patients',
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('Target Channel', fontsize=12)
axes[0].set_ylabel('Source Channel', fontsize=12)

# 2. Epilepsy Average
sns.heatmap(avg_epilepsy, ax=axes[1], cmap='viridis', square=True,
            vmin=0, vmax=vmax, xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
            cbar_kws={'label': 'Connectivity Strength'})
axes[1].set_title(f'Epilepsy Group Average\n{BAND_TO_ANALYZE}\nn={len(epilepsy_matrices)} patients',
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Target Channel', fontsize=12)
axes[1].set_ylabel('Source Channel', fontsize=12)

# 3. Difference (Epilepsy - Control)
max_diff = np.max(np.abs(diff))
sns.heatmap(diff, ax=axes[2], cmap='coolwarm', square=True, center=0,
            vmin=-max_diff, vmax=max_diff, xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
            cbar_kws={'label': 'Difference'})
axes[2].set_title(f'Difference (Epilepsy - Control)\nRed = Higher in Epilepsy',
                 fontsize=14, fontweight='bold')
axes[2].set_xlabel('Target Channel', fontsize=12)
axes[2].set_ylabel('Source Channel', fontsize=12)

plt.suptitle(f'Group-Level Connectivity Analysis\nFixed order p=13, Diagonal=0',
            fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'group_comparison_{BAND_TO_ANALYZE}.png',
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Figure 2: Statistical significance
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

# 1. P-values (log scale)
p_values_masked = p_values.copy()
p_values_masked[p_values_masked == 0] = 1
p_values_log = -np.log10(p_values_masked)

sns.heatmap(p_values_log, ax=axes[0], cmap='hot', square=True,
            xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
            cbar_kws={'label': '-log10(p-value)'})
axes[0].set_title(f'Statistical Significance\n{n_significant} connections p < {alpha_corrected:.6f}',
                 fontsize=14, fontweight='bold')
axes[0].set_xlabel('Target Channel', fontsize=12)
axes[0].set_ylabel('Source Channel', fontsize=12)

threshold_line = -np.log10(alpha_corrected)
axes[0].text(0.02, 0.98, f'Bonferroni threshold: -log10(p) = {threshold_line:.2f}',
            transform=axes[0].transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# 2. Effect sizes
sns.heatmap(cohens_d, ax=axes[1], cmap='coolwarm', square=True, center=0,
            vmin=-1, vmax=1, xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,
            cbar_kws={'label': "Cohen's d"})
axes[1].set_title(f"Effect Sizes (Epilepsy - Control)\n|d| > 0.5 = medium, |d| > 0.8 = large",
                 fontsize=14, fontweight='bold')
axes[1].set_xlabel('Target Channel', fontsize=12)
axes[1].set_ylabel('Source Channel', fontsize=12)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / f'group_statistics_{BAND_TO_ANALYZE}.png',
            dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# ==============================================================================
# SUMMARY REPORT
# ==============================================================================

print(f"\n{'='*80}")
print(f"GROUP-LEVEL ANALYSIS SUMMARY")
print(f"{'='*80}")
print(f"Band: {BAND_TO_ANALYZE}")
print()
print(f"ðŸ“Š Sample Sizes:")
print(f"  Control patients: {len(control_matrices)}")
print(f"  Epilepsy patients: {len(epilepsy_matrices)}")
print(f"  Total control epochs: {sum(control_epoch_counts)}")
print(f"  Total epilepsy epochs: {sum(epilepsy_epoch_counts)}")
print()
print(f"ðŸ“ˆ Connectivity Statistics:")
print(f"  Control mean: {np.mean(avg_control[avg_control > 0]):.4f} Â± {np.mean(se_control[se_control > 0]):.4f}")
print(f"  Epilepsy mean: {np.mean(avg_epilepsy[avg_epilepsy > 0]):.4f} Â± {np.mean(se_epilepsy[se_epilepsy > 0]):.4f}")
print(f"  Max increase: {np.max(diff):.4f}")
print(f"  Max decrease: {np.min(diff):.4f}")
print()
print(f"ðŸ“Š Statistical Tests:")
print(f"  Total comparisons: {n_comparisons}")
print(f"  Bonferroni Î±: {alpha_corrected:.6f}")
print(f"  Significant connections: {n_significant} ({n_significant/n_comparisons*100:.2f}%)")
print(f"  Mean |Cohen's d|: {np.mean(np.abs(cohens_d[cohens_d != 0])):.3f}")
print()

# Find top connections
diff_flat = diff.flatten()
indices_sorted = np.argsort(np.abs(diff_flat))[::-1]

print(f"ðŸ” TOP 10 MOST DIFFERENT CONNECTIONS:")
count = 0
for idx in indices_sorted:
    i = idx // n_channels
    j = idx % n_channels
    if i != j and count < 10:
        sig_marker = '***' if p_values[i, j] < alpha_corrected else ''
        print(f"  {CHANNEL_NAMES[i]:>4} â†’ {CHANNEL_NAMES[j]:<4}: "
              f"Î” = {diff[i,j]:+.4f}, "
              f"d = {cohens_d[i,j]:+.3f}, "
              f"p = {p_values[i,j]:.6f} {sig_marker}")
        count += 1
print()

print(f"ðŸ’¾ Saved plots to:")
print(f"  - {OUTPUT_DIR / f'group_comparison_{BAND_TO_ANALYZE}.png'}")
print(f"  - {OUTPUT_DIR / f'group_statistics_{BAND_TO_ANALYZE}.png'}")
print(f"{'='*80}\n")

# ==============================================================================
# SAVE RESULTS
# ==============================================================================

# Save numerical results
results = {
    'avg_control': avg_control,
    'avg_epilepsy': avg_epilepsy,
    'difference': diff,
    'p_values': p_values,
    'cohens_d': cohens_d,
    'significant_mask': significant_mask,
    'n_control': len(control_matrices),
    'n_epilepsy': len(epilepsy_matrices),
    'alpha_corrected': alpha_corrected
}

np.savez(OUTPUT_DIR / f'group_results_{BAND_TO_ANALYZE}.npz', **results)
print(f"ðŸ’¾ Saved numerical results to:")
print(f"  - {OUTPUT_DIR / f'group_results_{BAND_TO_ANALYZE}.npz'}")
print()