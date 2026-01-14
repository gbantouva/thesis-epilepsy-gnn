"""
CONNECTIVITY RESULTS ANALYZER - CORRECTED VERSION WITH GROUP COMPARISON
========================================================================
Generates statistics and plots for professor meeting
FIXED: Channel names (T1/T2 instead of A1/A2)
FIXED: Multi-band support
NEW: Figure 4 - Epilepsy vs Control per-band comparison
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# ============================================================================
# CONFIGURATION
# ============================================================================

# CORRECT channel names (matches preprocessing!)
CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                 'T5', 'P3', 'Pz', 'P4', 'T6', 
                 'O1', 'Oz', 'O2']

# ============================================================================
# LOAD DATA
# ============================================================================

# UPDATE THIS PATH TO YOUR RESULTS DIRECTORY
results_dir = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_bic_final_with_bands')
all_files = list(results_dir.rglob('*_graphs.npz'))

# Create output directory
output_dir = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\connectivity\results\connectivity_plots')
output_dir.mkdir(parents=True, exist_ok=True)

print(f"Found {len(all_files)} result files")

# ============================================================================
# COLLECT STATISTICS
# ============================================================================

all_orders = []
all_bics = []
max_pdc_values = []
max_dtf_values = []
mean_pdc_values = []
mean_dtf_values = []
n_epochs_per_file = []
epilepsy_orders = []
control_orders = []

# Per-band statistics (overall)
band_names = ['integrated', 'delta', 'theta', 'alpha', 'beta', 'gamma1', 'gamma2']
band_stats = {band: {'max_pdc': [], 'max_dtf': [], 'mean_pdc': [], 'mean_dtf': []} 
              for band in band_names}

# Per-band statistics separated by group (NEW!)
epilepsy_band_stats = {band: {'max_pdc': [], 'max_dtf': [], 'mean_pdc': [], 'mean_dtf': []} 
                       for band in band_names}
control_band_stats = {band: {'max_pdc': [], 'max_dtf': [], 'mean_pdc': [], 'mean_dtf': []} 
                      for band in band_names}

for f in all_files:
    try:
        data = np.load(f)
        
        # Determine group
        is_epilepsy = '00_epilepsy' in str(f)
        
        # Basic stats
        all_orders.extend(data['orders'])
        all_bics.extend(data['bic_values'])
        n_epochs_per_file.append(len(data['orders']))
        
        # Integrated band (primary analysis)
        max_pdc_values.extend(data['pdc_integrated'].max(axis=(1,2)))
        max_dtf_values.extend(data['dtf_integrated'].max(axis=(1,2)))
        mean_pdc_values.extend(data['pdc_integrated'].mean(axis=(1,2)))
        mean_dtf_values.extend(data['dtf_integrated'].mean(axis=(1,2)))
        
        # Per-band statistics
        for band in band_names:
            max_pdc = data[f'pdc_{band}'].max(axis=(1,2))
            max_dtf = data[f'dtf_{band}'].max(axis=(1,2))
            mean_pdc = data[f'pdc_{band}'].mean(axis=(1,2))
            mean_dtf = data[f'dtf_{band}'].mean(axis=(1,2))
            
            # Overall
            band_stats[band]['max_pdc'].extend(max_pdc)
            band_stats[band]['max_dtf'].extend(max_dtf)
            band_stats[band]['mean_pdc'].extend(mean_pdc)
            band_stats[band]['mean_dtf'].extend(mean_dtf)
            
            # Separated by group (NEW!)
            if is_epilepsy:
                epilepsy_band_stats[band]['max_pdc'].extend(max_pdc)
                epilepsy_band_stats[band]['max_dtf'].extend(max_dtf)
                epilepsy_band_stats[band]['mean_pdc'].extend(mean_pdc)
                epilepsy_band_stats[band]['mean_dtf'].extend(mean_dtf)
            else:
                control_band_stats[band]['max_pdc'].extend(max_pdc)
                control_band_stats[band]['max_dtf'].extend(max_dtf)
                control_band_stats[band]['mean_pdc'].extend(mean_pdc)
                control_band_stats[band]['mean_dtf'].extend(mean_dtf)
        
        # Group by label (for orders)
        if is_epilepsy:
            epilepsy_orders.extend(data['orders'])
        else:
            control_orders.extend(data['orders'])
            
    except Exception as e:
        print(f"Error loading {f.name}: {e}")

# Convert to arrays
all_orders = np.array(all_orders)
epilepsy_orders = np.array(epilepsy_orders)
control_orders = np.array(control_orders)
max_pdc_values = np.array(max_pdc_values)
max_dtf_values = np.array(max_dtf_values)

# ============================================================================
# PRINT STATISTICS
# ============================================================================

print("\n" + "="*80)
print("CONNECTIVITY ANALYSIS RESULTS SUMMARY")
print("="*80)

print("\n📁 FILE STATISTICS:")
print(f"  Successful files:     {len(all_files)}")
print(f"  Epochs per file:      {np.mean(n_epochs_per_file):.1f} ± {np.std(n_epochs_per_file):.1f}")
print(f"  Total epochs:         {len(all_orders):,}")
print(f"    Epilepsy epochs:    {len(epilepsy_orders):,}")
print(f"    Control epochs:     {len(control_orders):,}")

print("\n📊 MODEL ORDER STATISTICS (BIC-selected):")
print(f"  Overall:")
print(f"    Mean:   {all_orders.mean():.2f}")
print(f"    Median: {np.median(all_orders):.0f}")
print(f"    Std:    {all_orders.std():.2f}")
print(f"    Range:  {all_orders.min()}-{all_orders.max()}")

print(f"\n  By Group:")
print(f"    Epilepsy mean:  {epilepsy_orders.mean():.2f}")
print(f"    Control mean:   {control_orders.mean():.2f}")
print(f"    Difference:     {abs(epilepsy_orders.mean() - control_orders.mean()):.2f}")

print("\n  Top 5 Most Common Orders:")
unique_orders, counts = np.unique(all_orders, return_counts=True)
for i in range(min(5, len(unique_orders))):
    idx = np.argsort(-counts)[i]
    order = unique_orders[idx]
    count = counts[idx]
    percentage = 100 * count / len(all_orders)
    print(f"    p={order:2d}: {count:6,} epochs ({percentage:5.1f}%)")

print("\n⚠️  QUALITY INDICATORS (Integrated Band):")
print(f"  PDC statistics:")
print(f"    Mean max value:  {max_pdc_values.mean():.3f}")
print(f"    Epochs > 0.8:    {np.sum(max_pdc_values > 0.8):,} ({100*np.sum(max_pdc_values > 0.8)/len(max_pdc_values):.1f}%)")
print(f"    Epochs > 0.9:    {np.sum(max_pdc_values > 0.9):,} ({100*np.sum(max_pdc_values > 0.9)/len(max_pdc_values):.1f}%)")

print(f"\n  DTF statistics:")
print(f"    Mean max value:  {max_dtf_values.mean():.3f}")
print(f"    Epochs > 0.8:    {np.sum(max_dtf_values > 0.8):,} ({100*np.sum(max_dtf_values > 0.8)/len(max_dtf_values):.1f}%)")
print(f"    Epochs > 0.9:    {np.sum(max_dtf_values > 0.9):,} ({100*np.sum(max_dtf_values > 0.9)/len(max_dtf_values):.1f}%)")

print("\n🎵 PER-BAND STATISTICS:")
print(f"{'Band':<12} {'Freq (Hz)':<15} {'Mean Max PDC':<15} {'Mean Max DTF':<15}")
print("-" * 60)
band_ranges = {
    'integrated': '0.5-80',
    'delta': '0.5-4',
    'theta': '4-8',
    'alpha': '8-15',
    'beta': '15-30',
    'gamma1': '30-55',
    'gamma2': '65-80'
}
for band in band_names:
    mean_max_pdc = np.mean(band_stats[band]['max_pdc'])
    mean_max_dtf = np.mean(band_stats[band]['max_dtf'])
    print(f"{band:<12} {band_ranges[band]:<15} {mean_max_pdc:<15.3f} {mean_max_dtf:<15.3f}")

print("\n" + "="*80)

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

# Figure 1: Comprehensive Analysis (Original 2x3 grid)
fig = plt.figure(figsize=(16, 10))

# Plot 1: Order Distribution
ax1 = plt.subplot(2, 3, 1)
ax1.hist(all_orders, bins=range(2, 21), edgecolor='black', color='steelblue', alpha=0.7)
ax1.axvline(all_orders.mean(), color='red', linestyle='--', linewidth=2, 
            label=f'Mean={all_orders.mean():.1f}')
ax1.axvline(np.median(all_orders), color='orange', linestyle='--', linewidth=2, 
            label=f'Median={np.median(all_orders):.0f}')
ax1.set_xlabel('Model Order (p)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title(f'BIC-Selected Order Distribution\n(n={len(all_orders):,} epochs)', 
             fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Max PDC Distribution
ax2 = plt.subplot(2, 3, 2)
ax2.hist(max_pdc_values, bins=50, edgecolor='black', color='purple', alpha=0.7)
ax2.axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold 0.8')
ax2.axvline(max_pdc_values.mean(), color='orange', linestyle='--', linewidth=2, 
            label=f'Mean={max_pdc_values.mean():.2f}')
ax2.set_xlabel('Max PDC Value', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title(f'PDC Maximum Values (Integrated)\n{np.sum(max_pdc_values > 0.8):,} epochs above 0.8', 
             fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Max DTF Distribution
ax3 = plt.subplot(2, 3, 3)
ax3.hist(max_dtf_values, bins=50, edgecolor='black', color='teal', alpha=0.7)
ax3.axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold 0.8')
ax3.axvline(max_dtf_values.mean(), color='orange', linestyle='--', linewidth=2, 
            label=f'Mean={max_dtf_values.mean():.2f}')
ax3.set_xlabel('Max DTF Value', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title(f'DTF Maximum Values (Integrated)\n{np.sum(max_dtf_values > 0.8):,} epochs above 0.8', 
             fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Group Comparison
ax4 = plt.subplot(2, 3, 4)
ax4.hist([epilepsy_orders, control_orders], bins=range(2, 21), 
         label=['Epilepsy', 'Control'], alpha=0.6, color=['red', 'blue'], edgecolor='black')
ax4.set_xlabel('Model Order (p)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title(f'Order Distribution by Group\nEpilepsy: {epilepsy_orders.mean():.2f}, Control: {control_orders.mean():.2f}', 
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: PDC vs DTF Scatter
ax5 = plt.subplot(2, 3, 5)
ax5.scatter(max_dtf_values, max_pdc_values, alpha=0.3, s=1, c='steelblue')
ax5.plot([0, 1], [0, 1], 'r--', linewidth=1, label='y=x')
ax5.axhline(0.8, color='orange', linestyle='--', alpha=0.5, label='Threshold')
ax5.axvline(0.8, color='orange', linestyle='--', alpha=0.5)
ax5.set_xlabel('Max DTF Value', fontsize=11)
ax5.set_ylabel('Max PDC Value', fontsize=11)
ax5.set_title('PDC vs DTF Maximum Values\nCorrelation Check', fontsize=12, fontweight='bold')
ax5.legend()
ax5.grid(True, alpha=0.3)
ax5.set_xlim([0, 1])
ax5.set_ylim([0, 1])

# Plot 6: Mean Connectivity Values
ax6 = plt.subplot(2, 3, 6)
ax6.hist(mean_pdc_values, bins=50, alpha=0.6, label='PDC', color='purple', edgecolor='black')
ax6.hist(mean_dtf_values, bins=50, alpha=0.6, label='DTF', color='teal', edgecolor='black')
ax6.set_xlabel('Mean Connectivity Value', fontsize=11)
ax6.set_ylabel('Frequency', fontsize=11)
ax6.set_title(f'Mean Connectivity Values (Integrated)\nPDC: {np.mean(mean_pdc_values):.3f}, DTF: {np.mean(mean_dtf_values):.3f}', 
              fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'connectivity_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'connectivity_comprehensive_analysis.png'}")

# ============================================================================
# Figure 2: Order Comparison for Thesis
# ============================================================================

fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram comparison
axes[0].hist([epilepsy_orders, control_orders], bins=range(2, 21), 
            label=[f'Epilepsy (n={len(epilepsy_orders):,})', f'Control (n={len(control_orders):,})'], 
            alpha=0.6, color=['red', 'blue'], edgecolor='black')
axes[0].set_xlabel('Model Order (p)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('BIC-Selected Order Distribution by Group', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Box plot comparison
bp = axes[1].boxplot([epilepsy_orders, control_orders], tick_labels=['Epilepsy', 'Control'],
                      patch_artist=True, 
                      boxprops=dict(facecolor='lightblue'),
                      medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Model Order (p)', fontsize=12)
axes[1].set_title(f'Order Statistics\nEpilepsy: {epilepsy_orders.mean():.2f}±{epilepsy_orders.std():.2f}\nControl: {control_orders.mean():.2f}±{control_orders.std():.2f}', 
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'order_comparison_groups.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'order_comparison_groups.png'}")

# ============================================================================
# Figure 3: Per-Band Comparison (Overall)
# ============================================================================

fig3, axes = plt.subplots(2, 1, figsize=(14, 10))

# Convert band stats to arrays for plotting
band_labels = [band.capitalize() for band in band_names]
max_pdc_means = [np.mean(band_stats[band]['max_pdc']) for band in band_names]
max_dtf_means = [np.mean(band_stats[band]['max_dtf']) for band in band_names]
mean_pdc_means = [np.mean(band_stats[band]['mean_pdc']) for band in band_names]
mean_dtf_means = [np.mean(band_stats[band]['mean_dtf']) for band in band_names]

x = np.arange(len(band_names))
width = 0.35

# Plot maximum values
axes[0].bar(x - width/2, max_pdc_means, width, label='PDC', color='purple', alpha=0.7)
axes[0].bar(x + width/2, max_dtf_means, width, label='DTF', color='teal', alpha=0.7)
axes[0].set_ylabel('Mean Maximum Value', fontsize=12)
axes[0].set_title('Maximum Connectivity Values per Frequency Band', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(band_labels, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Plot mean values
axes[1].bar(x - width/2, mean_pdc_means, width, label='PDC', color='purple', alpha=0.7)
axes[1].bar(x + width/2, mean_dtf_means, width, label='DTF', color='teal', alpha=0.7)
axes[1].set_ylabel('Mean Value', fontsize=12)
axes[1].set_title('Average Connectivity Values per Frequency Band', fontsize=14, fontweight='bold')
axes[1].set_xticks(x)
axes[1].set_xticklabels(band_labels, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(output_dir / 'per_band_connectivity_comparison.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'per_band_connectivity_comparison.png'}")

# ============================================================================
# Figure 4: EPILEPSY vs CONTROL PER-BAND COMPARISON (NEW!)
# ============================================================================

fig4, axes = plt.subplots(2, 2, figsize=(16, 12))

# Prepare data for plotting
epilepsy_max_pdc = [np.mean(epilepsy_band_stats[band]['max_pdc']) for band in band_names]
control_max_pdc = [np.mean(control_band_stats[band]['max_pdc']) for band in band_names]

epilepsy_max_dtf = [np.mean(epilepsy_band_stats[band]['max_dtf']) for band in band_names]
control_max_dtf = [np.mean(control_band_stats[band]['max_dtf']) for band in band_names]

epilepsy_mean_pdc = [np.mean(epilepsy_band_stats[band]['mean_pdc']) for band in band_names]
control_mean_pdc = [np.mean(control_band_stats[band]['mean_pdc']) for band in band_names]

epilepsy_mean_dtf = [np.mean(epilepsy_band_stats[band]['mean_dtf']) for band in band_names]
control_mean_dtf = [np.mean(control_band_stats[band]['mean_dtf']) for band in band_names]

x = np.arange(len(band_names))
width = 0.35

# Panel 1: Max PDC
axes[0, 0].bar(x - width/2, epilepsy_max_pdc, width, label='Epilepsy', color='red', alpha=0.7, edgecolor='black')
axes[0, 0].bar(x + width/2, control_max_pdc, width, label='Control', color='blue', alpha=0.7, edgecolor='black')
axes[0, 0].set_ylabel('Mean Maximum PDC', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Maximum PDC Values: Epilepsy vs Control', fontsize=13, fontweight='bold')
axes[0, 0].set_xticks(x)
axes[0, 0].set_xticklabels(band_labels, rotation=45, ha='right')
axes[0, 0].legend(fontsize=11)
axes[0, 0].grid(True, alpha=0.3, axis='y')
axes[0, 0].set_ylim([0.7, 0.9])

# Add difference annotations
for i, band in enumerate(band_names):
    diff = epilepsy_max_pdc[i] - control_max_pdc[i]
    y_pos = max(epilepsy_max_pdc[i], control_max_pdc[i]) + 0.005
    color = 'red' if diff > 0 else 'blue'
    axes[0, 0].text(i, y_pos, f'{diff:+.3f}', ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

# Panel 2: Max DTF
axes[0, 1].bar(x - width/2, epilepsy_max_dtf, width, label='Epilepsy', color='red', alpha=0.7, edgecolor='black')
axes[0, 1].bar(x + width/2, control_max_dtf, width, label='Control', color='blue', alpha=0.7, edgecolor='black')
axes[0, 1].set_ylabel('Mean Maximum DTF', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Maximum DTF Values: Epilepsy vs Control', fontsize=13, fontweight='bold')
axes[0, 1].set_xticks(x)
axes[0, 1].set_xticklabels(band_labels, rotation=45, ha='right')
axes[0, 1].legend(fontsize=11)
axes[0, 1].grid(True, alpha=0.3, axis='y')
axes[0, 1].set_ylim([0.55, 0.75])

# Add difference annotations
for i, band in enumerate(band_names):
    diff = epilepsy_max_dtf[i] - control_max_dtf[i]
    y_pos = max(epilepsy_max_dtf[i], control_max_dtf[i]) + 0.005
    color = 'red' if diff > 0 else 'blue'
    axes[0, 1].text(i, y_pos, f'{diff:+.3f}', ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

# Panel 3: Mean PDC
axes[1, 0].bar(x - width/2, epilepsy_mean_pdc, width, label='Epilepsy', color='red', alpha=0.7, edgecolor='black')
axes[1, 0].bar(x + width/2, control_mean_pdc, width, label='Control', color='blue', alpha=0.7, edgecolor='black')
axes[1, 0].set_ylabel('Mean PDC Value', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Average PDC Values: Epilepsy vs Control', fontsize=13, fontweight='bold')
axes[1, 0].set_xticks(x)
axes[1, 0].set_xticklabels(band_labels, rotation=45, ha='right')
axes[1, 0].legend(fontsize=11)
axes[1, 0].grid(True, alpha=0.3, axis='y')
axes[1, 0].set_ylim([0.14, 0.18])

# Add difference annotations
for i, band in enumerate(band_names):
    diff = epilepsy_mean_pdc[i] - control_mean_pdc[i]
    y_pos = max(epilepsy_mean_pdc[i], control_mean_pdc[i]) + 0.001
    color = 'red' if diff > 0 else 'blue'
    axes[1, 0].text(i, y_pos, f'{diff:+.4f}', ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

# Panel 4: Mean DTF
axes[1, 1].bar(x - width/2, epilepsy_mean_dtf, width, label='Epilepsy', color='red', alpha=0.7, edgecolor='black')
axes[1, 1].bar(x + width/2, control_mean_dtf, width, label='Control', color='blue', alpha=0.7, edgecolor='black')
axes[1, 1].set_ylabel('Mean DTF Value', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Average DTF Values: Epilepsy vs Control', fontsize=13, fontweight='bold')
axes[1, 1].set_xticks(x)
axes[1, 1].set_xticklabels(band_labels, rotation=45, ha='right')
axes[1, 1].legend(fontsize=11)
axes[1, 1].grid(True, alpha=0.3, axis='y')
axes[1, 1].set_ylim([0.14, 0.18])

# Add difference annotations
for i, band in enumerate(band_names):
    diff = epilepsy_mean_dtf[i] - control_mean_dtf[i]
    y_pos = max(epilepsy_mean_dtf[i], control_mean_dtf[i]) + 0.001
    color = 'red' if diff > 0 else 'blue'
    axes[1, 1].text(i, y_pos, f'{diff:+.4f}', ha='center', va='bottom', fontsize=8, color=color, fontweight='bold')

plt.suptitle('Epilepsy vs Control: Per-Band Connectivity Comparison', 
             fontsize=16, fontweight='bold', y=0.995)
plt.tight_layout(rect=[0, 0, 1, 0.99])
plt.savefig(output_dir / 'epilepsy_vs_control_per_band.png', dpi=300, bbox_inches='tight')
print(f"✅ Saved: {output_dir / 'epilepsy_vs_control_per_band.png'}")

# ============================================================================
# PRINT PER-BAND GROUP COMPARISON STATISTICS
# ============================================================================

print("\n" + "="*80)
print("EPILEPSY vs CONTROL: PER-BAND COMPARISON")
print("="*80)
print(f"\n{'Band':<12} {'Measure':<10} {'Epilepsy':<12} {'Control':<12} {'Difference':<12} {'% Diff':<10}")
print("-" * 80)

for band in band_names:
    # Max PDC
    epi_max_pdc = np.mean(epilepsy_band_stats[band]['max_pdc'])
    con_max_pdc = np.mean(control_band_stats[band]['max_pdc'])
    diff_max_pdc = epi_max_pdc - con_max_pdc
    pct_diff_max_pdc = 100 * diff_max_pdc / con_max_pdc
    print(f"{band:<12} {'Max PDC':<10} {epi_max_pdc:<12.4f} {con_max_pdc:<12.4f} {diff_max_pdc:<+12.4f} {pct_diff_max_pdc:<+10.2f}%")
    
    # Max DTF
    epi_max_dtf = np.mean(epilepsy_band_stats[band]['max_dtf'])
    con_max_dtf = np.mean(control_band_stats[band]['max_dtf'])
    diff_max_dtf = epi_max_dtf - con_max_dtf
    pct_diff_max_dtf = 100 * diff_max_dtf / con_max_dtf
    print(f"{'':<12} {'Max DTF':<10} {epi_max_dtf:<12.4f} {con_max_dtf:<12.4f} {diff_max_dtf:<+12.4f} {pct_diff_max_dtf:<+10.2f}%")
    print()

print("="*80)

plt.show()

print("\n🎯 READY FOR PROFESSOR MEETING!")
print("   Generated 4 figures:")
print("   1. connectivity_comprehensive_analysis.png - Overall statistics")
print("   2. order_comparison_groups.png - Epilepsy vs Control")
print("   3. per_band_connectivity_comparison.png - Multi-band analysis (overall)")
print("   4. epilepsy_vs_control_per_band.png - Multi-band analysis (group comparison) ⭐ NEW!")