"""
CONNECTIVITY RESULTS ANALYZER
Generates statistics and plots for professor meeting
"""
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# Load all results
results_dir = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\connectivity\january_bic_final_with_bands')
all_files = list(results_dir.rglob('*_graphs.npz'))

print(f"Found {len(all_files)} result files")

# Collect statistics
all_orders = []
all_bics = []
max_pdc_values = []
max_dtf_values = []
mean_pdc_values = []
mean_dtf_values = []
n_epochs_per_file = []
epilepsy_orders = []
control_orders = []

for f in all_files:
    try:
        data = np.load(f)
        
        # Basic stats
        all_orders.extend(data['orders'])
        all_bics.extend(data['bic_values'])
        n_epochs_per_file.append(len(data['orders']))
        
        # Quality metrics
        #max_pdc_values.extend(data['pdc'].max(axis=(1,2)))
        #max_dtf_values.extend(data['dtf'].max(axis=(1,2)))
        #mean_pdc_values.extend(data['pdc'].mean(axis=(1,2)))
        #mean_dtf_values.extend(data['dtf'].mean(axis=(1,2)))

        # To this (assuming you want to check the broadband/integrated result):
        max_pdc_values.extend(data['pdc_integrated'].max(axis=(1,2)))
        max_dtf_values.extend(data['dtf_integrated'].max(axis=(1,2)))
        mean_pdc_values.extend(data['pdc_integrated'].mean(axis=(1,2)))
        mean_dtf_values.extend(data['dtf_integrated'].mean(axis=(1,2)))
        
        # Group by label
        if '00_epilepsy' in str(f):
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

# Print comprehensive statistics
print("\n" + "="*70)
print("CONNECTIVITY ANALYSIS RESULTS SUMMARY")
print("="*70)

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

print("\n⚠️  QUALITY INDICATORS:")
print(f"  PDC statistics:")
print(f"    Mean max value:  {max_pdc_values.mean():.3f}")
print(f"    Epochs > 0.8:    {np.sum(max_pdc_values > 0.8):,} ({100*np.sum(max_pdc_values > 0.8)/len(max_pdc_values):.1f}%)")
print(f"    Epochs > 0.9:    {np.sum(max_pdc_values > 0.9):,} ({100*np.sum(max_pdc_values > 0.9)/len(max_pdc_values):.1f}%)")

print(f"\n  DTF statistics:")
print(f"    Mean max value:  {max_dtf_values.mean():.3f}")
print(f"    Epochs > 0.8:    {np.sum(max_dtf_values > 0.8):,} ({100*np.sum(max_dtf_values > 0.8)/len(max_dtf_values):.1f}%)")
print(f"    Epochs > 0.9:    {np.sum(max_dtf_values > 0.9):,} ({100*np.sum(max_dtf_values > 0.9)/len(max_dtf_values):.1f}%)")

print("\n" + "="*70)

# Create comprehensive visualization
fig = plt.figure(figsize=(16, 10))

# Layout: 2 rows, 3 columns
# Row 1: Order distribution, PDC max values, DTF max values
# Row 2: Group comparison, PDC vs DTF scatter, Mean connectivity values

# Plot 1: Order Distribution
ax1 = plt.subplot(2, 3, 1)
ax1.hist(all_orders, bins=range(2, 17), edgecolor='black', color='steelblue', alpha=0.7)
ax1.axvline(all_orders.mean(), color='red', linestyle='--', linewidth=2, label=f'Mean={all_orders.mean():.1f}')
ax1.axvline(np.median(all_orders), color='orange', linestyle='--', linewidth=2, label=f'Median={np.median(all_orders):.0f}')
ax1.set_xlabel('Model Order (p)', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title(f'BIC-Selected Order Distribution\n(n={len(all_orders):,} epochs)', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3, axis='y')

# Plot 2: Max PDC Distribution
ax2 = plt.subplot(2, 3, 2)
ax2.hist(max_pdc_values, bins=50, edgecolor='black', color='purple', alpha=0.7)
ax2.axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold 0.8')
ax2.axvline(max_pdc_values.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean={max_pdc_values.mean():.2f}')
ax2.set_xlabel('Max PDC Value', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title(f'PDC Maximum Values\n{np.sum(max_pdc_values > 0.8)} epochs (>{0.8*100:.0f}%) above 0.8', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3, axis='y')

# Plot 3: Max DTF Distribution
ax3 = plt.subplot(2, 3, 3)
ax3.hist(max_dtf_values, bins=50, edgecolor='black', color='teal', alpha=0.7)
ax3.axvline(0.8, color='red', linestyle='--', linewidth=2, label='Threshold 0.8')
ax3.axvline(max_dtf_values.mean(), color='orange', linestyle='--', linewidth=2, label=f'Mean={max_dtf_values.mean():.2f}')
ax3.set_xlabel('Max DTF Value', fontsize=11)
ax3.set_ylabel('Frequency', fontsize=11)
ax3.set_title(f'DTF Maximum Values\n{np.sum(max_dtf_values > 0.8)} epochs (>{0.8*100:.0f}%) above 0.8', fontsize=12, fontweight='bold')
ax3.legend()
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Group Comparison
ax4 = plt.subplot(2, 3, 4)
ax4.hist([epilepsy_orders, control_orders], bins=range(2, 17), 
         label=['Epilepsy', 'Control'], alpha=0.6, color=['red', 'blue'], edgecolor='black')
ax4.set_xlabel('Model Order (p)', fontsize=11)
ax4.set_ylabel('Frequency', fontsize=11)
ax4.set_title(f'Order Distribution by Group\nEpilepsy: {epilepsy_orders.mean():.2f}, Control: {control_orders.mean():.2f}', 
              fontsize=12, fontweight='bold')
ax4.legend()
ax4.grid(True, alpha=0.3, axis='y')

# Plot 5: PDC vs DTF Scatter
ax5 = plt.subplot(2, 3, 5)
scatter = ax5.scatter(max_dtf_values, max_pdc_values, alpha=0.3, s=1, c='steelblue')
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
ax6.set_title(f'Mean Connectivity Values\nPDC: {np.mean(mean_pdc_values):.3f}, DTF: {np.mean(mean_dtf_values):.3f}', 
              fontsize=12, fontweight='bold')
ax6.legend()
ax6.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('connectivity_comprehensive_analysis.png', dpi=300, bbox_inches='tight')
print("\n✅ Saved: connectivity_comprehensive_analysis.png")

# Create order comparison figure for thesis
fig2, axes = plt.subplots(1, 2, figsize=(12, 5))

# Histogram comparison
axes[0].hist([epilepsy_orders, control_orders], bins=range(2, 17), 
            label=[f'Epilepsy (n={len(epilepsy_orders):,})', f'Control (n={len(control_orders):,})'], 
            alpha=0.6, color=['red', 'blue'], edgecolor='black')
axes[0].set_xlabel('Model Order (p)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('BIC-Selected Order Distribution by Group', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=11)
axes[0].grid(True, alpha=0.3, axis='y')

# Box plot comparison
axes[1].boxplot([epilepsy_orders, control_orders], labels=['Epilepsy', 'Control'],
               patch_artist=True, 
               boxprops=dict(facecolor='lightblue'),
               medianprops=dict(color='red', linewidth=2))
axes[1].set_ylabel('Model Order (p)', fontsize=12)
axes[1].set_title(f'Order Statistics\nEpilepsy: {epilepsy_orders.mean():.2f}±{epilepsy_orders.std():.2f}\nControl: {control_orders.mean():.2f}±{control_orders.std():.2f}', 
                 fontsize=13, fontweight='bold')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('order_comparison_groups.png', dpi=300, bbox_inches='tight')
print("✅ Saved: order_comparison_groups.png")

plt.show()

print("\n🎯 READY FOR PROFESSOR MEETING!")
print("   - Run this script to get statistics")
print("   - Check diagnostic_plots/ folder for example matrices")
print("   - Review the analysis figures created")
