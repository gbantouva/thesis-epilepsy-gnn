"""
BIC ORDER SELECTION VISUALIZATION
==================================
Visualizes how BIC selects the optimal MVAR order for connectivity analysis.
Shows BIC curves and selected orders across epochs.
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# ============================================================================
# CONFIGURATION
# ============================================================================

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',
                 'T1', 'T3', 'C3', 'Cz', 'C4', 'T4', 'T2',
                 'T5', 'P3', 'Pz', 'P4', 'T6', 
                 'O1', 'Oz', 'O2']

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_order_distribution(all_files, output_dir):
    """
    Plot 1: Overall order distribution across all patients/epochs.
    """
    all_orders = []
    epilepsy_orders = []
    control_orders = []
    
    print("Loading order statistics...")
    for f in tqdm(all_files):
        try:
            data = np.load(f)
            orders = data['orders']
            all_orders.extend(orders)
            
            if '00_epilepsy' in str(f):
                epilepsy_orders.extend(orders)
            else:
                control_orders.extend(orders)
        except Exception as e:
            print(f"Error loading {f.name}: {e}")
    
    all_orders = np.array(all_orders)
    epilepsy_orders = np.array(epilepsy_orders)
    control_orders = np.array(control_orders)
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Overall histogram
    axes[0, 0].hist(all_orders, bins=range(2, 21), edgecolor='black', 
                   color='steelblue', alpha=0.7)
    axes[0, 0].axvline(all_orders.mean(), color='red', linestyle='--', 
                      linewidth=2, label=f'Mean={all_orders.mean():.2f}')
    axes[0, 0].axvline(np.median(all_orders), color='orange', linestyle='--', 
                      linewidth=2, label=f'Median={np.median(all_orders):.0f}')
    axes[0, 0].set_xlabel('Model Order (p)', fontsize=12)
    axes[0, 0].set_ylabel('Frequency', fontsize=12)
    axes[0, 0].set_title(f'Overall BIC-Selected Order Distribution\n(n={len(all_orders):,} epochs)', 
                        fontsize=13, fontweight='bold')
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Group comparison
    axes[0, 1].hist([epilepsy_orders, control_orders], bins=range(2, 21),
                   label=[f'Epilepsy (n={len(epilepsy_orders):,})', 
                         f'Control (n={len(control_orders):,})'],
                   alpha=0.6, color=['red', 'blue'], edgecolor='black')
    axes[0, 1].set_xlabel('Model Order (p)', fontsize=12)
    axes[0, 1].set_ylabel('Frequency', fontsize=12)
    axes[0, 1].set_title('Order Distribution by Group', fontsize=13, fontweight='bold')
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Box plot comparison
    bp = axes[1, 0].boxplot([epilepsy_orders, control_orders], 
                           labels=['Epilepsy', 'Control'],
                           patch_artist=True,
                           boxprops=dict(facecolor='lightblue', alpha=0.7),
                           medianprops=dict(color='red', linewidth=2))
    axes[1, 0].set_ylabel('Model Order (p)', fontsize=12)
    axes[1, 0].set_title(f'Order Statistics by Group\nEpilepsy: {epilepsy_orders.mean():.2f}±{epilepsy_orders.std():.2f}\nControl: {control_orders.mean():.2f}±{control_orders.std():.2f}',
                        fontsize=13, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Percentage breakdown
    unique_orders, counts = np.unique(all_orders, return_counts=True)
    percentages = 100 * counts / len(all_orders)
    
    # Show top 10 most common orders
    top_n = min(10, len(unique_orders))
    top_indices = np.argsort(-counts)[:top_n]
    top_orders = unique_orders[top_indices]
    top_percentages = percentages[top_indices]
    
    axes[1, 1].barh(range(top_n), top_percentages, color='steelblue', alpha=0.7)
    axes[1, 1].set_yticks(range(top_n))
    axes[1, 1].set_yticklabels([f'p={int(o)}' for o in top_orders])
    axes[1, 1].set_xlabel('Percentage (%)', fontsize=12)
    axes[1, 1].set_title(f'Top {top_n} Most Common Orders', fontsize=13, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='x')
    
    # Add percentage labels
    for i, (order, pct) in enumerate(zip(top_orders, top_percentages)):
        axes[1, 1].text(pct + 0.5, i, f'{pct:.1f}%', 
                       va='center', fontsize=9)
    
    plt.tight_layout()
    save_path = output_dir / 'bic_order_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")
    
    # Print statistics
    print("\n" + "="*70)
    print("ORDER SELECTION STATISTICS")
    print("="*70)
    print(f"Total epochs analyzed: {len(all_orders):,}")
    print(f"\nOverall Statistics:")
    print(f"  Mean:   {all_orders.mean():.2f}")
    print(f"  Median: {np.median(all_orders):.0f}")
    print(f"  Std:    {all_orders.std():.2f}")
    print(f"  Range:  {all_orders.min()}-{all_orders.max()}")
    print(f"\nGroup Comparison:")
    print(f"  Epilepsy mean:  {epilepsy_orders.mean():.2f} ± {epilepsy_orders.std():.2f}")
    print(f"  Control mean:   {control_orders.mean():.2f} ± {control_orders.std():.2f}")
    print(f"  Difference:     {abs(epilepsy_orders.mean() - control_orders.mean()):.2f}")
    print(f"\nTop 5 Most Common Orders:")
    for i in range(min(5, len(unique_orders))):
        idx = top_indices[i]
        print(f"  p={int(unique_orders[idx]):2d}: {int(counts[idx]):6,} epochs ({percentages[idx]:5.1f}%)")
    print("="*70)


def plot_bic_values_distribution(all_files, output_dir):
    """
    Plot 2: Distribution of BIC values (shows model fit quality).
    """
    all_bic = []
    epilepsy_bic = []
    control_bic = []
    
    print("\nLoading BIC values...")
    for f in tqdm(all_files):
        try:
            data = np.load(f)
            bic_values = data['bic_values']
            all_bic.extend(bic_values)
            
            if '00_epilepsy' in str(f):
                epilepsy_bic.extend(bic_values)
            else:
                control_bic.extend(bic_values)
        except Exception as e:
            continue
    
    all_bic = np.array(all_bic)
    epilepsy_bic = np.array(epilepsy_bic)
    control_bic = np.array(control_bic)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Overall BIC distribution
    axes[0].hist(all_bic, bins=50, edgecolor='black', color='purple', alpha=0.7)
    axes[0].axvline(all_bic.mean(), color='red', linestyle='--', 
                   linewidth=2, label=f'Mean={all_bic.mean():.1f}')
    axes[0].set_xlabel('BIC Value', fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'BIC Value Distribution\n(Lower is better)', 
                     fontsize=13, fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Group comparison
    axes[1].hist([epilepsy_bic, control_bic], bins=50,
                label=[f'Epilepsy', f'Control'],
                alpha=0.6, color=['red', 'blue'], edgecolor='black')
    axes[1].set_xlabel('BIC Value', fontsize=12)
    axes[1].set_ylabel('Frequency', fontsize=12)
    axes[1].set_title(f'BIC Distribution by Group\nEpilepsy: {epilepsy_bic.mean():.1f}, Control: {control_bic.mean():.1f}',
                     fontsize=13, fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = output_dir / 'bic_value_distribution.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


def plot_order_vs_bic(all_files, output_dir):
    """
    Plot 3: Relationship between selected order and BIC value.
    """
    orders = []
    bic_values = []
    
    print("\nLoading order-BIC relationships...")
    for f in tqdm(all_files):
        try:
            data = np.load(f)
            orders.extend(data['orders'])
            bic_values.extend(data['bic_values'])
        except:
            continue
    
    orders = np.array(orders)
    bic_values = np.array(bic_values)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Scatter plot
    axes[0].scatter(orders, bic_values, alpha=0.1, s=1, c='steelblue')
    axes[0].set_xlabel('Selected Order (p)', fontsize=12)
    axes[0].set_ylabel('BIC Value', fontsize=12)
    axes[0].set_title('Order vs BIC Value\n(Each point = 1 epoch)', 
                     fontsize=13, fontweight='bold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot 2: Mean BIC per order
    unique_orders = np.unique(orders)
    mean_bic_per_order = []
    std_bic_per_order = []
    
    for order in unique_orders:
        mask = orders == order
        mean_bic_per_order.append(bic_values[mask].mean())
        std_bic_per_order.append(bic_values[mask].std())
    
    axes[1].errorbar(unique_orders, mean_bic_per_order, 
                    yerr=std_bic_per_order, fmt='o-', 
                    capsize=3, capthick=1, color='steelblue')
    axes[1].set_xlabel('Model Order (p)', fontsize=12)
    axes[1].set_ylabel('Mean BIC Value', fontsize=12)
    axes[1].set_title('Average BIC per Order\n(Error bars = Std Dev)', 
                     fontsize=13, fontweight='bold')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    save_path = output_dir / 'order_vs_bic_relationship.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {save_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Visualize BIC order selection statistics"
    )
    parser.add_argument("--input_dir", required=True, 
                       help="Directory containing *_graphs.npz files")
    parser.add_argument("--output_dir", required=True,
                       help="Where to save plots")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find all result files
    all_files = list(input_dir.rglob('*_graphs.npz'))
    
    if len(all_files) == 0:
        print(f"Error: No *_graphs.npz files found in {input_dir}")
        return
    
    print(f"Found {len(all_files)} result files")
    print("="*70)
    
    # Generate visualizations
    plot_order_distribution(all_files, output_dir)
    plot_bic_values_distribution(all_files, output_dir)
    plot_order_vs_bic(all_files, output_dir)
    
    print("\n" + "="*70)
    print("✅ BIC VISUALIZATION COMPLETE!")
    print("="*70)
    print("Generated plots:")
    print("  1. bic_order_distribution.png - How orders are distributed")
    print("  2. bic_value_distribution.png - BIC values across epochs")
    print("  3. order_vs_bic_relationship.png - Order selection quality")
    print(f"\nOutput directory: {output_dir}")
    print("="*70)


if __name__ == "__main__":
    main()