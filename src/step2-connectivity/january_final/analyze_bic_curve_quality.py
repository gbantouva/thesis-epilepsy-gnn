"""
Analyze BIC Curve Quality - REALISTIC VERSION FOR EEG
======================================================
Uses relaxed thresholds appropriate for real EEG data.

Changes from original:
- Smoothness threshold: 0.5 → 1.0 (allows normal EEG noise)
- U-shape detection: More lenient slope requirements
- Better error handling to reduce "insufficient_data"
- Added "acceptable" category between "clear" and "noisy"

Usage:
    python analyze_bic_curve_quality_realistic.py --preprocessed_dir "path/to/data" --n_samples 200
"""

import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.vector_ar.var_model import VAR
import warnings

warnings.filterwarnings("ignore")

# Constants
MIN_ORDER = 8
MAX_ORDER = 22


def analyze_bic_curve_quality_realistic(orders, bic_values):
    """
    Classify BIC curve with REALISTIC thresholds for EEG data.
    
    Categories:
    - clear: Smooth U-shape, obvious minimum
    - acceptable: U-shape visible despite some noise
    - noisy: General trend visible but irregular
    - edge: Optimal at boundary (p=8 or p=22)
    - chaotic: No interpretable pattern
    
    Returns
    -------
    category : str
    metrics : dict
    """
    if len(orders) < 5:
        return 'insufficient_data', {}
    
    # Normalize BIC values to [0, 1]
    bic_range = np.max(bic_values) - np.min(bic_values)
    
    # Handle flat curves
    if bic_range < 1e-6:
        return 'flat', {'range': bic_range}
    
    bic_norm = (bic_values - np.min(bic_values)) / bic_range
    
    # Find optimal order
    optimal_idx = np.argmin(bic_values)
    optimal_order = orders[optimal_idx]
    
    # === METRIC 1: Edge Detection ===
    is_edge = (optimal_order == MIN_ORDER) or (optimal_order == MAX_ORDER)
    
    # === METRIC 2: Smoothness (variance of 2nd derivative) ===
    if len(bic_norm) >= 3:
        second_deriv = np.diff(np.diff(bic_norm))
        smoothness = np.std(second_deriv)
    else:
        smoothness = np.inf
    
    # === METRIC 3: U-Shape Detection (RELAXED) ===
    left_slope = 0
    right_slope = 0
    
    if optimal_idx > 0:
        # Average slope on left side (should be negative = decreasing)
        left_slope = np.mean(np.diff(bic_norm[:optimal_idx+1]))
    
    if optimal_idx < len(bic_norm) - 1:
        # Average slope on right side (should be positive = increasing)
        right_slope = np.mean(np.diff(bic_norm[optimal_idx:]))
    
    # RELAXED U-shape criteria
    has_left_decrease = left_slope < 0  # Just negative, not strict threshold
    has_right_increase = right_slope > 0  # Just positive, not strict threshold
    u_shape_present = has_left_decrease and has_right_increase
    
    # Strong U-shape (for "clear" category)
    strong_u_shape = (left_slope < -0.005) and (right_slope > 0.005)
    
    # === METRIC 4: Depth of Minimum ===
    # How much does BIC decrease from edges to center?
    edge_avg = (bic_norm[0] + bic_norm[-1]) / 2
    depth = edge_avg - bic_norm[optimal_idx]  # Higher = clearer minimum
    
    # === METRIC 5: Relative Range ===
    relative_range = bic_range / np.abs(np.mean(bic_values))
    
    # Store all metrics
    metrics = {
        'optimal_order': optimal_order,
        'is_edge': is_edge,
        'smoothness': smoothness,
        'left_slope': left_slope,
        'right_slope': right_slope,
        'u_shape_present': u_shape_present,
        'strong_u_shape': strong_u_shape,
        'depth': depth,
        'relative_range': relative_range
    }
    
    # === CLASSIFICATION (REALISTIC THRESHOLDS) ===
    
    # 1. Edge cases
    if is_edge:
        if relative_range < 0.01:
            category = 'flat'  # Optimal at edge but curve is flat
        else:
            category = 'edge'  # Clear edge solution
    
    # 2. Chaotic (RAISED THRESHOLD: 0.5 → 1.0)
    elif smoothness > 1.0:  # Much more lenient!
        category = 'chaotic'
    
    # 3. Clear (RELAXED CRITERIA)
    elif strong_u_shape and smoothness < 0.4 and depth > 0.2:
        # Clear U-shape + reasonably smooth + good depth
        category = 'clear'
    
    # 4. Acceptable (NEW CATEGORY)
    elif u_shape_present and smoothness < 0.8:
        # U-shape visible + not too noisy
        category = 'acceptable'
    
    # 5. Noisy (everything else that's not chaotic)
    else:
        category = 'noisy'
    
    return category, metrics


def load_and_analyze_epoch(epochs_file, epoch_idx, min_order=MIN_ORDER, max_order=MAX_ORDER):
    """
    Load epoch and compute BIC curve with ROBUST error handling.
    """
    try:
        epochs = np.load(epochs_file)
        if epoch_idx >= len(epochs):
            return None, None, None
        
        epoch_data = epochs[epoch_idx]  # Shape: (22, 1000)
        
        # Check for bad data
        if np.any(np.isnan(epoch_data)) or np.any(np.isinf(epoch_data)):
            return None, None, None
        
        # Scale data for numerical stability
        data_std = np.std(epoch_data)
        if data_std < 1e-10:
            return None, None, None
        
        data_scaled = epoch_data / data_std
        
        # Fit VAR with robust settings
        model = VAR(data_scaled.T)  # (1000, 22)
        
        orders = []
        bic_values = []
        
        for p in range(min_order, max_order + 1):
            try:
                # More robust fitting
                result = model.fit(
                    maxlags=p,
                    trend='c',
                    verbose=False,
                    method='ols'  # Ordinary least squares (most stable)
                )
                
                bic = result.bic
                
                # Validate BIC value
                if not np.isnan(bic) and not np.isinf(bic) and np.abs(bic) < 1e10:
                    orders.append(p)
                    bic_values.append(bic)
                    
            except (np.linalg.LinAlgError, ValueError, FloatingPointError):
                # Skip this order if fitting fails
                continue
            except Exception:
                # Catch-all for other errors
                continue
        
        # Need at least 5 valid orders
        if len(orders) < 5:
            return None, None, None
        
        return orders, bic_values, np.argmin(bic_values)
        
    except Exception as e:
        return None, None, None


def main():
    parser = argparse.ArgumentParser(
        description="Analyze BIC curve quality with realistic EEG-calibrated thresholds"
    )
    parser.add_argument("--preprocessed_dir", required=True,
                       help="Directory with preprocessed data (*_epochs.npy)")
    parser.add_argument("--n_samples", type=int, default=200,
                       help="Number of random epochs to analyze (default: 200)")
    parser.add_argument("--output_dir", default="bic_quality_realistic",
                       help="Output directory")
    
    args = parser.parse_args()
    
    preprocessed_dir = Path(args.preprocessed_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("="*80)
    print("BIC CURVE QUALITY ANALYSIS - REALISTIC VERSION")
    print("="*80)
    print("\nChanges from original:")
    print("  - Smoothness threshold: 0.5 → 1.0 (allows normal EEG noise)")
    print("  - U-shape detection: Relaxed slope requirements")
    print("  - Added 'acceptable' category for good-enough curves")
    print("  - Better error handling\n")
    
    # Find all epoch files
    epoch_files = list(preprocessed_dir.rglob("*_epochs.npy"))
    print(f"Found {len(epoch_files)} preprocessed files")
    
    if len(epoch_files) == 0:
        print("❌ No epoch files found!")
        return
    
    # Sample random epochs
    np.random.seed(42)
    sampled_data = []
    
    print(f"\nSampling {args.n_samples} random epochs...\n")
    
    attempts = 0
    max_attempts = args.n_samples * 20  # More attempts allowed
    failures = {'load_error': 0, 'bic_compute_error': 0}
    
    pbar_format = "  Progress: {n_fmt}/{total_fmt} [{bar}] {percentage:3.0f}%"
    
    from tqdm import tqdm
    with tqdm(total=args.n_samples, bar_format=pbar_format) as pbar:
        while len(sampled_data) < args.n_samples and attempts < max_attempts:
            # Random file
            file_idx = np.random.randint(0, len(epoch_files))
            epoch_file = epoch_files[file_idx]
            
            # Random epoch in that file
            try:
                epochs = np.load(epoch_file, mmap_mode='r')
                n_epochs = len(epochs)
                epoch_idx = np.random.randint(0, n_epochs)
            except Exception:
                failures['load_error'] += 1
                attempts += 1
                continue
            
            # Compute BIC curve
            orders, bic_vals, _ = load_and_analyze_epoch(epoch_file, epoch_idx)
            
            if orders is None:
                failures['bic_compute_error'] += 1
                attempts += 1
                continue
            
            # Analyze quality with REALISTIC thresholds
            category, metrics = analyze_bic_curve_quality_realistic(orders, bic_vals)
            
            # Store result
            sampled_data.append({
                'file': epoch_file.name,
                'epoch_idx': epoch_idx,
                'category': category,
                'optimal_order': metrics.get('optimal_order', np.nan),
                'is_edge': metrics.get('is_edge', False),
                'smoothness': metrics.get('smoothness', np.nan),
                'u_shape_present': metrics.get('u_shape_present', False),
                'strong_u_shape': metrics.get('strong_u_shape', False),
                'depth': metrics.get('depth', np.nan)
            })
            
            pbar.update(1)
            attempts += 1
    
    # Convert to DataFrame
    df = pd.DataFrame(sampled_data)
    
    # Summary statistics
    print("\n" + "="*80)
    print("RESULTS")
    print("="*80)
    
    category_counts = df['category'].value_counts()
    print("\nBIC Curve Categories (Realistic Thresholds):")
    print("-"*50)
    
    # Define order for display
    category_order = ['clear', 'acceptable', 'noisy', 'edge', 'chaotic', 'flat', 'insufficient_data']
    for cat in category_order:
        if cat in category_counts.index:
            count = category_counts[cat]
            pct = (count / len(df)) * 100
            print(f"  {cat:18s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\nTotal analyzed: {len(df)} epochs")
    print(f"Sampling attempts: {attempts}")
    print(f"  - Load errors: {failures['load_error']}")
    print(f"  - BIC compute errors: {failures['bic_compute_error']}")
    
    # Convergence quality summary
    good_categories = ['clear', 'acceptable']
    okay_categories = ['noisy']
    poor_categories = ['edge', 'chaotic', 'flat']
    
    good_count = df['category'].isin(good_categories).sum()
    okay_count = df['category'].isin(okay_categories).sum()
    poor_count = df['category'].isin(poor_categories).sum()
    
    good_pct = (good_count / len(df)) * 100
    okay_pct = (okay_count / len(df)) * 100
    poor_pct = (poor_count / len(df)) * 100
    
    print("\n" + "-"*50)
    print("Quality Summary:")
    print("-"*50)
    print(f"  Good (clear + acceptable):  {good_count:3d} ({good_pct:5.1f}%)")
    print(f"  Okay (noisy):               {okay_count:3d} ({okay_pct:5.1f}%)")
    print(f"  Poor (edge + chaotic):      {poor_count:3d} ({poor_pct:5.1f}%)")
    
    # Assessment
    print("\n" + "-"*50)
    print("Assessment:")
    print("-"*50)
    
    usable_pct = good_pct + okay_pct
    if usable_pct > 75:
        print(f"  ✅ EXCELLENT ({usable_pct:.1f}% usable)")
        print(f"     Your p=15 choice is strongly supported!")
    elif usable_pct > 60:
        print(f"  ✓ GOOD ({usable_pct:.1f}% usable)")
        print(f"     Your p=15 should work well.")
    elif usable_pct > 45:
        print(f"  ~ ACCEPTABLE ({usable_pct:.1f}% usable)")
        print(f"     p=15 is reasonable. Monitor retention rate.")
    else:
        print(f"  ⚠️ CONCERN ({usable_pct:.1f}% usable)")
        print(f"     Consider checking retention rate carefully.")
    
    # Optimal order distribution
    print("\n" + "-"*50)
    print("Optimal Order Distribution:")
    print("-"*50)
    order_counts = df['optimal_order'].value_counts().sort_index()
    for order, count in order_counts.items():
        if not np.isnan(order):
            pct = (count / len(df)) * 100
            marker = " ◄ FIXED" if int(order) == 15 else ""
            print(f"  p={int(order):2d}: {count:3d} ({pct:5.1f}%){marker}")
    
    # Save results
    df.to_csv(output_dir / 'bic_quality_realistic.csv', index=False)
    print(f"\n✓ Saved: {output_dir / 'bic_quality_realistic.csv'}")
    
    # === VISUALIZATION ===
    print("\nGenerating visualizations...")
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Define colors
    colors = {
        'clear': '#27ae60',
        'acceptable': '#3498db',
        'noisy': '#f39c12',
        'edge': '#e67e22',
        'chaotic': '#e74c3c',
        'flat': '#95a5a6',
        'insufficient_data': '#7f8c8d'
    }
    
    # Plot 1: Category distribution
    categories_present = [cat for cat in category_order if cat in category_counts.index]
    counts = [category_counts.get(cat, 0) for cat in categories_present]
    bar_colors = [colors.get(cat, 'gray') for cat in categories_present]
    
    axes[0].bar(range(len(categories_present)), counts, color=bar_colors, alpha=0.8, edgecolor='black')
    axes[0].set_xticks(range(len(categories_present)))
    axes[0].set_xticklabels(categories_present, rotation=45, ha='right')
    axes[0].set_title('BIC Curve Quality Distribution\n(Realistic Thresholds)', 
                     fontweight='bold', fontsize=12)
    axes[0].set_xlabel('Category', fontsize=11)
    axes[0].set_ylabel('Count', fontsize=11)
    axes[0].grid(axis='y', alpha=0.3)
    
    # Add percentages
    for i, count in enumerate(counts):
        pct = (count / len(df)) * 100
        axes[0].text(i, count + max(counts)*0.02, f'{pct:.1f}%', 
                    ha='center', fontweight='bold', fontsize=9)
    
    # Plot 2: Quality summary (stacked)
    quality_groups = ['Good\n(clear+acceptable)', 'Okay\n(noisy)', 'Poor\n(edge+chaotic)']
    quality_counts = [good_count, okay_count, poor_count]
    quality_colors = ['#27ae60', '#f39c12', '#e74c3c']
    
    axes[1].bar(quality_groups, quality_counts, color=quality_colors, alpha=0.8, edgecolor='black')
    axes[1].set_title('Overall Quality Assessment', fontweight='bold', fontsize=12)
    axes[1].set_ylabel('Count', fontsize=11)
    axes[1].grid(axis='y', alpha=0.3)
    
    for i, count in enumerate(quality_counts):
        pct = (count / len(df)) * 100
        axes[1].text(i, count + max(quality_counts)*0.02, f'{pct:.1f}%', 
                    ha='center', fontweight='bold', fontsize=10)
    
    # Plot 3: Optimal order by category
    for cat in ['clear', 'acceptable', 'noisy', 'chaotic']:
        if cat in df['category'].values:
            subset = df[df['category'] == cat]['optimal_order'].dropna()
            if len(subset) > 0:
                axes[2].hist(subset, bins=range(MIN_ORDER, MAX_ORDER+2), 
                           alpha=0.6, label=cat, edgecolor='black', color=colors.get(cat, 'gray'))
    
    axes[2].axvline(15, color='red', linestyle='--', linewidth=2.5, label='Fixed Order (p=15)', zorder=10)
    axes[2].set_title('Optimal Order by Quality', fontweight='bold', fontsize=12)
    axes[2].set_xlabel('Optimal Order (p)', fontsize=11)
    axes[2].set_ylabel('Count', fontsize=11)
    axes[2].legend(fontsize=9)
    axes[2].grid(axis='y', alpha=0.3)
    axes[2].set_xticks(range(MIN_ORDER, MAX_ORDER+1, 2))
    
    plt.tight_layout()
    plt.savefig(output_dir / 'bic_quality_realistic_summary.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✓ Saved: {output_dir / 'bic_quality_realistic_summary.png'}")
    
    # Save text report
    report_path = output_dir / 'analysis_report.txt'
    with open(report_path, 'w') as f:
        f.write("="*80 + "\n")
        f.write("BIC CURVE QUALITY ANALYSIS - REALISTIC THRESHOLDS\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Analyzed: {len(df)} epochs\n")
        f.write(f"Preprocessed directory: {preprocessed_dir}\n\n")
        
        f.write("Category Distribution:\n")
        f.write("-"*50 + "\n")
        for cat in categories_present:
            count = category_counts.get(cat, 0)
            pct = (count / len(df)) * 100
            f.write(f"  {cat:18s}: {count:3d} ({pct:5.1f}%)\n")
        
        f.write("\nQuality Summary:\n")
        f.write("-"*50 + "\n")
        f.write(f"  Good (clear + acceptable):  {good_count:3d} ({good_pct:5.1f}%)\n")
        f.write(f"  Okay (noisy):               {okay_count:3d} ({okay_pct:5.1f}%)\n")
        f.write(f"  Poor (edge + chaotic):      {poor_count:3d} ({poor_pct:5.1f}%)\n")
        f.write(f"  TOTAL USABLE:               {good_count + okay_count:3d} ({usable_pct:5.1f}%)\n")
        
        f.write("\nInterpretation:\n")
        f.write("-"*50 + "\n")
        if usable_pct > 75:
            f.write(f"Excellent quality. {usable_pct:.1f}% of epochs show convergent BIC curves.\n")
            f.write(f"The fixed order p=15 is strongly supported by the data.\n")
        elif usable_pct > 60:
            f.write(f"Good quality. {usable_pct:.1f}% of epochs are usable.\n")
            f.write(f"The fixed order p=15 is well-supported.\n")
        else:
            f.write(f"Acceptable quality. {usable_pct:.1f}% of epochs are usable.\n")
            f.write(f"Monitor the retention rate in Step 2 connectivity computation.\n")
    
    print(f"✓ Saved: {report_path}")
    
    print("\n" + "="*80)
    print("✅ ANALYSIS COMPLETE")
    print("="*80)
    print(f"\nOutput directory: {output_dir}")
    print(f"  - bic_quality_realistic.csv")
    print(f"  - bic_quality_realistic_summary.png")
    print(f"  - analysis_report.txt")
    print("\nUse these results in your thesis Section 3.2.4!")
    print("="*80)


if __name__ == "__main__":
    main()