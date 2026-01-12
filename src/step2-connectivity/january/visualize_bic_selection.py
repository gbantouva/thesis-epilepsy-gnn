"""

BIC SELECTION VISUALIZATION

===========================

Shows how BIC selected optimal order for first 5 epochs of a patient.

Perfect for explaining methodology to professor!

"""



import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

from pathlib import Path

from statsmodels.tsa.vector_ar.var_model import VAR

from scipy import linalg

import warnings

warnings.filterwarnings("ignore")



# ==============================================================================

# CONFIGURATION

# ==============================================================================



# Choose a patient file

#PATIENT_FILE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001_epochs.npy')

PATIENT_FILE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_pp_balanced\00_epilepsy\aaaaaiek\s003_2010\02_tcp_le\aaaaaiek_s003_t000_epochs.npy')







# How many epochs to analyze

N_EPOCHS = 6



# BIC parameters (must match your batch processing!)

MIN_ORDER = 2

MAX_ORDER = 15

FS = 250.0

FREQ_RANGE = (0.5, 80)

NFFT = 512



# Channel names (excluding A1/A2 since you'll fix this)

CHANNEL_NAMES = ['Fp1', 'Fp2', 'F7', 'F3', 'Fz', 'F4', 'F8',

                 'T3', 'C3', 'Cz', 'C4', 'T4',

                 'T5', 'P3', 'Pz', 'P4', 'T6',

                 'O1', 'Oz', 'O2']



# ==============================================================================

# CONNECTIVITY COMPUTATION (Same as your batch script)

# ==============================================================================



def compute_dtf_pdc_from_var(coefs, fs=250.0, nfft=512):

    """Compute DTF and PDC from VAR coefficients."""

    p, K, _ = coefs.shape

    n_freqs = nfft // 2 + 1

    freqs = np.linspace(0, fs/2, n_freqs)

   

    A_f = np.zeros((n_freqs, K, K), dtype=complex)

    H_f = np.zeros((n_freqs, K, K), dtype=complex)

    I = np.eye(K)

   

    for f_idx, f in enumerate(freqs):

        A_sum = np.zeros((K, K), dtype=complex)

        for k in range(p):

            phase = np.exp(-1j * 2 * np.pi * f * (k + 1) / fs)

            A_sum += coefs[k] * phase

       

        A_f[f_idx] = I - A_sum

       

        try:

            H_f[f_idx] = linalg.inv(A_f[f_idx])

        except linalg.LinAlgError:

            H_f[f_idx] = linalg.pinv(A_f[f_idx])

   

    # PDC

    pdc = np.zeros((K, K, n_freqs))

    for f_idx in range(n_freqs):

        Af = A_f[f_idx]

        col_norms = np.sqrt(np.sum(np.abs(Af)**2, axis=0))

        col_norms[col_norms == 0] = 1e-10

        pdc[:, :, f_idx] = np.abs(Af) / col_norms[None, :]

   

    # DTF

    dtf = np.zeros((K, K, n_freqs))

    for f_idx in range(n_freqs):

        Hf = H_f[f_idx]

        row_norms = np.sqrt(np.sum(np.abs(Hf)**2, axis=1))

        row_norms[row_norms == 0] = 1e-10

        dtf[:, :, f_idx] = np.abs(Hf) / row_norms[:, None]

   

    return dtf, pdc, freqs





def analyze_single_epoch_with_bic(data, min_order=2, max_order=15):

    """

    Analyze one epoch and return BIC curve + connectivity.

    Returns dict with BIC values for all orders tested.

    """

    # Scale data (IMPORTANT: matches your batch processing)

    data_std = np.std(data)

    if data_std < 1e-10:

        return None

    data_scaled = data / data_std

   

    try:

        model = VAR(data_scaled.T)

       

        # Compute BIC for all orders

        bic_results = {}

        for p in range(min_order, max_order + 1):

            try:

                result = model.fit(maxlags=p, trend='c', verbose=False)

                bic_results[p] = result.bic

            except:

                pass

       

        if not bic_results:

            return None

       

        # Find best order

        best_order = min(bic_results, key=bic_results.get)

        best_bic = bic_results[best_order]

       

        # Fit final model with best order

        final_model = model.fit(maxlags=best_order, trend='c', verbose=False)

       

        # Compute connectivity

        dtf_spectrum, pdc_spectrum, freqs = compute_dtf_pdc_from_var(

            final_model.coefs, FS, NFFT

        )

       

        # Integrate over frequency band

        idx_band = np.where((freqs >= FREQ_RANGE[0]) & (freqs <= FREQ_RANGE[1]))[0]

        dtf_integrated = np.mean(dtf_spectrum[:, :, idx_band], axis=2)

        pdc_integrated = np.mean(pdc_spectrum[:, :, idx_band], axis=2)

       

        return {

            'bic_all': bic_results,

            'best_order': best_order,

            'best_bic': best_bic,

            'dtf': dtf_integrated,

            'pdc': pdc_integrated

        }

       

    except Exception as e:

        print(f"Error: {e}")

        return None





# ==============================================================================

# VISUALIZATION

# ==============================================================================



def create_bic_demonstration(epochs_data, n_epochs=5):

    """

    Create comprehensive figure showing BIC selection for multiple epochs.

    """

    # Analyze epochs

    results = []

    for i in range(min(n_epochs, len(epochs_data))):

        print(f"Analyzing epoch {i}...", end=" ")

        result = analyze_single_epoch_with_bic(

            epochs_data[i][:20, :],  # Exclude A1/A2 for clean demo

            MIN_ORDER,

            MAX_ORDER

        )

        if result:

            results.append((i, result))

            print(f"✓ p={result['best_order']}")

        else:

            print("✗ Failed")

   

    if len(results) == 0:

        print("❌ No valid epochs to visualize!")

        return

   

    # Create figure with subplots for each epoch

    n_valid = len(results)

    fig = plt.figure(figsize=(20, 4*n_valid))

   

    for plot_idx, (epoch_idx, result) in enumerate(results):

        # Extract data

        bic_all = result['bic_all']

        best_order = result['best_order']

        best_bic = result['best_bic']

        dtf = result['dtf']

        pdc = result['pdc']

       

        orders = sorted(bic_all.keys())

        bic_values = [bic_all[p] for p in orders]

       

        # Create 3 subplots per epoch: BIC curve, DTF, PDC

        gs = plt.GridSpec(n_valid, 3, figure=fig)

       

        # BIC Curve

        ax_bic = fig.add_subplot(gs[plot_idx, 0])

        ax_bic.plot(orders, bic_values, 'o-', linewidth=2, markersize=8,

                   color='steelblue', label='BIC values')

        ax_bic.plot(best_order, best_bic, 'r*', markersize=20,

                   label=f'Selected: p={best_order}', zorder=10)

        ax_bic.axvline(best_order, color='red', linestyle='--', alpha=0.3)

        ax_bic.set_xlabel('Model Order (p)', fontsize=11)

        ax_bic.set_ylabel('BIC Value', fontsize=11)

        ax_bic.set_title(f'Epoch {epoch_idx}: BIC Selection\nSelected p={best_order} (BIC={best_bic:.2f})',

                        fontsize=12, fontweight='bold')

        ax_bic.legend(fontsize=10)

        ax_bic.grid(True, alpha=0.3)

       

        # Annotate U-shape

        min_bic_order = orders[np.argmin(bic_values)]

        ax_bic.annotate('U-shaped curve\n(overfitting penalty)',

                       xy=(MAX_ORDER, bic_values[-1]),

                       xytext=(MAX_ORDER-2, bic_values[-1] + (max(bic_values)-min(bic_values))*0.1),

                       fontsize=9, ha='center',

                       arrowprops=dict(arrowstyle='->', color='gray', lw=1.5))

       

        # DTF Matrix

        ax_dtf = fig.add_subplot(gs[plot_idx, 1])

        global_max = max(dtf.max(), pdc.max())

        sns.heatmap(dtf, ax=ax_dtf, cmap='viridis', square=True,

                   xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,

                   vmin=0, vmax=global_max,

                   cbar_kws={'label': 'Strength'})

        ax_dtf.set_title(f'DTF Matrix (p={best_order})', fontsize=12, fontweight='bold')

        ax_dtf.tick_params(axis='x', rotation=45, labelsize=8)

        ax_dtf.tick_params(axis='y', rotation=0, labelsize=8)

       

        # PDC Matrix

        ax_pdc = fig.add_subplot(gs[plot_idx, 2])

        sns.heatmap(pdc, ax=ax_pdc, cmap='viridis', square=True,

                   xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,

                   vmin=0, vmax=global_max,

                   cbar_kws={'label': 'Strength'})

        ax_pdc.set_title(f'PDC Matrix (p={best_order})', fontsize=12, fontweight='bold')

        ax_pdc.tick_params(axis='x', rotation=45, labelsize=8)

        ax_pdc.tick_params(axis='y', rotation=0, labelsize=8)

   

    plt.tight_layout()

    output_file = f'bic_selection_demonstration_{n_valid}_epochs.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    print(f"\n✅ Saved: {output_file}")

    plt.show()





# ==============================================================================

# ALTERNATIVE: Single-Epoch Detailed View

# ==============================================================================



def create_single_epoch_detailed(epoch_data, epoch_idx=0):

    """

    Create detailed analysis for ONE epoch - perfect for presentation slide.

    """

    result = analyze_single_epoch_with_bic(

        epoch_data[:20, :],  # Exclude A1/A2

        MIN_ORDER,

        MAX_ORDER

    )

   

    if not result:

        print("❌ Failed to analyze epoch!")

        return

   

    bic_all = result['bic_all']

    best_order = result['best_order']

    best_bic = result['best_bic']

    dtf = result['dtf']

    pdc = result['pdc']

   

    orders = sorted(bic_all.keys())

    bic_values = [bic_all[p] for p in orders]

   

    # Create figure with 2x2 layout

    fig = plt.figure(figsize=(16, 12))

   

    # Large BIC curve (top left and center)

    ax1 = plt.subplot2grid((2, 3), (0, 0), colspan=2)

    ax1.plot(orders, bic_values, 'o-', linewidth=3, markersize=10,

            color='steelblue', label='BIC Score', zorder=5)

    ax1.plot(best_order, best_bic, 'r*', markersize=30,

            label=f'Optimal Order: p={best_order}', zorder=10)

    ax1.axvline(best_order, color='red', linestyle='--', linewidth=2, alpha=0.5)

   

    # Shade regions

    ax1.axvspan(MIN_ORDER, best_order-1, alpha=0.1, color='orange', label='Underfitting region')

    ax1.axvspan(best_order+1, MAX_ORDER, alpha=0.1, color='red', label='Overfitting region')

   

    ax1.set_xlabel('Model Order (p)', fontsize=14, fontweight='bold')

    ax1.set_ylabel('BIC Score (lower is better)', fontsize=14, fontweight='bold')

    ax1.set_title(f'BIC-Based Model Order Selection for Epoch {epoch_idx}\nOptimal Order: p={best_order}',

                 fontsize=16, fontweight='bold')

    ax1.legend(fontsize=12, loc='upper right')

    ax1.grid(True, alpha=0.3, linewidth=1.5)

    ax1.tick_params(labelsize=12)

   

    # Annotations

    ax1.annotate('Minimum BIC\n(Best Trade-off)',

                xy=(best_order, best_bic),

                xytext=(best_order, best_bic - (max(bic_values)-min(bic_values))*0.15),

                fontsize=11, ha='center', color='red', fontweight='bold',

                arrowprops=dict(arrowstyle='->', color='red', lw=2))

   

    # Statistics box

    stats_text = f"Statistics:\n"

    stats_text += f"Orders tested: {MIN_ORDER}-{MAX_ORDER}\n"

    stats_text += f"Selected order: {best_order}\n"

    stats_text += f"BIC range: {min(bic_values):.1f} to {max(bic_values):.1f}\n"

    stats_text += f"Parameters: {best_order * 20 * 20} coefficients"

   

    ax1.text(0.02, 0.98, stats_text, transform=ax1.transAxes,

            fontsize=10, verticalalignment='top',

            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

   

    # BIC Values Table (top right)

    ax2 = plt.subplot2grid((2, 3), (0, 2))

    ax2.axis('off')

    table_data = [[f"p={p}", f"{bic_all[p]:.2f}"] for p in orders]

    table = ax2.table(cellText=table_data, colLabels=['Order', 'BIC'],

                     cellLoc='center', loc='center',

                     colWidths=[0.3, 0.5])

    table.auto_set_font_size(False)

    table.set_fontsize(10)

    table.scale(1, 2)

   

    # Highlight selected row

    for i, p in enumerate(orders):

        if p == best_order:

            table[(i+1, 0)].set_facecolor('#ff9999')

            table[(i+1, 1)].set_facecolor('#ff9999')

   

    ax2.set_title('BIC Values\nfor All Orders', fontsize=12, fontweight='bold')

   

    # DTF Matrix (bottom left)

    ax3 = plt.subplot2grid((2, 3), (1, 0), colspan=1)

    global_max = max(dtf.max(), pdc.max())

    sns.heatmap(dtf, ax=ax3, cmap='viridis', square=True,

               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,

               vmin=0, vmax=global_max,

               cbar_kws={'label': 'Strength'})

    ax3.set_title(f'DTF Connectivity (p={best_order})', fontsize=12, fontweight='bold')

    ax3.tick_params(axis='x', rotation=45, labelsize=9)

    ax3.tick_params(axis='y', rotation=0, labelsize=9)

   

    # PDC Matrix (bottom middle)

    ax4 = plt.subplot2grid((2, 3), (1, 1), colspan=1)

    sns.heatmap(pdc, ax=ax4, cmap='viridis', square=True,

               xticklabels=CHANNEL_NAMES, yticklabels=CHANNEL_NAMES,

               vmin=0, vmax=global_max,

               cbar_kws={'label': 'Strength'})

    ax4.set_title(f'PDC Connectivity (p={best_order})', fontsize=12, fontweight='bold')

    ax4.tick_params(axis='x', rotation=45, labelsize=9)

    ax4.tick_params(axis='y', rotation=0, labelsize=9)

   

    # Value distributions (bottom right)

    ax5 = plt.subplot2grid((2, 3), (1, 2))

    ax5.hist(dtf.flatten(), bins=30, alpha=0.6, label='DTF', color='purple', edgecolor='black')

    ax5.hist(pdc.flatten(), bins=30, alpha=0.6, label='PDC', color='teal', edgecolor='black')

    ax5.set_xlabel('Connectivity Value', fontsize=11)

    ax5.set_ylabel('Frequency', fontsize=11)

    ax5.set_title(f'Value Distributions\nDTF mean: {dtf.mean():.3f}\nPDC mean: {pdc.mean():.3f}',

                 fontsize=12, fontweight='bold')

    ax5.legend(fontsize=10)

    ax5.grid(True, alpha=0.3, axis='y')

   

    plt.tight_layout()

    output_file = f'bic_detailed_2_epoch_{epoch_idx}.png'

    plt.savefig(output_file, dpi=300, bbox_inches='tight')

    print(f"✅ Saved: {output_file}")

    plt.show()





# ==============================================================================

# MAIN

# ==============================================================================



if __name__ == "__main__":

    print("="*70)

    print("BIC SELECTION VISUALIZATION FOR PROFESSOR MEETING")

    print("="*70)

   

    # Load data

    if not PATIENT_FILE.exists():

        print(f"❌ File not found: {PATIENT_FILE}")

        print("\nPlease update PATIENT_FILE path at top of script.")

        exit(1)

   

    print(f"\nLoading: {PATIENT_FILE.name}")

    epochs = np.load(PATIENT_FILE)

    print(f"✓ Loaded {len(epochs)} epochs, shape: {epochs.shape}")

   

    # Create visualizations

    print("\n" + "="*70)

    print("OPTION 1: Multiple Epochs Overview (5 epochs)")

    print("="*70)

    create_bic_demonstration(epochs, n_epochs=N_EPOCHS)

   

    print("\n" + "="*70)

    print("OPTION 2: Single Epoch Detailed Analysis")

    print("="*70)

    create_single_epoch_detailed(epochs[0], epoch_idx=0)

   

    print("\n" + "="*70)

    print("✅ DONE! You now have:")

    print("   1. bic_selection_demonstration_X_epochs.png (overview)")

    print("   2. bic_detailed_epoch_0.png (detailed single epoch)")

    print("\nBoth are perfect for showing your professor!")

    print("="*70)