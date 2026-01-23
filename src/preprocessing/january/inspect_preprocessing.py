"""
Comprehensive EEG Preprocessing Inspection Script

Validates preprocessing pipeline by comparing raw EDF with preprocessed data.
Generates a multi-page PDF report with all quality checks.

Usage:
    python inspect_preprocessing.py --edf raw_file.edf --preprocessed_dir output_dir --pid patient_id --out report_dir

Example:
    python inspect_preprocessing.py \
        --edf data_raw/DATA/00_epilepsy/patient001.edf \
        --preprocessed_dir data_pp \
        --pid patient001 \
        --out inspection_reports
"""

import argparse
import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import matplotlib.gridspec as gridspec
import json
import pickle
import warnings
warnings.filterwarnings('ignore')

import mne
from scipy import signal
from scipy.stats import zscore

# ============================================================================
# CONFIGURATION
# ============================================================================

CORE_CHS = [
    "Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8",
    "T1", "T3", "C3", "Cz", "C4", "T4", "T2",
    "T5", "P3", "Pz", "P4", "T6", "O1", "Oz", "O2"
]

# Color scheme
COLORS = {
    'before': '#e74c3c',      # Red
    'after': '#27ae60',       # Green
    'interpolated': '#f39c12', # Orange
    'real': '#3498db',        # Blue
    'rejected': '#95a5a6',    # Gray
    'kept': '#2ecc71',        # Light green
}


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def clean_channel_names(raw):
    """Standardize channel names (copied from preprocess_core.py)."""
    import re
    mapping = {
        orig: re.sub(r'^(?:EEG\s*)', '', orig)
        .replace('-LE', '')
        .replace('-REF', '')
        .strip()
        for orig in raw.ch_names
    }
    raw.rename_channels(mapping)
    
    core_ch_map = {ch.lower(): ch for ch in CORE_CHS}
    case_mapping = {}
    for ch in raw.ch_names:
        if ch.lower() in core_ch_map:
            case_mapping[ch] = core_ch_map[ch.lower()]
    
    if case_mapping:
        raw.rename_channels(case_mapping)


def compute_psd(data, sfreq, fmin=0.5, fmax=100):
    """Compute power spectral density using Welch's method."""
    freqs, psd = signal.welch(data, fs=sfreq, nperseg=min(256, data.shape[-1]))
    mask = (freqs >= fmin) & (freqs <= fmax)
    return freqs[mask], psd[..., mask]


def add_title_page(pdf, pid, edf_path, preprocessed_dir):
    """Add a title page to the PDF report."""
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.7, 'EEG Preprocessing Inspection Report', 
             ha='center', va='center', fontsize=24, fontweight='bold')
    fig.text(0.5, 0.55, f'Patient ID: {pid}', 
             ha='center', va='center', fontsize=16)
    fig.text(0.5, 0.45, f'Raw file: {edf_path.name}', 
             ha='center', va='center', fontsize=12, color='gray')
    fig.text(0.5, 0.38, f'Preprocessed dir: {preprocessed_dir}', 
             ha='center', va='center', fontsize=12, color='gray')
    
    # Add checklist
    checklist = """
    Quality Checks Included:
    ─────────────────────────────────────────
    1. Raw vs Preprocessed Time Series
    2. Power Spectral Density (Before/After)
    3. Notch Filter Verification (60 Hz)
    4. Bandpass Filter Verification (0.5-80 Hz)
    5. Channel-wise Variance Analysis
    6. Amplitude Distribution Histograms
    7. Interpolation Quality Assessment
    8. Epoch Overview Grid
    9. Epoch Amplitude Statistics
    10. Correlation Matrix (Before/After)
    11. Summary Statistics Table
    """
    fig.text(0.5, 0.15, checklist, ha='center', va='center', fontsize=10, 
             family='monospace', linespacing=1.5)
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# INSPECTION FUNCTIONS
# ============================================================================

def inspect_time_series(pdf, raw_data, raw_sfreq, raw_ch_names,
                        preprocessed_epochs, prep_sfreq, present_mask):
    """
    Page 1: Compare raw vs preprocessed time series.
    Shows first 10 seconds of a few channels.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('1. Time Series Comparison (First 50 Seconds)', fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 1, height_ratios=[1, 1], hspace=0.3)
    
    # Select channels to display (mix of frontal, central, occipital)
    display_chs = ['Fp1', 'F3', 'C3', 'P3', 'O1', 'Fz', 'Cz', 'Pz']
    
    # === RAW DATA ===
    ax1 = fig.add_subplot(gs[0])
    ax1.set_title('RAW (Before Preprocessing)', fontsize=12, color=COLORS['before'])
    
    t_raw = np.arange(min(int(50 * raw_sfreq), raw_data.shape[1])) / raw_sfreq
    offset = 0
    yticks, ylabels = [], []
    
    for i, ch in enumerate(display_chs):
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            trace = raw_data[idx, :len(t_raw)] * 1e6  # Convert to µV
            trace_centered = trace - np.mean(trace)
            ax1.plot(t_raw, trace_centered + offset, color=COLORS['before'], linewidth=0.5, alpha=0.8)
            yticks.append(offset)
            ylabels.append(ch)
            offset -= 150  # Spacing between channels
    
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(ylabels)
    ax1.set_xlabel('Time (s)')
    ax1.set_ylabel('Channel')
    ax1.set_xlim([0, 50])
    ax1.grid(True, alpha=0.3)
    
    # === PREPROCESSED DATA ===
    ax2 = fig.add_subplot(gs[1])
    ax2.set_title('PREPROCESSED (After Pipeline)', fontsize=12, color=COLORS['after'])
    
    #Concatenate multiple epochs to show 50 seconds
    samples_needed = int(50 * prep_sfreq)
    n_samples_per_epoch = preprocessed_epochs.shape[2]
    n_epochs_needed = min(int(np.ceil(samples_needed / n_samples_per_epoch)), len(preprocessed_epochs))

    # Concatenate epochs: (n_epochs, 22, n_times) -> (22, total_times)
    prep_data = preprocessed_epochs[:n_epochs_needed].transpose(1, 0, 2).reshape(22, -1)
    prep_data = prep_data[:, :samples_needed]  # Trim to exactly 50 seconds

    t_prep = np.arange(prep_data.shape[1]) / prep_sfreq
    offset = 0
    yticks, ylabels = [], []
    
    for i, ch in enumerate(display_chs):
        if ch in CORE_CHS:
            idx = CORE_CHS.index(ch)
            trace = prep_data[idx, :] * 1e6  # Convert to µV
            trace_centered = trace - np.mean(trace)
            
            # Color based on interpolation status
            color = COLORS['interpolated'] if not present_mask[idx] else COLORS['after']
            label_suffix = ' (interp)' if not present_mask[idx] else ''
            
            ax2.plot(t_prep, trace_centered + offset, color=color, linewidth=0.5, alpha=0.8)
            yticks.append(offset)
            ylabels.append(ch + label_suffix)
            offset -= 100
    
    ax2.set_yticks(yticks)
    ax2.set_yticklabels(ylabels)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Channel')
    ax2.set_xlim([0, 50])
    ax2.grid(True, alpha=0.3)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['before'], label='Raw'),
        Patch(facecolor=COLORS['after'], label='Preprocessed (real)'),
        Patch(facecolor=COLORS['interpolated'], label='Preprocessed (interpolated)')
    ]
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.98, 0.98))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_psd_comparison(pdf, raw_data, raw_sfreq, raw_ch_names,
                           preprocessed_epochs, prep_sfreq, present_mask):
    """
    Page 2: Power Spectral Density comparison (full spectrum).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('2. Power Spectral Density Comparison', fontsize=14, fontweight='bold')
    
    # Find common channels
    common_chs = [ch for ch in CORE_CHS if ch in raw_ch_names]
    raw_indices = [raw_ch_names.index(ch) for ch in common_chs]
    prep_indices = [CORE_CHS.index(ch) for ch in common_chs]
    
    # Compute PSD for raw
    raw_subset = raw_data[raw_indices, :]
    freqs_raw, psd_raw = compute_psd(raw_subset, raw_sfreq, fmin=0.1, fmax=100)
    
    # Compute PSD for preprocessed (average across epochs)
    prep_subset = preprocessed_epochs[:, prep_indices, :]
    psd_prep_list = []
    for ep in prep_subset:
        f, p = compute_psd(ep, prep_sfreq, fmin=0.1, fmax=100)
        psd_prep_list.append(p)
    freqs_prep = f
    psd_prep = np.mean(psd_prep_list, axis=0)
    
    # === Plot 1: Raw PSD (all channels) ===
    ax = axes[0, 0]
    for i, ch in enumerate(common_chs):
        ax.semilogy(freqs_raw, psd_raw[i], alpha=0.5, linewidth=0.8)
    ax.semilogy(freqs_raw, np.mean(psd_raw, axis=0), color='black', linewidth=2, label='Mean')
    ax.set_title('RAW - All Channels', color=COLORS['before'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.set_xlim([0, 100])
    ax.axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz')
    ax.axvspan(0, 0.5, alpha=0.2, color='gray', label='Below bandpass')
    ax.axvspan(80, 100, alpha=0.2, color='gray')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # === Plot 2: Preprocessed PSD (all channels) ===
    ax = axes[0, 1]
    for i, ch in enumerate(common_chs):
        ax.semilogy(freqs_prep, psd_prep[i], alpha=0.5, linewidth=0.8)
    ax.semilogy(freqs_prep, np.mean(psd_prep, axis=0), color='black', linewidth=2, label='Mean')
    ax.set_title('PREPROCESSED - All Channels', color=COLORS['after'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.set_xlim([0, 100])
    ax.axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz')
    ax.axvspan(0, 0.5, alpha=0.2, color='gray')
    ax.axvspan(80, 100, alpha=0.2, color='gray')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # === Plot 3: Mean PSD Overlay ===
    ax = axes[1, 0]
    ax.semilogy(freqs_raw, np.mean(psd_raw, axis=0), color=COLORS['before'], 
                linewidth=2, label='Before', alpha=0.8)
    ax.semilogy(freqs_prep, np.mean(psd_prep, axis=0), color=COLORS['after'], 
                linewidth=2, label='After', alpha=0.8)
    ax.set_title('Mean PSD Comparison')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.set_xlim([0, 100])
    ax.axvline(60, color='red', linestyle='--', alpha=0.5)
    ax.axvspan(0, 0.5, alpha=0.1, color='red', label='Filtered out')
    ax.axvspan(80, 100, alpha=0.1, color='red')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # === Plot 4: PSD Ratio (After/Before) ===
    ax = axes[1, 1]
    # Interpolate to common frequency grid
    from scipy.interpolate import interp1d
    f_interp = interp1d(freqs_raw, np.mean(psd_raw, axis=0), bounds_error=False, fill_value=np.nan)
    psd_raw_interp = f_interp(freqs_prep)
    
    with np.errstate(divide='ignore', invalid='ignore'):
        ratio = np.mean(psd_prep, axis=0) / psd_raw_interp
    
    ax.plot(freqs_prep, 10 * np.log10(ratio), color='purple', linewidth=2)
    ax.axhline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('Power Change (After/Before in dB)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power Change (dB)')
    ax.set_xlim([0, 100])
    ax.set_ylim([-40, 20])
    ax.axvline(60, color='red', linestyle='--', alpha=0.5, label='60 Hz notch')
    ax.axvspan(0, 0.5, alpha=0.2, color='gray', label='Highpass region')
    ax.axvspan(80, 100, alpha=0.2, color='gray', label='Lowpass region')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_notch_filter(pdf, raw_data, raw_sfreq, raw_ch_names,
                         preprocessed_epochs, prep_sfreq):
    """
    Page 3: Zoom in on 60 Hz to verify notch filter.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('3. Notch Filter Verification (60 Hz Power Line Noise)', fontsize=14, fontweight='bold')
    
    common_chs = [ch for ch in CORE_CHS if ch in raw_ch_names]
    raw_indices = [raw_ch_names.index(ch) for ch in common_chs]
    prep_indices = [CORE_CHS.index(ch) for ch in common_chs]
    
    # Compute high-resolution PSD around 60 Hz
    raw_subset = raw_data[raw_indices, :]
    freqs_raw, psd_raw = signal.welch(raw_subset, fs=raw_sfreq, nperseg=1024)
    
    prep_subset = preprocessed_epochs[:, prep_indices, :]
    psd_prep_list = []
    for ep in prep_subset:
        f, p = signal.welch(ep, fs=prep_sfreq, nperseg=min(1024, ep.shape[1]))
        psd_prep_list.append(p)
    freqs_prep = f
    psd_prep = np.mean(psd_prep_list, axis=0)
    
    # Zoom range
    zoom_range = (50, 70)
    
    # Raw
    ax = axes[0]
    mask = (freqs_raw >= zoom_range[0]) & (freqs_raw <= zoom_range[1])
    for i in range(len(common_chs)):
        ax.semilogy(freqs_raw[mask], psd_raw[i, mask], alpha=0.3, linewidth=1)
    ax.semilogy(freqs_raw[mask], np.mean(psd_raw[:, mask], axis=0), 
                color=COLORS['before'], linewidth=2, label='Mean')
    ax.axvline(60, color='red', linestyle='--', linewidth=2, label='60 Hz')
    ax.set_title('BEFORE Notch Filter', color=COLORS['before'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Preprocessed
    ax = axes[1]
    mask = (freqs_prep >= zoom_range[0]) & (freqs_prep <= zoom_range[1])
    for i in range(len(common_chs)):
        ax.semilogy(freqs_prep[mask], psd_prep[i, mask], alpha=0.3, linewidth=1)
    ax.semilogy(freqs_prep[mask], np.mean(psd_prep[:, mask], axis=0), 
                color=COLORS['after'], linewidth=2, label='Mean')
    ax.axvline(60, color='red', linestyle='--', linewidth=2, label='60 Hz')
    ax.set_title('AFTER Notch Filter', color=COLORS['after'])
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add annotation
    fig.text(0.5, 0.02, 
             '✓ If notch filter worked: 60 Hz peak should be greatly reduced or eliminated in the "AFTER" plot',
             ha='center', fontsize=10, style='italic')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_bandpass_filter(pdf, raw_data, raw_sfreq, raw_ch_names,
                            preprocessed_epochs, prep_sfreq, band=(0.5, 80)):
    """
    Page 4: Verify bandpass filter (0.5-80 Hz).
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'4. Bandpass Filter Verification ({band[0]}-{band[1]} Hz)', fontsize=14, fontweight='bold')
    
    common_chs = [ch for ch in CORE_CHS if ch in raw_ch_names]
    raw_indices = [raw_ch_names.index(ch) for ch in common_chs]
    prep_indices = [CORE_CHS.index(ch) for ch in common_chs]
    
    raw_subset = raw_data[raw_indices, :]
    freqs_raw, psd_raw = compute_psd(raw_subset, raw_sfreq, fmin=0.01, fmax=125)
    
    prep_subset = preprocessed_epochs[:, prep_indices, :]
    psd_prep_list = []
    for ep in prep_subset:
        f, p = compute_psd(ep, prep_sfreq, fmin=0.01, fmax=125)
        psd_prep_list.append(p)
    freqs_prep = f
    psd_prep = np.mean(psd_prep_list, axis=0)
    
    mean_raw = np.mean(psd_raw, axis=0)
    mean_prep = np.mean(psd_prep, axis=0)
    
    # Low frequency zoom (highpass check)
    ax = axes[0, 0]
    mask = freqs_raw < 5
    ax.semilogy(freqs_raw[mask], mean_raw[mask], color=COLORS['before'], linewidth=2, label='Before')
    mask = freqs_prep < 5
    ax.semilogy(freqs_prep[mask], mean_prep[mask], color=COLORS['after'], linewidth=2, label='After')
    ax.axvline(band[0], color='green', linestyle='--', linewidth=2, label=f'Highpass ({band[0]} Hz)')
    ax.set_title('Low Frequency (Highpass Check)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # High frequency zoom (lowpass check)
    ax = axes[0, 1]
    mask = freqs_raw > 60
    ax.semilogy(freqs_raw[mask], mean_raw[mask], color=COLORS['before'], linewidth=2, label='Before')
    mask = freqs_prep > 60
    ax.semilogy(freqs_prep[mask], mean_prep[mask], color=COLORS['after'], linewidth=2, label='After')
    ax.axvline(band[1], color='green', linestyle='--', linewidth=2, label=f'Lowpass ({band[1]} Hz)')
    ax.set_title('High Frequency (Lowpass Check)')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Full spectrum with passband marked
    ax = axes[1, 0]
    ax.semilogy(freqs_raw, mean_raw, color=COLORS['before'], linewidth=2, label='Before', alpha=0.7)
    ax.semilogy(freqs_prep, mean_prep, color=COLORS['after'], linewidth=2, label='After', alpha=0.7)
    ax.axvline(band[0], color='green', linestyle='--', linewidth=1)
    ax.axvline(band[1], color='green', linestyle='--', linewidth=1)
    ax.axvspan(band[0], band[1], alpha=0.1, color='green', label='Passband')
    ax.axvspan(0, band[0], alpha=0.1, color='red', label='Stopband')
    ax.axvspan(band[1], 125, alpha=0.1, color='red')
    ax.set_title('Full Spectrum with Passband')
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    
    # Frequency bands power comparison
    ax = axes[1, 1]
    bands_def = {
        'Delta (0.5-4)': (0.5, 4),
        'Theta (4-8)': (4, 8),
        'Alpha (8-13)': (8, 13),
        'Beta (13-30)': (13, 30),
        'Gamma (30-80)': (30, 80)
    }
    
    band_power_raw = []
    band_power_prep = []
    band_names = []
    
    for name, (fmin, fmax) in bands_def.items():
        mask_raw = (freqs_raw >= fmin) & (freqs_raw <= fmax)
        mask_prep = (freqs_prep >= fmin) & (freqs_prep <= fmax)
        band_power_raw.append(np.mean(mean_raw[mask_raw]))
        band_power_prep.append(np.mean(mean_prep[mask_prep]))
        band_names.append(name)
    
    x = np.arange(len(band_names))
    width = 0.35
    ax.bar(x - width/2, band_power_raw, width, label='Before', color=COLORS['before'], alpha=0.7)
    ax.bar(x + width/2, band_power_prep, width, label='After', color=COLORS['after'], alpha=0.7)
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(band_names, rotation=45, ha='right')
    ax.set_title('Power by Frequency Band')
    ax.set_ylabel('Mean Power (V²/Hz)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_channel_variance(pdf, raw_data, raw_sfreq, raw_ch_names,
                             preprocessed_epochs, prep_sfreq, present_mask):
    """
    Page 5: Channel-wise variance analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('5. Channel-wise Variance Analysis', fontsize=14, fontweight='bold')
    
    # Raw variance
    common_chs = [ch for ch in CORE_CHS if ch in raw_ch_names]
    raw_indices = [raw_ch_names.index(ch) for ch in common_chs]
    raw_var = np.var(raw_data[raw_indices, :], axis=1) * 1e12  # Convert to µV²
    
    # Preprocessed variance (mean across epochs)
    prep_var = np.var(preprocessed_epochs, axis=2).mean(axis=0) * 1e12
    
    # Plot 1: Raw channel variance
    ax = axes[0, 0]
    ax.bar(range(len(common_chs)), raw_var, color=COLORS['before'], alpha=0.7)
    ax.set_xticks(range(len(common_chs)))
    ax.set_xticklabels(common_chs, rotation=45, ha='right', fontsize=8)
    ax.set_title('RAW - Channel Variance', color=COLORS['before'])
    ax.set_ylabel('Variance (µV²)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Preprocessed channel variance
    ax = axes[0, 1]
    colors = [COLORS['interpolated'] if not present_mask[i] else COLORS['after'] for i in range(22)]
    ax.bar(range(22), prep_var, color=colors, alpha=0.7)
    ax.set_xticks(range(22))
    ax.set_xticklabels(CORE_CHS, rotation=45, ha='right', fontsize=8)
    ax.set_title('PREPROCESSED - Channel Variance', color=COLORS['after'])
    ax.set_ylabel('Variance (µV²)')
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add legend for interpolated
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=COLORS['after'], label='Real channels'),
        Patch(facecolor=COLORS['interpolated'], label='Interpolated')
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=8)
    
    # Plot 3: Variance comparison (common channels only)
    ax = axes[1, 0]
    prep_indices = [CORE_CHS.index(ch) for ch in common_chs]
    prep_var_common = prep_var[prep_indices]
    
    x = np.arange(len(common_chs))
    width = 0.35
    ax.bar(x - width/2, raw_var, width, label='Before', color=COLORS['before'], alpha=0.7)
    ax.bar(x + width/2, prep_var_common, width, label='After', color=COLORS['after'], alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(common_chs, rotation=45, ha='right', fontsize=8)
    ax.set_title('Variance Comparison (Common Channels)')
    ax.set_ylabel('Variance (µV²)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: Std deviation per channel (preprocessed)
    ax = axes[1, 1]
    prep_std = np.std(preprocessed_epochs, axis=2).mean(axis=0) * 1e6  # µV
    ax.bar(range(22), prep_std, color=colors, alpha=0.7)
    ax.axhline(np.mean(prep_std), color='black', linestyle='--', label=f'Mean: {np.mean(prep_std):.1f} µV')
    ax.set_xticks(range(22))
    ax.set_xticklabels(CORE_CHS, rotation=45, ha='right', fontsize=8)
    ax.set_title('PREPROCESSED - Channel Std Dev')
    ax.set_ylabel('Std Dev (µV)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_amplitude_distribution(pdf, raw_data, raw_ch_names,
                                   preprocessed_epochs, present_mask):
    """
    Page 6: Amplitude distribution histograms.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('6. Amplitude Distribution Analysis', fontsize=14, fontweight='bold')
    
    common_chs = [ch for ch in CORE_CHS if ch in raw_ch_names]
    raw_indices = [raw_ch_names.index(ch) for ch in common_chs]
    
    raw_subset = raw_data[raw_indices, :].flatten() * 1e6  # µV
    prep_data = preprocessed_epochs.flatten() * 1e6  # µV
    
    # Clip for visualization
    raw_clipped = np.clip(raw_subset, -500, 500)
    prep_clipped = np.clip(prep_data, -200, 200)
    
    # Plot 1: Raw amplitude histogram
    ax = axes[0, 0]
    ax.hist(raw_clipped, bins=100, color=COLORS['before'], alpha=0.7, density=True)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('RAW - Amplitude Distribution', color=COLORS['before'])
    ax.set_xlabel('Amplitude (µV)')
    ax.set_ylabel('Density')
    ax.set_xlim([-500, 500])
    
    stats_text = f'Mean: {np.mean(raw_subset):.1f} µV\nStd: {np.std(raw_subset):.1f} µV'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 2: Preprocessed amplitude histogram
    ax = axes[0, 1]
    ax.hist(prep_clipped, bins=100, color=COLORS['after'], alpha=0.7, density=True)
    ax.axvline(0, color='black', linestyle='-', alpha=0.3)
    ax.set_title('PREPROCESSED - Amplitude Distribution', color=COLORS['after'])
    ax.set_xlabel('Amplitude (µV)')
    ax.set_ylabel('Density')
    ax.set_xlim([-200, 200])
    
    stats_text = f'Mean: {np.mean(prep_data):.1f} µV\nStd: {np.std(prep_data):.1f} µV'
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Plot 3: Overlay comparison
    ax = axes[1, 0]
    ax.hist(raw_clipped, bins=100, color=COLORS['before'], alpha=0.5, density=True, label='Before')
    ax.hist(prep_clipped, bins=100, color=COLORS['after'], alpha=0.5, density=True, label='After')
    ax.set_title('Amplitude Distribution Comparison')
    ax.set_xlabel('Amplitude (µV)')
    ax.set_ylabel('Density')
    ax.set_xlim([-300, 300])
    ax.legend()
    
    # Plot 4: Q-Q plot (check for normality)
    ax = axes[1, 1]
    from scipy import stats as scipy_stats
    prep_sample = np.random.choice(prep_data, size=min(10000, len(prep_data)), replace=False)
    scipy_stats.probplot(prep_sample, dist="norm", plot=ax)
    ax.set_title('Q-Q Plot (Preprocessed vs Normal)')
    ax.get_lines()[0].set_markerfacecolor(COLORS['after'])
    ax.get_lines()[0].set_markeredgecolor(COLORS['after'])
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_interpolation_quality(pdf, preprocessed_epochs, present_mask):
    """
    Page 7: Interpolation quality assessment.
    """
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('7. Interpolation Quality Assessment', fontsize=14, fontweight='bold')
    
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.3)
    
    real_indices = np.where(present_mask)[0]
    interp_indices = np.where(~present_mask)[0]
    
    n_interp = len(interp_indices)
    n_real = len(real_indices)
    
    # Info text
    ax = fig.add_subplot(gs[0, 0])
    ax.axis('off')
    
    info_text = f"""
    Channel Status Summary
    ──────────────────────────────
    Total channels:      22 (CORE_CHS)
    Real channels:       {n_real}
    Interpolated:        {n_interp}
    
    Real channels:
    {', '.join([CORE_CHS[i] for i in real_indices])}
    
    Interpolated channels:
    {', '.join([CORE_CHS[i] for i in interp_indices]) if n_interp > 0 else 'None'}
    """
    ax.text(0.1, 0.9, info_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Variance comparison: real vs interpolated
    ax = fig.add_subplot(gs[0, 1])
    prep_var = np.var(preprocessed_epochs, axis=2).mean(axis=0) * 1e12
    
    if n_interp > 0:
        real_var = prep_var[real_indices]
        interp_var = prep_var[interp_indices]
        
        ax.boxplot([real_var, interp_var], labels=['Real', 'Interpolated'])
        ax.set_title('Variance: Real vs Interpolated Channels')
        ax.set_ylabel('Variance (µV²)')
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No interpolated channels', ha='center', va='center', fontsize=12)
        ax.set_title('Variance: Real vs Interpolated Channels')
    
    # Correlation with neighbors (for interpolated channels)
    ax = fig.add_subplot(gs[1, 0])
    
    if n_interp > 0:
        # Compute mean correlation of each interpolated channel with its neighbors
        mean_data = preprocessed_epochs.mean(axis=0)  # (22, n_times)
        corr_matrix = np.corrcoef(mean_data)
        
        interp_corrs = []
        for idx in interp_indices:
            # Get correlations with all other channels
            corrs = corr_matrix[idx, :]
            # Exclude self-correlation
            corrs = np.delete(corrs, idx)
            interp_corrs.append(np.mean(corrs))
        
        ax.bar(range(len(interp_indices)), interp_corrs, color=COLORS['interpolated'], alpha=0.7)
        ax.set_xticks(range(len(interp_indices)))
        ax.set_xticklabels([CORE_CHS[i] for i in interp_indices], rotation=45, ha='right')
        ax.axhline(0.5, color='red', linestyle='--', label='Threshold (0.5)')
        ax.set_title('Interpolated Channels: Mean Correlation with Others')
        ax.set_ylabel('Mean Correlation')
        ax.set_ylim([0, 1])
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
    else:
        ax.text(0.5, 0.5, 'No interpolated channels', ha='center', va='center', fontsize=12)
        ax.set_title('Interpolated Channels: Mean Correlation with Others')
    
    # Time series of interpolated channels
    ax = fig.add_subplot(gs[1, 1])
    
    if n_interp > 0:
        t = np.arange(preprocessed_epochs.shape[2]) / 250  # Assuming 250 Hz
        for i, idx in enumerate(interp_indices[:4]):  # Show max 4
            #trace = preprocessed_epochs[0, idx, :] * 1e6
            # NEW - uses epoch 5 (or last available if fewer epochs)
            epoch_to_show = min(5, len(preprocessed_epochs) - 1)
            trace = preprocessed_epochs[epoch_to_show, idx, :] * 1e6
            ax.plot(t, trace - i * 100, label=CORE_CHS[idx], alpha=0.8)
        ax.set_title('Interpolated Channel Waveforms (First Epoch)')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel('Amplitude (µV, offset)')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    else:
        ax.text(0.5, 0.5, 'No interpolated channels', ha='center', va='center', fontsize=12)
        ax.set_title('Interpolated Channel Waveforms')
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_epochs_overview(pdf, preprocessed_epochs, prep_sfreq, labels):
    """
    Page 8: Epoch overview grid.
    """
    n_epochs = len(preprocessed_epochs)
    n_show = min(20, n_epochs)
    
    fig, axes = plt.subplots(4, 5, figsize=(14, 10))
    fig.suptitle(f'8. Epoch Overview (Showing {n_show}/{n_epochs} Epochs)', fontsize=14, fontweight='bold')
    
    axes = axes.flatten()
    t = np.arange(preprocessed_epochs.shape[2]) / prep_sfreq
    
    # Select epochs to show (evenly spaced)
    indices = np.linspace(0, n_epochs - 1, n_show, dtype=int)
    
    for i, idx in enumerate(indices):
        ax = axes[i]
        
        # Plot mean across channels
        mean_trace = preprocessed_epochs[idx].mean(axis=0) * 1e6
        ax.plot(t, mean_trace, linewidth=0.5, color=COLORS['after'])
        
        # Add epoch info
        ax.set_title(f'Epoch {idx}', fontsize=8)
        ax.set_xlim([0, t[-1]])
        
        # Compute and show peak-to-peak
        ptp = np.ptp(preprocessed_epochs[idx]) * 1e6
        ax.text(0.95, 0.95, f'PTP: {ptp:.0f}µV', transform=ax.transAxes, 
                fontsize=6, ha='right', va='top')
        
        if i >= 15:  # Bottom row
            ax.set_xlabel('Time (s)', fontsize=7)
        if i % 5 == 0:  # Left column
            ax.set_ylabel('µV', fontsize=7)
        
        ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_epoch_statistics(pdf, preprocessed_epochs, threshold_uv):
    """
    Page 9: Epoch amplitude statistics and rejection info.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('9. Epoch Amplitude Statistics', fontsize=14, fontweight='bold')
    
    n_epochs = len(preprocessed_epochs)
    
    # Compute statistics per epoch
    ptp_per_epoch = np.ptp(preprocessed_epochs, axis=2).max(axis=1) * 1e6  # Max PTP across channels
    mean_per_epoch = np.abs(preprocessed_epochs).mean(axis=(1, 2)) * 1e6
    std_per_epoch = preprocessed_epochs.std(axis=(1, 2)) * 1e6
    
    # Plot 1: Peak-to-peak amplitude per epoch
    ax = axes[0, 0]
    ax.bar(range(n_epochs), ptp_per_epoch, color=COLORS['after'], alpha=0.7)
    if threshold_uv is not None:
        ax.axhline(threshold_uv, color='red', linestyle='--', linewidth=2, 
                   label=f'Rejection threshold: {threshold_uv:.0f} µV')
    ax.set_title('Peak-to-Peak Amplitude per Epoch')
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Max PTP (µV)')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Histogram of PTP values
    ax = axes[0, 1]
    ax.hist(ptp_per_epoch, bins=30, color=COLORS['after'], alpha=0.7, edgecolor='black')
    if threshold_uv is not None:
        ax.axvline(threshold_uv, color='red', linestyle='--', linewidth=2,
                   label=f'Rejection threshold: {threshold_uv:.0f} µV')
    ax.set_title('Distribution of Peak-to-Peak Amplitudes')
    ax.set_xlabel('Max PTP (µV)')
    ax.set_ylabel('Count')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Mean amplitude per epoch
    ax = axes[1, 0]
    ax.plot(range(n_epochs), mean_per_epoch, 'o-', color=COLORS['after'], 
            markersize=3, alpha=0.7)
    ax.set_title('Mean Absolute Amplitude per Epoch')
    ax.set_xlabel('Epoch Index')
    ax.set_ylabel('Mean |Amplitude| (µV)')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Statistics summary
    ax = axes[1, 1]
    ax.axis('off')
    
    stats_text = f"""
    Epoch Statistics Summary
    ────────────────────────────────────────
    Number of epochs:           {n_epochs}
    Epoch duration:             {preprocessed_epochs.shape[2] / 250:.1f} s
    
    Peak-to-Peak Amplitude:
      Min:                      {ptp_per_epoch.min():.1f} µV
      Max:                      {ptp_per_epoch.max():.1f} µV
      Mean:                     {ptp_per_epoch.mean():.1f} µV
      Std:                      {ptp_per_epoch.std():.1f} µV
      
    Rejection Threshold:        {threshold_uv:.1f} µV
    
    Mean Absolute Amplitude:
      Min:                      {mean_per_epoch.min():.2f} µV
      Max:                      {mean_per_epoch.max():.2f} µV
      Mean:                     {mean_per_epoch.mean():.2f} µV
      
    Standard Deviation:
      Min:                      {std_per_epoch.min():.2f} µV
      Max:                      {std_per_epoch.max():.2f} µV
      Mean:                     {std_per_epoch.mean():.2f} µV
    """
    ax.text(0.1, 0.95, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def inspect_correlation_matrix(pdf, raw_data, raw_ch_names, preprocessed_epochs, present_mask):
    """
    Page 10: Correlation matrix before and after.
    """
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('10. Channel Correlation Matrix', fontsize=14, fontweight='bold')
    
    common_chs = [ch for ch in CORE_CHS if ch in raw_ch_names]
    raw_indices = [raw_ch_names.index(ch) for ch in common_chs]
    
    # Raw correlation
    raw_subset = raw_data[raw_indices, :]
    corr_raw = np.corrcoef(raw_subset)
    
    ax = axes[0]
    im = ax.imshow(corr_raw, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(len(common_chs)))
    ax.set_yticks(range(len(common_chs)))
    ax.set_xticklabels(common_chs, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(common_chs, fontsize=7)
    ax.set_title('RAW - Channel Correlation', color=COLORS['before'])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Preprocessed correlation
    mean_prep = preprocessed_epochs.mean(axis=0)  # (22, n_times)
    corr_prep = np.corrcoef(mean_prep)
    
    ax = axes[1]
    im = ax.imshow(corr_prep, cmap='RdBu_r', vmin=-1, vmax=1, aspect='auto')
    ax.set_xticks(range(22))
    ax.set_yticks(range(22))
    
    # Mark interpolated channels
    labels = []
    for i, ch in enumerate(CORE_CHS):
        if not present_mask[i]:
            labels.append(f'{ch}*')
        else:
            labels.append(ch)
    
    ax.set_xticklabels(labels, rotation=45, ha='right', fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_title('PREPROCESSED - Channel Correlation (* = interpolated)', color=COLORS['after'])
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def create_summary_page(pdf, pid, raw_data, raw_sfreq, raw_ch_names,
                        preprocessed_epochs, prep_sfreq, present_mask, 
                        labels, threshold_uv):
    """
    Final page: Summary statistics table.
    """
    fig = plt.figure(figsize=(11, 8.5))
    fig.suptitle('11. Preprocessing Summary', fontsize=16, fontweight='bold')
    
    ax = fig.add_subplot(111)
    ax.axis('off')
    
    # Compute statistics
    common_chs = [ch for ch in CORE_CHS if ch in raw_ch_names]
    n_interp = (~present_mask).sum()
    
    raw_duration = raw_data.shape[1] / raw_sfreq
    prep_duration = preprocessed_epochs.shape[0] * preprocessed_epochs.shape[2] / prep_sfreq
    
    raw_ptp = np.ptp(raw_data) * 1e6
    prep_ptp = np.ptp(preprocessed_epochs) * 1e6
    
    summary = f"""
    ╔══════════════════════════════════════════════════════════════════════════════╗
    ║                         PREPROCESSING SUMMARY                                 ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  Patient ID:                  {pid:<47} ║
    ║  Label:                       {'EPILEPSY' if labels[0] == 1 else 'CONTROL':<47} ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  RAW DATA                                                                     ║
    ║  ─────────────────────────────────────────────────────────────────────────── ║
    ║  Channels:                    {len(raw_ch_names):<47} ║
    ║  CORE_CHS found:              {len(common_chs)}/22{' ':42} ║
    ║  Sampling rate:               {raw_sfreq:.1f} Hz{' ':40} ║
    ║  Duration:                    {raw_duration:.1f} s{' ':42} ║
    ║  Peak-to-peak:                {raw_ptp:.1f} µV{' ':40} ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  PREPROCESSED DATA                                                            ║
    ║  ─────────────────────────────────────────────────────────────────────────── ║
    ║  Channels:                    22 (standardized){' ':30} ║
    ║  Interpolated:                {n_interp:<47} ║
    ║  Sampling rate:               {prep_sfreq:.1f} Hz{' ':40} ║
    ║  Number of epochs:            {len(preprocessed_epochs):<47} ║
    ║  Epoch duration:              {preprocessed_epochs.shape[2] / prep_sfreq:.1f} s{' ':42} ║
    ║  Total duration:              {prep_duration:.1f} s{' ':41} ║
    ║  Peak-to-peak:                {prep_ptp:.1f} µV{' ':40} ║
    ║  Rejection threshold:         {threshold_uv:.1f} µV{' ':40} ║
    ╠══════════════════════════════════════════════════════════════════════════════╣
    ║  PIPELINE STEPS APPLIED                                                       ║
    ║  ─────────────────────────────────────────────────────────────────────────── ║
    ║  [✓] Channel name standardization                                             ║
    ║  [✓] Montage assignment (10-20 system)                                        ║
    ║  [✓] Bad channel interpolation (spherical spline)                             ║
    ║  [✓] Notch filter @ 60 Hz (zero-phase)                                        ║
    ║  [✓] Bandpass filter 0.5-80 Hz (zero-phase)                                   ║
    ║  [✓] Resampling to 250 Hz                                                     ║
    ║  [✓] Common average reference                                                 ║
    ║  [✓] Epoching (4s windows)                                                    ║
    ║  [✓] Artifact rejection ({threshold_uv:.0f} µV threshold){' ':34} ║
    ║  [✗] Linear detrending (skipped)                                              ║
    ║  [✗] Z-score normalization (skipped)                                          ║
    ╚══════════════════════════════════════════════════════════════════════════════╝
    """
    
    ax.text(0.5, 0.5, summary, transform=ax.transAxes, fontsize=9,
            verticalalignment='center', horizontalalignment='center',
            family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.5))
    
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Comprehensive EEG Preprocessing Inspection Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
    python inspect_preprocessing.py \\
        --edf data_raw/patient001.edf \\
        --preprocessed_dir data_pp \\
        --pid patient001 \\
        --out inspection_reports
        """
    )
    
    parser.add_argument("--edf", required=True, help="Path to raw EDF file")
    parser.add_argument("--preprocessed_dir", required=True, help="Directory containing preprocessed files")
    parser.add_argument("--pid", required=True, help="Patient ID (filename stem)")
    parser.add_argument("--out", required=True, help="Output directory for report")
    parser.add_argument("--sfreq", type=float, default=250.0, help="Preprocessed sampling rate (default: 250)")
    
    args = parser.parse_args()
    
    edf_path = Path(args.edf)
    prep_dir = Path(args.preprocessed_dir)
    out_dir = Path(args.out)
    pid = args.pid
    prep_sfreq = args.sfreq
    
    # Validate inputs
    if not edf_path.exists():
        print(f"Error: EDF file not found: {edf_path}")
        sys.exit(1)
    
    epochs_file = prep_dir / f"{pid}_epochs.npy"
    if not epochs_file.exists():
        print(f"Error: Preprocessed epochs not found: {epochs_file}")
        sys.exit(1)
    
    # Create output directory
    out_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*70)
    print("EEG PREPROCESSING INSPECTION")
    print("="*70)
    
    # ========== LOAD RAW DATA ==========
    print(f"\n[1/3] Loading raw EDF: {edf_path}")
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose="ERROR")
    raw.pick_types(eeg=True, exclude=['EOG', 'ECG', 'EMG', 'MISC', 'STIM'])
    clean_channel_names(raw)
    
    raw_data = raw.get_data()
    raw_sfreq = raw.info['sfreq']
    raw_ch_names = raw.ch_names
    
    print(f"  - Channels: {len(raw_ch_names)}")
    print(f"  - Sampling rate: {raw_sfreq} Hz")
    print(f"  - Duration: {raw_data.shape[1] / raw_sfreq:.1f} s")
    
    # ========== LOAD PREPROCESSED DATA ==========
    print(f"\n[2/3] Loading preprocessed data from: {prep_dir}")
    
    preprocessed_epochs = np.load(epochs_file)
    print(f"  - Epochs shape: {preprocessed_epochs.shape}")
    
    labels_file = prep_dir / f"{pid}_labels.npy"
    labels = np.load(labels_file) if labels_file.exists() else np.zeros(len(preprocessed_epochs))
    print(f"  - Label: {'EPILEPSY' if labels[0] == 1 else 'CONTROL'}")
    
    mask_file = prep_dir / f"{pid}_present_mask.npy"
    present_mask = np.load(mask_file) if mask_file.exists() else np.ones(22, dtype=bool)
    print(f"  - Interpolated channels: {(~present_mask).sum()}")
    
    # ------------------ CHANGED SECTION START ------------------
    # Calculate threshold based on 95th percentile of peak-to-peak amplitudes
    # (matching preprocess_core.py methodology)
    max_ptp_uv = np.ptp(preprocessed_epochs, axis=2).max(axis=1) * 1e6  # max ptp per epoch
    threshold_uv = float(np.percentile(max_ptp_uv, 95))

    print(f"  - Dynamic Threshold (95th percentile of max PTP): {threshold_uv:.2f} µV")
    print(f"  - Dynamic Threshold (95th percentile): {threshold_uv:.2f} µV")
    
    # ========== GENERATE REPORT ==========
    print(f"\n[3/3] Generating inspection report...")
    
    report_path = out_dir / f"{pid}_inspection_report.pdf"
    
    with PdfPages(report_path) as pdf:
        # Title page
        print("  - Title page")
        add_title_page(pdf, pid, edf_path, prep_dir)
        
        # Page 1: Time series
        print("  - Page 1: Time series comparison")
        inspect_time_series(pdf, raw_data, raw_sfreq, raw_ch_names,
                           preprocessed_epochs, prep_sfreq, present_mask)
        
        # Page 2: PSD comparison
        print("  - Page 2: PSD comparison")
        inspect_psd_comparison(pdf, raw_data, raw_sfreq, raw_ch_names,
                              preprocessed_epochs, prep_sfreq, present_mask)
        
        # Page 3: Notch filter
        print("  - Page 3: Notch filter verification")
        inspect_notch_filter(pdf, raw_data, raw_sfreq, raw_ch_names,
                            preprocessed_epochs, prep_sfreq)
        
        # Page 4: Bandpass filter
        print("  - Page 4: Bandpass filter verification")
        inspect_bandpass_filter(pdf, raw_data, raw_sfreq, raw_ch_names,
                               preprocessed_epochs, prep_sfreq)
        
        # Page 5: Channel variance
        print("  - Page 5: Channel variance analysis")
        inspect_channel_variance(pdf, raw_data, raw_sfreq, raw_ch_names,
                                preprocessed_epochs, prep_sfreq, present_mask)
        
        # Page 6: Amplitude distribution
        print("  - Page 6: Amplitude distribution")
        inspect_amplitude_distribution(pdf, raw_data, raw_ch_names,
                                       preprocessed_epochs, present_mask)
        
        # Page 7: Interpolation quality
        print("  - Page 7: Interpolation quality")
        inspect_interpolation_quality(pdf, preprocessed_epochs, present_mask)
        
        # Page 8: Epochs overview
        print("  - Page 8: Epochs overview")
        inspect_epochs_overview(pdf, preprocessed_epochs, prep_sfreq, labels)
        
        # Page 9: Epoch statistics
        print("  - Page 9: Epoch statistics")
        inspect_epoch_statistics(pdf, preprocessed_epochs, threshold_uv)
        
        # Page 10: Correlation matrix
        print("  - Page 10: Correlation matrix")
        inspect_correlation_matrix(pdf, raw_data, raw_ch_names, 
                                   preprocessed_epochs, present_mask)
        
        # Page 11: Summary
        print("  - Page 11: Summary")
        create_summary_page(pdf, pid, raw_data, raw_sfreq, raw_ch_names,
                           preprocessed_epochs, prep_sfreq, present_mask,
                           labels, threshold_uv)
    
    print("\n" + "="*70)
    print("INSPECTION COMPLETE")
    print("="*70)
    print(f"Report saved to: {report_path}")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()