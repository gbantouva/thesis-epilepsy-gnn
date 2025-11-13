"""
Single EDF file preprocessing script.

Usage:
  python src/preprocess_single.py --edf path/to/file.edf --out data_pp --psd_dir figures/psd
  
Example:
  python src/preprocess_single.py --edf data_raw/DATA/00_epilepsy/001_t001.edf --out data_pp --psd_dir figures/psd
  python src/preprocess_single.py --edf F:\October-Thesis\thesis-epilepsy-gnn\data_raw\DATA\00_epilepsy\aaaaaanr\s001_2003\02_tcp_le\aaaaaanr_s001_t001.edf --out F:\October-Thesis\thesis-epilepsy-gnn\test --psd_dir F:\October-Thesis\thesis-epilepsy-gnn\test\psd
This script:
1. Preprocesses a single EDF file using the optimized pipeline
2. Saves numpy arrays (epochs, labels, raw)
3. Saves metadata (channel info, present mask)
4. Optionally saves PSD plots before/after preprocessing
"""

import sys
from pathlib import Path

# Allow importing from src/ directory
sys.path.append(str(Path(__file__).resolve().parent))

import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
import json
from preprocess_core import preprocess_single


def main():
    """
    Main function to preprocess a single EEG EDF file.
    
    Saves the following files to output directory:
    - {pid}_epochs.npy          : Preprocessed epochs (n_epochs, 22, n_times)
    - {pid}_labels.npy          : Per-epoch labels (0=control, 1=epilepsy)
    - {pid}_raw.npy             : Preprocessed raw data (22, n_times)
    - {pid}_info.pkl            : MNE info object (contains metadata)
    - {pid}_present_mask.npy    : Boolean mask (True=real, False=interpolated)
    - {pid}_present_channels.json : List of channel names (always CORE_CHS)
    - {pid}_PSD_before.png      : PSD plot before preprocessing
    - {pid}_PSD_after.png       : PSD plot after preprocessing
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="Preprocess a single EEG EDF file for epilepsy detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage
  python src/preprocess_single.py --edf data_raw/DATA/00_epilepsy/001_t001.edf --out data_pp --psd_dir figures/psd
  
  # With custom parameters
  python src/preprocess_single.py --edf data.edf --out output --psd_dir figs --notch 50 --band 1 100 --resample 256
        """
    )
    
    parser.add_argument(
        "--edf", 
        required=True, 
        help="Path to input EDF file"
    )
    parser.add_argument(
        "--out", 
        required=True, 
        help="Output directory for preprocessed arrays and metadata"
    )
    parser.add_argument(
        "--psd_dir", 
        required=True, 
        help="Output directory for PSD figures"
    )
    parser.add_argument(
        "--notch", 
        type=float, 
        default=60.0, 
        help="Notch filter frequency in Hz (default: 60.0 for US power line)"
    )
    parser.add_argument(
        "--band", 
        type=float, 
        nargs=2, 
        default=[0.5, 100.0],
        metavar=('LOW', 'HIGH'),
        help="Bandpass filter frequencies in Hz (default: 0.5 100.0)"
    )
    parser.add_argument(
        "--resample", 
        type=float, 
        default=250.0,
        help="Target sampling rate in Hz (default: 250.0)"
    )
    parser.add_argument(
        "--epoch_len", 
        type=float, 
        default=4.0,
        help="Epoch duration in seconds (default: 4.0)"
    )
    parser.add_argument(
        "--epoch_overlap", 
        type=float, 
        default=0.0,
        help="Overlap between epochs in seconds (default: 0.0)"
    )
    parser.add_argument(
        "--reject_percentile", 
        type=float, 
        default=98.0,
        help="Percentile for adaptive rejection threshold (default: 98.0)"
    )
    parser.add_argument(
        "--reject_cap", 
        type=float, 
        default=None,
        help="Hard cap for rejection threshold in µV (default: None - no cap)"
    )
    parser.add_argument(
        "--no_psd", 
        action="store_true",
        help="Skip PSD computation and plotting"
    )
    parser.add_argument(
        "--quiet", 
        action="store_true",
        help="Suppress progress messages"
    )
    
    args = parser.parse_args()
    
    # Convert paths
    edf_path = Path(args.edf)
    out_dir = Path(args.out)
    psd_dir = Path(args.psd_dir)
    
    # Validate input file
    if not edf_path.exists():
        print(f"Error: EDF file not found: {edf_path}")
        sys.exit(1)
    
    # Create output directories
    out_dir.mkdir(parents=True, exist_ok=True)
    if not args.no_psd:
        psd_dir.mkdir(parents=True, exist_ok=True)
    
    # Get patient ID from filename
    pid = edf_path.stem
    
    # Print configuration
    if not args.quiet:
        print("\n" + "="*70)
        print("EEG PREPROCESSING - SINGLE FILE")
        print("="*70)
        print(f"Input file:          {edf_path}")
        print(f"Output directory:    {out_dir}")
        print(f"PSD directory:       {psd_dir}")
        print(f"Patient ID:          {pid}")
        print(f"Notch filter:        {args.notch} Hz")
        print(f"Bandpass filter:     {args.band[0]}-{args.band[1]} Hz")
        print(f"Resample rate:       {args.resample} Hz")
        print(f"Epoch length:        {args.epoch_len} s")
        print(f"Epoch overlap:       {args.epoch_overlap} s")
        print(f"Reject percentile:   {args.reject_percentile}th")
        if args.reject_cap is not None:
            print(f"Reject cap:          {args.reject_cap} µV")
        else:
            print(f"Reject cap:          None (no cap)")
        print("="*70)
    
    # Run preprocessing pipeline
    res = preprocess_single(
        edf_path,
        notch=args.notch,
        band=tuple(args.band),
        resample_hz=args.resample,
        epoch_len=args.epoch_len,
        epoch_overlap=args.epoch_overlap,
        reject_percentile=args.reject_percentile,
        reject_cap_uv=args.reject_cap,
        return_psd=not args.no_psd,
        verbose=not args.quiet
    )
    
    # Check if preprocessing failed
    if res is None:
        print(f"\n✗ Preprocessing failed for {pid}")
        print(f"  Skipping file save.")
        sys.exit(1)
    
    # Save preprocessed numpy arrays
    if not args.quiet:
        print("\nSaving preprocessed data...")
    
    np.save(out_dir / f"{pid}_epochs.npy", res["epochs"].get_data())
    if not args.quiet:
        print(f"  ✓ {pid}_epochs.npy")
    
    np.save(out_dir / f"{pid}_labels.npy", res["labels"])
    if not args.quiet:
        print(f"  ✓ {pid}_labels.npy")
    
    #np.save(out_dir / f"{pid}_raw.npy", res["raw_after"].get_data())
    #if not args.quiet:
    #    print(f"  ✓ {pid}_raw.npy")
    
    np.save(out_dir / f"{pid}_present_mask.npy", res["present_mask"])
    if not args.quiet:
        print(f"  ✓ {pid}_present_mask.npy")
    
    # Save metadata
    with open(out_dir / f"{pid}_info.pkl", "wb") as f:
        pickle.dump(res["raw_after"].info, f)
    if not args.quiet:
        print(f"  ✓ {pid}_info.pkl")
    
    with open(out_dir / f"{pid}_present_channels.json", "w", encoding="utf-8") as f:
        json.dump(res["present_channels"], f, ensure_ascii=False, indent=2)
    if not args.quiet:
        print(f"  ✓ {pid}_present_channels.json")
    
    # Save PSD plots
    if not args.no_psd:
        if not args.quiet:
            print("\nSaving PSD plots...")
        
        for tag, psd in [("before", res.get("psd_before")), 
                        ("after", res.get("psd_after"))]:
            if psd is None:
                continue
            
            try:
                # Create figure
                
                fig = psd.plot(show=False, average='mean', spatial_colors=False)
                #fig = psd.plot(show=False, spatial_colors=True)  # Per-channel colors

                # Add title and labels
                fig.suptitle(f"{pid} - PSD {tag.upper()}", fontsize=14, fontweight='bold')
                
                # Save figure
                output_path = psd_dir / f"{pid}_PSD_{tag}.png"
                fig.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                
                if not args.quiet:
                    print(f"  ✓ {pid}_PSD_{tag}.png")
                    
            except Exception as e:
                if not args.quiet:
                    print(f"  ✗ Failed to save PSD {tag}: {e}")
                plt.close('all')
    
    # Print summary
    if not args.quiet:
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"Patient ID:          {pid}")
        print(f"Label:               {'EPILEPSY' if res['labels'][0] == 1 else 'CONTROL'}")
        print(f"Epochs saved:        {len(res['epochs'])}")
        print(f"Channels:            {len(res['present_channels'])} (22 CORE_CHS)")
        print(f"Interpolated:        {(~res['present_mask']).sum()}")
        print(f"Rejection threshold: {res['threshold_uv']:.1f} µV")
        print(f"Epoch shape:         {res['epochs'].get_data().shape}")
        print(f"Output directory:    {out_dir}")
        print("="*70 + "\n")


if __name__ == "__main__":
    main()