"""
Copy Balanced Session Directories (Complete)
============================================

This script copies ENTIRE session directories for the selected balanced sessions.

Includes ALL preprocessing files:
- *_epochs.npy
- *_info.json
- *_labels.npy
- *_present_channels.npy
- *_present_mask.npy
- Any other files in the session directory

NO connectivity files (those are in connectivity_output, not data_pp)
"""

from pathlib import Path
import shutil
from tqdm import tqdm

# ============================================================================
# CONFIGURATION - CHANGE THESE!
# ============================================================================

# Path to balanced session list
BALANCED_LIST = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\balanced_epilepsy_sessions.txt')

# Source base directory (data_pp/version3)
SOURCE_BASE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_pp\version3')

# Destination directory (where to copy)
DEST_BASE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_pp\version3\balanced_data')

# ============================================================================
# SCRIPT
# ============================================================================

def get_session_directory(epoch_file_path):
    """
    Get the session directory from an epoch file path.
    
    Example:
        Input:  F:/data_pp/version3/00_epilepsy/aaaaaanr/s003_2013/01_tcp_ar/aaaaaanr_s003_t000_epochs.npy
        Output: F:/data_pp/version3/00_epilepsy/aaaaaanr/s003_2013/01_tcp_ar/
    """
    return Path(epoch_file_path).parent


def get_relative_path(full_path, base_path):
    """Get relative path from base."""
    try:
        return Path(full_path).relative_to(base_path)
    except ValueError:
        # If not relative, try with string manipulation
        full_str = str(full_path).replace('\\', '/')
        base_str = str(base_path).replace('\\', '/')
        if full_str.startswith(base_str):
            rel = full_str[len(base_str):].lstrip('/')
            return Path(rel)
        raise


def copy_directory_with_structure(src_dir, src_base, dest_base):
    """
    Copy entire directory preserving structure.
    
    Example:
        src_dir:   F:/data_pp/version3/00_epilepsy/aaaaaanr/s003_2013/01_tcp_ar
        src_base:  F:/data_pp/version3
        dest_base: F:/balanced_data
        result:    F:/balanced_data/00_epilepsy/aaaaaanr/s003_2013/01_tcp_ar
    """
    # Get relative path
    rel_path = get_relative_path(src_dir, src_base)
    
    # Create destination path
    dest_dir = dest_base / rel_path
    
    # Copy entire directory
    if src_dir.exists():
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)
        return True
    return False


def count_files_in_directory(directory):
    """Count files in a directory."""
    if directory.exists():
        return len(list(directory.glob('*.*')))
    return 0


def main():
    print("="*80)
    print("COPYING BALANCED SESSION DIRECTORIES")
    print("="*80)
    
    # Load balanced session list
    print(f"\n1. Loading balanced session list...")
    if not BALANCED_LIST.exists():
        print(f"   ❌ ERROR: List file not found: {BALANCED_LIST}")
        return
    
    with open(BALANCED_LIST, 'r') as f:
        balanced_sessions = [line.strip() for line in f]
    
    print(f"   ✓ Loaded {len(balanced_sessions)} epoch files")
    
    # Check source directory
    print(f"\n2. Source directory: {SOURCE_BASE}")
    if not SOURCE_BASE.exists():
        print(f"   ❌ ERROR: Source directory not found!")
        return
    print(f"   ✓ Directory found")
    
    # Create destination directory
    print(f"\n3. Destination directory: {DEST_BASE}")
    DEST_BASE.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Directory created/verified")
    
    # Extract unique session directories
    print(f"\n4. Finding session directories...")
    session_dirs = set()
    
    for epoch_path in balanced_sessions:
        session_dir = get_session_directory(epoch_path)
        session_dirs.add(session_dir)
    
    print(f"   ✓ Found {len(session_dirs)} unique session directories")
    
    # Copy session directories
    print(f"\n5. Copying session directories...")
    print(f"   This will copy ALL files in each session directory:")
    print(f"   - *_epochs.npy")
    print(f"   - *_info.json")
    print(f"   - *_labels.npy")
    print(f"   - *_present_channels.npy")
    print(f"   - *_present_mask.npy")
    print(f"   - Any other files\n")
    
    copied_sessions = 0
    copied_files = 0
    skipped = 0
    
    for session_dir in tqdm(session_dirs, desc="   Progress"):
        session_path = Path(session_dir)
        
        if not session_path.exists():
            print(f"\n   ⚠ Directory not found: {session_path}")
            skipped += 1
            continue
        
        try:
            # Count files before copying
            n_files = count_files_in_directory(session_path)
            
            # Copy entire directory
            if copy_directory_with_structure(session_path, SOURCE_BASE, DEST_BASE):
                copied_sessions += 1
                copied_files += n_files
        
        except Exception as e:
            print(f"\n   ❌ Error copying {session_path.name}: {e}")
            skipped += 1
    
    # Summary
    print(f"\n{'='*80}")
    print(f"COPY COMPLETE!")
    print(f"{'='*80}")
    print(f"\nCopied:")
    print(f"  Session directories: {copied_sessions}")
    print(f"  Total files:         {copied_files}")
    print(f"  Skipped:             {skipped}")
    
    print(f"\nDestination: {DEST_BASE}")
    
    # Verify copied files
    print(f"\n6. Verifying copied data...")
    copied_epochs = list(DEST_BASE.rglob('*_epochs.npy'))
    copied_info = list(DEST_BASE.rglob('*_info.json'))
    copied_labels = list(DEST_BASE.rglob('*_labels.npy'))
    copied_present_ch = list(DEST_BASE.rglob('*_present_channels.npy'))
    copied_present_mask = list(DEST_BASE.rglob('*_present_mask.npy'))
    
    print(f"   Epoch files:           {len(copied_epochs)}")
    print(f"   Info files:            {len(copied_info)}")
    print(f"   Labels files:          {len(copied_labels)}")
    print(f"   Present channels:      {len(copied_present_ch)}")
    print(f"   Present mask:          {len(copied_present_mask)}")
    
    # Calculate size
    print(f"\n7. Calculating copied data size...")
    total_size = sum(f.stat().st_size for f in DEST_BASE.rglob('*') if f.is_file())
    total_size_mb = total_size / (1024**2)
    total_size_gb = total_size / (1024**3)
    
    if total_size_gb >= 1:
        print(f"   Total size: {total_size_gb:.2f} GB")
    else:
        print(f"   Total size: {total_size_mb:.2f} MB")
    
    # Final check
    print(f"\n{'='*80}")
    if len(copied_epochs) == len(balanced_sessions):
        print(f"✅ SUCCESS! All {len(balanced_sessions)} sessions copied correctly!")
    else:
        print(f"⚠️  WARNING: Expected {len(balanced_sessions)} epoch files, found {len(copied_epochs)}")
        print(f"   Some files may have multiple epochs per session (e.g., t000, t001, etc.)")
        print(f"   This is NORMAL if your sessions have multiple EDF files.")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()