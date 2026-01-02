"""
Copy Selected RAW EDF Files to Balanced Dataset Folder
=======================================================

This script:
1. Reads balanced_epilepsy_edf_files.txt (list of selected files)
2. Copies each EDF file to a new directory
3. Preserves directory structure (patient/session/montage)

This creates a clean folder with ONLY the balanced dataset files!
"""

from pathlib import Path
import shutil
from tqdm import tqdm
import json

# ============================================================================
# CONFIGURATION
# ============================================================================

# Input: List of selected EDF files
BALANCED_LIST = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\analysis_results\balanced_epilepsy_edf_files.txt')

# Source base directory
SOURCE_BASE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_raw\DATA')

# Destination directory (where to copy files)
DEST_BASE = Path(r'F:\October-Thesis\thesis-epilepsy-gnn\data_raw_balanced\DATA')

# ============================================================================
# FUNCTIONS
# ============================================================================

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


def copy_file_with_structure(src_file, src_base, dest_base):
    """
    Copy file preserving directory structure.
    
    Example:
        src_file:  F:/data_raw/DATA/00_epilepsy/aaaaaanr/s003/01_tcp_ar/file.edf
        src_base:  F:/data_raw/DATA
        dest_base: F:/data_raw/DATA_balanced
        result:    F:/data_raw/DATA_balanced/00_epilepsy/aaaaaanr/s003/01_tcp_ar/file.edf
    """
    # Get relative path
    rel_path = get_relative_path(src_file, src_base)
    
    # Create destination path
    dest_file = dest_base / rel_path
    
    # Create parent directory if needed
    dest_file.parent.mkdir(parents=True, exist_ok=True)
    
    # Copy file
    if src_file.exists():
        shutil.copy2(src_file, dest_file)  # copy2 preserves metadata
        return True
    return False


def get_file_size(path):
    """Get file size in bytes."""
    if path.exists():
        return path.stat().st_size
    return 0


def format_size(bytes):
    """Format bytes to human-readable size."""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes < 1024.0:
            return f"{bytes:.2f} {unit}"
        bytes /= 1024.0
    return f"{bytes:.2f} TB"


def main():
    print("="*80)
    print("COPYING SELECTED RAW EDF FILES TO BALANCED DATASET FOLDER")
    print("="*80)
    
    # Load file list
    print(f"\n1. Loading file list from: {BALANCED_LIST}")
    if not BALANCED_LIST.exists():
        print(f"   ✗ ERROR: List file not found!")
        print(f"   Please run select_balanced_sessions_raw.py first.")
        return
    
    with open(BALANCED_LIST, 'r') as f:
        file_list = [line.strip() for line in f if line.strip()]
    
    print(f"   ✓ Loaded {len(file_list)} files")
    
    # Check source base
    print(f"\n2. Source directory: {SOURCE_BASE}")
    if not SOURCE_BASE.exists():
        print(f"   ✗ ERROR: Source directory not found!")
        return
    print(f"   ✓ Directory found")
    
    # Create destination directory
    print(f"\n3. Destination directory: {DEST_BASE}")
    DEST_BASE.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Directory created/verified")
    
    # Copy files
    print(f"\n4. Copying files...")
    print(f"   This preserves the directory structure:")
    print(f"   Source:      {SOURCE_BASE}")
    print(f"   Destination: {DEST_BASE}\n")
    
    copied_files = 0
    failed_files = 0
    total_size = 0
    
    for file_path_str in tqdm(file_list, desc="   Progress"):
        file_path = Path(file_path_str)
        
        try:
            # Get file size before copying
            file_size = get_file_size(file_path)
            
            # Copy file
            if copy_file_with_structure(file_path, SOURCE_BASE, DEST_BASE):
                copied_files += 1
                total_size += file_size
            else:
                print(f"\n   ⚠ File not found: {file_path.name}")
                failed_files += 1
        
        except Exception as e:
            print(f"\n   ✗ Error copying {file_path.name}: {e}")
            failed_files += 1
    
    # Summary
    print(f"\n{'='*80}")
    print("COPY COMPLETE!")
    print(f"{'='*80}")
    print(f"\nCopied files:     {copied_files}")
    print(f"Failed files:     {failed_files}")
    print(f"Total size:       {format_size(total_size)}")
    print(f"")
    print(f"Destination: {DEST_BASE}")
    
    # Verify structure
    print(f"\n5. Verifying copied structure...")
    copied_edf_files = list(DEST_BASE.rglob('*.edf'))
    print(f"   ✓ Found {len(copied_edf_files)} EDF files in destination")
    
    # Count by group
    epilepsy_files = [f for f in copied_edf_files if '00_epilepsy' in str(f)]
    control_files = [f for f in copied_edf_files if '01_no_epilepsy' in str(f)]
    
    print(f"   ✓ Epilepsy files: {len(epilepsy_files)}")
    print(f"   ✓ Control files:  {len(control_files)}")
    
    # Count patients
    patients = set()
    for edf_file in copied_edf_files:
        parts = edf_file.parts
        for i, part in enumerate(parts):
            if part in ['00_epilepsy', '01_no_epilepsy']:
                if i + 1 < len(parts):
                    patients.add(parts[i + 1])
    
    print(f"   ✓ Total patients: {len(patients)}")
    
    # Count sessions
    sessions = set()
    for edf_file in copied_edf_files:
        parts = edf_file.parts
        for part in parts:
            if part.startswith('s') and '_' in part:
                sessions.add(part)
    
    print(f"   ✓ Total sessions: {len(sessions)}")
    
    # Save statistics
    print(f"\n6. Saving copy statistics...")
    stats_file = DEST_BASE / 'copy_statistics.json'
    stats = {
        'timestamp': str(Path(__file__).stat().st_mtime),
        'source_base': str(SOURCE_BASE),
        'dest_base': str(DEST_BASE),
        'total_files_copied': copied_files,
        'total_files_failed': failed_files,
        'total_size_bytes': total_size,
        'total_size_formatted': format_size(total_size),
        'total_patients': len(patients),
        'total_sessions': len(sessions),
        'epilepsy_files': len(epilepsy_files),
        'control_files': len(control_files)
    }
    
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    print(f"   ✓ Saved: {stats_file}")
    
    # Final check
    print(f"\n{'='*80}")
    if copied_files == len(file_list):
        print(f"✅ SUCCESS! All {len(file_list)} files copied successfully!")
    else:
        print(f"⚠ WARNING: Expected {len(file_list)} files, copied {copied_files}")
        if failed_files > 0:
            print(f"   {failed_files} files failed to copy")
    print(f"{'='*80}")
    
    print(f"\nNext step:")
    print(f"  Run preprocessing on: {DEST_BASE}")


if __name__ == "__main__":
    main()