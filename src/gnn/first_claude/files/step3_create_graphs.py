"""
STEP 3: GRAPH CONSTRUCTION (22 CHANNELS)
========================================
Convert extracted features to PyTorch Geometric format with FILE-AWARE splitting.

CRITICAL: For control vs epilepsy, we use FILE-AWARE splitting:
- Train/val/test sets use DIFFERENT files
- This ensures the model generalizes to new recording sessions
- More realistic clinical scenario

Output:
- control_epilepsy_graphs.pt: All graphs
- train_graphs.pt, val_graphs.pt, test_graphs.pt
"""

import numpy as np
import torch
from torch_geometric.data import Data
from pathlib import Path
from tqdm import tqdm

# =============================================================================
# CONFIGURATION
# =============================================================================

FEATURES_FILE = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\control_vs_epilepsy\control_epilepsy_features.npz")
OUTPUT_DIR = Path(r"F:\October-Thesis\thesis-epilepsy-gnn\control_vs_epilepsy\pyg_dataset")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Graph options
THRESHOLD = 0.0       # 0.0 = keep all edges
SELF_LOOPS = False

# Split ratios (FILE-level)
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Reproducibility
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def adjacency_to_edge_index(adj_matrix, threshold=0.0):
    """Convert adjacency matrix to PyG edge_index format."""
    n_nodes = adj_matrix.shape[0]
    
    # Find edges above threshold (excluding diagonal)
    mask = (adj_matrix > threshold) & ~np.eye(n_nodes, dtype=bool)
    
    # Get source and target indices
    source, target = np.where(mask)
    edge_weights = adj_matrix[source, target]
    
    # Convert to tensors
    edge_index = torch.tensor(np.array([source, target]), dtype=torch.long)
    edge_attr = torch.tensor(edge_weights, dtype=torch.float32)
    
    return edge_index, edge_attr


def create_pyg_data(node_features, adjacency, label, threshold=0.0):
    """Create a single PyG Data object."""
    x = torch.tensor(node_features, dtype=torch.float32)
    y = torch.tensor(label, dtype=torch.long)
    edge_index, edge_attr = adjacency_to_edge_index(adjacency, threshold)
    
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y)


def file_aware_split(file_ids, labels, train_ratio=0.7, val_ratio=0.15):
    """
    Split data by FILES (not epochs) to ensure proper generalization.
    
    This is critical for medical ML:
    - Model must generalize to NEW recording sessions
    - Stratified by class to ensure balanced splits
    
    Returns
    -------
    train_idx, val_idx, test_idx : lists of epoch indices
    """
    
    # Get unique files per class
    unique_files = np.unique(file_ids)
    file_to_epochs = {f: np.where(file_ids == f)[0] for f in unique_files}
    
    # Group files by label
    control_files = [f for f in unique_files if labels[file_to_epochs[f][0]] == 0]
    epilepsy_files = [f for f in unique_files if labels[file_to_epochs[f][0]] == 1]
    
    print(f"\nFile-level split:")
    print(f"  Control files:  {len(control_files)}")
    print(f"  Epilepsy files: {len(epilepsy_files)}")
    
    # Shuffle files
    np.random.shuffle(control_files)
    np.random.shuffle(epilepsy_files)
    
    # Split files for each class
    def split_files(files):
        n = len(files)
        n_train = int(train_ratio * n)
        n_val = int(val_ratio * n)
        
        train = files[:n_train]
        val = files[n_train:n_train + n_val]
        test = files[n_train + n_val:]
        
        return train, val, test
    
    control_train, control_val, control_test = split_files(control_files)
    epilepsy_train, epilepsy_val, epilepsy_test = split_files(epilepsy_files)
    
    # Get epoch indices for each split
    def files_to_epoch_indices(file_list):
        indices = []
        for f in file_list:
            indices.extend(file_to_epochs[f].tolist())
        return indices
    
    train_idx = (files_to_epoch_indices(control_train) + 
                 files_to_epoch_indices(epilepsy_train))
    val_idx = (files_to_epoch_indices(control_val) + 
               files_to_epoch_indices(epilepsy_val))
    test_idx = (files_to_epoch_indices(control_test) + 
                files_to_epoch_indices(epilepsy_test))
    
    # Shuffle epoch indices
    np.random.shuffle(train_idx)
    np.random.shuffle(val_idx)
    np.random.shuffle(test_idx)
    
    # Print split statistics
    print(f"\nSplit statistics:")
    print(f"  Train: {len(train_idx)} epochs from {len(control_train) + len(epilepsy_train)} files")
    print(f"    - {len(control_train)} control, {len(epilepsy_train)} epilepsy")
    print(f"  Val:   {len(val_idx)} epochs from {len(control_val) + len(epilepsy_val)} files")
    print(f"    - {len(control_val)} control, {len(epilepsy_val)} epilepsy")
    print(f"  Test:  {len(test_idx)} epochs from {len(control_test) + len(epilepsy_test)} files")
    print(f"    - {len(control_test)} control, {len(epilepsy_test)} epilepsy")
    
    return train_idx, val_idx, test_idx


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    print("="*80)
    print("STEP 3: GRAPH CONSTRUCTION (22 CHANNELS)")
    print("="*80)
    
    # Load features
    print(f"\nLoading features from: {FEATURES_FILE}")
    data = np.load(FEATURES_FILE)
    
    node_features = data['node_features']  # (N, 22, F)
    adjacencies = data['adjacency']         # (N, 22, 22)
    labels = data['labels']                 # (N,)
    file_ids = data['file_ids']             # (N,) - string array
    
    n_samples = len(labels)
    print(f"  Samples: {n_samples:,}")
    print(f"  Node features: {node_features.shape}")
    print(f"  Adjacencies: {adjacencies.shape}")
    print(f"  Labels: Control={np.sum(labels==0):,}, Epilepsy={np.sum(labels==1):,}")
    print(f"  Unique files: {len(np.unique(file_ids))}")
    
    # Create PyG Data objects
    print(f"\nCreating PyG graphs (threshold={THRESHOLD})...")
    pyg_data_list = []
    
    for i in tqdm(range(n_samples), desc="Converting"):
        pyg_data = create_pyg_data(
            node_features[i],
            adjacencies[i],
            labels[i],
            threshold=THRESHOLD
        )
        pyg_data_list.append(pyg_data)
    
    # Sample info
    sample = pyg_data_list[0]
    avg_edges = np.mean([d.edge_index.shape[1] for d in pyg_data_list])
    print(f"\n📊 Graph structure:")
    print(f"  Nodes per graph: {sample.x.shape[0]} (22 EEG channels)")
    print(f"  Features per node: {sample.x.shape[1]}")
    print(f"  Avg edges per graph: {avg_edges:.0f}")
    
    # Save all graphs
    print(f"\nSaving all graphs...")
    torch.save(pyg_data_list, OUTPUT_DIR / 'control_epilepsy_graphs.pt')
    
    # FILE-AWARE SPLIT (critical for medical ML!)
    print(f"\n{'='*80}")
    print("CREATING FILE-AWARE SPLITS")
    print(f"{'='*80}")
    
    train_idx, val_idx, test_idx = file_aware_split(
        file_ids, labels, TRAIN_RATIO, VAL_RATIO
    )
    
    # Create split datasets
    train_data = [pyg_data_list[i] for i in train_idx]
    val_data = [pyg_data_list[i] for i in val_idx]
    test_data = [pyg_data_list[i] for i in test_idx]
    
    # Save splits
    torch.save(train_data, OUTPUT_DIR / 'train_graphs.pt')
    torch.save(val_data, OUTPUT_DIR / 'val_graphs.pt')
    torch.save(test_data, OUTPUT_DIR / 'test_graphs.pt')
    
    # Save indices and file mapping
    np.savez(
        OUTPUT_DIR / 'split_info.npz',
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        file_ids=file_ids,
        train_files=np.unique(file_ids[train_idx]),
        val_files=np.unique(file_ids[val_idx]),
        test_files=np.unique(file_ids[test_idx])
    )
    
    # Summary
    print(f"\n{'='*80}")
    print(f"✅ GRAPH CONSTRUCTION COMPLETE!")
    print(f"{'='*80}")
    
    print(f"\nClass distribution:")
    for split_name, split_data in [("Train", train_data), ("Val", val_data), ("Test", test_data)]:
        n_control = sum(1 for d in split_data if d.y == 0)
        n_epilepsy = sum(1 for d in split_data if d.y == 1)
        total = len(split_data)
        print(f"  {split_name:5s}: {total:5d} epochs | "
              f"Control: {n_control:4d} ({100*n_control/total:5.1f}%) | "
              f"Epilepsy: {n_epilepsy:4d} ({100*n_epilepsy/total:5.1f}%)")
    
    print(f"\n⚠️  IMPORTANT: Files are NOT shared between splits!")
    print(f"   This ensures the model generalizes to NEW recording sessions.")
    
    print(f"\nSaved to: {OUTPUT_DIR}")
    print("\nNext step: Run step4_ssl_pretrain.py")
