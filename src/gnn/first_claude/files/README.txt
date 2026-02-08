"""
╔════════════════════════════════════════════════════════════════════════════╗
║                                                                            ║
║     CONTROL VS EPILEPSY DETECTION (ADAPTED FOR YOUR PRE-COMPUTED DATA)    ║
║                                                                            ║
║  Graph-based Self-Supervised Learning for Epilepsy Detection             ║
║  Using Pre-Computed PDC Connectivity (22 Channels, 250 Hz)               ║
║                                                                            ║
╚════════════════════════════════════════════════════════════════════════════╝

═══════════════════════════════════════════════════════════════════════════
OVERVIEW
═══════════════════════════════════════════════════════════════════════════

This pipeline is ADAPTED for your specific setup:
✓ Pre-computed PDC connectivity (.npz files)
✓ 22 EEG channels (not 19)
✓ 250 Hz sampling rate (not 256)
✓ Directory structure: 00_epilepsy, 01_control
✓ Connectivity already computed with fixed order p=15

YOUR DATA STRUCTURE:
--------------------
connectivity/january_fixed_15/
├── 00_epilepsy/
│   └── patient_id/session/.../xxx_graphs.npz
└── 01_control/
    └── patient_id/session/.../xxx_graphs.npz

Each .npz file contains:
  - pdc_integrated, dtf_integrated
  - pdc_delta, pdc_theta, pdc_alpha, pdc_beta, pdc_gamma1, pdc_gamma2
  - labels, indices
  - Shape: (n_epochs, 22, 22)


═══════════════════════════════════════════════════════════════════════════
PIPELINE STEPS
═══════════════════════════════════════════════════════════════════════════

STEP 1: VALIDATE DATA
----------------------
python step1_validate_data.py

What it does:
- Scans your connectivity directory
- Identifies control vs epilepsy files
- Validates file structure
- Counts total epochs
- Generates file mapping

Expected output:
✓ Data validation report
✓ file_mapping.json


STEP 2: EXTRACT FEATURES
-------------------------
python step2_extract_features.py

What it does:
- Loads pre-computed PDC matrices from .npz files
- Extracts connectivity features (4 per channel):
  • out_strength, in_strength
  • out_degree, in_degree
- Optionally adds spectral/statistical features from epoch files
  • Spectral: 6 features (power bands)
  • Statistical: 5 features (mean, std, skew, kurt, line_length)
- Total: 4-15 features per channel depending on availability

Expected output:
✓ control_epilepsy_features.npz
✓ normalization_params.npz

Time: ~5-15 minutes


STEP 3: CREATE GRAPHS
----------------------
python step3_create_graphs.py

What it does:
- Converts features to PyTorch Geometric format
- Each epoch becomes a graph with 22 nodes
- FILE-AWARE splitting (critical!):
  • Different files in train/val/test
  • Ensures generalization to new sessions

Expected output:
✓ control_epilepsy_graphs.pt
✓ train_graphs.pt, val_graphs.pt, test_graphs.pt
✓ split_info.npz

Time: ~1-5 minutes


STEP 4: SSL PRE-TRAINING
-------------------------
python step4_ssl_pretrain.py

What it does:
- Self-supervised learning via contrastive loss
- Learns EEG patterns WITHOUT labels
- Creates robust embeddings

Expected output:
✓ pretrained_encoder.pt
✓ ssl_training_loss.png

Time: ~10-30 minutes (100 epochs)
GPU recommended but not required


STEP 5: TRAIN CLASSIFIER
-------------------------
python step5_train_classifier.py

What it does:
- Loads pre-trained encoder
- Adds classification head
- Trains for control vs epilepsy
- Evaluates on test set

Expected output:
✓ best_classifier.pt
✓ test_predictions.npz
✓ training_history.png
✓ confusion_matrix.png
✓ final_metrics.json

Time: ~5-15 minutes

THIS IS YOUR MAIN RESULT!


═══════════════════════════════════════════════════════════════════════════
KEY ADAPTATIONS FOR YOUR DATA
═══════════════════════════════════════════════════════════════════════════

1. PRE-COMPUTED CONNECTIVITY
   - No need to recalculate PDC
   - Directly loads from your .npz files
   - Uses 'pdc_integrated' by default (can change to specific bands)

2. 22 CHANNELS (not 19)
   - All scripts adapted for 22-channel EEG
   - Graph nodes: 22 (not 19)
   - Adjacency matrices: 22×22

3. 250 Hz SAMPLING
   - Frequency bands adjusted for 250 Hz
   - Welch PSD parameters optimized

4. DIRECTORY STRUCTURE
   - Automatically detects 00_epilepsy vs 01_control
   - Configurable patterns in step1_validate_data.py

5. FILE-AWARE SPLITTING
   - Uses file IDs instead of subject IDs
   - Ensures different recording sessions in train/test
   - More realistic evaluation


═══════════════════════════════════════════════════════════════════════════
CONFIGURATION
═══════════════════════════════════════════════════════════════════════════

UPDATE THESE PATHS IN EACH SCRIPT:
-----------------------------------

step1_validate_data.py:
  CONNECTIVITY_DIR = Path(r"F:\October-Thesis\...\connectivity\january_fixed_15")
  EPOCHS_DIR = Path(r"F:\October-Thesis\...\data_pp_balanced")
  OUTPUT_DIR = Path(r"F:\October-Thesis\...\control_vs_epilepsy")

step2_extract_features.py:
  CONNECTIVITY_DIR = Path(r"F:\October-Thesis\...\connectivity\january_fixed_15")
  EPOCHS_DIR = Path(r"F:\October-Thesis\...\data_pp_balanced")
  OUTPUT_DIR = Path(r"F:\October-Thesis\...\control_vs_epilepsy")
  
  # Feature options
  USE_SPECTRAL_FEATURES = True   # Requires epoch files
  USE_CONNECTIVITY_FEATURES = True
  PDC_BAND = 'pdc_integrated'    # Or: pdc_alpha, pdc_beta, etc.

step3_create_graphs.py:
  FEATURES_FILE = Path(r"...\control_vs_epilepsy\control_epilepsy_features.npz")
  OUTPUT_DIR = Path(r"...\control_vs_epilepsy\pyg_dataset")

step4_ssl_pretrain.py:
  GRAPHS_FILE = Path(r"...\pyg_dataset\control_epilepsy_graphs.pt")
  OUTPUT_DIR = Path(r"...\ssl_pretrained")

step5_train_classifier.py:
  DATA_DIR = Path(r"...\pyg_dataset")
  SSL_DIR = Path(r"...\ssl_pretrained")
  OUTPUT_DIR = Path(r"...\classifier")


═══════════════════════════════════════════════════════════════════════════
FEATURE OPTIONS
═══════════════════════════════════════════════════════════════════════════

In step2_extract_features.py, you can configure:

CONNECTIVITY ONLY (4 features per channel):
  USE_CONNECTIVITY_FEATURES = True
  USE_SPECTRAL_FEATURES = False
  EPOCHS_DIR = None

CONNECTIVITY + SPECTRAL (15 features per channel):
  USE_CONNECTIVITY_FEATURES = True
  USE_SPECTRAL_FEATURES = True
  EPOCHS_DIR = Path(r"...\data_pp_balanced")

Which PDC band to use:
  PDC_BAND = 'pdc_integrated'  # Default - all frequencies
  PDC_BAND = 'pdc_alpha'       # Alpha band only
  PDC_BAND = 'pdc_beta'        # Beta band only
  etc.

Recommendation: Start with connectivity only (faster), then try adding spectral.


═══════════════════════════════════════════════════════════════════════════
EXPECTED RESULTS
═══════════════════════════════════════════════════════════════════════════

Based on your pre-ictal/ictal work (79.5% acc, 86.3% AUC):

GOOD RESULTS:
- Test Accuracy: 75-90%
- Test AUC: 80-95%
- Train-Val Gap: <10%

WHAT MATTERS:
- High AUC (discriminative ability)
- Low generalization gap (not overfitting)
- Balanced confusion matrix


═══════════════════════════════════════════════════════════════════════════
COMPARISON WITH PRE-ICTAL/ICTAL WORK
═══════════════════════════════════════════════════════════════════════════

┌─────────────────────┬────────────────────┬──────────────────────────┐
│  Aspect             │  Pre-ictal/Ictal   │  Control/Epilepsy (THIS) │
├─────────────────────┼────────────────────┼──────────────────────────┤
│  Task               │  Seizure timing    │  Patient diagnosis       │
│  Channels           │  19                │  22 ✓                    │
│  Sampling rate      │  256 Hz            │  250 Hz ✓                │
│  Connectivity       │  Computed on-fly   │  Pre-computed ✓          │
│  Subjects           │  Epileptic only    │  Healthy + Epileptic ✓   │
│  Thesis alignment   │  Partial           │  Perfect ✓               │
│  Clinical relevance │  Seizure warning   │  Diagnosis ✓             │
└─────────────────────┴────────────────────┴──────────────────────────┘


═══════════════════════════════════════════════════════════════════════════
TROUBLESHOOTING
═══════════════════════════════════════════════════════════════════════════

PROBLEM: "No files found in 00_epilepsy"
SOLUTION: Check EPILEPSY_DIR_PATTERN and CONTROL_DIR_PATTERN in step1

PROBLEM: "pdc_integrated not found"
SOLUTION: Check available bands with print(conn_data.files)
          Update PDC_BAND in step2

PROBLEM: "Epoch/connectivity mismatch"
SOLUTION: This is OK - the code uses 'indices' from .npz to align
          Will skip mismatched epochs

PROBLEM: "CUDA out of memory"
SOLUTION: Reduce BATCH_SIZE in step4 and step5
          Or use CPU: DEVICE = torch.device('cpu')

PROBLEM: Low accuracy (<70%)
SOLUTION: - Try different PDC_BAND (alpha, beta often work well)
          - Add spectral features (USE_SPECTRAL_FEATURES=True)
          - Longer SSL pre-training (200 epochs)

PROBLEM: High train-val gap (>15%)
SOLUTION: - Increase dropout (0.6 instead of 0.5)
          - More data augmentation
          - Reduce model size


═══════════════════════════════════════════════════════════════════════════
QUICK START
═══════════════════════════════════════════════════════════════════════════

1. Update paths in ALL scripts (step1 through step5)

2. Run validation:
   python step1_validate_data.py

3. If validation passes, run pipeline:
   python step2_extract_features.py  # ~10 min
   python step3_create_graphs.py     # ~2 min
   python step4_ssl_pretrain.py      # ~20 min
   python step5_train_classifier.py  # ~10 min

4. Check results:
   - Look at: control_vs_epilepsy/classifier/final_metrics.json
   - View plots in: control_vs_epilepsy/classifier/

5. Compare approaches:
   - Connectivity only vs Connectivity + Spectral
   - Different PDC bands (integrated, alpha, beta)
   - With vs without SSL (USE_PRETRAINED=False)

Total time: ~1 hour


═══════════════════════════════════════════════════════════════════════════
THESIS PRESENTATION
═══════════════════════════════════════════════════════════════════════════

You now have TWO complete tasks:

1. Pre-ictal vs Ictal (COMPLETED)
   - 79.5% accuracy, 86.3% AUC
   - Seizure prediction application

2. Control vs Epilepsy (THIS PIPELINE)
   - TBD - run and see!
   - True epilepsy detection
   - Aligns perfectly with thesis title

Recommended thesis structure:
1. Intro: Graph-based SSL for epilepsy analysis
2. Method: GNN + SSL architecture (same for both tasks)
3. Task 1: Control vs Epilepsy (MAIN contribution)
4. Task 2: Pre-ictal vs Ictal (shows versatility)
5. Conclusion: SSL helps, GNNs work for EEG


═══════════════════════════════════════════════════════════════════════════
ADVANTAGES OF THIS ADAPTED PIPELINE
═══════════════════════════════════════════════════════════════════════════

✓ Uses your existing PDC computations (no redundant work)
✓ Handles 22 channels correctly
✓ Respects 250 Hz sampling rate
✓ File-aware splitting for proper evaluation
✓ Can use connectivity only (fast) or add spectral (better)
✓ Multiple PDC bands available for experimentation
✓ Same proven GNN+SSL architecture
✓ Perfect thesis alignment


═══════════════════════════════════════════════════════════════════════════
GOOD LUCK!
═══════════════════════════════════════════════════════════════════════════

Run the pipeline, get your results, and write your thesis! 🎓

Questions? Check the comments in each script for details.
"""