# H-FGAT: Hierarchical Fashion Graph Attention Network

**Original Target**: 16GB RAM machines  
**Updated**: June 13, 2026

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)
2. [Recent Training Improvements](#recent-training-improvements)
3. [Key Features](#key-features)
4. [Installation](#installation)
5. [Quick Start](#quick-start)
6. [Model Architecture](#model-architecture)
7. [Training & Output](#training--output)
8. [Evaluation Protocol](#evaluation-protocol)
9. [Inference Example](#inference-example)
10. [Technical Notes](#technical-notes)
11. [For Apple Silicon / high-memory users](#for-apple-silicon-high-memory-users)
12. [File Management & Git LFS](#file-management--git-lfs)
13. [Troubleshooting](#troubleshooting)
14. [References](#references)

---

## Project Overview

### What is H-FGAT?

This is a rewritten implementation of **Hierarchical Fashion Graph Attention Network** optimized for:
- **Original**: 16GB RAM machines
- **Updated**: 36GB RAM Apple Silicon with Metal Performance Shaders (MPS)

The implementation is based on the paper *Hybrid-hierarchical fashion graph attention network for compatibility-oriented and personalized outfit recommendation*, with training behavior aligned to the author's reference notebook after a detailed code comparison (see [`compare_team_and_author/analysis.md`](../compare_team_and_author/analysis.md)).

### Objectives of This Rewrite

1. **Preserve Original Pipeline**: Maintain the author's original architecture:
   - Multimodal item embedding (image + text + category)
   - Hierarchical graph construction (item-item, outfit-item, user-outfit)
   - Joint recommendation + outfit compatibility tasks

2. **Data Efficiency**:
   - Filter users by minimum interaction count; keep only related outfits and items
   - Only embed images from the filtered set with filename format: `item_id.png`

3. **Code Quality**:
   - Cleaner, modularized code structure
   - Produces complete `model.pt` / `best_model.pt` checkpoints for demo application use

4. **Hardware Optimization**:
   - Original: CPU/CUDA support for 16GB machines
   - Updated: MPS (Metal Performance Shaders) support for Apple M1/M2/M3/M4/M5 chips

---

## Recent Training Improvements

As of **June 13, 2026**, `hfgat_runall_rewrite_validate_fixed.ipynb` incorporates fixes from the author-vs-team analysis. These changes address overfitting (train loss dropping while val loss stays flat) and low Precision@10 compared to the paper.

| Fix | Change | Rationale |
|-----|--------|-----------|
| **Eval negatives** | `sampled_negatives=50` (was 99) | Matches author's evaluation protocol for fair metric comparison |
| **Recommendation scorer** | Dot product on L2-normalized embeddings (removed MLP scorer) | Reduces memorization; aligns with author's cosine-similarity design |
| **λ_comp** | `LAMBDA_COMP=0.5` (was 0.1) | Stronger compatibility regularization on item embeddings |
| **Data split** | Random 80/10/10 edge split (was per-user stratified holdout) | Larger, more reliable validation set (~68K val edges vs ~5.8K) |
| **Early stopping** | `PATIENCE=10` on val NDCG@10 | Stops training when validation ranking quality plateaus |
| **LR scheduler** | `ReduceLROnPlateau` (mode=max, factor=0.5, patience=5) | Halves learning rate when NDCG@10 stops improving |
| **Weight decay** | `WEIGHT_DECAY=1e-5` (was 1e-3) | Matches author; avoids crushing large embedding tables |
| **Gradient clipping** | `clip_grad_norm_(max_norm=1.0)` | Stabilizes training with large ResNet152/BERT projections |
| **L2 normalization** | `encode_items` output normalized | Consistent embedding scale across item/outfit/user layers |
| **Per-epoch graph forward** | One full graph propagation per epoch | Faster training; more stable gradients vs per-batch forward |
| **Training negatives** | `NEG_PER_POS=5` (was 1) | Stronger BPR signal; closer to evaluation difficulty |

For the full diff against the author's `fgat-session-3-train-model.ipynb`, see [`compare_team_and_author/analysis.md`](../compare_team_and_author/analysis.md).

---

## Key Features

✅ **Multimodal Embeddings**
- Image features: ResNet152 (2048-dim)
- Text features: BERT-base-Chinese (768-dim)
- Category features: One-hot encoding

✅ **Graph Neural Networks**
- Item-Item graph: Category similarity + outfit co-occurrence (top-10 neighbors per item)
- Outfit-Item graph: Which items compose each outfit
- User-Outfit graph: User-outfit interactions (train-only graph during training to avoid leakage)

✅ **Dual Tasks**
- Recommendation: BPR contrastive learning (user → outfit)
- Compatibility: BPR on outfit item sets (hard same-category negatives)

✅ **Hardware Support**
- MPS (Mac M-series GPU) — primary
- CUDA (NVIDIA GPU) — for Colab
- CPU (fallback) — works but slower

---

## Installation

### Prerequisites
- Python 3.8+
- 36GB+ RAM for high-memory systems (or 16GB+ for regular machines)
- 1TB+ SSD for models and caches

### Step 1: Environment Setup

```bash
# Navigate to project directory
cd /path/to/hfgat_rewrite_validate

# Create virtual environment (recommended)
python -m venv venv_hfgat
source venv_hfgat/bin/activate
```

### Step 2: Install Dependencies

**For Apple Silicon / high-memory systems**:
```bash
# Install PyTorch with MPS support (Mac GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu

# Verify MPS is available
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**For other platforms**:
```bash
# Standard PyTorch (with CUDA for Colab)
pip install torch torchvision torchaudio
```

### Step 3: Install Required Packages

```bash
pip install torch torchvision transformers pandas numpy pillow requests torch-geometric torch-scatter streamlit tqdm
```

### Step 4: Verify Installation

```bash
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'CUDA available: {torch.cuda.is_available()}')
print('✅ Installation successful!')
"
```

---

## Quick Start

### Dataset Preparation

Organize your data in this structure:

```
Dataset/
├── item_data.txt              # Columns: item_id, category, image_url, title
├── outfit_data.txt            # Columns: outfit_id, items (semicolon-separated)
├── user_data.txt              # Columns: user_id, outfits (semicolon-separated)
├── train_uo.txt               # Optional: user-outfit interactions (format: u1 o1 o2 o3)
└── fashion_item_images/       # Item images with naming: item_id.png
    ├── 123.png
    ├── 456.jpg
    └── ...
```

### Running Training

#### Option A: Jupyter Notebook (Interactive)

```bash
jupyter notebook hfgat_runall_rewrite_validate_fixed.ipynb
```

Run all cells from top to bottom. Training metrics are logged each epoch under **TRAIN WITH DETAILED METRICS**.

**Expected duration**: ~5–7 hours on Apple Silicon / high-memory systems (depends on `EPOCHS` and early stopping)

#### Option B: Python Script

```bash
jupyter nbconvert --to script hfgat_runall_rewrite_validate_fixed.ipynb
python hfgat_runall_rewrite_validate_fixed.py
```

#### Option C: Configuration (Cell 2)

Current default hyperparameters in the notebook:

```python
# Device (auto-detected)
DEVICE = "mps"  # mps > cuda > cpu

# Feature extraction
IMAGE_BACKBONE = "resnet152"
TEXT_MODEL_NAME = "bert-base-chinese"
IMAGE_BATCH_SIZE = 64
TEXT_BATCH_SIZE = 128

# Model
EMBED_DIM = 64
DROPOUT = 0.3

# Training (aligned with author comparison fixes)
LR = 0.001
WEIGHT_DECAY = 1e-5
EPOCHS = 50
BATCH_SIZE = 1024
LAMBDA_COMP = 0.5
NEG_PER_POS = 5
PATIENCE = 10

# Data filter
MIN_USER_INTERACTIONS = 4
MIN_TOP_NEIGHBORS = 10
```

---

## Model Architecture

### Multimodal Feature Extraction

```
Input Data
├── Images → ResNet152 → [2048-dim features]
├── Titles → BERT-base-Chinese → [768-dim features]
└── Categories → One-hot encoding → [C-dim features]
        ↓
    [Concatenate + Item Fusion MLP]
        ↓
    [Item Embeddings: 64-dim, L2-normalized]
```

### Hierarchical Graph Encoder (`HFGATLite`)

```
[Item Embeddings]  ← sparse item-item graph (category + co-occurrence)
        ↓
[Outfit Embeddings]  ← sparse outfit-item graph + learnable outfit base
        ↓
[User Embeddings]  ← sparse user-outfit graph (train edges only) + learnable user base
        ↓
┌──────────────────┬──────────────────┐
│  RECOMMENDATION  │  COMPATIBILITY   │
│  dot(u, o)       │  self-attn pool  │
│  (cosine sim)    │  + compat MLP    │
└──────────────────┴──────────────────┘
```

### Training Objectives

1. **Recommendation loss** (BPR):
   - Positive: observed user–outfit edge
   - Negative: `NEG_PER_POS` random outfits not in user's positive set
   - Score: dot product of L2-normalized user and outfit embeddings

2. **Compatibility loss** (BPR):
   - Positive: items in a real outfit
   - Negative: same-category hard negative (one item swapped)
   - Combined: `loss = rec_loss + LAMBDA_COMP * comp_loss`

### Paper Target Metrics (author reference)

From the author's reported results on the reference branch:

| Metric | Target |
|--------|--------|
| HR@10 | 0.4286 |
| Precision@10 | 0.4424 |
| Recall@10 | 0.1580 |
| NDCG@10 | 0.1340 |
| Compat Acc | 0.8956 |

Note: the author's public notebook has a validation bug (`TypeError` on epoch 1); see analysis doc for details. Use `sampled_negatives=50` when comparing metrics.

---

## Training & Output

### Output Files

After training completes, `output_hfgat_notebook/` contains:

```
output_hfgat_notebook/
├── model.pt                          # Final trained model (complete checkpoint)
├── best_model.pt                     # Best validation checkpoint (by NDCG@10)
├── best_validation_metrics.json      # Best metrics achieved
├── training_history_detailed.csv     # Per-epoch train/val loss and ranking metrics
├── split_stats.json                  # Train/val/test split information
│
├── subsample/                        # Filtered data artifacts
│   ├── item_sub.csv
│   ├── outfit_sub.csv
│   ├── user_sub.csv
│   ├── train_uo_sub.csv
│   └── subsample_stats.json
│
└── cache/                            # Feature cache (regenerable)
    └── item_features.pt             # Cached image/text/category features
```

### Per-Epoch Logged Metrics

Each epoch prints:

- `train_loss`, `train_rec_loss`, `train_comp_loss`
- `val_total_loss`, `val_rec_loss`, `val_comp_loss`
- `compat_acc`
- Ranking: `HR@10`, `Precision@10`, `Recall@10`, `NDCG@10`, `MRR@10`, `AUC`
- Current learning rate and early-stop patience counter

### Monitoring Training

**Option 1: VS Code Resource Monitor** — real-time CPU/memory  
**Option 2: Activity Monitor** — `open -a "Activity Monitor"`  
**Option 3: Command line** — `watch -n 1 'vm_stat | head -3'`

**Expected resource usage** (36GB RAM, MPS):
- Peak memory: ~20–24 GB
- GPU (Metal): high utilization during training batches

---

## Evaluation Protocol

To compare fairly with the author's reported numbers:

| Setting | Value | Notes |
|---------|-------|-------|
| Eval negatives | **50** per user | Author uses 50; 99 makes Precision@10 look ~2–3× lower |
| Top-K | 10 | HR@10, Precision@10, Recall@10, NDCG@10 |
| Split | Random **80/10/10** on user–outfit edges | ~543K / ~68K / ~68K edges |
| Train graph | Train edges only | Val/test edges excluded from user–outfit propagation |
| Early stopping | Best checkpoint by **NDCG@10** | Saved to `best_model.pt` |

---

## Inference Example

### Using Pre-trained Model

```python
import torch
from pathlib import Path

checkpoint = torch.load(
    Path("output_hfgat_notebook/best_model.pt"),
    map_location="cpu",
    weights_only=False,
)

model_state = checkpoint["model_state_dict"]
user2idx = checkpoint["mappings"]["user2idx"]
outfit2idx = checkpoint["mappings"]["outfit2idx"]
best_metrics = checkpoint.get("best_metrics", {})

print("Best NDCG@10:", best_metrics.get("NDCG@10"))
```

Recommendation scoring uses dot product on normalized user/outfit embeddings (same as training).

### Run Streamlit Demo

```bash
streamlit run sample_app.py
# Opens http://localhost:8501
```

---

## Technical Notes

### Configuration Options by Hardware

| Parameter | 16GB RAM | 36GB+ RAM (current defaults) | Notes |
|-----------|----------|-------------------------------|-------|
| IMAGE_BACKBONE | resnet50 | resnet152 | Deeper CNN |
| EMBED_DIM | 64–128 | 64 | Compressed embeddings reduce overfitting |
| IMAGE_BATCH_SIZE | 32 | 64 | Feature extraction |
| TEXT_BATCH_SIZE | 64 | 128 | Feature extraction |
| BATCH_SIZE | 512 | 1024 | Training batch size |
| EPOCHS | 30–40 | 50 | Early stopping may finish sooner |
| MIN_USER_INTERACTIONS | 4 | 4 | Users with fewer edges excluded |

### Negative Sampling Strategy

| Task | Strategy |
|------|----------|
| Recommendation (train) | `NEG_PER_POS=5` random outfits per positive |
| Recommendation (eval) | 50 sampled negatives + all val positives per user |
| Compatibility (train) | Hard negative: same-category item swapped into outfit |
| Compatibility (eval) | Held-out outfit split (80/20) |

### Data Split

- **User–outfit edges**: random shuffle, then 80% train / 10% val / 10% test
- **Compatibility outfits**: 80% train / 20% val by outfit index
- **User–outfit graph at train time**: built from **train edges only** (no val/test leakage)

---

## For Apple Silicon / high-memory users

### Performance highlights

- Automatic MPS detection (Metal Performance Shaders)
- Larger batch sizes (1024) and ResNet152 backbone
- Per-epoch graph forward reduces redundant sparse operations vs per-batch forward

### Recommended configuration (Cell 2)

```python
DEVICE = "mps"               # Auto-detected
IMAGE_BACKBONE = "resnet152"
EMBED_DIM = 64
BATCH_SIZE = 1024
EPOCHS = 50
LAMBDA_COMP = 0.5
NEG_PER_POS = 5
WEIGHT_DECAY = 1e-5
PATIENCE = 10
```

### Known MPS limitations

1. Some sparse tensor ops may fall back to CPU
2. Slight non-determinism between runs
3. Rare compatibility issues with certain PyTorch versions

**Workaround**: `DEVICE = "cpu"` for maximum stability (slower).

---

## File Management & Git LFS

### Large files

Track with Git LFS when committing:

```
output_hfgat_notebook/cache/item_features.pt
output_hfgat_notebook/best_model.pt
output_hfgat_notebook/model.pt
```

### Setup Git LFS

```bash
brew install git-lfs   # Mac
git lfs install
git lfs track "*.pt"
git add .gitattributes
```

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch torchvision` |
| `MPS not available` | Old PyTorch or unsupported OS | `pip install --upgrade torch` |
| `Out of memory` | Batch too large | Reduce `BATCH_SIZE` to 512 or 256 |
| Train loss ↓, val loss flat | Overfitting / weak regularization | Confirm `LAMBDA_COMP=0.5`, dot-product scorer, 80/10/10 split |
| Low Precision@10 vs paper | Different eval protocol | Use `sampled_negatives=50`; see analysis doc |
| `NaN in loss` | Gradient explosion | Gradient clipping is enabled (`max_norm=1.0`); try lower `LR` |
| Training stops early | Early stopping triggered | Normal if NDCG@10 plateaus; check `best_model.pt` |
| `verbose` scheduler warning | Older PyTorch | Safe to ignore or remove `verbose=True` from scheduler |

---

## References

- **H-FGAT Paper**: *Hybrid-hierarchical fashion graph attention network for compatibility-oriented and personalized outfit recommendation* — [ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2666827025001859)
- **Author reference code**: [kimnguyen branch](https://github.com/1653072/hcmus-master-is-research-methods/tree/kimnguyen) — `fgat-session-3-train-model.ipynb`
- **Team vs author analysis**: [`compare_team_and_author/analysis.md`](../compare_team_and_author/analysis.md)
- **PyTorch**: https://pytorch.org/docs/
- **PyTorch MPS**: https://pytorch.org/docs/stable/notes/mps.html

---

**Last Updated**: June 13, 2026 | **License**: [See original repository]
