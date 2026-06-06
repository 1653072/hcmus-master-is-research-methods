# H-FGAT: Hierarchical Fashion Graph Attention Network

**Original Target**: 16GB RAM machines  
**Updated**: June 6, 2026

---

## 📖 Table of Contents

1. [Project Overview](#project-overview)
2. [Key Features](#key-features)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Model Architecture](#model-architecture)
6. [Training & Output](#training--output)
7. [Inference Example](#inference-example)
8. [Technical Notes](#technical-notes)
9. [For Apple Silicon / high-memory users](#for-apple-silicon-high-memory-users)
10. [File Management & Git LFS](#file-management--git-lfs)

---

## Project Overview

### What is H-FGAT?

This is a rewritten implementation of **Hierarchical Fashion Graph Attention Network** optimized for:
- **Original**: 16GB RAM machines
- **Updated**: 36GB RAM Apple Silicon with Metal Performance Shaders (MPS)

### Objectives of This Rewrite

1. **Preserve Original Pipeline**: Maintain the author's original architecture:
   - Multimodal item embedding (image + text + category)
   - Hierarchical graph construction (item-item, outfit-item, user-outfit)
   - Joint recommendation + outfit compatibility tasks

2. **Data Efficiency**: 
   - Subsample ~30k users initially, keep only relevant outfits and items
   - Only embed images from subsampled set with filename format: `item_id.png`

3. **Code Quality**:
   - Cleaner, modularized code structure
   - Produces complete `model.pt` checkpoint for demo application use

4. **Hardware Optimization**:
   - Original: CPU/CUDA support for 16GB machines
   - Updated: MPS (Metal Performance Shaders) support for Apple M1/M2/M3/M4 chips

---

## Key Features

✅ **Multimodal Embeddings**
- Image features: ResNet152 (deepest network, 2048-dim)
- Text features: BERT-base-Chinese (768-dim)
- Category features: One-hot encoding

✅ **Graph Neural Networks**
- Item-Item graph: Category similarity + outfit co-occurrence
- Outfit-Item graph: Which items compose each outfit
- User-Outfit graph: User-outfit interactions

✅ **Dual Tasks**
- Recommendation: Predict next outfit for user
- Compatibility: Score outfit item combinations

✅ **Hardware Support**
- MPS (Mac M-series GPU) - ⚡ Primary
- CUDA (NVIDIA GPU) - For Colab
- CPU (Fallback) - Works but slower

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
pip install torch torchvision transformers pandas numpy pillow requests torch-geometric torch-scatter streamlit
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
# Open in VS Code or Jupyter
jupyter notebook hfgat_runall_rewrite_validate_fixed.ipynb

# Run cells sequentially (1→33)
```

**Expected duration**: ~6.5-8.5 hours on Apple Silicon / high-memory systems

#### Option B: Python Script

```bash
# Convert notebook to script
jupyter nbconvert --to script hfgat_runall_rewrite_validate_fixed.ipynb

# Run script
python hfgat_runall_rewrite_validate_fixed.py
```

#### Option C: Direct Configuration Override

Edit `Cell 1` of the notebook to customize:

```python
DEVICE = "mps"  # Auto-detected: mps > cuda > cpu
IMAGE_BACKBONE = "resnet152"  # 36GB RAM option
EMBED_DIM = 256  # 36GB RAM option
BATCH_SIZE = 1024  # 36GB RAM option
EPOCHS = 80  # 36GB RAM option
MIN_USER_INTERACTIONS = 5  # Data quality filter
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
    [Concatenate]
        ↓
    Item Fusion MLP (256-dim)
        ↓
    [Item Embeddings: 256-dim]
```

### Hierarchical Graph Attention

```
[Item Embeddings]
        ↓
    [Item-Item Graph]
    Category Edges + Co-occurrence
        ↓
    [Outfit Aggregation]
    Outfit-Item Graph
        ↓
    [User Aggregation]
    User-Outfit Graph
        ↓
┌───────────┬───────────┐
│ RECOMMENDATION │ COMPATIBILITY │
│   Task   │   Task   │
└───────────┴───────────┘
```

### Training Objectives

1. **Recommendation Loss**: BPR-style contrastive learning
   - Positive: user → outfit interaction
   - Negative: random outfit from different user

2. **Compatibility Loss**: Binary classification
   - Positive: items already in same outfit
   - Negative: items from different categories

---

## Training & Output

### Output Files

After training completes, `output_hfgat_notebook/` directory contains:

```
output_hfgat_notebook/
├── model.pt                          # Final trained model (complete checkpoint)
├── best_model.pt                     # Best validation checkpoint
├── best_validation_metrics.json      # Best metrics achieved
├── training_history.csv              # Training loss/metrics per epoch
├── training_history_detailed.csv     # Detailed training statistics
├── validation_report.json            # Final validation report
├── split_stats.json                  # Train/val/test split information
│
├── subsample/                        # Subsampled data
│   ├── item_sub.csv                 # Subsampled items
│   ├── outfit_sub.csv               # Subsampled outfits
│   ├── user_sub.csv                 # Subsampled users
│   ├── train_uo_sub.csv             # Subsampled interactions
│   └── subsample_stats.json         # Subsample statistics
│
├── cache/                            # Feature cache (regenerable)
│   └── item_features.pt             # Cached image/text/category features
│
├── cache_old/                        # Previous cache version
└── exported_embeddings/              # Final embeddings for inference
    ├── user_embeddings.pt           # User embeddings
    ├── item_embeddings.pt           # Item embeddings
    ├── outfit_embeddings.pt         # Outfit embeddings
    ├── user2idx.json                # User ID mappings
    ├── item2idx.json                # Item ID mappings
    ├── outfit2idx.json              # Outfit ID mappings
    └── item_meta_ordered.csv        # Item metadata
```

### Monitoring Training

**Option 1: VS Code Resource Monitor**
- Install "Resource Monitor" extension
- Shows real-time CPU/Memory

**Option 2: Activity Monitor**
```bash
open -a "Activity Monitor"
```

**Option 3: Command Line**
```bash
# Monitor power/memory
watch -n 1 'vm_stat | head -3'

# Apple Silicon specific
powermetrics -s cpu_power -n 1
```

**Expected Resource Usage**:
- Peak Memory: 20-24 GB (out of 36GB available)
- Active Memory: 15-18 GB
- CPU Cores: 8-10 cores (all P-cores)
- GPU (Metal): 70-80% utilization

---

## Inference Example

### Using Pre-trained Model

```python
import torch
from pathlib import Path

# Load model
model_path = Path("output_hfgat_notebook/model.pt")
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

# Extract components
model = checkpoint["model"]
user2idx = checkpoint["user2idx"]
item2idx = checkpoint["item2idx"]
user_embeddings = checkpoint["user_embeddings"]  # Pre-computed

# Recommend outfits for user_id="12345"
user_idx = user2idx["12345"]
user_vec = user_embeddings[user_idx]

# Score outfits
outfit_scores = torch.matmul(user_vec, outfit_embeddings.T)
top_k = torch.topk(outfit_scores, k=10)

print(f"Top 10 recommended outfits: {top_k.indices.tolist()}")
```

### Run Streamlit Demo

```bash
# Run interactive demo application
streamlit run sample_app.py

# Opens http://localhost:8501
```

---

## Technical Notes

### Configuration Options by Hardware

| Parameter | 16GB RAM | 36GB+ RAM | Notes |
|-----------|----------|-----------|-------|
| IMAGE_BACKBONE | resnet50 | resnet152 | Deeper network = better features |
| EMBED_DIM | 128 | 256 | Higher dimension = richer representation |
| HIDDEN_DIM | 128 | 256 | Larger MLP layers |
| IMAGE_BATCH_SIZE | 32 | 64 | Parallel image processing |
| TEXT_BATCH_SIZE | 64 | 128 | Parallel text processing |
| BATCH_SIZE | 512 | 1024 | Training batch size |
| EPOCHS | 30 | 80 | Training iterations |
| MIN_USER_INTERACTIONS | 4 | 5 | Data quality threshold |

### Feature Extraction Models

**Default (Current)**:
- Image: ResNet50 (2048-dim)
- Text: BERT-base-Chinese (768-dim)

**Alternative (high-memory capable)**:
- Image: ResNet152 (2048-dim, deeper)
- Text: BERT-large-Chinese (1024-dim, larger)

### Negative Sampling Strategy

- **Recommendation task**: Random outfit from different user
- **Compatibility task**: Item from different category

### Text Model Selection

- **Chinese titles**: Use `bert-base-chinese` (default)
- **English titles**: Use `bert-base-uncased`
- **Multilingual**: Use `bert-base-multilingual-uncased`

---

## For Apple Silicon / high-memory users

### 🚀 Performance Optimization Highlights

**GPU Acceleration**:
- Automatic MPS detection (Metal Performance Shaders)
- ~10-30x faster than CPU for neural networks
- Seamless fallback if unavailable

**Memory Efficiency**:
- 36GB RAM allows larger batch sizes
- 2x training parallelism (1024 vs 512 batch size)
- 2x embedding dimensions (256 vs 128)
- 2.7x training iterations (80 vs 30 epochs)

**Training Time**:
- **ResNet152 + BATCH_SIZE=1024**: ~18 sec/epoch
- **Total for 80 epochs**: ~6.5-8.5 hours
- **Faster than original**: ~2-3x speedup

### ⚠️ Known MPS Limitations

1. Some sparse tensor operations may fall back to CPU (expected, monitored)
2. Non-deterministic behavior (slight variations between runs)
3. Rare compatibility issues with certain PyTorch operations

**Workaround**: If issues occur, switch to CPU:
```python
DEVICE = "cpu"  # Slower but fully stable
```

### 📊 Recommended Configuration for high-memory systems

**In Cell 1 of notebook**:
```python
DEVICE = "mps"              # Auto-detected
IMAGE_BACKBONE = "resnet152" # Upgrade from resnet50
EMBED_DIM = 256             # Upgrade from 128
HIDDEN_DIM = 256            # Upgrade from 128
IMAGE_BATCH_SIZE = 64       # Upgrade from 32
TEXT_BATCH_SIZE = 128       # Upgrade from 64
BATCH_SIZE = 1024           # Upgrade from 512
EPOCHS = 80                 # Upgrade from 30
```

For detailed hardware optimization guidance, review the configuration examples in this README.

---

## File Management & Git LFS

### ⚠️ Large Files

Several files exceed 50MB and should be tracked with Git LFS:

```
output_hfgat_notebook/cache/item_features.pt           (160 MB)
output_hfgat_notebook/cache_old/item_features.pt       (143 MB)
output_hfgat_notebook/cache_old1/item_features.pt      (143 MB)
output_hfgat_notebook/best_model.pt                    (variable)
output_hfgat_notebook/model.pt                         (variable)
```

### Setup Git LFS

```bash
# Install Git LFS
brew install git-lfs  # Mac
sudo apt install git-lfs  # Linux

# Initialize in repository
cd /path/to/repository
git lfs install

# Track .pt files
git lfs track "*.pt"
git add .gitattributes
git commit -m "Track PyTorch .pt files with Git LFS"
```

For detailed Git LFS commands, use standard Git LFS documentation and repository tracking patterns.

---

## Troubleshooting

| Issue | Cause | Solution |
|-------|-------|----------|
| `ModuleNotFoundError: torch` | PyTorch not installed | `pip install torch torchvision` |
| `MPS not available` | PyTorch too old or Mac OS incompatible | `pip install --upgrade torch` |
| `Out of memory` | Batch sizes too large | Reduce by 50%: `BATCH_SIZE=512`, `IMAGE_BATCH_SIZE=32` |
| `Model not converging` | Learning rate too high | Reduce: `LR=0.00005` |
| `NaN in loss` | Gradient explosion | Reduce embedding dims: `EMBED_DIM=128` |
| `Slow feature extraction` | CPU fallback from MPS | Use `DEVICE="cpu"` for extraction only |

---

## References

- **H-FGAT Paper**: Hierarchical Dependency Attention Network for Fashion Outfit Recommendation
- **PyTorch Documentation**: https://pytorch.org/docs/
- **PyTorch MPS Guide**: https://pytorch.org/docs/stable/notes/mps.html
- **Torch Geometric**: https://pytorch-geometric.readthedocs.io/

---

**Last Updated**: June 6, 2026 | **License**: [See original repository]
