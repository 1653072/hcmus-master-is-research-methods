# HCMUS Master IS - Research Methods Course

**Description**: Repository for Research Methods course, storing codebase of AI models and deep learning research.

**Last Updated**: June 6, 2026

---

## 📂 Repository Structure

```
hcmus-master-is-research-methods/
├── hfgat_rewrite_validate/                 ← H-FGAT Project (Main focus)
│   ├── hfgat_runall_rewrite_validate_fixed.ipynb
│   ├── sample_app.py
│   ├── Dataset/
│   └── output_hfgat_notebook/
├── README.md                               ← Source of truth for repository documentation
└── [Other research projects]
```

---

## 🎯 Main Project: H-FGAT (Hierarchical Fashion Graph Attention Network)

### What is H-FGAT?

H-FGAT is a fashion outfit recommendation model using:
- **Multimodal Embeddings**: images + text + category
- **Graph Neural Networks**: item-item, outfit-item, user-outfit relationships
- **Dual tasks**: recommendation and outfit compatibility

### Rewrite Goals

This project is rewritten to:
- preserve the original pipeline layout
- subsample users and keep only relevant outfits and items
- embed only images present in the subsample set
- generate a complete `model.pt` checkpoint for demo use

### Hardware Optimization

- **16GB target**: ResNet50 + DistilBERT-style text encoder
- **36GB+ RAM target**: ResNet152 + larger batch sizes
- **Device support**: MPS (Mac) > CUDA (Colab) > CPU (fallback)

---

## 📚 What’s Included in This README

- Project overview and quick start
- Installation instructions
- Apple Silicon / MPS optimization notes
- Training and inference guidance
- Git LFS and large-file management
- Performance and hardware considerations
- Troubleshooting guidance

---

## Installation

### Prerequisites

- Python 3.8+
- 36GB+ RAM recommended for high-memory training
- 1TB+ SSD for checkpoints and cache

### Environment Setup

```bash
cd hfgat_rewrite_validate
python -m venv venv_hfgat
source venv_hfgat/bin/activate
```

### Install Dependencies

**Apple Silicon / MPS**:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

**Other platforms**:

```bash
pip install torch torchvision torchaudio
```

### Install Project Packages

```bash
pip install transformers pandas numpy pillow requests torch-geometric torch-scatter streamlit
```

### Verify Installation

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'MPS available: {torch.backends.mps.is_available()}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

---

## Quick Start

### Prepare the Dataset

Place these files under `hfgat_rewrite_validate/Dataset/`:

- `item_data.txt`
- `outfit_data.txt`
- `user_data.txt`
- `train_uo.txt` (optional)
- `fashion_item_images/` with image files named `item_id.png`

### Run Training

#### Option A: Notebook

```bash
cd hfgat_rewrite_validate
jupyter notebook hfgat_runall_rewrite_validate_fixed.ipynb
```

#### Option B: Python Script

```bash
cd hfgat_rewrite_validate
jupyter nbconvert --to script hfgat_runall_rewrite_validate_fixed.ipynb
python hfgat_runall_rewrite_validate_fixed.py
```

### Recommended Configuration

**High-memory (36GB+ RAM)**:

```python
DEVICE = "mps"
IMAGE_BACKBONE = "resnet152"
EMBED_DIM = 256
HIDDEN_DIM = 256
IMAGE_BATCH_SIZE = 64
TEXT_BATCH_SIZE = 128
BATCH_SIZE = 1024
EPOCHS = 80
MIN_USER_INTERACTIONS = 5
```

**Standard Machine (16GB RAM)**:

```python
DEVICE = "cuda" or "cpu"
IMAGE_BACKBONE = "resnet50"
EMBED_DIM = 128
HIDDEN_DIM = 128
IMAGE_BATCH_SIZE = 32
TEXT_BATCH_SIZE = 64
BATCH_SIZE = 512
EPOCHS = 30
MIN_USER_INTERACTIONS = 4
```

---

## Model Architecture

### Multimodal Feature Extraction

- Images → ResNet152 or ResNet50
- Text → BERT-base-Chinese (or language-appropriate model)
- Category → one-hot encoding

### Graph Structure

- Item-Item graph: category similarity + co-occurrence
- Outfit-Item graph: outfit item membership
- User-Outfit graph: user interaction edges

### Training Objectives

- Recommendation task: positive user-outfit pairs vs negative outfits
- Compatibility task: item pairs from same outfit vs different item category

---

## Training Output

Expected output folder: `hfgat_rewrite_validate/output_hfgat_notebook/`

Contains:

- `model.pt`
- `best_model.pt`
- `best_validation_metrics.json`
- `training_history.csv`
- `validation_report.json`
- `split_stats.json`
- `subsample/`
- `cache/`
- `exported_embeddings/`

---

## Inference Example

```python
import torch
from pathlib import Path

checkpoint = torch.load('hfgat_rewrite_validate/output_hfgat_notebook/model.pt', map_location='cpu')
# load model or embeddings from checkpoint as needed
```

Run the demo app:

```bash
cd hfgat_rewrite_validate
streamlit run sample_app.py
```

---

## Technical Notes

- Default image model for 16GB is `resnet50`
- Better quality on 36GB is `resnet152`
- Default text encoder for Chinese titles is `bert-base-chinese`
- If titles are in another language, choose a language-appropriate text model
- Negative sampling currently uses random outfits for recommendation and category-based item replacement for compatibility

---

## Git & Large File Management

### Large Files

Track these `.pt` files with Git LFS:

- `hfgat_rewrite_validate/output_hfgat_notebook/cache/item_features.pt`
- `hfgat_rewrite_validate/output_hfgat_notebook/cache_old/item_features.pt`
- `hfgat_rewrite_validate/output_hfgat_notebook/cache_old1/item_features.pt`
- `hfgat_rewrite_validate/output_hfgat_notebook/model.pt`
- `hfgat_rewrite_validate/output_hfgat_notebook/best_model.pt`

### Setup Git LFS

```bash
git lfs install
git lfs track "*.pt"
git add .gitattributes
git commit -m "Track PyTorch .pt files with Git LFS"
```

Use Git LFS cleanup commands and tracking guidance as needed from the repository context.

---

## Performance Overview

### Training Speed (high-memory / Apple Silicon)

| Configuration | Time per Epoch | 80 Epochs |
|---------------|----------------|-----------|
| ResNet152 + BATCH_SIZE=1024 | ~18 sec | ~6.5 hours |
| ResNet152 + BATCH_SIZE=512  | ~12 sec | ~4.5 hours |
| ResNet50 + BATCH_SIZE=512   | ~8 sec  | ~3 hours |

### Memory Usage

- Feature extraction: 18-22 GB
- Graph construction: 16-18 GB
- Training loops: 14-16 GB
- Total safe margin: <24 GB out of 36 GB

---

## Important Notes

- Verify `hfgat_rewrite_validate/Dataset/` exists and contains all required files
- Make sure `Dataset/fashion_item_images/` contains item image files
- Confirm MPS availability with `python -c "import torch; print(torch.backends.mps.is_available())"`
- If MPS causes issues, set `DEVICE = "cpu"`

---

## Resources

- PyTorch MPS: https://pytorch.org/docs/stable/notes/mps.html
- Torch Geometric: https://pytorch-geometric.readthedocs.io/
- Apple machine learning: https://developer.apple.com/machine-learning/

---

## Project Status

- ✅ `README.md` is now the source of truth
- ✅ H-FGAT documentation is consolidated here
- ✅ Hardware optimization notes and Git LFS guidance are included in this README

**Next step**: `cd hfgat_rewrite_validate && jupyter notebook hfgat_runall_rewrite_validate_fixed.ipynb`


