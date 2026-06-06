# H-FGAT Rewrite Validate

This folder contains a single-notebook H-FGAT rewrite plus a Streamlit demo app.

## Files

- `hfgat_runall_rewrite_validate_fixed.ipynb`: run-all training/export notebook.
- `sample_app.py`: Streamlit demo for user recommendation, outfit compatibility, and similar outfit lookup.
- `app_portability.py`: shared helpers for app-relative paths, device selection, and compatibility score handling.
- `translate.ipynb`: utility notebook to translate item titles.
- `Dataset/`: metadata files. Image files should be placed in `Dataset/fashion_item_images/`.
- `output_hfgat_notebook/`: model outputs and exported embeddings.

## Data layout

```text
Dataset/
├── item_data.txt
├── item_data_translated.csv
├── outfit_data.txt
├── user_data.txt
├── train_uo.txt                 # optional
└── fashion_item_images/
    ├── <item_id>.png
    ├── <item_id>.jpg
    └── ...
```

## Install

From the repository root:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git lfs install
git lfs pull
```

Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
git lfs install
git lfs pull
```

Use Python 3.10 or 3.11 locally when possible. Python 3.12 can work with recent PyTorch wheels, including current Colab runtimes, but Python 3.11 is usually the safer choice for Mac setup.

Google Colab fresh runtime setup:

```bash
!apt-get update -qq && apt-get install -y -qq git-lfs
!git lfs install
!git clone https://github.com/1653072/hcmus-master-is-research-methods.git
%cd hcmus-master-is-research-methods
!git checkout thienhuy
!git lfs pull
!python -m pip install -r requirements.txt
```

After that, open `hfgat_rewrite_validate/hfgat_runall_rewrite_validate_fixed.ipynb` or run the notebook cells from the checked-out repo. If you are already inside the repo in Colab, only the final two commands are needed.

## Run training/export

Open `hfgat_runall_rewrite_validate_fixed.ipynb` and run from top to bottom. The notebook uses app-local relative paths:

- `Dataset/`
- `output_hfgat_notebook/`

Device selection supports:

- `DEVICE = "auto"`: prefer CUDA, then CPU. This is the safest training default because the notebook uses sparse tensors.
- `DEVICE = "cuda"`: use CUDA if available, otherwise CPU.
- `DEVICE = "mps"`: use Apple Silicon MPS if available, otherwise CPU. Use this only if your local PyTorch build supports the sparse operations used by the notebook. On MacBook Pro M1/M2/M3, sparse tensor operations may still fail on MPS; set `DEVICE = "cpu"` locally or use CUDA in Colab for training.
- `DEVICE = "cpu"`: force CPU.

## Run Streamlit demo

From the repository root:

```bash
streamlit run hfgat_rewrite_validate/sample_app.py
```

The app resolves files relative to its own folder, so it does not depend on the current working directory.

## Notes

- If `.pt` files are tiny text files beginning with `version https://git-lfs.github.com/spec/v1`, run `git lfs pull`.
- If `torch.load` fails with `invalid load key, 'v'` or an unpickling error, check the `.pt` file contents. That usually means Git LFS left a pointer file instead of downloading the real tensor/checkpoint.
- If PyTorch install fails on Windows or CUDA machines, install the correct PyTorch build from https://pytorch.org/get-started/locally/, then rerun `python -m pip install -r requirements.txt`.
- The Streamlit app can load checkpoints whose compatibility head returns either one logit or multiple view logits.
