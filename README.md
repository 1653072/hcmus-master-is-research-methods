# HCMUS Master IS - Research Methods Course

Repository for the Research Methods course, used to store the H-FGAT fashion recommendation experiment and demo app.

## Current layout

```text
hfgat_rewrite_validate/
├── Dataset/
│   ├── item_data.txt
│   ├── item_data_translated.csv
│   ├── outfit_data.txt
│   └── user_data.txt
├── hfgat_runall_rewrite_validate_fixed.ipynb
├── output_hfgat_notebook/
├── sample_app.py
└── translate.ipynb
```

The previous nested folder `hfgat_rewrite_validate/hfgat_rewrite_validate/` has been removed on branch `thienhuy`.

## Large files

Model and tensor files are tracked with Git LFS:

```text
hfgat_rewrite_validate/output_hfgat_notebook/**/*.pt
```

Install and pull LFS files after cloning:

```bash
git lfs install
git lfs pull
```

Install Git LFS if needed:

- macOS: `brew install git-lfs`
- Windows: `choco install git-lfs` or `winget install GitHub.GitLFS`
- Ubuntu/Colab: `sudo apt-get install git-lfs`

If a `.pt` file opens as a small text file starting with `version https://git-lfs.github.com/spec/v1`, the real binary has not been pulled yet. PyTorch may then fail with errors such as `invalid load key, 'v'` or an unpickling error. From the repo root, run:

```bash
git lfs install
git lfs pull
git lfs ls-files
```

## Environment

Use Python 3.10 or 3.11 for the smoothest local setup. Python 3.12 can work with current PyTorch and Colab runtimes, but some ML packages publish wheels later for new Python releases. If installation fails on Python 3.12, create a Python 3.11 environment and reinstall.

Create a virtual environment, then install dependencies:

```bash
python3.11 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For GPU-specific PyTorch builds, follow the selector at https://pytorch.org/get-started/locally/ before installing the rest of the requirements.

Apple Silicon note for MacBook Pro M1/M2/M3: the Streamlit app can run on CPU or MPS, but the training notebook uses sparse tensor operations that may not be fully supported by PyTorch MPS. If you see an MPS sparse tensor error, set the notebook device to CPU locally, or use a CUDA GPU runtime in Google Colab for training.

## Run the demo app

The app now resolves `Dataset/` and `output_hfgat_notebook/` relative to `sample_app.py`, so it can be launched from the repo root:

```bash
streamlit run hfgat_rewrite_validate/sample_app.py
```

or from inside `hfgat_rewrite_validate/`:

```bash
streamlit run sample_app.py
```

## Google Colab quick start

```bash
!apt-get update -qq && apt-get install -y -qq git-lfs
!git lfs install
!git clone https://github.com/1653072/hcmus-master-is-research-methods.git
%cd hcmus-master-is-research-methods
!git checkout thienhuy
!git lfs pull
!python -m pip install -r requirements.txt
```

Then open `hfgat_rewrite_validate/hfgat_runall_rewrite_validate_fixed.ipynb` or run its cells after setting the runtime to GPU when available. If a `.pt` artifact still looks like a Git LFS pointer file, rerun `!git lfs pull` from `/content/hcmus-master-is-research-methods`.
