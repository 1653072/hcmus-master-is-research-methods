"""
Shared FGAT / H-HFGAT configuration for active-user notebooks.

Import in each notebook:
    import fgat_config as cfg
    from fgat_config import resolve_paths

Adjust hyperparameters here — all three notebooks read from this file.
"""
from __future__ import annotations

import os
from pathlib import Path

# ── Reproducibility ───────────────────────────────────────────────────
RANDOM_SEED = 42

# ── Data filtering ────────────────────────────────────────────────────
MIN_USER_INTERACTIONS = 4
SPLIT_MODE = "per_user"  # "per_user" | "edge" (edge-level random 80/10/10)

# Item-item top-K sparsification (NB2). None = disabled (full paper Eq.6 graph).
# Set to 10 later to enable quoc-tran-v4 style neighbor cap.
MIN_TOP_NEIGHBORS = None

# ── Feature extraction (Notebook 1) ───────────────────────────────────
FORCE_REBUILD_FEATURES = False
IMAGE_BACKBONE = "resnet152"
TEXT_MODEL_NAME = "bert-base-chinese"
IMAGE_BATCH_SIZE = 64
TEXT_BATCH_SIZE = 128
# None = ad-hoc: no custom cap; tokenizer uses BERT model max length (512 for bert-base-chinese)
MAX_TEXT_LENGTH = None
USE_PRETRAINED_BACKBONES = True

VISUAL_DIM = 2048
TEXT_DIM = 768

# ── Model ─────────────────────────────────────────────────────────────
EMBED_DIM = 64
NUM_HEADS = 4
DROPOUT = 0.3
MAX_OUTFIT_ITEMS_FOR_COMP = 10

# ── Training (Notebook 3) ─────────────────────────────────────────────
EPOCHS = 40
LR = 0.001
WEIGHT_DECAY = 1e-5
LAMBDA_COMP = 0.2
BATCH_SIZE = 512
NEG_PER_POS = 1
PATIENCE = 10
EVAL_EVERY = 1          # evaluate metrics every N epochs (1 = every epoch)
EARLY_STOP_METRIC = "HR@K"  # early-stop checkpoint selection metric
TOP_K = 10
EVAL_NEG_SAMPLES = 50   # negatives per user during eval

# ── Paths (resolved at runtime) ───────────────────────────────────────
KAGGLE_DATA = Path("/kaggle/input/datasets/kiettruonglifeez/recsys-fgat")
KAGGLE_DATASET_URL = "https://www.kaggle.com/datasets/kiettruonglifeez/recsys-fgat"
LOCAL_DATA = Path("hfgat_rewrite_validate") / "Dataset"
LOCAL_OUTPUT = Path("output_fgat_active_user")
KAGGLE_OUTPUT = Path("/kaggle/working")

# Required under DATA_ROOT for Notebook 1 (local / Kaggle)
REQUIRED_RAW_FILES = ("item_data.txt", "outfit_data.txt", "user_data.txt")
OPTIONAL_RAW_FILES = ("train_uo.txt",)  # derived from user_data if absent

OUTPUT_SUBDIRS = (
    "embeddings",
    "matrices",
    "subsample",
    "splits",
    "models",
    "images",
    "cache",
)


def _missing_raw_files(data_root: Path) -> list[str]:
    return [name for name in REQUIRED_RAW_FILES if not (data_root / name).is_file()]


def validate_raw_dataset(data_root: Path) -> None:
    """Raise FileNotFoundError with download hints if metadata .txt files are absent."""
    missing = _missing_raw_files(data_root)
    if not missing:
        return
    has_images = (data_root / "fashion_item_images").is_dir()
    hint = (
        f"\n\nMissing raw metadata under: {data_root}\n"
        f"  missing: {', '.join(missing)}\n"
    )
    if has_images:
        hint += (
            "  note: fashion_item_images/ is present but the .txt files are not.\n"
            "        Images alone are not enough — download the text files too.\n"
        )
    hint += (
        f"\nDownload the Kaggle dataset: {KAGGLE_DATASET_URL}\n"
        "  Place these files directly in DATA_ROOT:\n"
        "    item_data.txt, outfit_data.txt, user_data.txt, train_uo.txt (optional)\n"
        "\nCLI (after `pip install kaggle` and ~/.kaggle/kaggle.json):\n"
        f"  kaggle datasets download -d kiettruonglifeez/recsys-fgat -p /tmp/recsys-fgat --unzip\n"
        f"  cp /tmp/recsys-fgat/*.txt {data_root}/\n"
        "\nOr set FGAT_DATA_ROOT to a folder that already contains the .txt files."
    )
    raise FileNotFoundError(hint)


def resolve_paths(repo_root: Path | None = None) -> dict:
    """Resolve DATA / OUTPUT / CACHE paths for Kaggle or local laptop."""
    repo_root = (repo_root or Path.cwd()).resolve()

    if KAGGLE_DATA.exists():
        data_root = KAGGLE_DATA
        output_root = KAGGLE_OUTPUT
        image_root = output_root / "images"
    else:
        env_data = os.environ.get("FGAT_DATA_ROOT")
        if env_data:
            local_data = Path(env_data).expanduser().resolve()
        else:
            local_data = (repo_root / LOCAL_DATA).resolve()
        if not local_data.is_dir():
            raise FileNotFoundError(
                f"Dataset directory not found: {local_data}\n"
                f"Create it or set FGAT_DATA_ROOT, or use Kaggle path {KAGGLE_DATA}."
            )
        validate_raw_dataset(local_data)
        data_root = local_data
        output_root = repo_root / LOCAL_OUTPUT
        image_root = data_root / "fashion_item_images"

    for sub in OUTPUT_SUBDIRS:
        (output_root / sub).mkdir(parents=True, exist_ok=True)

    cache_dir = output_root / "cache"

    return {
        "REPO_ROOT": repo_root,
        "DATA_ROOT": data_root,
        "OUTPUT_ROOT": output_root,
        "IMAGE_ROOT": image_root,
        "CACHE_DIR": cache_dir,
        "DATA_PATH": str(data_root) + "/",
        "OUTPUT_PATH": str(output_root) + "/",
        "IMAGE_DIR": str(image_root) + "/",
        "FEATURE_CACHE_PATH": cache_dir / "item_features.pt",
        "BEST_STATE_PATH": output_root / "models" / "best_model.pt",
    }


def print_config_summary() -> None:
    """Print a short summary of active hyperparameters."""
    print("── fgat_config ──")
    print(f"  MIN_USER_INTERACTIONS={MIN_USER_INTERACTIONS}  SPLIT_MODE={SPLIT_MODE}")
    print(f"  MIN_TOP_NEIGHBORS={MIN_TOP_NEIGHBORS}  (None=disabled)")
    print(f"  EPOCHS={EPOCHS}  LR={LR}  WEIGHT_DECAY={WEIGHT_DECAY}  LAMBDA_COMP={LAMBDA_COMP}")
    print(f"  BATCH_SIZE={BATCH_SIZE}  NEG_PER_POS={NEG_PER_POS}  PATIENCE={PATIENCE}")
    print(f"  EVAL_EVERY={EVAL_EVERY}  EARLY_STOP_METRIC={EARLY_STOP_METRIC}")
    print(f"  IMAGE_BATCH_SIZE={IMAGE_BATCH_SIZE}  TEXT_BATCH_SIZE={TEXT_BATCH_SIZE}")
    txt_len = "ad-hoc (BERT max)" if MAX_TEXT_LENGTH is None else str(MAX_TEXT_LENGTH)
    print(f"  MAX_TEXT_LENGTH={txt_len}  FORCE_REBUILD_FEATURES={FORCE_REBUILD_FEATURES}")
