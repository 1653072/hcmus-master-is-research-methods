from dataclasses import dataclass
from math import exp
from pathlib import Path
import re
from types import SimpleNamespace
from typing import Iterable, Optional, Sequence

_GIT_LFS_POINTER_VERSION = "version https://git-lfs.github.com/spec/v1"
_ITEM_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_IMAGE_EXTENSION_PATTERN = re.compile(r"^\.[A-Za-z0-9]+$")


@dataclass(frozen=True)
class AppPaths:
    app_dir: Path
    artifact_dir: Path
    export_dir: Path
    image_dir: Path


def build_paths(app_dir: Path) -> AppPaths:
    app_dir = Path(app_dir).expanduser()
    if not app_dir.is_absolute():
        app_dir = Path.cwd() / app_dir
    artifact_dir = app_dir / "output_hfgat_notebook"
    return AppPaths(
        app_dir=app_dir,
        artifact_dir=artifact_dir,
        export_dir=artifact_dir / "exported_embeddings",
        image_dir=app_dir / "Dataset" / "fashion_item_images",
    )


def is_git_lfs_pointer_file(path: Path) -> bool:
    path = Path(path)
    if not path.is_file():
        return False

    try:
        content = path.read_bytes()[:512].decode("utf-8")
    except (OSError, UnicodeDecodeError):
        return False

    lines = content.splitlines()
    return (
        len(lines) >= 3
        and lines[0] == _GIT_LFS_POINTER_VERSION
        and lines[1].startswith("oid sha256:")
        and lines[2].startswith("size ")
    )


def normalize_item_image_filename(item_id: str, extension: str) -> str:
    item_id = str(item_id).strip()
    extension = str(extension).strip().lower()
    if extension and not extension.startswith("."):
        extension = f".{extension}"

    if (
        not item_id
        or item_id in {".", ".."}
        or "/" in item_id
        or "\\" in item_id
        or not _ITEM_ID_PATTERN.fullmatch(item_id)
    ):
        raise ValueError(f"Unsafe item image id: {item_id!r}")
    if not _IMAGE_EXTENSION_PATTERN.fullmatch(extension):
        raise ValueError(f"Unsafe item image extension: {extension!r}")

    return f"{item_id}{extension}"


def resolve_device(
    requested: str = "auto",
    *,
    cuda_available: Optional[bool] = None,
    mps_available: Optional[bool] = None,
):
    try:
        import torch
    except ModuleNotFoundError:
        torch = None

    requested = (requested or "auto").lower()
    if cuda_available is None:
        cuda_available = bool(torch and torch.cuda.is_available())
    if mps_available is None:
        mps_available = bool(torch and hasattr(torch.backends, "mps") and torch.backends.mps.is_available())

    def device(name: str):
        return torch.device(name) if torch else SimpleNamespace(type=name)

    if requested == "cuda":
        return device("cuda" if cuda_available else "cpu")
    if requested == "mps":
        return device("mps" if mps_available else "cpu")
    if requested == "cpu":
        return device("cpu")
    if cuda_available:
        return device("cuda")
    if mps_available:
        return device("mps")
    return device("cpu")


def _flatten(values) -> Sequence[float]:
    if hasattr(values, "detach"):
        values = values.detach().cpu().reshape(-1).tolist()
    elif isinstance(values, Iterable) and not isinstance(values, (str, bytes)):
        flattened = []
        for value in values:
            if isinstance(value, Iterable) and not isinstance(value, (str, bytes)):
                flattened.extend(_flatten(value))
            else:
                flattened.append(float(value))
        values = flattened
    else:
        values = [float(values)]
    return [float(value) for value in values]


def compatibility_probability(logits) -> float:
    values = _flatten(logits)
    if not values:
        return 0.0
    # Some checkpoints predict multiple compatibility views. Average them for
    # the app-level binary compatibility score.
    score = sum(values) / len(values)
    return 1.0 / (1.0 + exp(-score))


def infer_compat_head(model_state_dict) -> tuple[bool, int]:
    if "compat_mlp.3.weight" in model_state_dict:
        return True, int(model_state_dict["compat_mlp.3.weight"].shape[0])
    if "compat_mlp.2.weight" in model_state_dict:
        return False, int(model_state_dict["compat_mlp.2.weight"].shape[0])
    return False, 1
