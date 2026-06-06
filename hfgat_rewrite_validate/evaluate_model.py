import argparse
import json
from pathlib import Path
from typing import Any, Dict

from app_portability import build_paths, is_git_lfs_pointer_file


METRIC_ORDER = [
    "compat_acc",
    "AUC",
    "Precision@10",
    "HR@10",
    "Recall@10",
    "NDCG@10",
    "MRR@10",
    "val_total_loss",
    "val_rec_loss",
    "val_comp_loss",
]
PERCENT_METRICS = {"compat_acc", "AUC", "Precision@10", "HR@10", "Recall@10", "NDCG@10", "MRR@10"}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Missing {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def load_report(work_dir: Path) -> Dict[str, Any]:
    report_path = work_dir / "validation_report.json"
    if report_path.exists():
        return load_json(report_path)

    model_path = work_dir / "model.pt"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Missing {report_path} and {model_path}. Run the notebook training/export first."
        )
    if is_git_lfs_pointer_file(model_path):
        raise RuntimeError(f"{model_path} is a Git LFS pointer. Run `git lfs pull` first.")

    import torch

    checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
    report = checkpoint.get("validation_report")
    if not report:
        raise KeyError(f"{model_path} does not contain validation_report.")
    return report


def build_summary(report: Dict[str, Any]) -> Dict[str, Any]:
    metrics = report.get("best_metrics") or {}
    split_stats = report.get("split_stats") or {}
    summary = {
        "best_epoch": report.get("best_epoch", metrics.get("epoch")),
        "best_metric": report.get("best_metric"),
        "metrics": {name: metrics[name] for name in METRIC_ORDER if name in metrics},
        "split_stats": split_stats,
    }
    return summary


def format_metric(name: str, value: Any) -> str:
    if isinstance(value, (int, float)) and name in PERCENT_METRICS:
        return f"{value:.4f} ({value * 100:.2f}%)"
    if isinstance(value, (int, float)):
        return f"{value:.4f}"
    return str(value)


def print_summary(summary: Dict[str, Any]) -> None:
    print("H-FGAT validation report")
    print("========================")
    print(f"Best epoch : {summary.get('best_epoch')}")
    print(f"Best metric: {summary.get('best_metric')}")
    print()
    print("Metrics")
    for name, value in summary["metrics"].items():
        print(f"- {name}: {format_metric(name, value)}")

    split_stats = summary.get("split_stats") or {}
    if split_stats:
        print()
        print("Split")
        for name in ["train_edges", "val_edges", "test_edges", "train_users", "val_users", "test_users"]:
            if name in split_stats:
                print(f"- {name}: {split_stats[name]}")


def parse_args() -> argparse.Namespace:
    default_paths = build_paths(Path(__file__).parent)
    parser = argparse.ArgumentParser(description="Print saved H-FGAT validation metrics.")
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=default_paths.artifact_dir,
        help="Directory containing validation_report.json and model.pt.",
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = load_report(args.work_dir)
    summary = build_summary(report)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_summary(summary)


if __name__ == "__main__":
    main()
