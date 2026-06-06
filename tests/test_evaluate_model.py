import json
import tempfile
import unittest
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "hfgat_rewrite_validate"))

from evaluate_model import build_summary, format_metric, load_report


class EvaluateModelTests(unittest.TestCase):
    def test_build_summary_extracts_saved_validation_metrics(self):
        report = {
            "best_epoch": 11,
            "best_metric": "NDCG@10",
            "best_metrics": {
                "epoch": 11,
                "compat_acc": 0.7879,
                "AUC": 0.8625,
                "NDCG@10": 0.4014,
                "ignored": 123,
            },
            "split_stats": {"val_users": 5543},
        }

        summary = build_summary(report)

        self.assertEqual(summary["best_epoch"], 11)
        self.assertEqual(summary["best_metric"], "NDCG@10")
        self.assertEqual(summary["metrics"], {"compat_acc": 0.7879, "AUC": 0.8625, "NDCG@10": 0.4014})
        self.assertEqual(summary["split_stats"], {"val_users": 5543})

    def test_format_metric_includes_percentage_for_score_metrics_only(self):
        self.assertEqual(format_metric("compat_acc", 0.787924), "0.7879 (78.79%)")
        self.assertEqual(format_metric("val_total_loss", 0.4382817), "0.4383")

    def test_load_report_prefers_json_file_without_torch(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            expected = {"best_epoch": 1, "best_metrics": {"compat_acc": 0.5}}
            (work_dir / "validation_report.json").write_text(json.dumps(expected), encoding="utf-8")

            self.assertEqual(load_report(work_dir), expected)

    def test_load_report_detects_lfs_pointer_model_when_json_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            work_dir = Path(tmp)
            (work_dir / "model.pt").write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0123456789abcdef\n"
                "size 123456\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(RuntimeError, "Git LFS pointer"):
                load_report(work_dir)


if __name__ == "__main__":
    unittest.main()
