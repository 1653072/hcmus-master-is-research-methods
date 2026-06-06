import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hfgat_rewrite_validate.app_portability import (
    build_paths,
    compatibility_probability,
    is_git_lfs_pointer_file,
    normalize_item_image_filename,
    resolve_device,
)


class AppPortabilityTests(unittest.TestCase):
    def test_build_paths_are_relative_to_app_directory(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_dir = Path(tmp) / "hfgat_rewrite_validate"
            paths = build_paths(app_dir)

        self.assertEqual(paths.artifact_dir, app_dir / "output_hfgat_notebook")
        self.assertEqual(paths.export_dir, app_dir / "output_hfgat_notebook" / "exported_embeddings")
        self.assertEqual(paths.image_dir, app_dir / "Dataset" / "fashion_item_images")

    def test_resolve_device_prefers_available_backend_and_supports_auto(self):
        self.assertEqual(resolve_device("cpu").type, "cpu")
        self.assertEqual(resolve_device("cuda", cuda_available=False, mps_available=False).type, "cpu")
        self.assertEqual(resolve_device("mps", cuda_available=False, mps_available=True).type, "mps")
        self.assertEqual(resolve_device("auto", cuda_available=True, mps_available=True).type, "cuda")
        self.assertEqual(resolve_device("auto", cuda_available=False, mps_available=True).type, "mps")

    def test_compatibility_probability_handles_single_and_multi_view_logits(self):
        single = [0.0]
        multi = [0.0, 2.0, -1.0]

        self.assertAlmostEqual(compatibility_probability(single), 0.5)
        self.assertGreater(compatibility_probability(multi), 0.5)

    def test_git_lfs_pointer_file_detection(self):
        with tempfile.TemporaryDirectory() as tmp:
            pointer = Path(tmp) / "model.pt"
            pointer.write_text(
                "version https://git-lfs.github.com/spec/v1\n"
                "oid sha256:0123456789abcdef\n"
                "size 123456\n",
                encoding="utf-8",
            )
            real_file = Path(tmp) / "real.pt"
            real_file.write_bytes(b"PK\x03\x04not a pointer")

            self.assertTrue(is_git_lfs_pointer_file(pointer))
            self.assertFalse(is_git_lfs_pointer_file(real_file))
            self.assertFalse(is_git_lfs_pointer_file(Path(tmp) / "missing.pt"))

    def test_item_image_filename_normalizes_common_ids(self):
        self.assertEqual(normalize_item_image_filename(" ABC-123_45 ", ".jpg"), "ABC-123_45.jpg")
        self.assertEqual(normalize_item_image_filename("sku.001", "PNG"), "sku.001.png")

    def test_item_image_filename_rejects_path_traversal(self):
        for bad_item_id in ("../secret", "..", "a/b", r"a\\b", "/absolute", "C:\\absolute"):
            with self.subTest(item_id=bad_item_id):
                with self.assertRaises(ValueError):
                    normalize_item_image_filename(bad_item_id, ".jpg")

        for bad_extension in ("../jpg", "jpg/../../png", ".tar.gz", ""):
            with self.subTest(extension=bad_extension):
                with self.assertRaises(ValueError):
                    normalize_item_image_filename("item123", bad_extension)


if __name__ == "__main__":
    unittest.main()
