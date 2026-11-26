import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from prepare_r2s100k import PALETTE, _convert_split, _label_lookup, _pair_split
from r2s100k_metrics import confusion_matrix, metrics_for_pair


class R2S100KDataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.lookup = _label_lookup()

    def _make_pair(self, root, rgb, image_size=(2, 2), filename='road image.jpg'):
        image_root = root / 'train'
        label_root = root / 'train_labels' / 'Train-Labels'
        image_root.mkdir(parents=True, exist_ok=True)
        label_root.mkdir(parents=True, exist_ok=True)
        Image.new('RGB', image_size, (30, 40, 50)).save(image_root / filename)
        Image.fromarray(np.asarray(rgb, dtype=np.uint8), mode='RGB').save(label_root / f'{filename}___fuse.png')

    def test_rgb_palette_converts_to_class_ids_and_keeps_spaces(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            rgb = np.asarray([[PALETTE[0], PALETTE[1]], [PALETTE[14], PALETTE[3]]], dtype=np.uint8)
            self._make_pair(root, rgb)
            pairs = _pair_split(root, 'train')
            records, counts = _convert_split(root, root / 'processed', 'train', pairs, self.lookup)
            self.assertEqual(records[0]['sample_id'], 'train/road image.jpg')
            self.assertEqual(records[0]['image'], 'train/road image.jpg')
            mask = np.asarray(Image.open(root / 'processed' / records[0]['label']))
            np.testing.assert_array_equal(mask, np.asarray([[0, 1], [14, 3]], dtype=np.uint8))
            self.assertEqual(counts[14], 1)

    def test_unknown_color_fails_instead_of_becoming_background(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_pair(root, np.full((2, 2, 3), (1, 2, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, 'unknown RGB colors'):
                _convert_split(root, root / 'processed', 'train', _pair_split(root, 'train'), self.lookup)

    def test_size_mismatch_and_missing_labels_fail(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._make_pair(root, np.zeros((1, 2, 3), dtype=np.uint8))
            with self.assertRaisesRegex(ValueError, 'size mismatch'):
                _convert_split(root, root / 'processed', 'train', _pair_split(root, 'train'), self.lookup)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'train').mkdir()
            Image.new('RGB', (1, 1)).save(root / 'train' / 'missing.jpg')
            (root / 'train_labels' / 'Train-Labels').mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, 'without labels'):
                _pair_split(root, 'train')


class R2S100KMetricTests(unittest.TestCase):
    def test_known_confusion_ignore_and_missing_class(self):
        target = np.asarray([[0, 1], [1, 255]], dtype=np.uint8)
        prediction = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
        matrix = confusion_matrix(target, prediction, nclass=3)
        np.testing.assert_array_equal(matrix[:2, :2], np.asarray([[1, 0], [1, 1]]))
        self.assertEqual(matrix.sum(), 3)
        result = metrics_for_pair(target, prediction, nclass=3)
        self.assertAlmostEqual(result['pixel_accuracy'], 2 / 3)
        self.assertAlmostEqual(result['iou'][0], 0.5)
        self.assertAlmostEqual(result['iou'][1], 0.5)
        self.assertTrue(np.isnan(result['iou'][2]))
        self.assertEqual(result['classes_in_miou'], 2)
        self.assertEqual(result['target_pixel_count'], 3)
        self.assertEqual(result['coverage'], 1.0)

    def test_confidence_ignore_reports_coverage_with_gt_denominator(self):
        target = np.asarray([[0, 1], [1, 255]], dtype=np.uint8)
        prediction = np.asarray([[0, 255], [1, 255]], dtype=np.uint8)
        result = metrics_for_pair(target, prediction, nclass=3)
        self.assertAlmostEqual(result['coverage'], 2 / 3)
        self.assertAlmostEqual(result['pixel_accuracy'], 1.0)
        self.assertEqual(result['evaluated_pixel_count'], 2)

    def test_all_predictions_ignored_has_zero_coverage_and_no_pixel_accuracy(self):
        target = np.asarray([[0, 1]], dtype=np.uint8)
        prediction = np.asarray([[255, 255]], dtype=np.uint8)
        result = metrics_for_pair(target, prediction, nclass=2)
        self.assertEqual(result['coverage'], 0.0)
        self.assertIsNone(result['pixel_accuracy'])
        self.assertEqual(result['classes_in_miou'], 0)
        self.assertIsNone(result['miou_with_background'])


if __name__ == '__main__':
    unittest.main()
