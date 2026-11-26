import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import evaluate_pseudo_masks_r2s100k


class R2S100KEvaluatorCLITests(unittest.TestCase):
    def test_saved_masks_generate_grouped_reports_and_confidence_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            labels_root = root / 'processed'
            prediction_root = root / 'predictions'
            report_root = root / 'metrics'
            gt_dir = labels_root / 'labels' / 'train'
            full_dir = prediction_root / 'full' / 'train'
            confident_dir = prediction_root / 'confident' / 'train'
            for folder in (gt_dir, full_dir, confident_dir):
                folder.mkdir(parents=True)
            target = np.asarray([[0, 1], [1, 255]], dtype=np.uint8)
            full = np.asarray([[0, 0], [1, 1]], dtype=np.uint8)
            confident = np.asarray([[0, 255], [1, 255]], dtype=np.uint8)
            for folder, mask in ((gt_dir, target), (full_dir, full), (confident_dir, confident)):
                Image.fromarray(mask, mode='L').save(folder / 'sample.png')
            gt_manifest = root / 'eval_train.jsonl'
            gt_manifest.write_text(json.dumps({
                'sample_id': 'train/sample.jpg', 'image': 'train/sample.jpg',
                'label': 'labels/train/sample.png',
            }) + '\n')
            labeled_manifest = root / 'labeled_val.jsonl'
            labeled_manifest.write_text(json.dumps({
                'sample_id': 'val/sample.jpg', 'image': 'val/sample.jpg', 'label': 'labels/val/sample.png',
            }) + '\n')
            prediction_manifest = root / 'predictions.jsonl'
            prediction_manifest.write_text(json.dumps({
                'sample_id': 'train/sample.jpg', 'mask_full': 'full/train/sample.png',
                'mask_confident': 'confident/train/sample.png',
            }) + '\n')
            argv = [
                'evaluate_pseudo_masks_r2s100k.py', '--gt-manifest', str(gt_manifest),
                '--labels-root', str(labels_root), '--labeled-manifest', str(labeled_manifest),
                '--prediction-manifest', str(prediction_manifest), '--prediction-root', str(prediction_root),
                '--output-root', str(report_root),
            ]
            with patch('sys.argv', argv):
                evaluate_pseudo_masks_r2s100k.main()
            summary = json.loads((report_root / 'metrics_summary.json').read_text())
            self.assertEqual(summary['num_images'], 1)
            self.assertEqual(summary['train_images_also_labeled_from_val'], 1)
            self.assertAlmostEqual(summary['confidence_filtered_mask']['coverage'], 2 / 3)
            self.assertEqual(summary['groups']['train_images_also_labeled_from_val']['num_images'], 1)
            self.assertTrue((report_root / 'metrics_per_class.csv').exists())
            self.assertTrue((report_root / 'metrics_per_image.csv').exists())


if __name__ == '__main__':
    unittest.main()
