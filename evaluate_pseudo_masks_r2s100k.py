#!/usr/bin/env python3
"""Evaluate saved R2S100K pseudo masks against the separate train GT manifest."""

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from r2s100k_metrics import confusion_matrix, summarize_confusion
from util.classes import CLASSES


def _metric_row(summary, class_names):
    row = {
        'pixel_accuracy': summary['pixel_accuracy'],
        'coverage': summary['coverage'],
        'miou_with_background': summary['miou_with_background'],
        'miou_without_background': summary['miou_without_background'],
        'classes_in_miou': summary['classes_in_miou'],
        'target_pixel_count': summary['target_pixel_count'],
        'evaluated_pixel_count': summary['evaluated_pixel_count'],
    }
    for index, name in enumerate(class_names):
        row[f'iou_{index}_{name}'] = _finite(summary['iou'][index])
        row[f'dice_{index}_{name}'] = _finite(summary['dice'][index])
    return row


def _finite(value):
    return float(value) if np.isfinite(value) else None


def _read_manifest(path):
    records = []
    with path.open(encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON') from exc
    if not records:
        raise ValueError(f'Manifest has no records: {path}')
    return records


def _load_mask(path):
    with Image.open(path) as image:
        return np.asarray(image.convert('L'), dtype=np.uint8)


def _write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f'No rows to write: {path}')
    with path.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gt-manifest', type=Path, required=True)
    parser.add_argument('--labels-root', type=Path, required=True)
    parser.add_argument('--labeled-manifest', type=Path, default=None,
                        help='Used to report clean train images separately from train/val duplicates')
    parser.add_argument('--prediction-manifest', type=Path, required=True)
    parser.add_argument('--prediction-root', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--limit', type=int, default=0, help='Evaluate the first N GT records (smoke test)')
    args = parser.parse_args()

    gt_records = _read_manifest(args.gt_manifest)
    prediction_records = _read_manifest(args.prediction_manifest)
    if args.limit:
        gt_records = gt_records[:args.limit]
    predictions = {record['sample_id']: record for record in prediction_records}
    if len(predictions) != len(prediction_records):
        raise ValueError('Prediction manifest contains duplicate sample_id values')
    missing = [record['sample_id'] for record in gt_records if record['sample_id'] not in predictions]
    if missing:
        raise ValueError(f'Missing predictions for {len(missing)} GT records; examples: {missing[:5]}')

    class_names = CLASSES['r2s100k']
    labeled_names = set()
    if args.labeled_manifest:
        labeled_names = {Path(record['image']).name for record in _read_manifest(args.labeled_manifest)}
    full_total = np.zeros((len(class_names), len(class_names)), dtype=np.int64)
    confident_total = np.zeros_like(full_total)
    target_pixel_total = 0
    groups = {}
    per_image_rows = []
    for index, gt_record in enumerate(gt_records):
        prediction_record = predictions[gt_record['sample_id']]
        gt_path = args.labels_root / gt_record['label']
        target = _load_mask(gt_path)
        full = _load_mask(args.prediction_root / prediction_record['mask_full'])
        confident = _load_mask(args.prediction_root / prediction_record['mask_confident'])
        if target.shape != full.shape or target.shape != confident.shape:
            raise ValueError(f"Mask size mismatch for {gt_record['sample_id']}: gt={target.shape}, full={full.shape}, confident={confident.shape}")

        valid_gt = target != 255
        total_pixels = int(valid_gt.sum())
        target_pixel_total += total_pixels
        full_confusion = confusion_matrix(target, full)
        confident_confusion = confusion_matrix(target, confident)
        full_total += full_confusion
        confident_total += confident_confusion
        if labeled_names:
            group = ('train_images_also_labeled_from_val' if Path(gt_record['image']).name in labeled_names
                     else 'clean_unlabeled_train')
        else:
            group = 'all_train'
        group_data = groups.setdefault(group, {
            'num_images': 0,
            'target_pixel_count': 0,
            'full': np.zeros_like(full_total),
            'confident': np.zeros_like(full_total),
        })
        group_data['num_images'] += 1
        group_data['target_pixel_count'] += total_pixels
        group_data['full'] += full_confusion
        group_data['confident'] += confident_confusion
        for variant, matrix in (('full', full_confusion), ('confident', confident_confusion)):
            summary = summarize_confusion(matrix, total_pixels, len(class_names))
            row = {'sample_id': gt_record['sample_id'], 'group': group, 'variant': variant}
            row.update(_metric_row(summary, class_names))
            per_image_rows.append(row)
        if (index + 1) % 500 == 0 or index + 1 == len(gt_records):
            print(f'Evaluated {index + 1}/{len(gt_records)} images')

    full_summary = summarize_confusion(full_total, target_pixel_total, len(class_names))
    confident_summary = summarize_confusion(confident_total, target_pixel_total, len(class_names))
    group_reports = {}
    for group, values in groups.items():
        group_full = summarize_confusion(values['full'], values['target_pixel_count'], len(class_names))
        group_confident = summarize_confusion(values['confident'], values['target_pixel_count'], len(class_names))
        group_reports[group] = {
            'num_images': values['num_images'],
            'full_mask': _metric_row(group_full, class_names),
            'confidence_filtered_mask': _metric_row(group_confident, class_names),
        }
    report = {
        'dataset': 'R2S100K',
        'gt_manifest': str(args.gt_manifest.resolve()),
        'prediction_manifest': str(args.prediction_manifest.resolve()),
        'prediction_metadata': None,
        'num_images': len(gt_records),
        'labeled_val_manifest': str(args.labeled_manifest.resolve()) if args.labeled_manifest else None,
        'train_images_also_labeled_from_val': len(labeled_names & {Path(record['image']).name for record in gt_records}),
        'ignore_index': 255,
        'aggregation': 'sum confusion matrices over pixels before computing metrics',
        'class_names': class_names,
        'full_mask': _metric_row(full_summary, class_names),
        'confidence_filtered_mask': _metric_row(confident_summary, class_names),
        'groups': group_reports,
        'confidence_filtered_denominator': 'pixels with prediction != 255; coverage denominator is all non-ignore GT pixels',
    }
    metadata_path = args.prediction_root / 'metadata.json'
    if metadata_path.exists():
        report['prediction_metadata'] = json.loads(metadata_path.read_text(encoding='utf-8'))
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / 'metrics_summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    class_rows = []
    all_group_summaries = {'all_train': {'full': full_summary, 'confident': confident_summary}}
    for group, values in groups.items():
        all_group_summaries[group] = {
            'full': summarize_confusion(values['full'], values['target_pixel_count'], len(class_names)),
            'confident': summarize_confusion(values['confident'], values['target_pixel_count'], len(class_names)),
        }
    for group, variants in all_group_summaries.items():
        for variant, summary in variants.items():
            for class_id in range(len(class_names)):
                class_rows.append({
                    'group': group,
                    'variant': variant,
                    'class_id': class_id,
                    'class_name': class_names[class_id],
                    'iou': _finite(summary['iou'][class_id]),
                    'dice': _finite(summary['dice'][class_id]),
                    'intersection': int(summary['intersection'][class_id]),
                    'union': int(summary['union'][class_id]),
                    'target_pixels': int(summary['target_pixels'][class_id]),
                    'predicted_pixels': int(summary['predicted_pixels'][class_id]),
                })
    _write_csv(args.output_root / 'metrics_per_class.csv', class_rows)
    _write_csv(args.output_root / 'metrics_per_image.csv', per_image_rows)
    print(json.dumps({
        'num_images': len(gt_records),
        'full_miou': full_summary['miou_with_background'],
        'full_miou_without_background': full_summary['miou_without_background'],
        'confident_miou': confident_summary['miou_with_background'],
        'confident_coverage': confident_summary['coverage'],
        'reports': str(args.output_root.resolve()),
    }, indent=2))


if __name__ == '__main__':
    main()
