#!/usr/bin/env python3
"""Validate R2S100K pairs, convert RGB labels to class IDs, and write manifests."""

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from util.classes import CLASSES


PALETTE = [
    (0, 0, 0), (2, 79, 59), (17, 163, 74), (112, 84, 62), (225, 148, 79),
    (120, 114, 104), (166, 130, 95), (128, 222, 91), (119, 61, 128),
    (93, 86, 176), (140, 160, 222), (234, 133, 5), (156, 28, 39),
    (99, 122, 130), (123, 43, 31),
]
IGNORE_INDEX = 255
LABEL_SUFFIX = '___fuse.png'


def _key(rgb):
    return (rgb[..., 0].astype(np.uint32) << 16) | (rgb[..., 1].astype(np.uint32) << 8) | rgb[..., 2]


def _label_lookup():
    lookup = np.full(1 << 24, -1, dtype=np.int16)
    for class_id, color in enumerate(PALETTE):
        lookup[(color[0] << 16) | (color[1] << 8) | color[2]] = class_id
    return lookup


def _files(folder):
    return sorted(path for path in folder.rglob('*') if path.is_file())


def _pair_split(root, split):
    image_root = root / split
    label_root = root / f'{split}_labels'
    label_folder = label_root / 'Train-Labels' if split == 'train' and (label_root / 'Train-Labels').is_dir() else label_root
    if not image_root.is_dir() or not label_folder.is_dir():
        raise FileNotFoundError(f'Expected image/label folders: {image_root} and {label_folder}')

    images = _files(image_root)
    labels = {}
    duplicates = []
    for path in _files(label_folder):
        if not path.name.endswith(LABEL_SUFFIX):
            continue
        name = path.name[:-len(LABEL_SUFFIX)]
        if name in labels:
            duplicates.append(name)
        labels[name] = path
    if duplicates:
        raise ValueError(f'{split}: duplicate label basenames: {duplicates[:5]}')

    image_by_name = {}
    for path in images:
        if path.name in image_by_name:
            raise ValueError(f'{split}: duplicate image basename: {path.name}')
        image_by_name[path.name] = path
    missing_labels = sorted(image_by_name.keys() - labels.keys())
    orphan_labels = sorted(labels.keys() - image_by_name.keys())
    if missing_labels or orphan_labels:
        raise ValueError(
            f'{split}: {len(missing_labels)} images without labels, {len(orphan_labels)} orphan labels. '
            f'Examples: missing={missing_labels[:5]}, orphan={orphan_labels[:5]}'
        )
    return [(image_by_name[name], labels[name]) for name in sorted(image_by_name)]


def _write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8') as stream:
        for record in records:
            stream.write(json.dumps(record, ensure_ascii=False) + '\n')


def _convert_split(root, output_root, split, pairs, lookup):
    pixel_counts = Counter()
    records = []
    destination = output_root / 'labels' / split
    destination.mkdir(parents=True, exist_ok=True)
    for index, (image_path, label_path) in enumerate(pairs, start=1):
        with Image.open(image_path) as image:
            image_size = image.size
        with Image.open(label_path) as label_image:
            label = np.asarray(label_image.convert('RGB'), dtype=np.uint8)
        if image_size != (label.shape[1], label.shape[0]):
            raise ValueError(
                f'{split}: image/mask size mismatch for {image_path.name}: '
                f'{image_size} vs {(label.shape[1], label.shape[0])}'
            )

        encoded = _key(label)
        ids = lookup[encoded]
        unknown = ids < 0
        if np.any(unknown):
            colors, counts = np.unique(label[unknown].reshape(-1, 3), axis=0, return_counts=True)
            examples = [(tuple(int(v) for v in color), int(count)) for color, count in zip(colors[:20], counts[:20])]
            raise ValueError(f'{split}: unknown RGB colors in {label_path}: {examples}')
        values, counts = np.unique(ids, return_counts=True)
        pixel_counts.update({int(value): int(count) for value, count in zip(values, counts)})

        relative_image = image_path.relative_to(root).as_posix()
        sample_id = f'{split}/{image_path.relative_to(root / split).as_posix()}'
        relative_label = f'labels/{split}/{image_path.relative_to(root / split).with_suffix(".png").as_posix()}'
        destination_label = output_root / relative_label
        destination_label.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(ids.astype(np.uint8), mode='L').save(destination_label)
        records.append({
            'sample_id': sample_id,
            'image': relative_image,
            'label': relative_label,
            'width': image_size[0],
            'height': image_size[1],
        })
        if index % 500 == 0 or index == len(pairs):
            print(f'{split}: converted {index}/{len(pairs)} masks')
    return records, pixel_counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True, help='Root containing train/, val/, and *_labels/')
    parser.add_argument('--output-root', type=Path, required=True, help='Separate root for class-ID masks and manifests')
    parser.add_argument('--expected-train', type=int, default=9562, help='Expected train pair count; use 0 to disable')
    parser.add_argument('--expected-val', type=int, default=1366, help='Expected val pair count; use 0 to disable')
    parser.add_argument('--include-overlap', action='store_true', help='Use all train images as unlabeled, including val duplicates')
    args = parser.parse_args()

    root = args.data_root.resolve()
    output_root = args.output_root.resolve()
    train_pairs = _pair_split(root, 'train')
    val_pairs = _pair_split(root, 'val')
    if args.expected_train and len(train_pairs) != args.expected_train:
        raise ValueError(f'Expected {args.expected_train} train pairs, found {len(train_pairs)}')
    if args.expected_val and len(val_pairs) != args.expected_val:
        raise ValueError(f'Expected {args.expected_val} val pairs, found {len(val_pairs)}')

    train_names = {image.name for image, _ in train_pairs}
    val_names = {image.name for image, _ in val_pairs}
    overlapping_names = train_names & val_names
    print(f'Paired data: train={len(train_pairs)}, val={len(val_pairs)}, matching basenames={len(overlapping_names)}')

    lookup = _label_lookup()
    val_records, val_counts = _convert_split(root, output_root, 'val', val_pairs, lookup)
    train_records, train_counts = _convert_split(root, output_root, 'train', train_pairs, lookup)
    overlap_ids = {f'train/{name}' for name in overlapping_names}
    clean_unlabeled = [record for record in train_records if record['sample_id'] not in overlap_ids]

    manifests = output_root / 'manifests'
    _write_jsonl(manifests / 'labeled_val.jsonl', val_records)
    clean_unlabeled_images = [{'sample_id': record['sample_id'], 'image': record['image']} for record in clean_unlabeled]
    all_train_images = [{'sample_id': record['sample_id'], 'image': record['image']} for record in train_records]
    _write_jsonl(manifests / 'unlabeled_train.jsonl', all_train_images if args.include_overlap else clean_unlabeled_images)
    _write_jsonl(manifests / 'unlabeled_train_all.jsonl', all_train_images)
    _write_jsonl(manifests / 'images_train.jsonl', all_train_images)
    _write_jsonl(manifests / 'eval_train.jsonl', train_records)

    color_counts = {
        'train': {CLASSES['r2s100k'][i]: train_counts.get(i, 0) for i in range(len(PALETTE))},
        'val': {CLASSES['r2s100k'][i]: val_counts.get(i, 0) for i in range(len(PALETTE))},
    }
    metadata = {
        'dataset': 'R2S100K',
        'num_classes': len(PALETTE),
        'ignore_index': IGNORE_INDEX,
        'palette_rgb': [list(color) for color in PALETTE],
        'class_names': CLASSES['r2s100k'],
        'counts': {
            'train_images': len(train_records),
            'val_images': len(val_records),
            'train_val_matching_basenames': len(overlapping_names),
            'unlabeled_train': len(train_records) if args.include_overlap else len(clean_unlabeled),
        },
        'class_pixel_counts': color_counts,
        'include_overlap_in_unlabeled': args.include_overlap,
    }
    (output_root / 'dataset_metadata.json').write_text(json.dumps(metadata, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    print(json.dumps(metadata['counts'], indent=2))
    print(f'Converted masks and manifests saved under: {output_root}')
    print('Unknown colors: 0 (conversion stops at the first mask containing an unknown color)')


if __name__ == '__main__':
    main()
