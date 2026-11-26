#!/usr/bin/env python3
"""Generate full and confidence-filtered R2S100K pseudo masks."""

import argparse
import json
from pathlib import Path
import random

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import torch.nn.functional as F
import yaml

from dataset.r2s100k import IMAGENET_MEAN, IMAGENET_STD, read_manifest
from model.semseg.dpt import DPT
from train_r2s100k import MODEL_CONFIGS
from util.classes import CLASSES


PALETTE = np.asarray([
    (0, 0, 0), (2, 79, 59), (17, 163, 74), (112, 84, 62), (225, 148, 79),
    (120, 114, 104), (166, 130, 95), (128, 222, 91), (119, 61, 128),
    (93, 86, 176), (140, 160, 222), (234, 133, 5), (156, 28, 39),
    (99, 122, 130), (123, 43, 31),
], dtype=np.uint8)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/r2s100k.yaml'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--labels-root', type=Path, default=None, help='Prepared output root containing labels/')
    parser.add_argument('--manifest', type=Path, required=True, help='Image-only manifest to export')
    parser.add_argument('--gt-manifest', type=Path, default=None, help='Separate GT manifest used only for sampled visualizations')
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--weights', choices=('model_ema', 'student'), default='model_ema')
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--stride', type=int, default=None)
    parser.add_argument('--confidence-threshold', type=float, default=None)
    parser.add_argument('--inference-batch-size', type=int, default=4)
    parser.add_argument('--num-visualizations', type=int, default=32)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--limit', type=int, default=0, help='Only export the first N records (smoke test)')
    parser.add_argument('--disable-amp', action='store_true')
    return parser.parse_args()


def _make_model(cfg, state):
    size = cfg['backbone'].rsplit('_', 1)[-1]
    model = DPT(**{**MODEL_CONFIGS[size], 'nclass': int(cfg['nclass'])})
    model.load_state_dict(state)
    return model.cuda().eval()


def _starts(length, crop, stride):
    if length <= crop:
        return [0]
    points = list(range(0, length - crop + 1, stride))
    last = length - crop
    if points[-1] != last:
        points.append(last)
    return points


def _tile_positions(height, width, crop, stride):
    return [(y, x) for y in _starts(height, crop, stride) for x in _starts(width, crop, stride)]


def _image_tensor(array):
    value = array.astype(np.float32) / 255.0
    value = (value - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(value.transpose(2, 0, 1).copy())


def _infer(model, image, cfg, crop, stride, batch_size, use_amp):
    original_h, original_w = image.shape[:2]
    pad_h, pad_w = max(0, crop - original_h), max(0, crop - original_w)
    if pad_h or pad_w:
        image = np.pad(image, ((0, pad_h), (0, pad_w), (0, 0)), mode='edge')
    height, width = image.shape[:2]
    positions = _tile_positions(height, width, crop, stride)
    probability_sum = np.zeros((int(cfg['nclass']), height, width), dtype=np.float32)
    count = np.zeros((height, width), dtype=np.float32)
    for start in range(0, len(positions), batch_size):
        group = positions[start:start + batch_size]
        batch = torch.stack([_image_tensor(image[y:y + crop, x:x + crop]) for y, x in group]).cuda()
        with torch.inference_mode(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=use_amp):
            logits = model(batch)
            probabilities = logits.float().softmax(dim=1)
        if probabilities.shape[-2:] != (crop, crop):
            probabilities = F.interpolate(probabilities, (crop, crop), mode='bilinear', align_corners=True)
        probabilities = probabilities.cpu().numpy()
        for (y, x), tile_probability in zip(group, probabilities):
            probability_sum[:, y:y + crop, x:x + crop] += tile_probability
            count[y:y + crop, x:x + crop] += 1.0
    average = probability_sum / np.maximum(count[None], 1.0)
    average = average[:, :original_h, :original_w]
    prediction = average.argmax(axis=0).astype(np.uint8)
    confidence = average.max(axis=0)
    return prediction, confidence


def _colorize(mask):
    ids = np.asarray(mask, dtype=np.uint8)
    rgb = np.zeros((*ids.shape, 3), dtype=np.uint8)
    valid = ids < len(PALETTE)
    rgb[valid] = PALETTE[ids[valid]]
    rgb[~valid] = (255, 0, 255)
    return Image.fromarray(rgb, mode='RGB')


def _write_visualization(path, image_path, gt_path, prediction_path):
    image = Image.open(image_path).convert('RGB')
    prediction = Image.open(prediction_path).convert('L')
    panels = [('Image', image), ('Pseudo mask', _colorize(prediction))]
    if gt_path is not None:
        ground_truth = Image.open(gt_path).convert('L')
        gt_color = _colorize(ground_truth)
        overlay = Image.blend(image, _colorize(prediction), alpha=0.45)
        panels = [('Image', image), ('Ground truth', gt_color), ('Pseudo mask', _colorize(prediction)), ('Overlay', overlay)]
    thumb_w = 640
    thumb_h = max(1, int(image.height * thumb_w / image.width))
    tile = Image.new('RGB', (thumb_w, thumb_h + 32), 'white')
    canvas = Image.new('RGB', (thumb_w * 2, (thumb_h + 32) * 2), '#d9d9d9')
    draw = ImageDraw.Draw(canvas)
    for index, (title, panel) in enumerate(panels):
        panel = panel.resize((thumb_w, thumb_h), Image.Resampling.NEAREST if title != 'Image' else Image.Resampling.BILINEAR)
        x, y = (index % 2) * thumb_w, (index // 2) * (thumb_h + 32)
        canvas.paste(tile, (x, y))
        canvas.paste(panel, (x, y + 32))
        draw.text((x + 8, y + 8), title, fill='black', font=ImageFont.load_default())
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path, quality=92)


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    cfg['crop_size'] = int(args.crop_size or cfg['crop_size'])
    cfg['nclass'] = int(cfg.get('nclass', 15))
    stride = int(args.stride or cfg.get('inference_stride', cfg['crop_size'] * 2 // 3))
    threshold = float(args.confidence_threshold if args.confidence_threshold is not None else cfg.get('conf_thresh', 0.95))
    if cfg['crop_size'] % 14 or stride <= 0 or stride > cfg['crop_size']:
        raise ValueError('crop-size must be divisible by 14 and stride must be in (0, crop-size]')
    if not torch.cuda.is_available():
        raise RuntimeError('Pseudo-mask export requires CUDA. Activate a CUDA-enabled PyTorch environment.')
    records = read_manifest(args.manifest)
    if args.limit:
        records = records[:args.limit]
    if not records:
        raise ValueError('No image records to export')
    if args.inference_batch_size < 1 or args.num_visualizations < 0 or not 0.0 <= threshold <= 1.0:
        raise ValueError('inference-batch-size and visualization count must be positive/non-negative; threshold must be in [0,1]')
    sample_ids = [record['sample_id'] for record in records]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError('Image manifest contains duplicate sample_id values')
    gt_by_id = {}
    if args.gt_manifest:
        gt_by_id = {item['sample_id']: item for item in read_manifest(args.gt_manifest)}
    if args.num_visualizations and not args.gt_manifest:
        raise ValueError('--gt-manifest is required to save GT/pseudo-mask visualizations')
    if args.gt_manifest and not args.labels_root:
        raise ValueError('--labels-root is required when --gt-manifest is provided')
    rng = random.Random(args.seed)
    selected_indices = set(rng.sample(range(len(records)), min(args.num_visualizations, len(records))))

    checkpoint = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if args.weights not in checkpoint:
        raise KeyError(f"Checkpoint does not contain requested weights '{args.weights}'")
    model = _make_model(cfg, checkpoint[args.weights])
    use_amp = bool(cfg.get('amp', True) and not args.disable_amp)

    output_root = args.output_root
    full_root, confident_root = output_root / 'full', output_root / 'confident'
    manifest_path = output_root / 'predictions.jsonl'
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    prediction_records = []
    for index, record in enumerate(records):
        image_path = Path(args.data_root) / record['image']
        with Image.open(image_path) as opened:
            image = np.asarray(opened.convert('RGB'))
        prediction, confidence = _infer(model, image, cfg, cfg['crop_size'], stride,
                                        args.inference_batch_size, use_amp)
        relative = Path(record['image']).with_suffix('.png')
        full_path = full_root / relative
        confident_path = confident_root / relative
        full_path.parent.mkdir(parents=True, exist_ok=True)
        confident_path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(prediction, mode='L').save(full_path)
        filtered = prediction.copy()
        filtered[confidence < threshold] = 255
        Image.fromarray(filtered, mode='L').save(confident_path)
        output_record = {
            'sample_id': record['sample_id'],
            'image': record['image'],
            'mask_full': full_path.relative_to(output_root).as_posix(),
            'mask_confident': confident_path.relative_to(output_root).as_posix(),
        }
        prediction_records.append(output_record)
        if index in selected_indices:
            gt_path = None
            if record['sample_id'] in gt_by_id:
                gt_record = gt_by_id[record['sample_id']]
                if 'label' in gt_record:
                    gt_path = Path(args.labels_root) / gt_record['label']
            if gt_path is None and args.gt_manifest:
                raise ValueError(f"GT manifest has no resolvable label for {record['sample_id']}; use --labels-root")
            stem = f'{index:05d}_{Path(record["image"]).stem[:48]}.jpg'
            _write_visualization(output_root / 'visualizations' / stem, image_path, gt_path, full_path)
        if (index + 1) % 100 == 0 or index + 1 == len(records):
            print(f'Exported {index + 1}/{len(records)} masks')

    with manifest_path.open('w', encoding='utf-8') as stream:
        for item in prediction_records:
            stream.write(json.dumps(item, ensure_ascii=False) + '\n')
    metadata = {
        'dataset': 'R2S100K',
        'checkpoint': str(args.checkpoint.resolve()),
        'weights': args.weights,
        'seed': args.seed,
        'num_images': len(records),
        'crop_size': cfg['crop_size'],
        'stride': stride,
        'confidence_threshold': threshold,
        'class_names': CLASSES['r2s100k'],
        'palette_rgb': PALETTE.tolist(),
        'amp': use_amp,
        'prediction_manifest': manifest_path.name,
    }
    (output_root / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    print(f'Prediction manifest: {manifest_path}')
    print(f'Full masks: {full_root}; confidence-filtered masks: {confident_root}')


if __name__ == '__main__':
    main()
