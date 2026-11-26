"""Manifest-based R2S100K loading and UniMatch V2 augmentations."""

import json
import math
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
import torch
from torch.utils.data import Dataset


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def read_manifest(path):
    records = []
    with open(path, encoding='utf-8') as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f'{path}:{line_number}: invalid JSON: {exc}') from exc
            if 'sample_id' not in record or 'image' not in record:
                raise ValueError(f'{path}:{line_number}: each record needs sample_id and image')
            records.append(record)
    if not records:
        raise ValueError(f'Manifest has no records: {path}')
    return records


def _resolve(root, relative_path):
    path = Path(relative_path)
    return path if path.is_absolute() else Path(root) / path


def _to_tensor(image):
    array = np.asarray(image.convert('RGB'), dtype=np.float32) / 255.0
    array = (array - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(array.transpose(2, 0, 1).copy())


def _jitter(image):
    operations = [
        lambda im: ImageEnhance.Brightness(im).enhance(random.uniform(0.6, 1.4)),
        lambda im: ImageEnhance.Contrast(im).enhance(random.uniform(0.6, 1.4)),
        lambda im: ImageEnhance.Color(im).enhance(random.uniform(0.6, 1.4)),
    ]
    random.shuffle(operations)
    for operation in operations:
        image = operation(image)
    if random.random() < 0.2:
        image = ImageOps.grayscale(image).convert('RGB')
    if random.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 2.0)))
    return image


def _resize_and_crop(image, mask, size, ratio_range=(0.5, 2.0)):
    width, height = image.size
    scale = random.uniform(*ratio_range)
    resized_w = max(1, int(round(width * scale)))
    resized_h = max(1, int(round(height * scale)))
    image = image.resize((resized_w, resized_h), Image.Resampling.BILINEAR)
    if mask is not None:
        mask = mask.resize((resized_w, resized_h), Image.Resampling.NEAREST)

    pad_w, pad_h = max(0, size - resized_w), max(0, size - resized_h)
    if pad_w or pad_h:
        image = ImageOps.expand(image, border=(0, 0, pad_w, pad_h), fill=0)
        if mask is not None:
            mask = ImageOps.expand(mask, border=(0, 0, pad_w, pad_h), fill=255)
    valid = Image.new('L', image.size, 255)
    if pad_w or pad_h:
        valid = Image.new('L', (resized_w, resized_h), 255)
        valid = ImageOps.expand(valid, border=(0, 0, pad_w, pad_h), fill=0)

    width, height = image.size
    x = random.randint(0, width - size)
    y = random.randint(0, height - size)
    box = (x, y, x + size, y + size)
    image = image.crop(box)
    valid = valid.crop(box)
    if mask is not None:
        mask = mask.crop(box)
    if random.random() < 0.5:
        image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        valid = valid.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if mask is not None:
            mask = mask.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    return image, mask, valid


class R2S100KLabeledDataset(Dataset):
    def __init__(self, manifest, data_root, labels_root, crop_size, records=None):
        self.records = records if records is not None else read_manifest(manifest)
        self.data_root = Path(data_root)
        self.labels_root = Path(labels_root)
        self.crop_size = int(crop_size)
        missing = [item.get('sample_id', '?') for item in self.records if 'label' not in item]
        if missing:
            raise ValueError(f'Labeled manifest has records without label paths: {missing[:3]}')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(_resolve(self.data_root, record['image'])).convert('RGB')
        mask_path = _resolve(self.labels_root, record['label'])
        mask = Image.open(mask_path).convert('L')
        if image.size != mask.size:
            raise ValueError(f"Image/mask size mismatch for {record['sample_id']}: {image.size} vs {mask.size}")
        image, mask, _ = _resize_and_crop(image, mask, self.crop_size)
        mask_tensor = torch.from_numpy(np.asarray(mask, dtype=np.uint8).copy()).long()
        return _to_tensor(image), mask_tensor


class R2S100KUnlabeledDataset(Dataset):
    def __init__(self, manifest, data_root, crop_size, records=None):
        self.records = records if records is not None else read_manifest(manifest)
        self.data_root = Path(data_root)
        self.crop_size = int(crop_size)
        if any('label' in item for item in self.records):
            raise ValueError('Unlabeled manifest must not contain label paths')

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = Image.open(_resolve(self.data_root, record['image'])).convert('RGB')
        weak, _, valid = _resize_and_crop(image, None, self.crop_size)
        strong1, strong2 = weak.copy(), weak.copy()
        if random.random() < 0.8:
            strong1 = _jitter(strong1)
        else:
            strong1 = strong1.filter(ImageFilter.GaussianBlur(radius=0.0))
        if random.random() < 0.8:
            strong2 = _jitter(strong2)
        else:
            strong2 = strong2.filter(ImageFilter.GaussianBlur(radius=0.0))
        valid_tensor = torch.from_numpy((np.asarray(valid, dtype=np.uint8) != 0).astype(np.uint8))
        return _to_tensor(weak), _to_tensor(strong1), _to_tensor(strong2), valid_tensor


def repeat_records(records, count):
    """Repeat a labeled list to pair each epoch with the unlabeled loader length."""
    if not records or count < 1:
        raise ValueError('records and count must be non-empty/positive')
    return (records * math.ceil(count / len(records)))[:count]
