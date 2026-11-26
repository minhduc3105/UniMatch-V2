#!/usr/bin/env python3
"""Train UniMatch V2 on R2S100K without running validation."""

import argparse
from copy import deepcopy
import csv
import json
import logging
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
import yaml

from dataset.r2s100k import R2S100KLabeledDataset, R2S100KUnlabeledDataset, read_manifest, repeat_records
from dataset.transform import obtain_cutmix_box_hw
from model.semseg.dpt import DPT
from util.classes import CLASSES


MODEL_CONFIGS = {
    'small': {'encoder_size': 'small', 'features': 64, 'out_channels': [48, 96, 192, 384]},
    'base': {'encoder_size': 'base', 'features': 128, 'out_channels': [96, 192, 384, 768]},
    'large': {'encoder_size': 'large', 'features': 256, 'out_channels': [256, 512, 1024, 1024]},
    'giant': {'encoder_size': 'giant', 'features': 384, 'out_channels': [1536, 1536, 1536, 1536]},
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=Path('configs/r2s100k.yaml'))
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--labels-root', type=Path, required=True, help='Prepared output root containing labels/')
    parser.add_argument('--labeled-manifest', type=Path, required=True)
    parser.add_argument('--unlabeled-manifest', type=Path, required=True)
    parser.add_argument('--pretrained', type=Path, default=None)
    parser.add_argument('--output-root', type=Path, required=True)
    parser.add_argument('--resume', type=Path, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--crop-size', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--num-workers', type=int, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--max-steps', type=int, default=0, help='Stop after this many total optimizer steps (smoke test)')
    parser.add_argument('--limit-labeled', type=int, default=0, help='Use the first N labeled manifest records')
    parser.add_argument('--limit-unlabeled', type=int, default=0, help='Use the first N unlabeled manifest records')
    parser.add_argument('--disable-amp', action='store_true')
    parser.add_argument('--allow-overlap', action='store_true', help='Acknowledge shared val/train image IDs in the manifests')
    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_pretrained(model, path):
    checkpoint = torch.load(path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ('state_dict', 'teacher', 'model'):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f'Pretrained checkpoint does not contain a state dict: {path}')
    state = {}
    for key, value in checkpoint.items():
        clean_key = key.removeprefix('module.').removeprefix('backbone.')
        state[clean_key] = value
    model.backbone.load_state_dict(state)


def _model(cfg, pretrained):
    size = cfg['backbone'].rsplit('_', 1)[-1]
    if size not in MODEL_CONFIGS:
        raise ValueError(f"Unsupported DINOv2 backbone: {cfg['backbone']}")
    model = DPT(**{**MODEL_CONFIGS[size], 'nclass': cfg['nclass']})
    if not Path(pretrained).is_file():
        raise FileNotFoundError(f'Pretrained DINOv2 weights not found: {pretrained}')
    _load_pretrained(model, pretrained)
    return model


def _write_step(writer, values):
    write_header = not writer.exists() or writer.stat().st_size == 0
    with writer.open('a', newline='', encoding='utf-8') as stream:
        output = csv.DictWriter(stream, fieldnames=['step', 'epoch', 'loss', 'loss_labeled', 'loss_unlabeled',
                                                     'pseudo_coverage', 'learning_rate', 'gpu_memory_gb'])
        if write_header:
            output.writeheader()
        output.writerow(values)


def main():
    args = parse_args()
    cfg = yaml.safe_load(args.config.read_text(encoding='utf-8'))
    cfg['nclass'] = int(cfg.get('nclass', 15))
    cfg['epochs'] = int(args.epochs or cfg['epochs'])
    cfg['crop_size'] = int(args.crop_size or cfg['crop_size'])
    cfg['batch_size'] = int(args.batch_size or cfg['batch_size'])
    cfg['num_workers'] = int(args.num_workers if args.num_workers is not None else cfg.get('num_workers', 4))
    cfg['seed'] = int(args.seed if args.seed is not None else cfg.get('seed', 42))
    cfg['amp'] = bool(cfg.get('amp', True) and not args.disable_amp)
    cfg['ignore_index'] = int(cfg.get('ignore_index', 255))
    cfg['conf_thresh'] = float(cfg.get('conf_thresh', 0.95))
    pretrained = args.pretrained or Path(cfg['pretrained_path'])
    args.output_root.mkdir(parents=True, exist_ok=True)

    if not torch.cuda.is_available():
        raise RuntimeError('R2S100K UniMatch training requires CUDA. Activate a CUDA-enabled PyTorch environment.')
    if cfg['crop_size'] % 14:
        raise ValueError(f"crop-size must be divisible by 14 for DINOv2; got {cfg['crop_size']}")
    if cfg['batch_size'] < 2:
        raise ValueError('batch-size must be at least 2 because UniMatch CutMix exchanges views across samples')
    if args.max_steps < 0 or args.limit_labeled < 0 or args.limit_unlabeled < 0:
        raise ValueError('max-steps and manifest limits must be non-negative')

    seed_everything(cfg['seed'])
    torch.backends.cudnn.benchmark = True
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
    logger = logging.getLogger('r2s100k')

    labeled_records = read_manifest(args.labeled_manifest)
    unlabeled_records = read_manifest(args.unlabeled_manifest)
    if args.limit_labeled:
        labeled_records = labeled_records[:args.limit_labeled]
    if args.limit_unlabeled:
        unlabeled_records = unlabeled_records[:args.limit_unlabeled]
    if not labeled_records or not unlabeled_records:
        raise ValueError('Both labeled and unlabeled manifests must contain at least one record')
    if any('label' in record for record in unlabeled_records):
        raise ValueError('The unlabeled manifest contains label paths; use unlabeled_train.jsonl')
    labeled_ids = {record['sample_id'] for record in labeled_records}
    unlabeled_names = {Path(record['image']).name for record in unlabeled_records}
    overlapping = labeled_ids & {f"val/{name}" for name in unlabeled_names}
    if overlapping and not args.allow_overlap:
        raise ValueError(f'Labeled and unlabeled manifests overlap by sample id: {sorted(overlapping)[:5]}')
    if overlapping:
        logger_message = f'WARNING: {len(overlapping)} labeled image IDs also occur in the unlabeled manifest.'
        print(logger_message)

    # Match the labeled loader length to the unlabeled set, as in UniMatch V2.
    labeled_repeated = repeat_records(labeled_records, len(unlabeled_records))
    labeled_data = R2S100KLabeledDataset(None, args.data_root, args.labels_root, cfg['crop_size'], labeled_repeated)
    unlabeled_data = R2S100KUnlabeledDataset(None, args.data_root, cfg['crop_size'], unlabeled_records)
    labeled_loader = DataLoader(labeled_data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                                num_workers=cfg['num_workers'], pin_memory=True)
    unlabeled_loader = DataLoader(unlabeled_data, batch_size=cfg['batch_size'], shuffle=True, drop_last=True,
                                  num_workers=cfg['num_workers'], pin_memory=True)
    steps_per_epoch = len(unlabeled_loader)
    if steps_per_epoch == 0:
        raise ValueError(f"Unlabeled set is smaller than batch-size {cfg['batch_size']}")

    model = _model(cfg, pretrained).cuda().train()
    model_ema = deepcopy(model).eval()
    for param in model_ema.parameters():
        param.requires_grad_(False)
    optimizer = torch.optim.AdamW(
        [
            {'params': [parameter for parameter in model.backbone.parameters() if parameter.requires_grad],
             'lr': cfg['lr']},
            {'params': [parameter for name, parameter in model.named_parameters()
                        if 'backbone' not in name and parameter.requires_grad],
             'lr': cfg['lr'] * float(cfg.get('lr_multi', 40.0))},
        ],
        lr=cfg['lr'], betas=(0.9, 0.999), weight_decay=float(cfg.get('weight_decay', 0.01)),
    )
    scaler = torch.amp.GradScaler('cuda', enabled=cfg['amp'])
    criterion_l = nn.CrossEntropyLoss(ignore_index=cfg['ignore_index'])
    criterion_u = nn.CrossEntropyLoss(reduction='none')
    total_steps = max(1, steps_per_epoch * cfg['epochs'])
    global_step = 0
    first_epoch = 0
    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu', weights_only=False)
        model.load_state_dict(checkpoint['student'])
        model_ema.load_state_dict(checkpoint['model_ema'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        if checkpoint.get('scaler'):
            scaler.load_state_dict(checkpoint['scaler'])
        global_step = int(checkpoint.get('global_step', 0))
        first_epoch = int(checkpoint.get('epoch', -1)) + 1
        rng_state = checkpoint.get('rng_state')
        if rng_state:
            random.setstate(rng_state['python'])
            np.random.set_state(rng_state['numpy'])
            torch.set_rng_state(rng_state['torch'])
            if torch.cuda.is_available() and rng_state.get('cuda'):
                torch.cuda.set_rng_state_all(rng_state['cuda'])
        logger.info('Resumed %s after epoch %d at step %d', args.resume, first_epoch, global_step)

    (args.output_root / 'run_config.json').write_text(json.dumps({
        'config': cfg,
        'data_root': str(args.data_root.resolve()),
        'labels_root': str(args.labels_root.resolve()),
        'labeled_manifest': str(args.labeled_manifest.resolve()),
        'unlabeled_manifest': str(args.unlabeled_manifest.resolve()),
        'pretrained': str(Path(pretrained).resolve()),
        'labeled_count': len(labeled_records),
        'unlabeled_count': len(unlabeled_records),
        'max_steps': args.max_steps,
    }, indent=2) + '\n', encoding='utf-8')
    metrics_path = args.output_root / 'train_metrics.csv'
    existing_outputs = [path.name for path in (args.output_root / 'latest.pth', args.output_root / 'final.pth', metrics_path)
                        if path.exists()]
    if not args.resume and existing_outputs:
        raise FileExistsError(f'Training output already contains {existing_outputs}; choose a fresh output-root or pass --resume')

    logger.info('Classes: %s', CLASSES['r2s100k'])
    logger.info('Train without validation: labeled=%d, unlabeled=%d, crop=%d, batch/GPU=%d, epochs=%d',
                len(labeled_records), len(unlabeled_records), cfg['crop_size'], cfg['batch_size'], cfg['epochs'])
    gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    logger.info('Device: %s (%.1f GB), AMP=%s', torch.cuda.get_device_name(), gpu_memory_gb, cfg['amp'])

    stop = False
    for epoch in range(first_epoch, cfg['epochs']):
        model.train()
        loader = zip(labeled_loader, unlabeled_loader)
        running = []
        started = time.time()
        for step_in_epoch, ((img_x, mask_x), (img_u_w, img_u_s1, img_u_s2, valid_u)) in enumerate(loader):
            torch.cuda.reset_peak_memory_stats()
            progress = global_step / total_steps
            lr = float(cfg['lr']) * (1.0 - min(progress, 1.0)) ** 0.9
            optimizer.param_groups[0]['lr'] = lr
            optimizer.param_groups[1]['lr'] = lr * float(cfg.get('lr_multi', 40.0))

            img_x = img_x.cuda(non_blocking=True)
            mask_x = mask_x.cuda(non_blocking=True)
            img_u_w, img_u_s1, img_u_s2 = (tensor.cuda(non_blocking=True) for tensor in (img_u_w, img_u_s1, img_u_s2))
            valid_u = valid_u.cuda(non_blocking=True).bool()
            box1 = torch.stack([obtain_cutmix_box_hw(cfg['crop_size'], cfg['crop_size']) for _ in range(cfg['batch_size'])]).cuda()
            box2 = torch.stack([obtain_cutmix_box_hw(cfg['crop_size'], cfg['crop_size']) for _ in range(cfg['batch_size'])]).cuda()

            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg['amp']):
                pred_u_w = model_ema(img_u_w)
                prob_u_w = pred_u_w.float().softmax(dim=1)
                conf_u_w, mask_u_w = prob_u_w.max(dim=1)

            img_s1 = img_u_s1.clone()
            img_s2 = img_u_s2.clone()
            box1_img, box2_img = box1.unsqueeze(1).bool(), box2.unsqueeze(1).bool()
            img_s1[box1_img.expand_as(img_s1)] = img_s1.flip(0)[box1_img.expand_as(img_s1)]
            img_s2[box2_img.expand_as(img_s2)] = img_s2.flip(0)[box2_img.expand_as(img_s2)]
            
            mask1, conf1, valid1 = mask_u_w.clone(), conf_u_w.clone(), valid_u.clone()
            mask2, conf2, valid2 = mask_u_w.clone(), conf_u_w.clone(), valid_u.clone()
            mask1[box1.bool()] = mask_u_w.flip(0)[box1.bool()]
            conf1[box1.bool()] = conf_u_w.flip(0)[box1.bool()]
            valid1[box1.bool()] = valid_u.flip(0)[box1.bool()]
            mask2[box2.bool()] = mask_u_w.flip(0)[box2.bool()]
            conf2[box2.bool()] = conf_u_w.flip(0)[box2.bool()]
            valid2[box2.bool()] = valid_u.flip(0)[box2.bool()]

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type='cuda', dtype=torch.float16, enabled=cfg['amp']):
                pred_x = model(img_x)
                pred_s1, pred_s2 = model(torch.cat((img_s1, img_s2), dim=0), comp_drop=True).chunk(2)
                loss_x = criterion_l(pred_x.float(), mask_x)
                valid_area1 = valid1.sum().clamp_min(1)
                valid_area2 = valid2.sum().clamp_min(1)
                loss_s1 = criterion_u(pred_s1.float(), mask1)
                loss_s1 = (loss_s1 * (conf1.ge(cfg['conf_thresh']) & valid1)).sum() / valid_area1
                loss_s2 = criterion_u(pred_s2.float(), mask2)
                loss_s2 = (loss_s2 * (conf2.ge(cfg['conf_thresh']) & valid2)).sum() / valid_area2
                loss_u = (loss_s1 + loss_s2) / 2.0
                loss = (loss_x + loss_u) / 2.0

            if not torch.isfinite(loss):
                raise FloatingPointError(f'Non-finite loss at epoch={epoch}, step={global_step}: {loss.item()}')
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            ema_ratio = min(1.0 - 1.0 / (global_step + 1.0), 0.996)
            with torch.no_grad():
                for param, param_ema in zip(model.parameters(), model_ema.parameters()):
                    param_ema.mul_(ema_ratio).add_(param.detach(), alpha=1.0 - ema_ratio)
                for buffer, buffer_ema in zip(model.buffers(), model_ema.buffers()):
                    buffer_ema.copy_(buffer)

            coverage = ((conf_u_w >= cfg['conf_thresh']) & valid_u).sum().float() / valid_u.sum().clamp_min(1)
            memory_gb = torch.cuda.max_memory_allocated() / (1024 ** 3)
            values = {
                'step': global_step + 1,
                'epoch': epoch,
                'loss': float(loss.item()),
                'loss_labeled': float(loss_x.item()),
                'loss_unlabeled': float(loss_u.item()),
                'pseudo_coverage': float(coverage.item()),
                'learning_rate': lr,
                'gpu_memory_gb': memory_gb,
            }
            _write_step(metrics_path, values)
            running.append(values)
            global_step += 1
            if global_step % 20 == 0 or global_step == 1:
                logger.info('epoch=%d step=%d/%d loss=%.4f labeled=%.4f unlabeled=%.4f coverage=%.3f lr=%.2e gpu=%.2fGB',
                            epoch + 1, step_in_epoch + 1, steps_per_epoch, values['loss'], values['loss_labeled'],
                            values['loss_unlabeled'], values['pseudo_coverage'], lr, memory_gb)
            if args.max_steps and global_step >= args.max_steps:
                stop = True
                break

        if not running:
            raise RuntimeError('No training steps completed in the epoch')
        avg_loss = float(np.mean([row['loss'] for row in running]))
        logger.info('epoch=%d completed (%d steps, %.1fs), mean loss=%.4f', epoch + 1, len(running),
                    time.time() - started, avg_loss)
        checkpoint = {
            'student': model.state_dict(),
            'model_ema': model_ema.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scaler': scaler.state_dict(),
            'epoch': epoch,
            'global_step': global_step,
            'config': cfg,
            'seed': cfg['seed'],
            'labeled_manifest': str(args.labeled_manifest.resolve()),
            'unlabeled_manifest': str(args.unlabeled_manifest.resolve()),
            'pretrained': str(Path(pretrained).resolve()),
            'rng_state': {
                'python': random.getstate(),
                'numpy': np.random.get_state(),
                'torch': torch.get_rng_state(),
                'cuda': torch.cuda.get_rng_state_all(),
            },
        }
        torch.save(checkpoint, args.output_root / 'latest.pth')
        if stop or epoch + 1 == cfg['epochs']:
            torch.save(checkpoint, args.output_root / 'final.pth')
        if stop:
            break

    logger.info('Training finished at step %d. Checkpoints: %s', global_step, args.output_root)


if __name__ == '__main__':
    main()
