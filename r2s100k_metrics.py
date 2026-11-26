"""Numpy-only metrics for full and confidence-filtered R2S100K masks."""

import numpy as np


def confusion_matrix(target, prediction, nclass=15, ignore_index=255):
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    if target.shape != prediction.shape:
        raise ValueError(f'Target/prediction size mismatch: {target.shape} vs {prediction.shape}')
    target_valid = target != ignore_index
    if np.any((target[target_valid] < 0) | (target[target_valid] >= nclass)):
        raise ValueError('Ground-truth mask contains class IDs outside the configured class range')
    if np.any((prediction != ignore_index) & ((prediction < 0) | (prediction >= nclass))):
        raise ValueError('Prediction contains class IDs outside the configured class range')
    valid = target_valid & (prediction != ignore_index)
    encoded = nclass * target[valid] + prediction[valid]
    return np.bincount(encoded, minlength=nclass * nclass).reshape(nclass, nclass).astype(np.int64)


def summarize_confusion(confusion, total_target_pixels, nclass=15):
    intersection = np.diag(confusion).astype(np.float64)
    target = confusion.sum(axis=1).astype(np.float64)
    predicted = confusion.sum(axis=0).astype(np.float64)
    union = target + predicted - intersection
    iou = np.divide(intersection, union, out=np.full(nclass, np.nan), where=union > 0)
    dice_denominator = target + predicted
    dice = np.divide(2 * intersection, dice_denominator, out=np.full(nclass, np.nan), where=dice_denominator > 0)
    accuracy = float(intersection.sum() / confusion.sum()) if confusion.sum() else None
    defined = np.isfinite(iou)
    defined_no_bg = defined.copy()
    if nclass:
        defined_no_bg[0] = False
    return {
        'confusion_matrix': confusion,
        'intersection': intersection,
        'target_pixels': target,
        'predicted_pixels': predicted,
        'union': union,
        'iou': iou,
        'dice': dice,
        'pixel_accuracy': accuracy,
        'miou_with_background': float(np.nanmean(iou)) if np.any(defined) else None,
        'miou_without_background': float(np.nanmean(iou[defined_no_bg])) if np.any(defined_no_bg) else None,
        'classes_in_miou': int(defined.sum()),
        'classes_in_miou_without_background': int(defined_no_bg.sum()),
        'target_pixel_count': int(total_target_pixels),
        'evaluated_pixel_count': int(confusion.sum()),
        'coverage': float(confusion.sum() / total_target_pixels) if total_target_pixels else None,
    }


def metrics_for_pair(target, prediction, nclass=15, ignore_index=255):
    target = np.asarray(target, dtype=np.int64)
    prediction = np.asarray(prediction, dtype=np.int64)
    total_target_pixels = int(np.count_nonzero(target != ignore_index))
    matrix = confusion_matrix(target, prediction, nclass, ignore_index)
    return summarize_confusion(matrix, total_target_pixels, nclass)
