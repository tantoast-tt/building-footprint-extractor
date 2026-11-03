# -*- coding: utf-8 -*-
"""이진 마스크 평가 지표. numpy 기반이라 torch 없이도 동작."""
import numpy as np


def _binarize(a, thr=0.5):
    return (np.asarray(a) > thr).astype(np.uint8)


def iou_score(pred, target, thr=0.5, eps=1e-7):
    p, t = _binarize(pred, thr), _binarize(target, thr)
    inter = np.logical_and(p, t).sum()
    union = np.logical_or(p, t).sum()
    return float((inter + eps) / (union + eps))


def f1_score(pred, target, thr=0.5, eps=1e-7):
    p, t = _binarize(pred, thr), _binarize(target, thr)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, 1 - t).sum()
    fn = np.logical_and(1 - p, t).sum()
    return float((2 * tp + eps) / (2 * tp + fp + fn + eps))


def precision_recall(pred, target, thr=0.5, eps=1e-7):
    p, t = _binarize(pred, thr), _binarize(target, thr)
    tp = np.logical_and(p, t).sum()
    fp = np.logical_and(p, 1 - t).sum()
    fn = np.logical_and(1 - p, t).sum()
    return float((tp + eps) / (tp + fp + eps)), float((tp + eps) / (tp + fn + eps))


def evaluate_pair(pred, target, thr=0.5):
    prec, rec = precision_recall(pred, target, thr)
    return {
        "iou": iou_score(pred, target, thr),
        "f1": f1_score(pred, target, thr),
        "precision": prec,
        "recall": rec,
    }
