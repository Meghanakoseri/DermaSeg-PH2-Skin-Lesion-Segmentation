from __future__ import annotations

import torch


def confusion_counts(prediction: torch.Tensor, target: torch.Tensor):
    prediction = prediction.bool()
    target = target.bool()
    tp = (prediction & target).sum().item()
    fp = (prediction & ~target).sum().item()
    fn = (~prediction & target).sum().item()
    tn = (~prediction & ~target).sum().item()
    return tp, fp, fn, tn


def metrics_from_counts(tp: int, fp: int, fn: int, tn: int, eps: float = 1e-8):
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)
    iou = tp / (tp + fp + fn + eps)
    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "iou": iou,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }
