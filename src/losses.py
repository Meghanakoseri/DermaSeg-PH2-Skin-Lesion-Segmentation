from __future__ import annotations

import torch
from torch import nn
import torch.nn.functional as F


def soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    target_float = (target == 1).float()
    dims = (1, 2)
    intersection = (probabilities * target_float).sum(dim=dims)
    denominator = probabilities.sum(dim=dims) + target_float.sum(dim=dims)
    dice = (2.0 * intersection + eps) / (denominator + eps)
    return 1.0 - dice.mean()


def soft_iou_loss(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    target_float = (target == 1).float()
    dims = (1, 2)
    intersection = (probabilities * target_float).sum(dim=dims)
    union = probabilities.sum(dim=dims) + target_float.sum(dim=dims) - intersection
    iou = (intersection + eps) / (union + eps)
    return 1.0 - iou.mean()


def focal_tversky_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    alpha: float = 0.5,
    beta: float = 0.5,
    gamma: float = 0.75,
    eps: float = 1e-6,
):
    probabilities = torch.softmax(logits, dim=1)[:, 1]
    target_float = (target == 1).float()
    dims = (1, 2)
    tp = (probabilities * target_float).sum(dim=dims)
    fp = (probabilities * (1 - target_float)).sum(dim=dims)
    fn = ((1 - probabilities) * target_float).sum(dim=dims)
    tversky = (tp + eps) / (tp + alpha * fp + beta * fn + eps)
    return ((1 - tversky) ** gamma).mean()


class SegmentationLoss(nn.Module):
    def __init__(self, name: str, overlap_weight: float = 0.5):
        super().__init__()
        self.name = name
        self.overlap_weight = overlap_weight
        self.ce = nn.CrossEntropyLoss()

    def forward(self, logits, target):
        ce_loss = self.ce(logits, target.long())
        if self.name == "cross_entropy":
            return ce_loss
        if self.name == "ce_dice":
            overlap = soft_dice_loss(logits, target)
        elif self.name == "ce_iou":
            overlap = soft_iou_loss(logits, target)
        elif self.name == "ce_tversky":
            overlap = focal_tversky_loss(logits, target, gamma=1.0)
        elif self.name == "focal_tversky":
            overlap = focal_tversky_loss(logits, target)
        else:
            raise ValueError(f"Unknown loss: {self.name}")
        return (1 - self.overlap_weight) * ce_loss + self.overlap_weight * overlap
