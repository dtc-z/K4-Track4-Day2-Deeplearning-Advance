"""Classification losses and batch-level Mixup/CutMix."""
from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def build_criterion(kind: str = "ce", **kw):
    kind = kind.lower()
    if kind == "ce":
        smoothing = float(kw.get("smoothing", kw.get("label_smoothing", 0.0)))
        return nn.CrossEntropyLoss(label_smoothing=smoothing)
    if kind == "ls":
        return LabelSmoothingCE(float(kw.get("smoothing", kw.get("label_smoothing", 0.1))))
    if kind == "focal":
        return FocalLoss(gamma=float(kw.get("gamma", 2.0)), alpha=kw.get("alpha"))
    if kind == "ce_weighted":
        weight = kw.get("weight")
        if weight is None:
            raise ValueError("ce_weighted requires weight=<class weight tensor>")
        return nn.CrossEntropyLoss(weight=torch.as_tensor(weight, dtype=torch.float32))
    raise ValueError(f"Unknown loss {kind!r}; choose ce, ls, focal, or ce_weighted")


class LabelSmoothingCE(nn.Module):
    """Cross entropy with uniform label smoothing."""

    def __init__(self, smoothing: float = 0.1):
        super().__init__()
        if not 0.0 <= smoothing < 1.0:
            raise ValueError("smoothing must be in [0, 1)")
        self.smoothing = float(smoothing)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return F.cross_entropy(logits, target.long(), label_smoothing=self.smoothing)


class FocalLoss(nn.Module):
    """Multiclass focal loss; gamma=0 reduces exactly to (weighted) CE."""

    def __init__(self, gamma: float = 2.0, alpha=None):
        super().__init__()
        if gamma < 0:
            raise ValueError("gamma must be non-negative")
        self.gamma = float(gamma)
        if alpha is None:
            self.register_buffer("alpha", None)
        else:
            self.register_buffer("alpha", torch.as_tensor(alpha, dtype=torch.float32))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.long()
        log_probs = F.log_softmax(logits, dim=1)
        log_pt = log_probs.gather(1, target[:, None]).squeeze(1)
        pt = log_pt.exp()
        focal = (1.0 - pt).clamp_min(0).pow(self.gamma)
        loss = -focal * log_pt
        if self.alpha is not None:
            if self.alpha.numel() != logits.shape[1]:
                raise ValueError(f"alpha must contain {logits.shape[1]} class weights")
            loss = loss * self.alpha.to(logits.device, logits.dtype)[target]
        return loss.mean()


def class_weights(counts, beta: float = 0.0) -> torch.Tensor:
    counts = torch.as_tensor(counts, dtype=torch.float64)
    if counts.ndim != 1 or counts.numel() != 9 or torch.any(counts <= 0):
        raise ValueError("counts must contain 9 positive training class counts")
    if beta < 0 or beta >= 1:
        raise ValueError("beta must be in [0, 1); use 0 for inverse frequency")
    if beta == 0:
        weights = counts.reciprocal()
    else:
        # Numerically stable form of (1-beta)/(1-beta**n).
        log_beta = np.log(beta)
        denominator = -torch.expm1(counts * log_beta)
        weights = (1.0 - beta) / denominator
    weights = weights / weights.mean()
    return weights.to(dtype=torch.float32)


def mix_batch(x: torch.Tensor, y: torch.Tensor, alpha: float = 1.0, mode: str = "cutmix"):
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    if x.ndim != 4 or y.ndim != 1 or len(x) != len(y):
        raise ValueError("expected x=(N,C,H,W), y=(N,) with matching batch size")
    if mode not in {"mixup", "cutmix"}:
        raise ValueError("mode must be 'mixup' or 'cutmix'")
    if len(x) < 2:
        return x, (y, y, 1.0)
    lam = float(np.random.beta(alpha, alpha))
    permutation = torch.randperm(len(x), device=x.device)
    y_a, y_b = y, y[permutation]
    if mode == "mixup":
        return lam * x + (1.0 - lam) * x[permutation], (y_a, y_b, lam)

    _, _, height, width = x.shape
    ratio = float(np.sqrt(1.0 - lam))
    box_w = int(round(width * ratio))
    box_h = int(round(height * ratio))
    center_x = int(torch.randint(width, (1,), device=x.device).item())
    center_y = int(torch.randint(height, (1,), device=x.device).item())
    x1, x2 = max(center_x - box_w // 2, 0), min(center_x + box_w // 2, width)
    y1, y2 = max(center_y - box_h // 2, 0), min(center_y + box_h // 2, height)
    mixed = x.clone()
    mixed[:, :, y1:y2, x1:x2] = x[permutation, :, y1:y2, x1:x2]
    actual_area = max(0, x2 - x1) * max(0, y2 - y1)
    lam_actual = 1.0 - actual_area / float(height * width)
    return mixed, (y_a, y_b, lam_actual)


def mixed_loss(criterion, logits: torch.Tensor, targets):
    y_a, y_b, lam = targets
    return float(lam) * criterion(logits, y_a) + (1.0 - float(lam)) * criterion(logits, y_b)
