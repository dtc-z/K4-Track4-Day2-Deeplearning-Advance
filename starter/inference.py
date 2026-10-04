"""Validation-only TTA, calibration, ensembling, and Conv/BatchNorm fusion."""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F


def _to_numpy(value):
    return value.detach().float().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)


@torch.inference_mode()
def predict_logits(model, loader, device, view=None):
    model.eval()
    device = torch.device(device)
    filenames, labels, outputs = [], [], []
    for images, target, batch_names in loader:
        images = images.to(device, non_blocking=True)
        if view is not None:
            images = view(images)
        logits = model(images)
        filenames.extend(str(name) for name in batch_names)
        labels.append(target.cpu().numpy())
        outputs.append(logits.float().cpu().numpy())
    if not outputs:
        raise ValueError("loader produced no batches")
    return filenames, np.concatenate(labels), np.concatenate(outputs)


def view_identity(x):
    return x


def view_hflip(x):
    if x.ndim != 4:
        raise ValueError("expected a batch of images shaped (N,C,H,W)")
    return torch.flip(x, dims=(-1,))


def views_multicrop(x, crop: int):
    if x.ndim != 4:
        raise ValueError("expected x shaped (N,C,H,W)")
    height, width = x.shape[-2:]
    if crop > min(height, width) or crop < 1:
        raise ValueError(f"crop={crop} does not fit input {height}x{width}")
    positions = [(0, 0), (0, width - crop), (height - crop, 0),
                 (height - crop, width - crop), ((height - crop) // 2, (width - crop) // 2)]
    return [x[:, :, top:top + crop, left:left + crop] for top, left in positions]


def views_multiscale(x, sizes):
    if x.ndim != 4:
        raise ValueError("expected x shaped (N,C,H,W)")
    views = []
    for size in sizes:
        if int(size) < 1:
            raise ValueError("all sizes must be positive")
        views.append(F.interpolate(x, size=(int(size), int(size)), mode="bilinear",
                                   align_corners=False, antialias=True))
    return views


def _softmax(logits):
    if torch.is_tensor(logits):
        return torch.softmax(logits, dim=1)
    values = np.asarray(logits, dtype=np.float64)
    values = values - values.max(axis=1, keepdims=True)
    exps = np.exp(values)
    return exps / exps.sum(axis=1, keepdims=True)


def aggregate_views(logits_per_view, space: str = "prob"):
    if not logits_per_view:
        raise ValueError("at least one view is required")
    space = space.lower()
    if space not in {"prob", "logit"}:
        raise ValueError("space must be 'prob' or 'logit'")
    arrays = [torch.as_tensor(value) for value in logits_per_view]
    shape = arrays[0].shape
    if len(shape) != 2 or any(value.shape != shape for value in arrays):
        raise ValueError("all views must have matching (N,K) logits")
    if space == "prob":
        probs = torch.stack([torch.softmax(value, dim=1) for value in arrays]).mean(0)
    else:
        probs = torch.softmax(torch.stack(arrays).mean(0), dim=1)
    probs = probs / probs.sum(1, keepdim=True).clamp_min(1e-12)
    return _to_numpy(probs)


def ensemble_probs(list_of_probs):
    if not list_of_probs:
        raise ValueError("at least one probability matrix is required")
    arrays = [np.asarray(p, dtype=np.float64) for p in list_of_probs]
    shape = arrays[0].shape
    if len(shape) != 2 or any(p.shape != shape for p in arrays):
        raise ValueError("all probability matrices must have the same (N,K) shape and image order")
    if any(not np.isfinite(p).all() or (p < 0).any() for p in arrays):
        raise ValueError("probabilities must be finite and non-negative")
    probs = np.mean(arrays, axis=0)
    return probs / np.clip(probs.sum(axis=1, keepdims=True), 1e-12, None)


def fit_temperature(val_logits, val_labels) -> float:
    logits = torch.as_tensor(val_logits, dtype=torch.float64, device="cpu")
    labels = torch.as_tensor(val_labels, dtype=torch.long, device="cpu")
    if logits.ndim != 2 or labels.ndim != 1 or len(logits) != len(labels):
        raise ValueError("val_logits must be (N,K) and val_labels must be (N,)")
    if len(labels) == 0:
        raise ValueError("cannot fit temperature on an empty validation set")
    log_temperature = torch.nn.Parameter(torch.zeros((), dtype=torch.float64))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100,
                                  line_search_fn="strong_wolfe")

    def closure():
        optimizer.zero_grad()
        temperature = log_temperature.clamp(-5.0, 5.0).exp()
        loss = F.cross_entropy(logits / temperature, labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(log_temperature.detach().clamp(-5.0, 5.0).exp())


def apply_temperature(logits, T: float):
    if T <= 0 or not np.isfinite(T):
        raise ValueError("temperature must be finite and positive")
    scaled = torch.as_tensor(logits, dtype=torch.float64) / float(T)
    return torch.softmax(scaled, dim=1).cpu().numpy()


def _fuse_pairs(parent):
    fused_count = 0
    names = list(parent._modules.keys())
    for child in list(parent.children()):
        fused_count += _fuse_pairs(child)
    for left_name, right_name in zip(names, names[1:]):
        left = parent._modules[left_name]
        right = parent._modules[right_name]
        if isinstance(left, torch.nn.Conv2d) and isinstance(right, torch.nn.BatchNorm2d):
            if not right.track_running_stats or right.running_mean is None:
                continue
            from torch.nn.utils.fusion import fuse_conv_bn_eval
            parent._modules[left_name] = fuse_conv_bn_eval(left, right)
            parent._modules[right_name] = torch.nn.Identity()
            fused_count += 1
    return fused_count


@torch.no_grad()
def fuse_conv_bn(model):
    """Return a fused copy and print the maximum eval-output difference."""
    reference = copy.deepcopy(model).eval()
    fused = copy.deepcopy(model).eval()
    device = next(fused.parameters()).device
    dtype = next(fused.parameters()).dtype
    input_size = 224
    cfg = getattr(model, "pretrained_cfg", {}) or getattr(model, "default_cfg", {}) or {}
    if isinstance(cfg, dict):
        size = cfg.get("input_size")
        if size and len(size) == 3:
            input_size = int(size[-1])
    sample = torch.randn(1, 3, input_size, input_size, device=device, dtype=dtype)
    before = reference(sample)
    fused_count = _fuse_pairs(fused)
    after = fused(sample)
    max_error = float((before.float() - after.float()).abs().max())
    print(f"Conv/BatchNorm fusion max absolute error: {max_error:.3e}")
    if fused_count == 0:
        print("No adjacent Conv2d/BatchNorm2d pairs were found (not applicable for this architecture).")
    return fused
