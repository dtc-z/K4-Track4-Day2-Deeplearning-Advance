"""Backbone construction and parameter/FLOP accounting using timm."""
from __future__ import annotations

import warnings

import torch

SUGGESTED_BACKBONES = {
    "resnet50": "resnet50",
    "resnext50": "resnext50_32x4d",
    "convnext_tiny": "convnext_tiny",
    "deit_small": "deit_small_patch16_224",
    "swin_tiny": "swin_tiny_patch4_window7_224",
    "efficientnet_b0": "efficientnet_b0",
    "mobilenetv3": "mobilenetv3_large_100",
}


def _timm():
    try:
        import timm
    except Exception as exc:
        raise RuntimeError("timm is required. Install requirements.txt in Colab/Kaggle.") from exc
    return timm


def build_model(name: str, pretrained: bool = True, num_classes: int = 9,
                drop_rate: float = 0.0, init: str = "finetune"):
    if init not in {"scratch", "frozen", "finetune"}:
        raise ValueError("init must be scratch, frozen, or finetune")
    timm = _timm()
    use_pretrained = bool(pretrained and init != "scratch")
    model = timm.create_model(name, pretrained=use_pretrained,
                              num_classes=num_classes, drop_rate=drop_rate)
    cfg = getattr(model, "pretrained_cfg", None) or getattr(model, "default_cfg", {}) or {}
    if isinstance(cfg, dict):
        tag = cfg.get("tag") or cfg.get("hf_hub_id") or cfg.get("url") or "timm-default"
    else:
        tag = (getattr(cfg, "tag", None) or getattr(cfg, "hf_hub_id", None)
               or getattr(cfg, "url", None) or str(cfg))
    model._lab_pretrained_tag = tag if use_pretrained else "random-init"
    model._lab_init = init
    if init == "frozen":
        freeze_backbone(model)
    return model


def _classifier(model):
    if not hasattr(model, "get_classifier"):
        raise TypeError(f"{type(model).__name__} does not expose get_classifier(); cannot identify head")
    classifier = model.get_classifier()
    if not isinstance(classifier, torch.nn.Module):
        raise TypeError("model.get_classifier() did not return a torch module")
    return classifier


def freeze_backbone(model) -> None:
    classifier_ids = {id(p) for p in _classifier(model).parameters()}
    for parameter in model.parameters():
        parameter.requires_grad_(id(parameter) in classifier_ids)
    model._lab_frozen_backbone = True


def set_frozen_backbone_eval(model) -> None:
    """Keep all frozen feature layers (including BatchNorm) in eval mode."""
    if getattr(model, "_lab_frozen_backbone", False):
        model.eval()
        _classifier(model).train()


def param_groups(model, lr_backbone: float, lr_head: float, weight_decay: float):
    head_ids = {id(p) for p in _classifier(model).parameters()}
    grouped: dict[tuple[str, float], list[torch.nn.Parameter]] = {}
    for parameter in model.parameters():
        if not parameter.requires_grad:
            continue
        is_head = id(parameter) in head_ids
        decay = float(weight_decay) if (is_head or parameter.ndim > 1) else 0.0
        lr = float(lr_head if is_head else lr_backbone)
        grouped.setdefault(("head" if is_head else "backbone", decay), []).append(parameter)
    groups = []
    for (kind, decay), params in grouped.items():
        groups.append({"params": params,
                       "lr": float(lr_head if kind == "head" else lr_backbone),
                       "weight_decay": decay})
    if not groups:
        raise ValueError("No trainable parameters found")
    return groups


def count_params(model) -> float:
    return sum(parameter.numel() for parameter in model.parameters()) / 1e6


def count_gmacs(model, img_size: int = 224) -> float:
    """Estimate MACs with fvcore when installed, otherwise thop; return GMAC/image.

    fvcore counts FLOPs, so divide by two to report multiply-accumulate operations.
    Unsupported ops are ignored and the tool is recorded in the run metadata.
    """
    original_device = next(model.parameters()).device
    was_training = model.training
    model.eval()
    sample = torch.zeros(1, 3, img_size, img_size, device=original_device)
    fvcore_error = None
    try:
        try:
            from fvcore.nn import FlopCountAnalysis
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                flops = float(FlopCountAnalysis(model, sample).total())
            model._lab_gmac_tool = "fvcore FLOPs / 2"
            return flops / 2e9
        except ImportError as exc:
            fvcore_error = exc
        except Exception as exc:
            fvcore_error = exc
        try:
            from thop import profile
            macs, _ = profile(model, inputs=(sample,), verbose=False)
            model._lab_gmac_tool = "thop MACs"
            return float(macs) / 1e9
        except Exception as thop_error:
            raise RuntimeError(f"GMAC counting failed with fvcore ({fvcore_error}) and thop ({thop_error})") from thop_error
    finally:
        model.train(was_training)
