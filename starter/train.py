"""Shared, resumable training pipeline for the DeepWeeds lab."""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import get_args, get_origin, get_type_hints, Union

import numpy as np
import pandas as pd
import torch
from torch.nn.utils import clip_grad_norm_

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next((parent for parent in SCRIPT_PATH.parents if (parent / "eval.py").is_file()), SCRIPT_PATH.parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from . import dataset as dataset_lib, losses, model as model_lib
except ImportError:  # Supports ``python starter/train.py`` as well as package imports.
    import dataset as dataset_lib
    import losses
    import model as model_lib

from eval import compute_metrics, save_predictions


@dataclass
class Config:
    exp_id: str = "T00"
    seed: int = 0
    fold: int = 0
    backbone: str = "resnet50"
    init: str = "finetune"             # scratch | frozen | finetune
    drop_rate: float = 0.0
    img_size: int = 224
    aug: str = "basic"                 # basic | color | trivial | randaug | none
    sampler: str | None = None         # None | balanced
    mix: str | None = None             # None | mixup | cutmix
    mix_alpha: float = 1.0
    loss: str = "ce"                   # ce | ls | focal | ce_weighted
    label_smoothing: float = 0.0
    focal_gamma: float = 2.0
    class_weight_beta: float | None = None
    epochs: int = 12
    batch_size: int = 64
    lr_backbone: float = 1e-4
    lr_head: float = 1e-3
    weight_decay: float = 0.05
    warmup_epochs: float = 1.0
    ema_decay: float | None = None
    amp: bool = True
    num_workers: int = 2
    images_dir: str = "data/images"
    labels_dir: str = "data/labels"
    out_dir: str = "runs"
    pred_dir: str = "predictions"
    curves_dir: str = "curves"
    device: str = "auto"
    resume: str | None = None
    overwrite_run: bool = False
    save_val_predictions: bool = True
    save_test_predictions: bool = False


def run_dir(cfg: Config) -> Path:
    return Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}"


def pred_path(cfg: Config, split: str) -> Path:
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    return Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_{split}.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def _device(cfg: Config) -> torch.device:
    if cfg.device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(cfg.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but no CUDA device is available")
    return device


def build_optimizer(model, cfg: Config):
    groups = model_lib.param_groups(model, cfg.lr_backbone, cfg.lr_head, cfg.weight_decay)
    return torch.optim.AdamW(groups, betas=(0.9, 0.999), eps=1e-8)


def build_scheduler(optimizer, cfg: Config, steps_per_epoch: int):
    total_steps = max(1, int(cfg.epochs * steps_per_epoch))
    warmup_steps = min(total_steps, max(0, int(round(cfg.warmup_epochs * steps_per_epoch))))
    min_ratio = 1e-3

    def multiplier(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(min_ratio, min(1.0, (step + 1) / warmup_steps))
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, progress))
        return min_ratio + (1.0 - min_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, multiplier)


class EMA:
    """EMA model parameters with current (non-averaged) normalization buffers."""

    def __init__(self, model, decay: float):
        import copy
        if not 0.0 < decay < 1.0:
            raise ValueError("ema_decay must be in (0, 1)")
        self.decay = float(decay)
        self.ema_model = copy.deepcopy(model).eval()
        for parameter in self.ema_model.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def update(self, model) -> None:
        source_parameters = dict(model.named_parameters())
        for name, target in self.ema_model.named_parameters():
            source = source_parameters[name].detach()
            target.mul_(self.decay).add_(source, alpha=1.0 - self.decay)
        source_buffers = dict(model.named_buffers())
        for name, target in self.ema_model.named_buffers():
            target.copy_(source_buffers[name])

    def copy_to(self, model) -> None:
        model.load_state_dict(self.ema_model.state_dict())


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16,
                          enabled=bool(enabled and device.type == "cuda"))


def _make_scaler(enabled: bool):
    try:
        return torch.amp.GradScaler("cuda", enabled=enabled)
    except (AttributeError, TypeError):
        return torch.cuda.amp.GradScaler(enabled=enabled)


def train_one_epoch(model, loader, criterion, optimizer, scheduler, scaler, cfg: Config,
                    device, ema: EMA | None = None) -> dict:
    model.train()
    model_lib.set_frozen_backbone_eval(model)
    total_loss = 0.0
    total_items = 0
    started = time.perf_counter()
    for images, labels, _ in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        targets = None
        if cfg.mix:
            images, targets = losses.mix_batch(images, labels, cfg.mix_alpha, cfg.mix)
        with _autocast(device, cfg.amp):
            logits = model(images)
            loss = losses.mixed_loss(criterion, logits, targets) if targets is not None else criterion(logits, labels)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        clip_grad_norm_([p for p in model.parameters() if p.requires_grad], max_norm=5.0)
        scale_before_step = scaler.get_scale()
        scaler.step(optimizer)
        scaler.update()
        # GradScaler skips optimizer.step() on overflow and lowers its scale.
        # Do not advance the LR schedule for an update that did not happen.
        if scaler.get_scale() >= scale_before_step:
            scheduler.step()
        if ema is not None:
            ema.update(model)
        count = int(labels.shape[0])
        total_loss += float(loss.detach()) * count
        total_items += count
    return {
        "train_loss": total_loss / max(1, total_items),
        "lr": float(optimizer.param_groups[0]["lr"]),
        "epoch_seconds": time.perf_counter() - started,
    }


@torch.inference_mode()
def evaluate(model, loader, criterion, device):
    model.eval()
    filenames, labels_all, logits_all = [], [], []
    total_loss, total_items = 0.0, 0
    for images, labels, batch_names in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images)
        loss = criterion(logits, labels)
        total_loss += float(loss) * len(labels)
        total_items += len(labels)
        filenames.extend(str(name) for name in batch_names)
        labels_all.append(labels.cpu().numpy())
        logits_all.append(logits.float().cpu().numpy())
    return (filenames, np.concatenate(labels_all), np.concatenate(logits_all),
            total_loss / max(1, total_items))


def _softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=np.float64)
    z = z - z.max(axis=1, keepdims=True)
    exp = np.exp(z)
    return exp / exp.sum(axis=1, keepdims=True)


def plot_curves(history: list[dict], path: str | Path, title: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    frame = pd.DataFrame(history)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, left = plt.subplots(figsize=(9, 5.5))
    epochs = frame["epoch"]
    left.plot(epochs, frame["train_loss"], marker="o", label="Train loss")
    left.plot(epochs, frame["val_loss"], marker="o", label="Val loss")
    left.set_xlabel("Epoch")
    left.set_ylabel("Loss")
    left.grid(alpha=0.25)
    right = left.twinx()
    right.plot(epochs, frame["val_macro_f1"], color="tab:green", marker="s", label="Val macro-F1")
    right.set_ylabel("Validation macro-F1")
    lines, labels = left.get_legend_handles_labels()
    lines2, labels2 = right.get_legend_handles_labels()
    left.legend(lines + lines2, labels + labels2, loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _atomic_torch_save(payload, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def _load_checkpoint(path: str | Path, device):
    try:
        return torch.load(path, map_location=device, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=device)


def _runtime_versions() -> dict:
    import importlib.metadata
    versions = {"python": sys.version.split()[0], "torch": torch.__version__,
                "numpy": np.__version__, "pandas": pd.__version__}
    for package in ("timm", "torchvision", "scikit-learn", "matplotlib"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def run(cfg: Config) -> dict:
    if cfg.epochs < 1 or cfg.batch_size < 1:
        raise ValueError("epochs and batch_size must be positive")
    set_seed(cfg.seed)
    device = _device(cfg)
    out_dir = run_dir(cfg)
    test_prediction_path = pred_path(cfg, "test")
    if cfg.save_test_predictions and test_prediction_path.exists():
        raise FileExistsError(f"Refusing to overwrite {test_prediction_path}; test may be run only once per seed")
    if (out_dir / "summary.json").exists() and not (cfg.resume or cfg.overwrite_run):
        raise FileExistsError(f"Run artifacts already exist in {out_dir}; choose a new exp_id/seed or set overwrite_run=true")
    out_dir.mkdir(parents=True, exist_ok=True)
    config_path = out_dir / "config.json"
    config_json = dataclasses.asdict(cfg)
    config_json.update({"device_resolved": str(device), "versions": _runtime_versions()})
    config_path.write_text(json.dumps(config_json, indent=2, ensure_ascii=False), encoding="utf-8")

    train_df, val_df, test_df = dataset_lib.load_split(cfg.labels_dir, cfg.fold)
    split_summary = dataset_lib.check_split(train_df, val_df, test_df, cfg.images_dir)
    train_loader = dataset_lib.make_loader(
        train_df, cfg.images_dir, dataset_lib.build_transforms(True, cfg.img_size, cfg.aug),
        cfg.batch_size, True, cfg.sampler, cfg.num_workers, cfg.seed,
    )
    val_loader = dataset_lib.make_loader(
        val_df, cfg.images_dir, dataset_lib.build_transforms(False, cfg.img_size),
        cfg.batch_size, False, None, cfg.num_workers, cfg.seed,
    )

    model = model_lib.build_model(cfg.backbone, pretrained=cfg.init != "scratch",
                                  num_classes=dataset_lib.NUM_CLASSES, drop_rate=cfg.drop_rate, init=cfg.init)
    model.to(device)
    params_m = model_lib.count_params(model)
    gmacs = model_lib.count_gmacs(model, cfg.img_size)
    config_json.update({"weight_tag": getattr(model, "_lab_pretrained_tag", "unknown"),
                        "num_params_m": float(params_m), "gmac": float(gmacs),
                        "gmac_tool": getattr(model, "_lab_gmac_tool", "unknown")})
    config_path.write_text(json.dumps(config_json, indent=2, ensure_ascii=False), encoding="utf-8")
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg, len(train_loader))
    class_counts = [int((pd.to_numeric(train_df["Label"]) == i).sum()) for i in range(9)]
    weights = None
    if cfg.loss == "ce_weighted":
        weights = losses.class_weights(class_counts, cfg.class_weight_beta or 0.0)
        weights = weights.to(device)
    criterion = losses.build_criterion(
        cfg.loss, smoothing=cfg.label_smoothing, gamma=cfg.focal_gamma, weight=weights,
    ).to(device)
    scaler = _make_scaler(cfg.amp and device.type == "cuda")
    ema = EMA(model, cfg.ema_decay) if cfg.ema_decay is not None else None

    checkpoint_path = out_dir / "best.pt"
    last_path = out_dir / "last.pt"
    history: list[dict] = []
    best_f1 = -float("inf")
    best_epoch = 0
    start_epoch = 0
    if cfg.resume:
        checkpoint = _load_checkpoint(cfg.resume, device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        scheduler.load_state_dict(checkpoint["scheduler_state"])
        if checkpoint.get("scaler_state"):
            scaler.load_state_dict(checkpoint["scaler_state"])
        if ema is not None and checkpoint.get("ema_state") is not None:
            ema.ema_model.load_state_dict(checkpoint["ema_state"])
        history = checkpoint.get("history", [])
        start_epoch = int(checkpoint["epoch"])
        best_f1 = float(checkpoint.get("best_f1", -float("inf")))
        best_epoch = int(checkpoint.get("best_epoch", 0))
        print(f"Resuming {cfg.exp_id}/seed{cfg.seed} at epoch {start_epoch + 1}")

    for epoch in range(start_epoch, cfg.epochs):
        # Deterministic epoch-specific randomness also makes a resumed run repeatable.
        epoch_seed = cfg.seed + epoch
        random.seed(epoch_seed)
        np.random.seed(epoch_seed)
        torch.manual_seed(epoch_seed)
        train_loader.generator.manual_seed(epoch_seed)
        if isinstance(train_loader.sampler, torch.utils.data.WeightedRandomSampler):
            train_loader.sampler.generator.manual_seed(epoch_seed)
        train_values = train_one_epoch(model, train_loader, criterion, optimizer, scheduler,
                                       scaler, cfg, device, ema)
        eval_model = ema.ema_model if ema is not None else model
        names, y_true, val_logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
        val_probs = _softmax(val_logits)
        metrics = compute_metrics(y_true, val_probs.argmax(1), val_probs)
        row = {
            "epoch": epoch + 1, **train_values, "val_loss": val_loss,
            "val_top1": metrics["top1"], "val_macro_f1": metrics["macro_f1"],
            "val_balanced_acc": metrics["balanced_acc"], "val_ece": metrics["ece"],
            "lr_backbone": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(row)
        pd.DataFrame(history).to_csv(out_dir / "history.csv", index=False)
        improved = metrics["macro_f1"] > best_f1 + 1e-12
        if improved:
            best_f1 = float(metrics["macro_f1"])
            best_epoch = epoch + 1
            best_payload = {
                "epoch": best_epoch, "best_f1": best_f1,
                "model_state": model.state_dict(),
                "ema_state": ema.ema_model.state_dict() if ema is not None else None,
                "val_metrics": {key: float(metrics[key]) for key in ("top1", "macro_f1", "balanced_acc", "ece", "nll")},
            }
            _atomic_torch_save(best_payload, checkpoint_path)
        last_payload = {
            "epoch": epoch + 1, "best_epoch": best_epoch, "best_f1": best_f1,
            "model_state": model.state_dict(),
            "ema_state": ema.ema_model.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict(), "scheduler_state": scheduler.state_dict(),
            "scaler_state": scaler.state_dict(), "history": history,
        }
        _atomic_torch_save(last_payload, last_path)
        print(f"{cfg.exp_id} seed={cfg.seed} epoch {epoch + 1}/{cfg.epochs} "
              f"train_loss={row['train_loss']:.4f} val_loss={val_loss:.4f} "
              f"val_macro_f1={row['val_macro_f1']:.4f} val_top1={row['val_top1']:.4f} "
              f"seconds={row['epoch_seconds']:.1f}")

    if not checkpoint_path.exists():
        raise RuntimeError("No best checkpoint was produced")
    best = _load_checkpoint(checkpoint_path, device)
    model.load_state_dict(best["model_state"])
    if ema is not None and best["ema_state"] is not None:
        ema.ema_model.load_state_dict(best["ema_state"])
        eval_model = ema.ema_model
    else:
        eval_model = model
    names, y_true, val_logits, val_loss = evaluate(eval_model, val_loader, criterion, device)
    val_probs = _softmax(val_logits)
    val_metrics = compute_metrics(y_true, val_probs.argmax(1), val_probs)
    np.savez_compressed(out_dir / "val_logits.npz", filenames=np.asarray(names),
                        y_true=y_true, logits=val_logits)
    if cfg.save_val_predictions:
        save_predictions(pred_path(cfg, "val"), names, y_true, val_probs)

    if cfg.save_test_predictions:
        test_loader = dataset_lib.make_loader(
            test_df, cfg.images_dir, dataset_lib.build_transforms(False, cfg.img_size),
            cfg.batch_size, False, None, cfg.num_workers, cfg.seed,
        )
        test_names, y_test, test_logits, test_loss = evaluate(eval_model, test_loader, criterion, device)
        test_probs = _softmax(test_logits)
        np.savez_compressed(out_dir / "test_logits.npz", filenames=np.asarray(test_names),
                            y_true=y_test, logits=test_logits)
        save_predictions(test_prediction_path, test_names, y_test, test_probs)
        test_metrics = compute_metrics(y_test, test_probs.argmax(1), test_probs)
    else:
        test_metrics, test_loss = None, None

    epoch_seconds = [float(item["epoch_seconds"]) for item in history]
    summary = {
        "exp_id": cfg.exp_id, "seed": cfg.seed, "fold": cfg.fold,
        "backbone": cfg.backbone, "weight_tag": getattr(model, "_lab_pretrained_tag", "unknown"),
        "init": cfg.init, "best_epoch": best_epoch,
        "macro_f1_val": float(val_metrics["macro_f1"]), "top1_val": float(val_metrics["top1"]),
        "balanced_acc_val": float(val_metrics["balanced_acc"]), "ece_val": float(val_metrics["ece"]),
        "val_loss": float(val_loss), "macro_f1_val_per_class": val_metrics["f1"].tolist(),
        "precision_val_per_class": val_metrics["precision"].tolist(),
        "recall_val_per_class": val_metrics["recall"].tolist(),
        "num_params_m": float(params_m), "gmac": float(gmacs),
        "gmac_tool": getattr(model, "_lab_gmac_tool", "unknown"),
        "mean_epoch_seconds": float(np.mean(epoch_seconds)) if epoch_seconds else None,
        "train_seconds": float(sum(epoch_seconds)), "device": str(device),
        "versions": _runtime_versions(), "split_summary": split_summary,
        "test_saved": bool(cfg.save_test_predictions),
    }
    if test_metrics is not None:
        summary.update({
            "macro_f1_test": float(test_metrics["macro_f1"]), "top1_test": float(test_metrics["top1"]),
            "balanced_acc_test": float(test_metrics["balanced_acc"]), "ece_test": float(test_metrics["ece"]),
            "test_loss": float(test_loss), "macro_f1_test_per_class": test_metrics["f1"].tolist(),
            "precision_test_per_class": test_metrics["precision"].tolist(),
            "recall_test_per_class": test_metrics["recall"].tolist(),
        })
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    curves_path = Path(cfg.curves_dir) / f"{cfg.exp_id}_seed{cfg.seed}.png"
    plot_curves(history, curves_path, f"{cfg.exp_id} · {cfg.backbone} · seed {cfg.seed}")
    print(f"Best checkpoint: {checkpoint_path} (epoch {best_epoch}, val macro-F1 {best_f1:.4f})")
    return summary


def _parse_bool(value: str) -> bool:
    lowered = value.lower()
    if lowered in {"true", "1", "yes", "on"}:
        return True
    if lowered in {"false", "0", "no", "off"}:
        return False
    raise ValueError(f"expected a boolean, received {value!r}")


def _parse_value(value: str, annotation, default):
    if value.lower() in {"none", "null"}:
        if default is not None:
            raise ValueError(f"{value!r} is not valid for a non-optional field")
        return None
    args = [arg for arg in get_args(annotation) if arg is not type(None)]
    target = args[0] if args else annotation
    if target is bool or isinstance(default, bool):
        return _parse_bool(value)
    if target is int or isinstance(default, int) and not isinstance(default, bool):
        return int(value)
    if target is float or isinstance(default, float):
        return float(value)
    return value


def parse_overrides(pairs: list[str]) -> dict:
    fields = {field.name: field for field in dataclasses.fields(Config)}
    annotations = get_type_hints(Config)
    overrides = {}
    for pair in pairs:
        if "=" not in pair:
            raise ValueError(f"Expected KEY=VALUE, received {pair!r}")
        key, value = pair.split("=", 1)
        if key not in fields:
            raise ValueError(f"Unknown Config field {key!r}; valid fields: {', '.join(fields)}")
        default = fields[key].default
        overrides[key] = _parse_value(value, annotations[key], default)
    return overrides


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one DeepWeeds configuration")
    parser.add_argument("--set", nargs="+", required=True, metavar="KEY=VALUE",
                        help="Config overrides, e.g. exp_id=B01 backbone=resnet50 seed=0")
    args = parser.parse_args()
    cfg = Config(**parse_overrides(args.set))
    print(json.dumps(run(cfg), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
