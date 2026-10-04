"""Compare inference methods on validation, then run one selected method on test once."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next((parent for parent in SCRIPT_PATH.parents if (parent / "eval.py").is_file()), SCRIPT_PATH.parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from . import benchmark, dataset, inference, model as model_lib
    from .train import Config, _softmax, _load_checkpoint
except ImportError:
    import benchmark
    import dataset
    import inference
    import model as model_lib
    from train import Config, _softmax, _load_checkpoint

from eval import compute_metrics, save_predictions


def _load_run(run_dir: Path, device: torch.device):
    config_data = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
    allowed = {field.name for field in dataclasses.fields(Config)}
    cfg = Config(**{key: value for key, value in config_data.items() if key in allowed})
    model = model_lib.build_model(cfg.backbone, pretrained=False, num_classes=dataset.NUM_CLASSES,
                                  drop_rate=cfg.drop_rate, init=cfg.init).to(device)
    checkpoint = _load_checkpoint(run_dir / "best.pt", device)
    if cfg.ema_decay is not None and checkpoint.get("ema_state") is not None:
        model.load_state_dict(checkpoint["ema_state"])
    else:
        model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model, cfg


def _loader(split_df, cfg, split: str, method: str = "oneview"):
    if method == "fivecrop":
        from torchvision import transforms
        resize_size = round(cfg.img_size * 256 / 224)
        transform = transforms.Compose([transforms.Resize(resize_size), transforms.ToTensor(),
                                        transforms.Normalize(dataset.IMAGENET_MEAN, dataset.IMAGENET_STD)])
    else:
        transform = dataset.build_transforms(False, cfg.img_size)
    return dataset.make_loader(split_df, cfg.images_dir, transform, cfg.batch_size,
                               train=False, num_workers=cfg.num_workers, seed=cfg.seed)


@torch.inference_mode()
def _predict(model, loader, device, method: str, scales: tuple[int, ...] = (), crop_size: int | None = None):
    model.eval()
    filenames, labels, all_view_logits = [], [], None
    dtype = "fp16" if method == "fp16" else "fp32"
    if dtype == "fp16" and torch.device(device).type != "cuda":
        raise RuntimeError("fp16 validation requires a CUDA device")
    for images, target, batch_names in loader:
        images = images.to(device, non_blocking=True)
        filenames.extend(str(name) for name in batch_names)
        labels.append(target.cpu().numpy())
        if method == "fivecrop":
            crop_size = int(crop_size or images.shape[-1])
            view_batches = inference.views_multicrop(images, crop_size)
        elif method == "hflip-prob" or method == "hflip-logit":
            view_batches = [images, inference.view_hflip(images)]
        elif method == "multiscale":
            view_batches = inference.views_multiscale(images, scales)
        else:
            view_batches = [images]
        if all_view_logits is None:
            all_view_logits = [[] for _ in view_batches]
        if len(all_view_logits) != len(view_batches):
            raise RuntimeError("view count changed between batches")
        for output_index, view in enumerate(view_batches):
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.float16,
                                enabled=(dtype == "fp16" and torch.device(device).type == "cuda")):
                output = model(view)
            all_view_logits[output_index].append(output.float().cpu().numpy())
    if all_view_logits is None:
        raise ValueError("empty data loader")
    view_logits = [np.concatenate(parts) for parts in all_view_logits]
    y_true = np.concatenate(labels)
    if method in {"hflip-prob", "fivecrop", "multiscale"}:
        probs = inference.aggregate_views(view_logits, space="prob")
        combined_logits = np.log(np.clip(probs, 1e-12, 1.0))
    elif method == "hflip-logit":
        probs = inference.aggregate_views(view_logits, space="logit")
        combined_logits = np.mean(view_logits, axis=0)
    else:
        combined_logits = view_logits[0]
        probs = _softmax(combined_logits)
    return filenames, y_true, combined_logits, probs


def _metrics(y_true, probs):
    return compute_metrics(y_true, probs.argmax(axis=1), probs)


def _row(exp_id, seed, method, k, metrics, latency=None, temperature=None, note=""):
    latency = latency or {}
    return {
        "exp_id": exp_id, "seed": seed, "method": method, "K": k,
        "macro_f1_val": float(metrics["macro_f1"]), "top1_val": float(metrics["top1"]),
        "balanced_acc_val": float(metrics["balanced_acc"]), "ece_val": float(metrics["ece"]),
        "nll_val": float(metrics["nll"]), "temperature": temperature,
        "latency_p50_ms": latency.get("p50"), "latency_p95_ms": latency.get("p95"),
        "latency_p99_ms": latency.get("p99"), "throughput_img_s": latency.get("images_per_s"),
        "gpu": latency.get("gpu"), "dtype": latency.get("dtype", "fp32"),
        "batch": latency.get("batch", 1), "img_size": latency.get("img_size"),
        "torch": latency.get("torch", torch.__version__), "includes_preprocessing": False,
        "note": note,
    }


def compare(run_dir: Path, output_csv: Path, device: str = "auto", iters: int = 50,
            warmup: int = 10, scales: tuple[int, ...] = (192, 224, 256), benchmark_enabled: bool = True):
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    model, cfg = _load_run(run_dir, target)
    _, val_df, _ = dataset.load_split(cfg.labels_dir, cfg.fold)
    loader = _loader(val_df, cfg, "val")
    rows = []
    base_values = {}
    methods = [("oneview", 1), ("hflip-prob", 2), ("hflip-logit", 2),
               ("fivecrop", 5), ("multiscale", len(scales))]
    for method, k in methods:
        try:
            method_loader = _loader(val_df, cfg, "val", method)
            names, y_true, logits, probs = _predict(model, method_loader, target, method, scales, cfg.img_size)
        except (RuntimeError, ValueError) as exc:
            rows.append({"exp_id": cfg.exp_id, "seed": cfg.seed, "method": method,
                         "note": f"unsupported: {exc}"})
            continue
        metrics = _metrics(y_true, probs)
        base_values[method] = (names, y_true, logits, probs, metrics)
        latency = None
        if benchmark_enabled:
            if k == 1:
                latency = benchmark.latency_report(model, 1, cfg.img_size, "fp32", str(target), warmup, iters)
            else:
                latency = benchmark.tta_latency(model, k, batch_size=1, img_size=cfg.img_size,
                                                dtype="fp32", device=str(target), warmup=warmup, iters=iters)
        rows.append(_row(cfg.exp_id, cfg.seed, method, k, metrics, latency))

    if "oneview" not in base_values:
        raise RuntimeError("The required one-view validation result could not be produced")
    names, y_true, logits, probs, metrics = base_values["oneview"]
    temperature = inference.fit_temperature(logits, y_true)
    calibrated = inference.apply_temperature(logits, temperature)
    calibrated_metrics = _metrics(y_true, calibrated)
    rows.append(_row(cfg.exp_id, cfg.seed, "temperature", 1, calibrated_metrics,
                     temperature=temperature, note="T fitted on validation logits only"))

    try:
        fused = inference.fuse_conv_bn(model)
        fused_logits = _predict(fused, loader, target, "oneview", scales)[2]
        fused_probs = _softmax(fused_logits)
        fused_metrics = _metrics(y_true, fused_probs)
        fused_latency = (benchmark.latency_report(fused, 1, cfg.img_size, "fp32", str(target), warmup, iters)
                         if benchmark_enabled else None)
        rows.append(_row(cfg.exp_id, cfg.seed, "bn-fused", 1, fused_metrics, fused_latency,
                         note="No quality change expected; fusion applies only to adjacent Conv-BN pairs"))
    except (RuntimeError, ValueError) as exc:
        rows.append({"exp_id": cfg.exp_id, "seed": cfg.seed, "method": "bn-fused", "note": f"unsupported: {exc}"})

    if benchmark_enabled:
        fp16_latency = benchmark.latency_report(model, 1, cfg.img_size, "fp16", str(target), warmup, iters) if target.type == "cuda" else None
        if fp16_latency is not None:
            fp16_logits = _predict(model, loader, target, "fp16", scales)[2]
            rows.append(_row(cfg.exp_id, cfg.seed, "fp16", 1, _metrics(y_true, _softmax(fp16_logits)), fp16_latency))
        # A larger batch gives a throughput point for deployment comparison.
        throughput = None
        for throughput_batch in (32, 16, 8, 4, 2, 1):
            try:
                throughput = benchmark.latency_report(model, throughput_batch, cfg.img_size,
                                                      "fp32", str(target), warmup, iters)
                break
            except torch.cuda.OutOfMemoryError:
                if target.type != "cuda" or throughput_batch == 1:
                    raise
                torch.cuda.empty_cache()
        if throughput is None:
            raise RuntimeError("Could not find a batch size for throughput measurement")
        for row in rows:
            if row.get("method") == "oneview":
                row["throughput_img_s"] = throughput["images_per_s"]
                row["throughput_batch"] = throughput["batch"]

    frame = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    if output_csv.exists():
        previous = pd.read_csv(output_csv)
        frame = pd.concat([previous, frame], ignore_index=True)
        frame = frame.drop_duplicates(["exp_id", "seed", "method"], keep="last")
    frame.to_csv(output_csv, index=False)
    print(frame.to_string(index=False))
    return frame


def evaluate_final(run_dir: Path, split: str, method: str, exp_id: str | None,
                   device: str = "auto", output_dir: Path = Path("predictions"),
                   scales: tuple[int, ...] = (192, 224, 256), save_uncalibrated: bool = True):
    if split not in {"val", "test"}:
        raise ValueError("split must be val or test")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    target = torch.device(device)
    model, cfg = _load_run(run_dir, target)
    if method.endswith("-temp"):
        base_method = method[:-5]
        use_temperature = True
    else:
        base_method = method
        use_temperature = False
    allowed = {"oneview", "hflip-prob", "hflip-logit", "fivecrop", "multiscale", "bn-fused", "fp16"}
    if base_method not in allowed:
        raise ValueError(f"Unsupported method {method!r}; choose one of {sorted(allowed)} and optional -temp")
    final_id = exp_id or cfg.exp_id
    seed = cfg.seed
    test_path = output_dir / f"{final_id}_seed{seed}_test.csv"
    uncal_path = output_dir / f"{final_id}_uncal_seed{seed}_test.csv"
    if split == "test" and test_path.exists():
        raise FileExistsError(f"Refusing to overwrite {test_path}; test may be run only once per seed")
    if split == "test" and use_temperature and save_uncalibrated and uncal_path.exists():
        raise FileExistsError(f"Refusing to overwrite {uncal_path}")
    val_df = dataset.load_split(cfg.labels_dir, cfg.fold)[1]
    val_loader = _loader(val_df, cfg, "val", base_method)
    eval_model = inference.fuse_conv_bn(model) if base_method == "bn-fused" else model
    val_names, val_y, val_logits, val_probs = _predict(eval_model, val_loader, target, base_method, scales, cfg.img_size)
    temperature = inference.fit_temperature(val_logits, val_y) if use_temperature else None
    if use_temperature:
        val_uncal = val_probs
        val_probs = inference.apply_temperature(val_logits, temperature)
    val_metrics = _metrics(val_y, val_probs)
    output_dir.mkdir(parents=True, exist_ok=True)
    val_path = output_dir / f"{final_id}_seed{seed}_val.csv"
    save_predictions(val_path, val_names, val_y, val_probs)
    result = {"exp_id": final_id, "seed": seed, "method": method,
              "temperature": temperature, "macro_f1_val": val_metrics["macro_f1"],
              "top1_val": val_metrics["top1"], "ece_val": val_metrics["ece"],
              "val_predictions": str(val_path)}
    if split == "test":
        _, _, test_df = dataset.load_split(cfg.labels_dir, cfg.fold)
        test_loader = _loader(test_df, cfg, "test", base_method)
        test_names, test_y, test_logits, test_probs = _predict(eval_model, test_loader, target, base_method, scales, cfg.img_size)
        if use_temperature:
            if save_uncalibrated:
                save_predictions(uncal_path, test_names, test_y, test_probs)
                result["uncal_test_predictions"] = str(uncal_path)
            test_probs = inference.apply_temperature(test_logits, temperature)
        save_predictions(test_path, test_names, test_y, test_probs)
        test_metrics = _metrics(test_y, test_probs)
        result.update({"macro_f1_test": test_metrics["macro_f1"], "top1_test": test_metrics["top1"],
                       "ece_test": test_metrics["ece"], "test_predictions": str(test_path)})
    result_path = run_dir / f"inference_{split}_{method}.json"
    result_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    compare_parser = subparsers.add_parser("compare", help="compare methods using validation only")
    compare_parser.add_argument("--run-dir", required=True)
    compare_parser.add_argument("--out", default="inference_results.csv")
    compare_parser.add_argument("--device", default="auto")
    compare_parser.add_argument("--iters", type=int, default=50)
    compare_parser.add_argument("--warmup", type=int, default=10)
    compare_parser.add_argument("--scales", nargs="+", type=int, default=[192, 224, 256])
    compare_parser.add_argument("--skip-latency", action="store_true")
    final_parser = subparsers.add_parser("final", help="apply a preselected method to val or test")
    final_parser.add_argument("--run-dir", required=True)
    final_parser.add_argument("--split", choices=("val", "test"), required=True)
    final_parser.add_argument("--method", default="oneview", help="e.g. oneview, hflip-prob, oneview-temp")
    final_parser.add_argument("--exp-id", help="output prediction ID; defaults to the run's exp_id")
    final_parser.add_argument("--output-dir", default="predictions")
    final_parser.add_argument("--device", default="auto")
    final_parser.add_argument("--scales", nargs="+", type=int, default=[192, 224, 256])
    args = parser.parse_args()
    if args.command == "compare":
        compare(Path(args.run_dir), Path(args.out), args.device, args.iters, args.warmup,
                tuple(args.scales), not args.skip_latency)
    else:
        evaluate_final(Path(args.run_dir), args.split, args.method, args.exp_id,
                       args.device, Path(args.output_dir), tuple(args.scales))


if __name__ == "__main__":
    main()
