"""Synchronized latency benchmarks for batch-1 and throughput measurements."""
from __future__ import annotations

import copy
import time

import numpy as np
import torch


def bench(fn, warmup: int = 10, iters: int = 100, sync=None) -> dict:
    if warmup < 10:
        raise ValueError("Use at least 10 warmup iterations")
    if iters < 50:
        raise ValueError("Use at least 50 timed iterations")
    for _ in range(warmup):
        fn()
    samples = np.empty(iters, dtype=np.float64)
    for index in range(iters):
        if sync is not None:
            sync()
        start = time.perf_counter()
        fn()
        if sync is not None:
            sync()
        samples[index] = (time.perf_counter() - start) * 1000.0
    return {"p50": float(np.percentile(samples, 50)),
            "p95": float(np.percentile(samples, 95)),
            "p99": float(np.percentile(samples, 99)),
            "mean": float(samples.mean()), "n": int(iters), "warmup": int(warmup)}


def latency_report(model, batch_size: int, img_size: int, dtype: str = "fp32", device: str = "cuda",
                   warmup: int = 10, iters: int = 100) -> dict:
    if batch_size < 1 or img_size < 1:
        raise ValueError("batch_size and img_size must be positive")
    target = torch.device(device)
    if target.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for latency benchmark but unavailable")
    dtype = dtype.lower()
    if dtype not in {"fp32", "amp", "fp16"}:
        raise ValueError("dtype must be fp32, amp, or fp16")
    if dtype == "fp16" and target.type != "cuda":
        raise ValueError("fp16 benchmark is supported here only on CUDA")
    benchmark_model = copy.deepcopy(model).to(target).eval()
    if dtype == "fp16":
        benchmark_model.half()
        input_dtype = torch.float16
    else:
        benchmark_model.float()
        input_dtype = torch.float32
    batch = torch.randn(batch_size, 3, img_size, img_size, device=target, dtype=input_dtype)
    sync = (lambda: torch.cuda.synchronize(target)) if target.type == "cuda" else None

    @torch.inference_mode()
    def forward():
        with torch.autocast(device_type=target.type, dtype=torch.float16,
                            enabled=(dtype == "amp" and target.type == "cuda")):
            return benchmark_model(batch)

    result = bench(forward, warmup=warmup, iters=iters, sync=sync)
    gpu_name = torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU"
    result.update({"gpu": gpu_name, "dtype": dtype, "batch": int(batch_size),
                   "img_size": int(img_size), "images_per_s": float(batch_size / (result["p50"] / 1000.0)),
                   "torch": torch.__version__, "device": str(target),
                   "batch_norm_fused": False, "includes_preprocessing": False,
                   "warmup": int(warmup), "n": int(iters)})
    return result


def tta_latency(model, k_views: int, **kw) -> dict:
    if k_views < 1:
        raise ValueError("k_views must be positive")
    target = torch.device(kw.get("device", "cuda"))
    dtype = kw.get("dtype", "fp32").lower()
    benchmark_model = copy.deepcopy(model).to(target).eval()
    size = int(kw["img_size"])
    batch_size = int(kw["batch_size"])
    if dtype == "fp16":
        benchmark_model.half()
        images = torch.randn(batch_size, 3, size, size, device=target, dtype=torch.float16)
    else:
        images = torch.randn(batch_size, 3, size, size, device=target, dtype=torch.float32)
    sync = (lambda: torch.cuda.synchronize(target)) if target.type == "cuda" else None

    @torch.inference_mode()
    def forward_all_views():
        for _ in range(k_views):
            with torch.autocast(device_type=target.type, dtype=torch.float16,
                                enabled=(dtype == "amp" and target.type == "cuda")):
                benchmark_model(images)

    timed = bench(forward_all_views, warmup=kw.get("warmup", 10),
                  iters=kw.get("iters", 100), sync=sync)
    gpu_name = torch.cuda.get_device_name(target) if target.type == "cuda" else "CPU"
    return {**timed, "gpu": gpu_name, "dtype": dtype, "batch": batch_size,
            "img_size": size, "k_views": int(k_views), "images_per_s": batch_size / (timed["p50"] / 1000.0),
            "torch": torch.__version__, "device": str(target), "batch_norm_fused": False,
            "includes_preprocessing": False}
