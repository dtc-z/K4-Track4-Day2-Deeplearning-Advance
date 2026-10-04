"""Measure synchronized batch-1 latency for completed backbone runs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from starter.benchmark import latency_report
from starter.inference_eval import _load_run


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", default="runs")
    parser.add_argument("--out", default="inference_results.csv")
    parser.add_argument("--ids", nargs="+", default=["B01", "B02", "B03", "B04", "B05"])
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iters", type=int, default=50)
    args = parser.parse_args()
    target = torch.device(args.device)
    rows = []
    for exp_id in args.ids:
        run_dir = Path(args.runs_root) / exp_id / "seed0"
        summary = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        model, cfg = _load_run(run_dir, target)
        measured = latency_report(model, 1, cfg.img_size, "fp32", str(target), args.warmup, args.iters)
        rows.append({
            "exp_id": exp_id, "seed": 0, "method": "oneview", "K": 1,
            "macro_f1_val": summary["macro_f1_val"], "top1_val": summary["top1_val"],
            "ece_val": summary["ece_val"], "latency_p50_ms": measured["p50"],
            "latency_p95_ms": measured["p95"], "latency_p99_ms": measured["p99"],
            "throughput_img_s": measured["images_per_s"], "gpu": measured["gpu"],
            "dtype": measured["dtype"], "batch": 1, "img_size": cfg.img_size,
            "torch": measured["torch"], "includes_preprocessing": False,
            "note": f"warmup={args.warmup}; timed={args.iters}; forward only",
        })
        print(f"{exp_id}: val macro-F1={summary['macro_f1_val']:.4f}, "
              f"batch-1 p95={measured['p95']:.2f} ms ({measured['gpu']})", flush=True)
        del model
        if target.type == "cuda":
            torch.cuda.empty_cache()

    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    new = pd.DataFrame(rows)
    if output.exists():
        new = pd.concat([pd.read_csv(output), new], ignore_index=True)
        new = new.drop_duplicates(["exp_id", "seed", "method"], keep="last")
    new.to_csv(output, index=False)


if __name__ == "__main__":
    main()
