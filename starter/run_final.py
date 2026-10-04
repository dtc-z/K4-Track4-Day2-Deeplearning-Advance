"""Retrain the val-selected recipe and T00 baseline with >=3 seeds; does not touch test."""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next((parent for parent in SCRIPT_PATH.parents if (parent / "eval.py").is_file()), SCRIPT_PATH.parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
try:
    from .train import Config, run
except ImportError:
    from train import Config, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selected-run", required=True,
                        help="config.json from the recipe chosen using validation results")
    parser.add_argument("--final-id", default="F01")
    parser.add_argument("--baseline-id", default="T00")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--out-dir", default="runs_final")
    parser.add_argument("--pred-dir", default="predictions")
    parser.add_argument("--curves-dir", default="curves")
    parser.add_argument("--overwrite-runs", action="store_true")
    args = parser.parse_args()
    if len(set(args.seeds)) < 3:
        raise ValueError("The final and baseline need at least three different seeds")
    selected_path = Path(args.selected_run)
    selected_data = json.loads(selected_path.read_text(encoding="utf-8"))
    allowed = {field.name for field in dataclasses.fields(Config)}
    selected = {key: value for key, value in selected_data.items() if key in allowed}
    selected.update({"out_dir": args.out_dir, "pred_dir": args.pred_dir,
                     "curves_dir": args.curves_dir, "save_test_predictions": False,
                     "save_val_predictions": True, "resume": None,
                     "overwrite_run": args.overwrite_runs})
    chosen_template = Config(**selected)
    baseline_template = Config(
        exp_id=args.baseline_id, seed=0, fold=chosen_template.fold,
        backbone=chosen_template.backbone, init="finetune", drop_rate=0.0,
        img_size=chosen_template.img_size, aug="basic", sampler=None, mix=None,
        loss="ce", label_smoothing=0.0, focal_gamma=chosen_template.focal_gamma,
        class_weight_beta=None, epochs=chosen_template.epochs, batch_size=chosen_template.batch_size,
        lr_backbone=chosen_template.lr_backbone, lr_head=chosen_template.lr_head,
        weight_decay=chosen_template.weight_decay, warmup_epochs=chosen_template.warmup_epochs,
        ema_decay=None, amp=chosen_template.amp, num_workers=chosen_template.num_workers,
        images_dir=chosen_template.images_dir, labels_dir=chosen_template.labels_dir,
        out_dir=args.out_dir, pred_dir=args.pred_dir, curves_dir=args.curves_dir,
        device=chosen_template.device, save_val_predictions=True, save_test_predictions=False,
        overwrite_run=args.overwrite_runs,
    )
    planned = []
    for seed in args.seeds:
        planned.append(dataclasses.replace(chosen_template, exp_id=args.final_id, seed=seed))
        planned.append(dataclasses.replace(baseline_template, seed=seed))
    active = []
    for cfg in planned:
        out_path = Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}" / "summary.json"
        if out_path.exists() and not args.overwrite_runs:
            print(f"Skipping completed {cfg.exp_id} seed={cfg.seed}")
            continue
        last_path = Path(cfg.out_dir) / cfg.exp_id / f"seed{cfg.seed}" / "last.pt"
        if last_path.exists() and not args.overwrite_runs:
            cfg = dataclasses.replace(cfg, resume=str(last_path))
        test_path = Path(cfg.pred_dir) / f"{cfg.exp_id}_seed{cfg.seed}_test.csv"
        if test_path.exists():
            raise FileExistsError(f"A test prediction already exists: {test_path}; refusing to repeat it.")
        active.append(cfg)
    print("Final training plan (validation only; test remains untouched):")
    for cfg in planned:
        print(f"  {cfg.exp_id} seed={cfg.seed}: backbone={cfg.backbone}, init={cfg.init}, "
              f"aug={cfg.aug}, loss={cfg.loss}, mix={cfg.mix}, EMA={cfg.ema_decay}")
    for cfg in active:
        summary = run(cfg)
        print(f"Completed {cfg.exp_id} seed={cfg.seed}: val macro-F1={summary['macro_f1_val']:.4f}")
    print("Final retraining complete. Compare the selected inference method on validation before running test.")


if __name__ == "__main__":
    main()
