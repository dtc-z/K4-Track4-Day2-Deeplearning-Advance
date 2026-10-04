"""Run the prescribed backbone sweep or controlled training ablations on val only."""
from __future__ import annotations

import argparse
import sys
from dataclasses import replace
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next((parent for parent in SCRIPT_PATH.parents if (parent / "eval.py").is_file()), SCRIPT_PATH.parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

try:
    from .train import Config, run
except ImportError:
    from train import Config, run

BACKBONES = [
    ("B01", "resnet50"),
    ("B02", "resnext50_32x4d"),
    ("B03", "convnext_tiny"),
    ("B04", "deit_small_patch16_224"),
    ("B05", "mobilenetv3_large_100"),
]


def recipe_configs(backbone: str, seed: int, epochs: int, batch_size: int,
                   num_workers: int) -> list[tuple[str, Config]]:
    """Return a one-factor-at-a-time ablation plan plus one combination run."""
    base = Config(exp_id="T00", backbone=backbone, seed=seed, epochs=epochs,
                  batch_size=batch_size, num_workers=num_workers)
    variants = [
        ("T00", base),
        ("T01_scratch", replace(base, exp_id="T01_scratch", init="scratch")),
        ("T02_frozen", replace(base, exp_id="T02_frozen", init="frozen")),
        ("T03_color", replace(base, exp_id="T03_color", aug="color")),
        ("T04_randaug", replace(base, exp_id="T04_randaug", aug="randaug")),
        ("T05_label_smoothing", replace(base, exp_id="T05_label_smoothing", loss="ls", label_smoothing=0.1)),
        ("T06_focal", replace(base, exp_id="T06_focal", loss="focal", focal_gamma=2.0)),
        ("T07_weighted_ce", replace(base, exp_id="T07_weighted_ce", loss="ce_weighted", class_weight_beta=0.0)),
        ("T08_balanced_sampler", replace(base, exp_id="T08_balanced_sampler", sampler="balanced")),
        ("T09_cutmix", replace(base, exp_id="T09_cutmix", mix="cutmix", mix_alpha=1.0)),
        ("T10_mixup", replace(base, exp_id="T10_mixup", mix="mixup", mix_alpha=0.2)),
        ("T11_ema", replace(base, exp_id="T11_ema", ema_decay=0.999)),
        ("T12_cutmix_ls_ema", replace(base, exp_id="T12_cutmix_ls_ema", mix="cutmix",
                                        loss="ls", label_smoothing=0.1, ema_decay=0.999)),
    ]
    return variants


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("backbones", "ablations"))
    parser.add_argument("--backbone", default="resnet50",
                        help="backbone selected on validation for the ablations stage")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--dry-run", action="store_true", help="print planned runs without training")
    parser.add_argument("--ids", nargs="*", help="optional experiment IDs to run")
    parser.add_argument("--resume", help="resume the single selected experiment from a last.pt checkpoint")
    args = parser.parse_args()
    if args.stage == "backbones":
        configs = [(exp_id, Config(exp_id=exp_id, backbone=name, seed=args.seed,
                                   epochs=args.epochs, batch_size=args.batch_size,
                                   num_workers=args.num_workers))
                   for exp_id, name in BACKBONES]
    else:
        configs = recipe_configs(args.backbone, args.seed, args.epochs,
                                 args.batch_size, args.num_workers)
    if args.ids:
        requested = set(args.ids)
        known = {exp_id for exp_id, _ in configs}
        unknown = requested - known
        if unknown:
            raise ValueError(f"Unknown IDs {sorted(unknown)}; available: {sorted(known)}")
        configs = [(exp_id, cfg) for exp_id, cfg in configs if exp_id in requested]
    if args.resume:
        if len(configs) != 1:
            raise ValueError("--resume requires exactly one selected experiment via --ids")
        exp_id, cfg = configs[0]
        configs[0] = (exp_id, replace(cfg, resume=args.resume))
    print(f"Planned {len(configs)} validation-only runs:")
    for exp_id, cfg in configs:
        print(f"  {exp_id:20s} backbone={cfg.backbone:28s} init={cfg.init:9s} "
              f"aug={cfg.aug:8s} loss={cfg.loss:12s} mix={cfg.mix}")
    if args.dry_run:
        return
    for _, cfg in configs:
        result = run(cfg)
        print(f"Completed {result['exp_id']} seed={result['seed']} "
              f"val macro-F1={result['macro_f1_val']:.4f}")


if __name__ == "__main__":
    main()
