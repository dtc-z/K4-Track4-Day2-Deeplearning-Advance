"""Run fold-0 integrity checks, class/sample EDA, and optional pipeline smoke checks."""
from __future__ import annotations

import argparse
import json
import random
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
    from . import dataset, losses, model as model_lib
except ImportError:
    import dataset
    import losses
    import model as model_lib


def _save_class_plot(frames, names, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    combined = pd.concat(frames, keys=("train", "val", "test"), names=("split", "row"))
    counts = combined.groupby(["split", "Label"]).size().unstack(fill_value=0)
    counts = counts.reindex(index=("train", "val", "test"), columns=range(dataset.NUM_CLASSES), fill_value=0)
    ax = counts.rename(columns={i: names[i] for i in range(dataset.NUM_CLASSES)}).plot(
        kind="bar", figsize=(13, 5), width=0.82
    )
    ax.set_title("DeepWeeds fold 0 class distribution")
    ax.set_xlabel("Split")
    ax.set_ylabel("Images")
    ax.legend(title="Class", bbox_to_anchor=(1.02, 1), loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(output, dpi=160, bbox_inches="tight")
    plt.close()
    return counts


def _save_sample_grid(frames, images_dir: Path, names, output: Path, transform=None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rows, cols = dataset.NUM_CLASSES, 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 25))
    for class_id in range(rows):
        candidates = pd.concat(frames, ignore_index=True)
        candidates = candidates[pd.to_numeric(candidates["Label"]) == class_id].head(cols)
        for col in range(cols):
            axis = axes[class_id, col]
            axis.axis("off")
            if col >= len(candidates):
                continue
            filename = str(candidates.iloc[col]["Filename"])
            with Image.open(images_dir / filename) as image:
                image = image.convert("RGB")
                if transform is not None:
                    tensor = transform(image)
                    mean = torch.tensor(dataset.IMAGENET_MEAN)[:, None, None]
                    std = torch.tensor(dataset.IMAGENET_STD)[:, None, None]
                    image = (tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy()
                axis.imshow(image)
            axis.set_title(f"{names[class_id]}\n{filename}", fontsize=8)
    fig.suptitle("Three examples per class (random training augmentation)", y=1.002)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _smoke_check(train_df, images_dir: Path, backbone: str, device: torch.device,
                 overfit_steps: int, batch_size: int, output_dir: Path):
    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(7)
    uniform_logits = torch.zeros(3, dataset.NUM_CLASSES)
    ce = torch.nn.CrossEntropyLoss()(uniform_logits, torch.tensor([0, 1, 2]))
    expected = float(np.log(dataset.NUM_CLASSES))
    if abs(float(ce) - expected) > 1e-6:
        raise AssertionError("Uniform nine-class logits did not produce ln(9) CE")
    focal = losses.FocalLoss(gamma=0.0)
    sample_logits = torch.randn(8, dataset.NUM_CLASSES)
    sample_labels = torch.randint(dataset.NUM_CLASSES, (8,))
    focal_error = abs(float(focal(sample_logits, sample_labels) - torch.nn.functional.cross_entropy(sample_logits, sample_labels)))
    if focal_error >= 1e-6:
        raise AssertionError(f"Focal loss gamma=0 differs from CE by {focal_error}")
    images, labels, _ = next(iter(dataset.make_loader(
        train_df, images_dir, dataset.build_transforms(True, 224, "basic"),
        batch_size, True, num_workers=0, seed=7,
    )))
    images, labels = images[:batch_size].to(device), labels[:batch_size].to(device)
    model = model_lib.build_model(backbone, pretrained=False, num_classes=dataset.NUM_CLASSES, init="scratch").to(device)
    model.train()
    with torch.no_grad():
        initial_loss = float(torch.nn.functional.cross_entropy(model(images), labels))
    # Repeatedly train on one fixed, tiny batch as a deliberate overfit check.
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.0)
    for _ in range(overfit_steps):
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.cross_entropy(model(images), labels)
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final_loss = float(torch.nn.functional.cross_entropy(model(images), labels))
    if final_loss >= initial_loss:
        raise AssertionError(f"One-batch overfit check did not improve loss: {initial_loss:.4f}->{final_loss:.4f}")

    # A synthetic CutMix batch verifies the area-corrected coefficient remains valid.
    mixed, (target_a, target_b, lam) = losses.mix_batch(
        torch.rand(8, 3, 32, 32, device=device), torch.arange(8, device=device) % 9,
        alpha=1.0, mode="cutmix",
    )
    if mixed.shape != (8, 3, 32, 32) or not 0.0 <= lam <= 1.0 or target_a.shape != target_b.shape:
        raise AssertionError("CutMix smoke check failed")
    report = {"uniform_ce": float(ce), "expected_ln9": expected, "focal_gamma0_abs_error": focal_error,
              "backbone": backbone, "initial_loss": initial_loss,
              "overfit_loss": final_loss, "overfit_steps": overfit_steps,
              "cutmix_lambda": float(lam), "device": str(device)}
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "pipeline_smoke.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("Pipeline smoke check:", json.dumps(report, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--images-dir", default="data/images")
    parser.add_argument("--labels-dir", default="data/labels")
    parser.add_argument("--out-dir", default="eda")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--smoke-test", action="store_true")
    parser.add_argument("--backbone", default="mobilenetv3_large_100")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overfit-steps", type=int, default=75)
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()
    out_dir, images_dir = Path(args.out_dir), Path(args.images_dir)
    train_df, val_df, test_df = dataset.load_split(args.labels_dir, args.fold)
    split_report = dataset.check_split(train_df, val_df, test_df, images_dir)
    label_map = pd.read_csv(Path(args.labels_dir) / "labels.csv").drop_duplicates("Label").sort_values("Label")
    names = [str(item) for item in label_map["Species"].tolist()]
    if len(names) != dataset.NUM_CLASSES:
        raise ValueError("labels.csv must define all 9 classes")
    counts = _save_class_plot([train_df, val_df, test_df], names, out_dir / "class_distribution.png")
    all_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    total_counts = [int((pd.to_numeric(all_df["Label"]) == i).sum()) for i in range(dataset.NUM_CLASSES)]
    expected = dataset.EXPECTED_CLASS_COUNTS
    class_report = {names[i]: {"observed": total_counts[i], "paper_table": expected[i],
                               "matches": total_counts[i] == expected[i]}
                    for i in range(dataset.NUM_CLASSES)}
    print("Full-dataset counts vs paper Table 1:")
    print(pd.DataFrame(class_report).T.to_string())
    _save_sample_grid([train_df], images_dir, names, out_dir / "train_samples.png")
    _save_sample_grid([train_df], images_dir, names, out_dir / "augmentation_samples.png",
                      transform=dataset.build_transforms(True, 224, "color"))
    report = {"split": split_report, "full_class_counts": class_report}
    (out_dir / "eda_summary.json").write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    if args.smoke_test:
        device = torch.device("cuda" if args.device == "auto" and torch.cuda.is_available() else
                              "cpu" if args.device == "auto" else args.device)
        _smoke_check(train_df, images_dir, args.backbone, device,
                     args.overfit_steps, args.batch_size, out_dir)
    print(f"EDA artifacts saved in {out_dir.resolve()}")


if __name__ == "__main__":
    main()
