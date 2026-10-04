"""Build the required results.xlsx and validation/test comparison figures from real run artifacts."""
from __future__ import annotations

import argparse
import json
import re
import sys
from copy import copy
from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl.styles import PatternFill

SCRIPT_PATH = Path(__file__).resolve()
ROOT = next((parent for parent in SCRIPT_PATH.parents if (parent / "eval.py").is_file()), SCRIPT_PATH.parents[1])
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eval import check_against_csv, compute_metrics, load_names, read_pred


SHEETS = {
    "Backbones": ["exp_id", "backbone", "weight_tag", "params_m", "GMAC", "img_size", "epochs",
                  "seed", "macro_f1_val", "top1_val", "train_sec_per_epoch", "latency_batch1_ms", "notes"],
    "Training": ["exp_id", "backbone", "axis", "changed_from_T00", "seed", "macro_f1_val",
                 "top1_val", "delta_macro_f1_vs_T00", "chinee_apple_f1", "snake_weed_f1", "notes"],
    "Inference": ["exp_id", "method", "checkpoint", "K", "macro_f1_val", "top1_val", "ECE_val",
                  "p50_ms_batch1", "p95_ms_batch1", "p99_ms_batch1", "throughput_img_s", "relative_cost_vs_I00", "notes"],
    "Final": ["exp_id", "configuration", "seed", "macro_f1_val", "macro_f1_test", "top1_test", "ECE_test",
              "test_macro_f1_mean_std", "test_top1_mean_std", "n_seeds"],
    "PerClass": ["exp_id", "class", "n_test", "precision", "recall", "F1"],
    "Latency": ["configuration", "GPU", "dtype", "batch", "img_size", "batch_norm_fused", "p50_ms", "p95_ms",
                "p99_ms", "images_per_s", "torch", "includes_preprocessing"],
    "Summary": ["rank", "exp_id", "method", "backbone", "macro_f1_val", "top1_val", "p95_ms_batch1", "params_m", "GMAC", "notes"],
}


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _find_runs(roots: list[Path]):
    summaries, configs = {}, {}
    for root in roots:
        if not root.exists():
            continue
        for summary_path in root.glob("**/summary.json"):
            summary = _json(summary_path)
            exp_id, seed = summary.get("exp_id"), int(summary.get("seed", 0))
            key = (exp_id, seed)
            summaries[key] = summary
            config_path = summary_path.with_name("config.json")
            if config_path.exists():
                configs[key] = _json(config_path)
    return summaries, configs


def _prediction_groups(prediction_dir: Path, split: str, reference_csv: Path | None):
    grouped = {}
    pattern = re.compile(r"^(?P<exp>.+)_seed(?P<seed>\d+)_(?P<split>val|test)\.csv$")
    for path in sorted(prediction_dir.glob(f"*_seed*_{split}.csv")):
        match = pattern.match(path.name)
        if not match:
            continue
        exp_id, seed = match.group("exp"), int(match.group("seed"))
        if "_uncal" in exp_id:
            continue
        prediction = read_pred(str(path))
        if reference_csv is not None:
            check_against_csv(prediction, str(reference_csv), split)
        metrics = compute_metrics(prediction.y_true, prediction.y_pred, prediction.probs)
        grouped.setdefault(exp_id, {})[seed] = (prediction, metrics)
    return grouped


def _mean_std(values):
    array = np.asarray(values, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std(ddof=1)) if len(array) > 1 else float("nan")
    return mean, std


def _experiment_axis(exp_id: str, config: dict):
    mapping = {
        "T01_scratch": ("A · initialization", "finetune → scratch"),
        "T02_frozen": ("A · initialization", "finetune → frozen backbone"),
        "T03_color": ("B · augmentation", "basic → color jitter"),
        "T04_randaug": ("B · augmentation", "basic → RandAugment"),
        "T05_label_smoothing": ("C · loss", "CE → label smoothing 0.1"),
        "T06_focal": ("C · loss", "CE → focal gamma=2"),
        "T07_weighted_ce": ("C · loss", "CE → inverse-frequency weighted CE"),
        "T08_balanced_sampler": ("D · sampler", "none → balanced sampler"),
        "T09_cutmix": ("B · augmentation", "none → CutMix"),
        "T10_mixup": ("B · augmentation", "none → Mixup"),
        "T11_ema": ("F · regularization", "EMA off → decay 0.999"),
        "T12_cutmix_ls_ema": ("combination", "CutMix + label smoothing + EMA"),
    }
    return mapping.get(exp_id, ("baseline", json.dumps({key: config.get(key) for key in
                            ("init", "aug", "loss", "sampler", "mix", "ema_decay")}, ensure_ascii=False)))


def _plot_confusion(exp_id, records, names, output: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    matrices = [record[1]["confusion"] for record in records.values()]
    matrix = np.mean(matrices, axis=0)
    fig, axis = plt.subplots(figsize=(10, 8))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(names)), names, rotation=45, ha="right")
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title(f"{exp_id}: mean test confusion counts across seeds")
    fig.colorbar(image, ax=axis, label="Mean image count")
    threshold = matrix.max() * 0.55
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            axis.text(col, row, f"{matrix[row, col]:.1f}", ha="center", va="center",
                      color="white" if matrix[row, col] > threshold else "black", fontsize=7)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _plot_tradeoff(inference_df: pd.DataFrame, output: Path):
    if inference_df.empty or "p95_ms_batch1" not in inference_df:
        return
    data = inference_df.dropna(subset=["p95_ms_batch1", "macro_f1_val"])
    if data.empty:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 6))
    axis.scatter(data["p95_ms_batch1"], data["macro_f1_val"])
    for _, row in data.iterrows():
        axis.annotate(f"{row['exp_id']}:{row['method']}",
                      (row["p95_ms_batch1"], row["macro_f1_val"]), fontsize=7, alpha=0.8)
    axis.set_xlabel("Batch-1 p95 latency (ms; preprocessing excluded)")
    axis.set_ylabel("Validation macro-F1")
    axis.set_title("Validation accuracy–latency trade-off")
    axis.grid(alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=170, bbox_inches="tight")
    plt.close(fig)


def build_workbook(runs_dirs: list[str | Path], prediction_dir: str | Path,
                   labels_csv: str | Path, test_csv: str | Path,
                   inference_csv: str | Path | None, output: str | Path,
                   curves_dir: str | Path = "curves") -> dict[str, pd.DataFrame]:
    runs_dirs = [Path(path) for path in runs_dirs]
    prediction_dir = Path(prediction_dir)
    labels_csv, test_csv = Path(labels_csv), Path(test_csv)
    summaries, configs = _find_runs(runs_dirs)
    val_groups = _prediction_groups(prediction_dir, "val", None)
    test_groups = _prediction_groups(prediction_dir, "test", test_csv)
    names = load_names(str(labels_csv))

    inference_data = pd.read_csv(inference_csv) if inference_csv and Path(inference_csv).exists() else pd.DataFrame()
    inference_rows = []
    if not inference_data.empty:
        for _, row in inference_data.iterrows():
            inference_rows.append({
                "exp_id": row.get("exp_id"), "method": row.get("method"),
                "checkpoint": f"{row.get('exp_id')}/seed{row.get('seed')}", "K": row.get("K"),
                "macro_f1_val": row.get("macro_f1_val"), "top1_val": row.get("top1_val"),
                "ECE_val": row.get("ece_val"), "p50_ms_batch1": row.get("latency_p50_ms"),
                "p95_ms_batch1": row.get("latency_p95_ms"), "p99_ms_batch1": row.get("latency_p99_ms"),
                "throughput_img_s": row.get("throughput_img_s"), "relative_cost_vs_I00": None,
                "notes": row.get("note", ""),
            })
        inference_frame = pd.DataFrame(inference_rows)
        for exp_id, group in inference_frame.groupby("exp_id"):
            base = group[group["method"] == "oneview"]
            if not base.empty and float(base.iloc[0]["p50_ms_batch1"] or 0) > 0:
                base_cost = float(base.iloc[0]["p50_ms_batch1"])
                inference_frame.loc[group.index, "relative_cost_vs_I00"] = (
                    inference_frame.loc[group.index, "p50_ms_batch1"] / base_cost
                )
    else:
        inference_frame = pd.DataFrame(columns=SHEETS["Inference"])

    backbone_rows, training_rows = [], []
    baseline_by_model = {}
    for (exp_id, seed), summary in summaries.items():
        config = configs.get((exp_id, seed), {})
        if exp_id.startswith("T00"):
            baseline_by_model[(summary.get("backbone"), seed)] = summary
    for (exp_id, seed), summary in sorted(summaries.items()):
        config = configs.get((exp_id, seed), {})
        common = {
            "exp_id": exp_id, "backbone": summary.get("backbone"), "weight_tag": summary.get("weight_tag"),
            "params_m": summary.get("num_params_m"), "GMAC": summary.get("gmac"),
            "img_size": config.get("img_size"), "epochs": config.get("epochs"), "seed": seed,
            "macro_f1_val": summary.get("macro_f1_val"), "top1_val": summary.get("top1_val"),
        }
        if exp_id.startswith("B"):
            one = inference_frame[(inference_frame["exp_id"] == exp_id) & (inference_frame["method"] == "oneview")]
            latency = one.iloc[0]["p50_ms_batch1"] if not one.empty else None
            backbone_rows.append({**common, "train_sec_per_epoch": summary.get("mean_epoch_seconds"),
                                  "latency_batch1_ms": latency, "notes": f"Device: {summary.get('device')}; {summary.get('gmac_tool')}"})
        if exp_id.startswith("T"):
            axis, changed = _experiment_axis(exp_id, config)
            base = baseline_by_model.get((summary.get("backbone"), seed))
            delta = summary.get("macro_f1_val") - base.get("macro_f1_val") if base else None
            f1 = summary.get("macro_f1_val_per_class") or [None] * 9
            training_rows.append({
                "exp_id": exp_id, "backbone": summary.get("backbone"), "axis": axis,
                "changed_from_T00": changed, "seed": seed,
                "macro_f1_val": summary.get("macro_f1_val"), "top1_val": summary.get("top1_val"),
                "delta_macro_f1_vs_T00": delta, "chinee_apple_f1": f1[0], "snake_weed_f1": f1[7],
                "notes": f"Best epoch {summary.get('best_epoch')}; time {summary.get('train_seconds')} sec",
            })

    final_rows, per_class_rows = [], []
    confusion_candidates = []
    for exp_id, seed_records in sorted(test_groups.items()):
        config = next((value for (run_id, _), value in configs.items() if run_id == exp_id), {})
        per_seed_metrics = {seed: metrics for seed, (_, metrics) in seed_records.items()}
        mf1_mean, mf1_std = _mean_std([metric["macro_f1"] for metric in per_seed_metrics.values()])
        acc_mean, acc_std = _mean_std([metric["top1"] for metric in per_seed_metrics.values()])
        for seed, (prediction, metrics) in sorted(seed_records.items()):
            val_entry = val_groups.get(exp_id, {}).get(seed)
            val_macro = val_entry[1]["macro_f1"] if val_entry else None
            final_rows.append({
                "exp_id": exp_id, "configuration": f"{config.get('backbone', 'unknown')} / {config.get('loss', 'unknown')} / {config.get('aug', 'unknown')}",
                "seed": seed, "macro_f1_val": val_macro, "macro_f1_test": metrics["macro_f1"],
                "top1_test": metrics["top1"], "ECE_test": metrics["ece"],
                "test_macro_f1_mean_std": None, "test_top1_mean_std": None,
                "n_seeds": len(seed_records),
            })
        final_rows.append({
            "exp_id": exp_id, "configuration": f"{config.get('backbone', 'unknown')} / {config.get('loss', 'unknown')} / {config.get('aug', 'unknown')}",
            "seed": "mean ± std (sample)",
            "macro_f1_val": np.mean([val_groups.get(exp_id, {}).get(seed, (None, {"macro_f1": np.nan}))[1]["macro_f1"]
                                     for seed in seed_records]),
            "macro_f1_test": mf1_mean, "top1_test": acc_mean,
            "ECE_test": np.mean([metric["ece"] for metric in per_seed_metrics.values()]),
            "test_macro_f1_mean_std": f"{mf1_mean:.4f} ± {mf1_std:.4f}",
            "test_top1_mean_std": f"{acc_mean:.4f} ± {acc_std:.4f}", "n_seeds": len(seed_records),
        })
        mean_confusion = np.mean([metric["confusion"] for metric in per_seed_metrics.values()], axis=0)
        confusion_candidates.append((exp_id, mean_confusion, len(seed_records)))
        vector_metrics = [metric for metric in per_seed_metrics.values()]
        for class_index, class_name in enumerate(names):
            per_class_rows.append({
                "exp_id": exp_id, "class": class_name,
                "n_test": float(np.mean([metric["support"][class_index] for metric in vector_metrics])),
                "precision": float(np.mean([metric["precision"][class_index] for metric in vector_metrics])),
                "recall": float(np.mean([metric["recall"][class_index] for metric in vector_metrics])),
                "F1": float(np.mean([metric["f1"][class_index] for metric in vector_metrics])),
            })
    if confusion_candidates:
        final_id, _, _ = max(confusion_candidates, key=lambda item: item[2])
        _plot_confusion(final_id, test_groups[final_id], names, Path(curves_dir) / f"{final_id}_confusion_matrix.png")

    latency_rows = []
    if not inference_data.empty:
        for _, row in inference_data.iterrows():
            latency_rows.append({
                "configuration": f"{row.get('exp_id')} · {row.get('method')}", "GPU": row.get("gpu"),
                "dtype": row.get("dtype"), "batch": row.get("batch"), "img_size": row.get("img_size"),
                "batch_norm_fused": row.get("method") == "bn-fused", "p50_ms": row.get("latency_p50_ms"),
                "p95_ms": row.get("latency_p95_ms"), "p99_ms": row.get("latency_p99_ms"),
                "images_per_s": row.get("throughput_img_s"), "torch": row.get("torch"),
                "includes_preprocessing": row.get("includes_preprocessing", False),
            })

    summary_rows = []
    for (exp_id, seed), summary in summaries.items():
        if exp_id.startswith(("B", "T", "F")):
            config = configs.get((exp_id, seed), {})
            one = inference_frame[(inference_frame["exp_id"] == exp_id) & (inference_frame["method"] == "oneview")]
            summary_rows.append({"exp_id": exp_id, "method": "oneview", "backbone": summary.get("backbone"),
                                 "macro_f1_val": summary.get("macro_f1_val"), "top1_val": summary.get("top1_val"),
                                 "p95_ms_batch1": one.iloc[0]["p95_ms_batch1"] if not one.empty else None,
                                 "params_m": summary.get("num_params_m"), "GMAC": summary.get("gmac"),
                                 "notes": f"{summary.get('weight_tag')} · seed {seed}"})
    if not inference_frame.empty:
        for _, row in inference_frame.iterrows():
            summary_rows.append({"exp_id": row.get("exp_id"), "method": row.get("method"),
                                 "backbone": None, "macro_f1_val": row.get("macro_f1_val"),
                                 "top1_val": row.get("top1_val"), "p95_ms_batch1": row.get("p95_ms_batch1"),
                                 "params_m": None, "GMAC": None, "notes": row.get("notes")})
    summary_frame = pd.DataFrame(summary_rows)
    if not summary_frame.empty:
        summary_frame = summary_frame.sort_values("macro_f1_val", ascending=False, na_position="last").head(10)
        summary_frame.insert(0, "rank", range(1, len(summary_frame) + 1))
    _plot_tradeoff(inference_frame, Path(curves_dir) / "inference_tradeoff.png")

    tables = {
        "Backbones": pd.DataFrame(backbone_rows, columns=SHEETS["Backbones"]),
        "Training": pd.DataFrame(training_rows, columns=SHEETS["Training"]),
        "Inference": inference_frame.reindex(columns=SHEETS["Inference"]),
        "Final": pd.DataFrame(final_rows, columns=SHEETS["Final"]),
        "PerClass": pd.DataFrame(per_class_rows, columns=SHEETS["PerClass"]),
        "Latency": pd.DataFrame(latency_rows, columns=SHEETS["Latency"]),
        "Summary": summary_frame.reindex(columns=SHEETS["Summary"]),
    }
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        for sheet_name, frame in tables.items():
            frame.to_excel(writer, sheet_name=sheet_name, index=False)
            worksheet = writer.book[sheet_name]
            worksheet.freeze_panes = "A2"
            worksheet.auto_filter.ref = worksheet.dimensions
            for column_cells in worksheet.columns:
                width = min(48, max(12, max(len(str(cell.value or "")) for cell in column_cells) + 2))
                worksheet.column_dimensions[column_cells[0].column_letter].width = width
            for cell in worksheet[1]:
                font = copy(cell.font)
                font.bold = True
                font.color = "FFFFFF"
                cell.font = font
                cell.fill = PatternFill("solid", fgColor="244062")
    print(f"Wrote {output.resolve()}")
    return tables


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", nargs="+", default=["runs", "runs_final"])
    parser.add_argument("--predictions", default="predictions")
    parser.add_argument("--labels", default="data/labels/labels.csv")
    parser.add_argument("--test-csv", default="data/labels/test_subset0.csv")
    parser.add_argument("--inference-csv", default="inference_results.csv")
    parser.add_argument("--output", default="results.xlsx")
    parser.add_argument("--curves", default="curves")
    args = parser.parse_args()
    build_workbook(args.runs, args.predictions, args.labels, args.test_csv,
                   args.inference_csv, args.output, args.curves)


if __name__ == "__main__":
    main()
