"""DeepWeeds CSV validation, image transforms, datasets, and dataloaders."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from PIL import Image

NUM_CLASSES = 9
CLASS_NAMES = [
    "Chinee Apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly Acacia",
    "Rubber Vine", "Siam Weed", "Snake Weed", "Negatives",
]
EXPECTED_CLASS_COUNTS = [1125, 1064, 1031, 1022, 1062, 1009, 1074, 1016, 9106]
EXPECTED_DATASET_SIZE = 17509
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
SPLIT_COLUMNS = ("Filename", "Label")


def load_split(labels_dir: str | Path, fold: int = 0):
    """Read the three author-provided CSVs without resampling or filtering."""
    labels_dir = Path(labels_dir)
    if fold not in range(5):
        raise ValueError(f"fold must be 0..4, received {fold}")
    paths = [labels_dir / f"{part}_subset{fold}.csv" for part in ("train", "val", "test")]
    frames = []
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Missing split CSV: {path}. Run prepare_data.py first.")
        frame = pd.read_csv(path)
        missing = [column for column in SPLIT_COLUMNS if column not in frame.columns]
        if missing:
            raise ValueError(f"{path}: missing columns {missing}; expected {SPLIT_COLUMNS}")
        # Validate integer labels without rewriting the source DataFrame.
        labels = pd.to_numeric(frame["Label"], errors="raise")
        if not np.equal(labels, np.floor(labels)).all() or not labels.between(0, NUM_CLASSES - 1).all():
            raise ValueError(f"{path}: Label values must be integers in 0..{NUM_CLASSES - 1}")
        if frame["Filename"].isna().any() or frame["Filename"].duplicated().any():
            raise ValueError(f"{path}: Filename contains missing or duplicate values")
        frames.append(frame)
    return tuple(frames)


def check_split(train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame,
                images_dir: str | Path) -> dict:
    """Run the required split integrity and image-existence checks and print counts."""
    images_dir = Path(images_dir)
    parts = {"train": train_df, "val": val_df, "test": test_df}
    names = {key: set(df["Filename"].astype(str)) for key, df in parts.items()}
    overlap = {
        "train_val": sorted(names["train"] & names["val"]),
        "train_test": sorted(names["train"] & names["test"]),
        "val_test": sorted(names["val"] & names["test"]),
    }
    nonempty_overlap = {key: len(value) for key, value in overlap.items() if value}
    if nonempty_overlap:
        raise ValueError(f"Fold {0} split overlap by filename: {nonempty_overlap}")

    union = set.union(*names.values())
    if len(union) != EXPECTED_DATASET_SIZE:
        raise ValueError(f"Fold split union contains {len(union)} images; expected {EXPECTED_DATASET_SIZE}")

    missing_files = [str(images_dir / name) for group in names.values() for name in group
                     if not (images_dir / name).is_file()]
    if missing_files:
        preview = missing_files[:10]
        raise FileNotFoundError(f"{len(missing_files)} CSV images are missing under {images_dir}; first: {preview}")

    n = {key: len(df) for key, df in parts.items()}
    expected_ratio = {"train": 0.60, "val": 0.20, "test": 0.20}
    ratios = {key: value / EXPECTED_DATASET_SIZE for key, value in n.items()}
    off_ratio = {key: round(ratios[key] - expected_ratio[key], 4)
                 for key in parts if abs(ratios[key] - expected_ratio[key]) > 0.01}
    if off_ratio:
        raise ValueError(f"Split proportions differ from 60/20/20 by more than 1 percentage point: {off_ratio}")
    per_class = {
        key: {CLASS_NAMES[i]: int((pd.to_numeric(df["Label"]) == i).sum()) for i in range(NUM_CLASSES)}
        for key, df in parts.items()
    }
    result = {"n": n, "ratios": ratios, "per_class": per_class,
              "overlap": {key: 0 for key in overlap}, "union": len(union),
              "missing_images": 0, "expected_total": EXPECTED_DATASET_SIZE}
    print(f"Split sizes: {n} (fractions: { {k: round(v, 4) for k, v in ratios.items()} })")
    print("Class counts:")
    print(pd.DataFrame(per_class).to_string())
    print(f"Pairwise overlap: {result['overlap']}; union={len(union)}; missing images=0")
    return result


def _transforms_module():
    try:
        from torchvision import transforms
    except Exception as exc:
        raise RuntimeError("torchvision is required. Install requirements.txt in Colab/Kaggle.") from exc
    return transforms


def build_transforms(train: bool, img_size: int = 224, aug: str = "basic"):
    """Create train or deterministic validation transforms; no vertical flips are used."""
    if img_size < 32:
        raise ValueError("img_size must be at least 32")
    transforms = _transforms_module()
    if train:
        ops = [transforms.RandomResizedCrop(img_size, scale=(0.70, 1.0)),
               transforms.RandomHorizontalFlip(p=0.5)]
        if aug == "basic":
            pass
        elif aug == "color":
            ops.append(transforms.ColorJitter(brightness=0.2, contrast=0.2,
                                               saturation=0.2, hue=0.04))
        elif aug == "trivial":
            if not hasattr(transforms, "TrivialAugmentWide"):
                raise RuntimeError("TrivialAugmentWide is unavailable; install a recent torchvision.")
            ops.append(transforms.TrivialAugmentWide())
        elif aug == "randaug":
            ops.append(transforms.RandAugment(num_ops=2, magnitude=9))
        elif aug == "none":
            ops = [transforms.Resize(round(img_size * 256 / 224)), transforms.CenterCrop(img_size)]
        else:
            raise ValueError(f"Unknown augmentation {aug!r}; choose basic, color, trivial, randaug, or none")
    else:
        # The source images are 256x256. Scaling before center-crop preserves the
        # same relative crop (256->224) at every requested training resolution.
        resize_size = round(img_size * 256 / 224)
        ops = [transforms.Resize(resize_size), transforms.CenterCrop(img_size)]
    ops.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(ops)


class DeepWeedsDataset(Dataset):
    """Image dataset returning ``(normalized_tensor, integer_label, filename)``."""

    def __init__(self, df: pd.DataFrame, images_dir: str | Path, transform=None):
        missing = [column for column in ("Filename", "Label") if column not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns {missing}")
        self.df = df.reset_index(drop=True)
        self.images_dir = Path(images_dir)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, i: int):
        row = self.df.iloc[i]
        filename = str(row["Filename"])
        path = self.images_dir / filename
        with Image.open(path) as image:
            image = image.convert("RGB")
            tensor = self.transform(image) if self.transform is not None else image.copy()
        return tensor, int(row["Label"]), filename


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % (2 ** 32)
    np.random.seed(seed)
    random.seed(seed)


def make_loader(df: pd.DataFrame, images_dir: str | Path, transform, batch_size: int,
                train: bool, sampler: str | None = None, num_workers: int = 2,
                seed: int = 0) -> DataLoader:
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if sampler not in (None, "balanced"):
        raise ValueError("sampler must be None or 'balanced'")
    dataset = DeepWeedsDataset(df, images_dir, transform)
    generator = torch.Generator().manual_seed(int(seed))
    weighted_sampler = None
    if train and sampler == "balanced":
        labels = pd.to_numeric(df["Label"]).to_numpy(dtype=np.int64)
        counts = np.bincount(labels, minlength=NUM_CLASSES)
        weights = 1.0 / np.maximum(counts, 1)
        sample_weights = torch.as_tensor(weights[labels], dtype=torch.double)
        weighted_sampler = WeightedRandomSampler(sample_weights, num_samples=len(labels),
                                                 replacement=True, generator=generator)
    return DataLoader(
        dataset, batch_size=batch_size, shuffle=bool(train and weighted_sampler is None),
        sampler=weighted_sampler, num_workers=max(0, int(num_workers)),
        pin_memory=torch.cuda.is_available(), drop_last=bool(train and len(dataset) > batch_size),
        worker_init_fn=_seed_worker, generator=generator,
        persistent_workers=False,
    )
