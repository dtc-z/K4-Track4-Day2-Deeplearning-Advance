"""Verify images.zip, extract DeepWeeds images, and fetch the author's fold-0 CSVs."""
from __future__ import annotations

import argparse
import hashlib
import shutil
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parent
EXPECTED_MD5 = "b7b30f96d466fba86016aa5a26606e0f"
IMAGE_URL = "https://zenodo.org/records/7939060/files/images.zip?download=1"
LABEL_BASE = "https://raw.githubusercontent.com/AlexOlsen/DeepWeeds/master/labels"
REQUIRED_LABELS = ("labels.csv", "train_subset0.csv", "val_subset0.csv", "test_subset0.csv")
EXPECTED_IMAGES = 17509


def md5_file(path: Path) -> str:
    digest = hashlib.md5()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".download")
    request = urllib.request.Request(url, headers={"User-Agent": "DeepWeeds-Day2-Lab/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output)
        if temporary.stat().st_size == 0:
            raise RuntimeError(f"Downloaded empty file from {url}")
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def resolve_archive(argument: str | None, data_dir: Path) -> Path:
    if argument:
        return Path(argument).expanduser().resolve()
    candidates = [REPO / "images.zip", data_dir / "images.zip"]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    destination = data_dir / "images.zip"
    print(f"images.zip not found; downloading {IMAGE_URL}")
    download(IMAGE_URL, destination)
    return destination


def extract_images(archive: Path, images_dir: Path, force: bool = False) -> int:
    if not archive.is_file():
        raise FileNotFoundError(archive)
    observed_md5 = md5_file(archive)
    if observed_md5.lower() != EXPECTED_MD5:
        raise ValueError(f"images.zip MD5 mismatch: expected {EXPECTED_MD5}, got {observed_md5}")
    images_dir.mkdir(parents=True, exist_ok=True)
    existing = list(images_dir.glob("*.jpg"))
    if not force and len(existing) == EXPECTED_IMAGES:
        print(f"Found {len(existing)} extracted JPGs under {images_dir}; extraction skipped")
        return len(existing)

    with zipfile.ZipFile(archive) as bundle:
        corrupt = bundle.testzip()
        if corrupt is not None:
            raise ValueError(f"Corrupt member in archive: {corrupt}")
        members = [info for info in bundle.infolist() if not info.is_dir()
                   and PurePosixPath(info.filename).suffix.lower() in {".jpg", ".jpeg"}]
        if len(members) != EXPECTED_IMAGES:
            raise ValueError(f"Archive contains {len(members)} JPEGs; expected {EXPECTED_IMAGES}")
        target_names = [PurePosixPath(info.filename).name for info in members]
        if len(set(target_names)) != len(target_names):
            raise ValueError("Archive contains duplicate JPEG basenames; refusing to flatten it")
        for info, basename in zip(members, target_names):
            relative = PurePosixPath(info.filename)
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(f"Unsafe path in archive: {info.filename}")
            target = images_dir / basename
            with bundle.open(info) as source, target.open("wb") as output:
                shutil.copyfileobj(source, output)
    print(f"Verified MD5 {observed_md5}; extracted {len(members)} images to {images_dir}")
    return len(members)


def download_labels(labels_dir: Path, force: bool = False) -> None:
    labels_dir.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_LABELS:
        destination = labels_dir / name
        if destination.exists() and not force:
            print(f"Keeping existing {destination}")
            continue
        download(f"{LABEL_BASE}/{name}", destination)
        print(f"Downloaded {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", help="path to the supplied images.zip; defaults to repo/images.zip")
    parser.add_argument("--data-dir", default=str(REPO / "data"), help="directory for images and labels")
    parser.add_argument("--force-extract", action="store_true", help="extract images again")
    parser.add_argument("--force-labels", action="store_true", help="download CSVs again")
    args = parser.parse_args()
    data_dir = Path(args.data_dir).expanduser().resolve()
    archive = resolve_archive(args.archive, data_dir)
    extract_images(archive, data_dir / "images", args.force_extract)
    download_labels(data_dir / "labels", args.force_labels)
    print("Data preparation complete. Next run: python starter/eda.py --smoke-test")


if __name__ == "__main__":
    main()
