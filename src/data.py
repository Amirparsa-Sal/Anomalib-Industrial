"""Dataset discovery, validation splitting, and data preparation.

Handles the multi-view dataset structure where each sample consists of
5 images from different perspectives. The split logic ensures all views
of a sample stay together to prevent data leakage.
"""

import os
import re
import shutil
from pathlib import Path
from typing import Optional

import numpy as np


def discover_classes(data_dir: str) -> list[str]:
    """Find all class directories in the dataset root.

    Args:
        data_dir: Root directory of the dataset.

    Returns:
        Sorted list of class directory names (e.g., ['class_01', 'class_02', ...]).
    """
    data_path = Path(data_dir)
    classes = sorted([
        d.name for d in data_path.iterdir()
        if d.is_dir() and d.name.startswith("class_")
    ])
    return classes


def get_sample_ids(image_dir: str) -> list[str]:
    """Extract unique sample IDs from a directory of multi-view images.

    Image naming convention: img_<hash>_view<N>.png
    Sample ID is the hash portion that groups all views together.

    Args:
        image_dir: Directory containing image files.

    Returns:
        Sorted list of unique sample IDs.
    """
    pattern = re.compile(r"^img_([a-f0-9]+)_view\d+\.png$")
    sample_ids = set()

    dir_path = Path(image_dir)
    if not dir_path.exists():
        return []

    for f in dir_path.iterdir():
        if f.is_file():
            match = pattern.match(f.name)
            if match:
                sample_ids.add(match.group(1))

    return sorted(sample_ids)


def get_views_for_sample(image_dir: str, sample_id: str) -> list[Path]:
    """Get all view image paths for a given sample ID.

    Args:
        image_dir: Directory containing images.
        sample_id: The hash identifier of the sample.

    Returns:
        List of Path objects for all views of this sample.
    """
    dir_path = Path(image_dir)
    views = sorted(dir_path.glob(f"img_{sample_id}_view*.png"))
    return views


def create_validation_split(
    data_dir: str,
    class_name: str,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> dict:
    """Create train/val split for a single class, keeping multi-view samples together.

    Validation set contains:
    - 20% of normal samples (all views kept together)
    - ALL anomalous samples (with their ground truth masks)

    Args:
        data_dir: Root dataset directory.
        class_name: Name of the class (e.g., 'class_01').
        val_ratio: Fraction of normal samples for validation.
        seed: Random seed for reproducibility.

    Returns:
        Dictionary with keys:
            - 'train_normal': list of image paths for training
            - 'val_normal': list of image paths for validation (normal)
            - 'val_anomalous': list of (image_path, mask_path) tuples
            - 'test': list of image paths for test set
    """
    class_dir = Path(data_dir) / class_name
    good_dir = class_dir / "train" / "good"
    test_dir = class_dir / "test"

    # Get all normal sample IDs and split
    normal_ids = get_sample_ids(str(good_dir))
    rng = np.random.default_rng(seed)
    rng.shuffle(normal_ids)

    n_val = max(1, int(len(normal_ids) * val_ratio))
    val_ids = set(normal_ids[:n_val])
    train_ids = set(normal_ids[n_val:])

    # Collect training normal images
    train_normal = []
    for sid in sorted(train_ids):
        train_normal.extend(get_views_for_sample(str(good_dir), sid))

    # Collect validation normal images
    val_normal = []
    for sid in sorted(val_ids):
        val_normal.extend(get_views_for_sample(str(good_dir), sid))

    # Collect all anomalous samples with their masks
    val_anomalous = []
    anomaly_dirs = sorted([
        d for d in (class_dir / "train").iterdir()
        if d.is_dir() and d.name.startswith("anomaly_")
    ])

    for anomaly_dir in anomaly_dirs:
        mask_dir = class_dir / "ground_truth_train" / anomaly_dir.name
        sample_ids = get_sample_ids(str(anomaly_dir))

        for sid in sample_ids:
            views = get_views_for_sample(str(anomaly_dir), sid)
            masks = get_views_for_sample(str(mask_dir), sid)

            for view_img, view_mask in zip(views, masks):
                val_anomalous.append((view_img, view_mask))

    # Collect test images
    test_images = sorted(test_dir.glob("*.png")) if test_dir.exists() else []

    return {
        "train_normal": train_normal,
        "val_normal": val_normal,
        "val_anomalous": val_anomalous,
        "test": test_images,
    }


def prepare_data_directory(
    split_info: dict,
    prepared_dir: str,
    class_name: str,
) -> dict:
    """Create a prepared directory structure with symlinks for Anomalib.

    Creates the following structure:
        prepared_dir/class_name/
            train/normal/          -> training normal images (symlinks)
            val/normal/            -> validation normal images (symlinks)
            val/anomalous/         -> validation anomalous images (symlinks)
            val_masks/anomalous/   -> ground truth masks (symlinks)

    Args:
        split_info: Output from create_validation_split().
        prepared_dir: Root directory for prepared data.
        class_name: Name of the class.

    Returns:
        Dictionary with paths to prepared directories.
    """
    base = Path(prepared_dir) / class_name

    # Define directory structure
    dirs = {
        "train_normal": base / "train" / "normal",
        "val_normal": base / "val" / "normal",
        "val_anomalous": base / "val" / "anomalous",
        "val_masks": base / "val_masks" / "anomalous",
    }

    # Clean and recreate
    if base.exists():
        shutil.rmtree(base)

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # Create symlinks for training normal images
    for img_path in split_info["train_normal"]:
        dst = dirs["train_normal"] / img_path.name
        os.symlink(img_path.resolve(), dst)

    # Create symlinks for validation normal images
    for img_path in split_info["val_normal"]:
        dst = dirs["val_normal"] / img_path.name
        os.symlink(img_path.resolve(), dst)

    # Create symlinks for validation anomalous images and masks
    for img_path, mask_path in split_info["val_anomalous"]:
        dst_img = dirs["val_anomalous"] / img_path.name
        dst_mask = dirs["val_masks"] / mask_path.name
        os.symlink(img_path.resolve(), dst_img)
        os.symlink(mask_path.resolve(), dst_mask)

    return {k: str(v) for k, v in dirs.items()}


def get_image_id(image_path: str | Path) -> str:
    """Extract the submission image ID from a file path.

    Converts a filename like 'img_abc123_view1.png' to 'img_abc123_view1'.

    Args:
        image_path: Path to the image file.

    Returns:
        Image ID string for submission.
    """
    return Path(image_path).stem
