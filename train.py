"""Training script for anomaly detection models.

Trains anomaly detection models (PatchCore, PaDIM, FastFlow, DRAEM) on the
multi-view industrial dataset using the Anomalib framework. Handles per-class
training with custom validation splits that respect multi-view sample grouping.
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import Folder, PredictDataset
from anomalib.engine import Engine

from src.data import (
    create_validation_split,
    discover_classes,
    get_image_id,
    prepare_data_directory,
)
from src.metrics import (
    compute_anomaly_activity,
    compute_per_image_ap,
    compute_pixel_metrics,
)
from src.models import get_default_max_epochs, get_model
from src.visualization import save_validation_results


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train anomaly detection models on multi-view industrial dataset."
    )

    # Required arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["patchcore", "padim", "fastflow", "draem"],
        help="Anomaly detection algorithm to use.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of the dataset.",
    )

    # Optional arguments
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (square). Default: 224.",
    )
    parser.add_argument(
        "--model-output-path",
        type=str,
        default=None,
        help="Directory to save trained model checkpoints.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store validation visualizations and stats.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Specific classes to train (e.g., class_01 class_02). Default: all.",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.2,
        help="Fraction of normal samples for validation. Default: 0.2.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility. Default: 42.",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=None,
        help="Max training epochs. Default: algorithm-specific.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Training batch size. Default: 32.",
    )

    # Algorithm-specific parameters
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone network (e.g., wide_resnet50_2, resnet18).",
    )
    parser.add_argument(
        "--coreset-sampling-ratio",
        type=float,
        default=0.1,
        help="PatchCore: coreset sampling ratio. Default: 0.1.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=9,
        help="PatchCore: number of nearest neighbors. Default: 9.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="PaDIM: number of features for dimensionality reduction.",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=8,
        help="FastFlow: number of normalizing flow steps. Default: 8.",
    )
    parser.add_argument(
        "--anomaly-source-path",
        type=str,
        default=None,
        help="DRAEM: path to external texture images for anomaly generation.",
    )

    return parser.parse_args()


def build_model_kwargs(args: argparse.Namespace) -> dict:
    """Extract algorithm-specific kwargs from parsed arguments."""
    kwargs = {}

    if args.backbone:
        kwargs["backbone"] = args.backbone

    if args.algorithm == "patchcore":
        kwargs["coreset_sampling_ratio"] = args.coreset_sampling_ratio
        kwargs["num_neighbors"] = args.num_neighbors
    elif args.algorithm == "padim":
        if args.n_features is not None:
            kwargs["n_features"] = args.n_features
    elif args.algorithm == "fastflow":
        kwargs["flow_steps"] = args.flow_steps
    elif args.algorithm == "draem":
        if args.anomaly_source_path:
            kwargs["anomaly_source_path"] = args.anomaly_source_path

    return kwargs


def run_inference_on_images(
    engine: Engine,
    model,
    image_paths: list[Path],
    img_size: int,
    ckpt_path: str,
) -> list[np.ndarray]:
    """Run model inference on a list of images and return anomaly maps.

    Args:
        engine: Anomalib Engine instance.
        model: Trained model instance.
        image_paths: List of image file paths.
        img_size: Image size used during training.
        ckpt_path: Path to model checkpoint.

    Returns:
        List of anomaly maps as numpy arrays, shape (H, W) in [0, 1].
    """
    if not image_paths:
        return []

    # Create a temporary directory with symlinks for PredictDataset
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "images"
        tmp_path.mkdir()
        for img_path in image_paths:
            dst = tmp_path / img_path.name
            dst.symlink_to(img_path.resolve())

        dataset = PredictDataset(
            path=str(tmp_path),
            image_size=(img_size, img_size),
        )

        predictions = engine.predict(
            model=model,
            dataset=dataset,
            ckpt_path=ckpt_path,
        )

    # Extract anomaly maps and match to original order
    anomaly_maps = {}
    if predictions is not None:
        for pred in predictions:
            fname = Path(pred.image_path).name
            amap = pred.anomaly_map
            if isinstance(amap, torch.Tensor):
                amap = amap.squeeze().cpu().numpy()
            # Normalize to [0, 1]
            if amap.max() > amap.min():
                amap = (amap - amap.min()) / (amap.max() - amap.min())
            else:
                amap = np.zeros_like(amap)
            anomaly_maps[fname] = amap

    # Return in original order
    result = []
    for img_path in image_paths:
        if img_path.name in anomaly_maps:
            result.append(anomaly_maps[img_path.name])
        else:
            result.append(np.zeros((img_size, img_size), dtype=np.float32))

    return result


def train_class(
    class_name: str,
    args: argparse.Namespace,
    model_kwargs: dict,
) -> dict:
    """Train a model for a single class and evaluate on validation set.

    Args:
        class_name: Name of the class (e.g., 'class_01').
        args: Parsed command-line arguments.
        model_kwargs: Algorithm-specific model parameters.

    Returns:
        Dictionary with validation metrics.
    """
    print(f"\n{'='*60}")
    print(f"Training {args.algorithm.upper()} on {class_name}")
    print(f"{'='*60}")

    # Step 1: Create validation split
    split_info = create_validation_split(
        data_dir=args.data_dir,
        class_name=class_name,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    print(f"  Training samples: {len(split_info['train_normal'])} images")
    print(f"  Val normal: {len(split_info['val_normal'])} images")
    print(f"  Val anomalous: {len(split_info['val_anomalous'])} images")
    print(f"  Test images: {len(split_info['test'])} images")

    # Step 2: Prepare data directory
    prepared_dir = Path(args.data_dir).parent / "prepared_data"
    dir_paths = prepare_data_directory(split_info, str(prepared_dir), class_name)

    # Step 3: Create Anomalib Folder datamodule
    # Use prepared directory with pre-split data
    datamodule = Folder(
        name=class_name,
        root=str(prepared_dir / class_name),
        normal_dir="train/normal",
        abnormal_dir="val/anomalous",
        normal_test_dir="val/normal",
        abnormal_test_dir="val/anomalous",
        mask_dir="val_masks/anomalous",
        image_size=(args.img_size, args.img_size),
        train_batch_size=args.batch_size,
        eval_batch_size=args.batch_size,
        test_split_mode="from_dir",
        val_split_mode="same_as_test",
        val_split_ratio=0.5,
    )

    # Step 4: Create model and engine
    model = get_model(args.algorithm, **model_kwargs)
    max_epochs = args.max_epochs or get_default_max_epochs(args.algorithm)

    # Determine checkpoint directory
    if args.model_output_path:
        ckpt_dir = Path(args.model_output_path) / class_name
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        default_root_dir = str(ckpt_dir)
    else:
        default_root_dir = str(Path("results") / args.algorithm / class_name)

    engine = Engine(
        max_epochs=max_epochs,
        default_root_dir=default_root_dir,
        accelerator="auto",
        devices=1,
    )

    # Step 5: Train
    print(f"  Training for {max_epochs} epoch(s)...")
    engine.fit(datamodule=datamodule, model=model)

    # Find the checkpoint
    ckpt_path = _find_best_checkpoint(default_root_dir)
    print(f"  Checkpoint saved: {ckpt_path}")

    # Step 6: Run validation inference
    print("  Running validation inference...")
    val_normal_paths = split_info["val_normal"]
    val_anomalous_paths = [p for p, _ in split_info["val_anomalous"]]
    val_mask_paths = [m for _, m in split_info["val_anomalous"]]

    normal_maps = run_inference_on_images(
        engine, model, val_normal_paths, args.img_size, ckpt_path
    )
    anomalous_maps = run_inference_on_images(
        engine, model, val_anomalous_paths, args.img_size, ckpt_path
    )

    # Load ground truth masks
    gt_masks = []
    for mask_path in val_mask_paths:
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is not None:
            mask = cv2.resize(mask, (args.img_size, args.img_size), interpolation=cv2.INTER_NEAREST)
            mask = (mask > 127).astype(np.float32)
        else:
            mask = np.zeros((args.img_size, args.img_size), dtype=np.float32)
        gt_masks.append(mask)

    # Step 7: Compute metrics
    all_pred_maps = normal_maps + anomalous_maps
    all_gt_masks = [np.zeros((args.img_size, args.img_size), dtype=np.float32)] * len(normal_maps) + gt_masks
    metrics = compute_pixel_metrics(all_pred_maps, all_gt_masks)

    print(f"  Validation Metrics:")
    for name, value in metrics.items():
        print(f"    {name}: {value:.6f}")

    # Step 8: Save visualizations if output_dir specified
    if args.output_dir:
        print("  Saving validation visualizations...")

        # Build entries for visualization
        normal_entries = []
        for img_path, pred_map in zip(val_normal_paths, normal_maps):
            activity = compute_anomaly_activity(pred_map)
            normal_entries.append({
                "image_path": str(img_path),
                "pred_map": pred_map,
                "gt_mask": np.zeros((args.img_size, args.img_size), dtype=np.float32),
                "score": activity,
            })

        anomalous_entries = []
        for img_path, pred_map, gt_mask in zip(val_anomalous_paths, anomalous_maps, gt_masks):
            ap = compute_per_image_ap(pred_map, gt_mask)
            anomalous_entries.append({
                "image_path": str(img_path),
                "pred_map": pred_map,
                "gt_mask": gt_mask,
                "score": ap,
            })

        save_validation_results(
            normal_entries=normal_entries,
            anomalous_entries=anomalous_entries,
            metrics=metrics,
            output_dir=args.output_dir,
            class_name=class_name,
            algorithm=args.algorithm,
        )

    # Save training config alongside checkpoint
    config = {
        "algorithm": args.algorithm,
        "class_name": class_name,
        "img_size": args.img_size,
        "max_epochs": max_epochs,
        "model_kwargs": model_kwargs,
        "metrics": metrics,
        "checkpoint_path": ckpt_path,
    }
    config_path = Path(default_root_dir) / "training_config.json"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    return metrics


def _find_best_checkpoint(root_dir: str) -> str:
    """Find the best model checkpoint in the results directory.

    Searches recursively for .ckpt files, preferring 'best' or 'last' named ones.
    """
    root = Path(root_dir)
    ckpt_files = list(root.rglob("*.ckpt"))

    if not ckpt_files:
        raise FileNotFoundError(f"No checkpoint found in {root_dir}")

    # Prefer files with 'best' in name, then 'last', then most recent
    for keyword in ["best", "last"]:
        matches = [f for f in ckpt_files if keyword in f.name.lower()]
        if matches:
            return str(matches[0])

    # Return most recently modified
    return str(max(ckpt_files, key=lambda f: f.stat().st_mtime))


def main():
    args = parse_args()
    model_kwargs = build_model_kwargs(args)

    # Discover classes
    if args.classes:
        classes = args.classes
    else:
        classes = discover_classes(args.data_dir)

    if not classes:
        print(f"Error: No classes found in {args.data_dir}")
        sys.exit(1)

    print(f"Algorithms: {args.algorithm}")
    print(f"Classes: {classes}")
    print(f"Image size: {args.img_size}x{args.img_size}")

    # Train on each class
    all_metrics = {}
    for class_name in classes:
        metrics = train_class(class_name, args, model_kwargs)
        all_metrics[class_name] = metrics

    # Print summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for class_name, metrics in all_metrics.items():
        print(f"  {class_name}: AP={metrics['AP']:.4f}  AUPRC={metrics['AUPRC']:.4f}  AUROC={metrics['AUROC']:.4f}")

    mean_ap = np.mean([m["AP"] for m in all_metrics.values()])
    print(f"\n  Mean AP: {mean_ap:.4f}")


if __name__ == "__main__":
    main()
