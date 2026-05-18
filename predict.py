"""Prediction script for generating test submissions.

Loads trained anomaly detection models and runs inference on test images,
producing submission CSV files with q8rle-encoded anomaly maps.
"""

import argparse
import csv
import sys
import tempfile
from pathlib import Path

import cv2
import numpy as np
import torch
from anomalib.data import PredictDataset
from anomalib.engine import Engine

from src.data import discover_classes, get_image_id
from src.encoding import float_matrix_to_q8rle
from src.models import get_model
from src.visualization import save_prediction_visualization


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run inference on test images and generate submission file."
    )

    # Required arguments
    parser.add_argument(
        "--algorithm",
        type=str,
        required=True,
        choices=["patchcore", "padim", "fastflow", "draem"],
        help="Anomaly detection algorithm used for training.",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        required=True,
        help="Root directory of the dataset.",
    )
    parser.add_argument(
        "--model-input-path",
        type=str,
        required=True,
        help="Directory containing trained model checkpoints (one per class).",
    )

    # Optional arguments
    parser.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Image size (square). Must match training. Default: 224.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Directory to store prediction visualizations.",
    )
    parser.add_argument(
        "--submission-path",
        type=str,
        default="submission.csv",
        help="Path for the output submission CSV. Default: submission.csv.",
    )
    parser.add_argument(
        "--classes",
        type=str,
        nargs="+",
        default=None,
        help="Specific classes to predict. Default: all.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Inference batch size. Default: 32.",
    )

    # Algorithm-specific (needed for model instantiation)
    parser.add_argument(
        "--backbone",
        type=str,
        default=None,
        help="Backbone network (must match training).",
    )
    parser.add_argument(
        "--coreset-sampling-ratio",
        type=float,
        default=0.1,
        help="PatchCore: coreset sampling ratio.",
    )
    parser.add_argument(
        "--num-neighbors",
        type=int,
        default=9,
        help="PatchCore: number of nearest neighbors.",
    )
    parser.add_argument(
        "--n-features",
        type=int,
        default=None,
        help="PaDIM: number of features.",
    )
    parser.add_argument(
        "--flow-steps",
        type=int,
        default=8,
        help="FastFlow: number of flow steps.",
    )
    parser.add_argument(
        "--anomaly-source-path",
        type=str,
        default=None,
        help="DRAEM: path to texture images.",
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


def find_checkpoint(model_input_path: str, class_name: str) -> str:
    """Find the model checkpoint for a given class.

    Searches in model_input_path/class_name/ for .ckpt files.

    Args:
        model_input_path: Root directory with saved models.
        class_name: Class to find checkpoint for.

    Returns:
        Path to the checkpoint file.

    Raises:
        FileNotFoundError: If no checkpoint is found.
    """
    search_dir = Path(model_input_path) / class_name
    if not search_dir.exists():
        # Try without class subdirectory
        search_dir = Path(model_input_path)

    ckpt_files = list(search_dir.rglob("*.ckpt"))

    if not ckpt_files:
        raise FileNotFoundError(
            f"No checkpoint found for {class_name} in {search_dir}"
        )

    # Prefer 'best' then 'last' then most recent
    for keyword in ["best", "last"]:
        matches = [f for f in ckpt_files if keyword in f.name.lower()]
        if matches:
            return str(matches[0])

    return str(max(ckpt_files, key=lambda f: f.stat().st_mtime))


def run_test_inference(
    model,
    engine: Engine,
    test_images: list[Path],
    img_size: int,
    ckpt_path: str,
) -> dict[str, np.ndarray]:
    """Run inference on test images and return anomaly maps.

    Args:
        model: Model instance.
        engine: Engine instance.
        test_images: List of test image paths.
        img_size: Image size for inference.
        ckpt_path: Path to model checkpoint.

    Returns:
        Dictionary mapping image filename to anomaly map (H, W) in [0, 1].
    """
    if not test_images:
        return {}

    # Create a temporary directory with symlinks
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / "images"
        tmp_path.mkdir()
        for img_path in test_images:
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

    # Collect anomaly maps
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

    return anomaly_maps


def predict_class(
    class_name: str,
    args: argparse.Namespace,
    model_kwargs: dict,
) -> list[tuple[str, str]]:
    """Run prediction for a single class.

    Args:
        class_name: Name of the class.
        args: Parsed arguments.
        model_kwargs: Model configuration.

    Returns:
        List of (image_id, q8rle_string) tuples for submission.
    """
    print(f"\n{'='*60}")
    print(f"Predicting {args.algorithm.upper()} on {class_name}")
    print(f"{'='*60}")

    # Find test images
    test_dir = Path(args.data_dir) / class_name / "test"
    if not test_dir.exists():
        print(f"  Warning: No test directory for {class_name}")
        return []

    test_images = sorted(test_dir.glob("*.png"))
    print(f"  Test images: {len(test_images)}")

    # Find checkpoint
    ckpt_path = find_checkpoint(args.model_input_path, class_name)
    print(f"  Checkpoint: {ckpt_path}")

    # Create model and engine
    model = get_model(args.algorithm, **model_kwargs)
    engine = Engine(accelerator="auto", devices=1)

    # Run inference
    print("  Running inference...")
    anomaly_maps = run_test_inference(
        model, engine, test_images, args.img_size, ckpt_path
    )

    # Generate submission entries
    submissions = []
    for img_path in test_images:
        fname = img_path.name
        image_id = get_image_id(fname)

        if fname in anomaly_maps:
            amap = anomaly_maps[fname]
            # Resize to original image dimensions for submission
            original = cv2.imread(str(img_path))
            if original is not None:
                orig_h, orig_w = original.shape[:2]
                amap_resized = cv2.resize(
                    amap, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
                )
            else:
                amap_resized = amap
        else:
            # Fallback: all zeros
            original = cv2.imread(str(img_path))
            if original is not None:
                orig_h, orig_w = original.shape[:2]
            else:
                orig_h, orig_w = args.img_size, args.img_size
            amap_resized = np.zeros((orig_h, orig_w), dtype=np.float32)

        q8rle_str = float_matrix_to_q8rle(amap_resized)
        submissions.append((image_id, q8rle_str))

    # Save visualizations if requested
    if args.output_dir:
        print("  Saving prediction visualizations...")
        vis_dir = Path(args.output_dir) / class_name / "test"
        vis_dir.mkdir(parents=True, exist_ok=True)

        for img_path in test_images:
            fname = img_path.name
            if fname in anomaly_maps:
                amap = anomaly_maps[fname]
                save_prediction_visualization(
                    image_path=str(img_path),
                    pred_map=amap,
                    output_path=vis_dir / f"{img_path.stem}_vis.png",
                    gt_mask=None,
                )

    print(f"  Generated {len(submissions)} submission entries")
    return submissions


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

    print(f"Algorithm: {args.algorithm}")
    print(f"Classes: {classes}")
    print(f"Image size: {args.img_size}x{args.img_size}")
    print(f"Model path: {args.model_input_path}")

    # Run predictions for each class
    all_submissions = []
    for class_name in classes:
        try:
            submissions = predict_class(class_name, args, model_kwargs)
            all_submissions.extend(submissions)
        except FileNotFoundError as e:
            print(f"  Error: {e}")
            continue

    # Write submission CSV
    submission_path = Path(args.submission_path)
    submission_path.parent.mkdir(parents=True, exist_ok=True)

    with open(submission_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        for image_id, label in sorted(all_submissions):
            writer.writerow([image_id, label])

    print(f"\n{'='*60}")
    print(f"Submission saved: {submission_path}")
    print(f"Total entries: {len(all_submissions)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
