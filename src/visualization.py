"""Visualization utilities for anomaly detection results.

Generates overlay images showing original images alongside ground truth masks
and predicted segmentation maps. Also creates summary images for top/bottom
predictions ranked by AP (anomalous) or activity (normal).
"""

from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np


def create_overlay(
    image: np.ndarray,
    pred_map: np.ndarray,
    gt_mask: np.ndarray | None = None,
    alpha: float = 0.4,
) -> np.ndarray:
    """Create a visualization with original, prediction overlay, and optional GT.

    Args:
        image: Original image, shape (H, W, 3), BGR format.
        pred_map: Predicted anomaly map, shape (H, W), values in [0, 1].
        gt_mask: Ground truth binary mask, shape (H, W). None for test images.
        alpha: Transparency for overlay blending.

    Returns:
        Combined visualization image as numpy array.
    """
    h, w = image.shape[:2]

    # Convert prediction to heatmap
    pred_heatmap = cv2.applyColorMap(
        (pred_map * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    pred_overlay = cv2.addWeighted(image, 1 - alpha, pred_heatmap, alpha, 0)

    if gt_mask is not None:
        # GT mask as green overlay
        gt_vis = image.copy()
        gt_colored = np.zeros_like(image)
        gt_colored[:, :, 1] = 255  # Green channel
        mask_bool = gt_mask > 0
        gt_vis[mask_bool] = cv2.addWeighted(
            image[mask_bool], 1 - alpha,
            gt_colored[mask_bool], alpha, 0
        )
        combined = np.hstack([image, gt_vis, pred_overlay])
    else:
        combined = np.hstack([image, pred_overlay])

    return combined


def save_prediction_visualization(
    image_path: str | Path,
    pred_map: np.ndarray,
    output_path: str | Path,
    gt_mask: np.ndarray | None = None,
) -> None:
    """Save a single prediction visualization to disk.

    Args:
        image_path: Path to the original image.
        pred_map: Predicted anomaly map, shape (H, W), values in [0, 1].
        output_path: Where to save the visualization.
        gt_mask: Optional ground truth mask.
    """
    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Cannot read image: {image_path}")

    # Resize prediction map to match image if needed
    h, w = image.shape[:2]
    if pred_map.shape != (h, w):
        pred_map = cv2.resize(pred_map, (w, h), interpolation=cv2.INTER_LINEAR)
    if gt_mask is not None and gt_mask.shape != (h, w):
        gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)

    overlay = create_overlay(image, pred_map, gt_mask)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), overlay)


def create_topk_summary(
    entries: list[dict],
    title: str,
    output_path: str | Path,
    k: int = 5,
    mode: str = "top",
) -> None:
    """Create a summary image showing top-k or bottom-k predictions.

    Args:
        entries: List of dicts with keys: 'image_path', 'pred_map', 'gt_mask'
            (optional), 'score' (AP or activity score).
        title: Title for the summary figure.
        output_path: Where to save the summary image.
        k: Number of predictions to show.
        mode: 'top' for highest scores, 'bottom' for lowest scores.
    """
    if not entries:
        return

    # Sort by score
    sorted_entries = sorted(entries, key=lambda x: x["score"], reverse=(mode == "top"))
    selected = sorted_entries[:k]

    n_cols = 3 if selected[0].get("gt_mask") is not None else 2
    fig, axes = plt.subplots(len(selected), n_cols, figsize=(4 * n_cols, 4 * len(selected)))

    if len(selected) == 1:
        axes = axes.reshape(1, -1)

    col_titles = ["Original", "Ground Truth", "Prediction"] if n_cols == 3 else ["Original", "Prediction"]

    for i, entry in enumerate(selected):
        image = cv2.imread(str(entry["image_path"]))
        if image is None:
            continue
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]

        pred_map = entry["pred_map"]
        if pred_map.shape != (h, w):
            pred_map = cv2.resize(pred_map, (w, h), interpolation=cv2.INTER_LINEAR)

        # Original image
        axes[i, 0].imshow(image_rgb)
        axes[i, 0].set_title(f"{Path(entry['image_path']).name}\nScore: {entry['score']:.4f}")
        axes[i, 0].axis("off")

        col_idx = 1
        if n_cols == 3:
            # Ground truth
            gt_mask = entry.get("gt_mask")
            if gt_mask is not None:
                if gt_mask.shape != (h, w):
                    gt_mask = cv2.resize(gt_mask, (w, h), interpolation=cv2.INTER_NEAREST)
                axes[i, col_idx].imshow(gt_mask, cmap="gray", vmin=0, vmax=1)
            else:
                axes[i, col_idx].imshow(np.zeros((h, w)), cmap="gray", vmin=0, vmax=1)
            axes[i, col_idx].set_title("Ground Truth")
            axes[i, col_idx].axis("off")
            col_idx += 1

        # Prediction heatmap
        axes[i, col_idx].imshow(image_rgb, alpha=0.6)
        axes[i, col_idx].imshow(pred_map, cmap="jet", alpha=0.4, vmin=0, vmax=1)
        axes[i, col_idx].set_title("Prediction")
        axes[i, col_idx].axis("off")

    # Set column titles
    for j, ct in enumerate(col_titles):
        axes[0, j].set_title(ct + "\n" + axes[0, j].get_title(), fontweight="bold")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


def save_validation_results(
    normal_entries: list[dict],
    anomalous_entries: list[dict],
    metrics: dict[str, float],
    output_dir: str | Path,
    class_name: str,
    algorithm: str,
) -> None:
    """Save complete validation results including visualizations and stats.

    Creates:
    - Individual prediction visualizations (normal/ and anomalous/ subdirs)
    - Top-5 and Bottom-5 summary images for both categories
    - Text file with model statistics

    Args:
        normal_entries: List of dicts for normal validation predictions.
        anomalous_entries: List of dicts for anomalous validation predictions.
        metrics: Dictionary of computed metrics.
        output_dir: Root output directory.
        class_name: Name of the class being evaluated.
        algorithm: Algorithm name used.
    """
    output_path = Path(output_dir) / class_name
    normal_dir = output_path / "normal"
    anomalous_dir = output_path / "anomalous"
    normal_dir.mkdir(parents=True, exist_ok=True)
    anomalous_dir.mkdir(parents=True, exist_ok=True)

    # Save individual visualizations for normal predictions
    for entry in normal_entries:
        fname = Path(entry["image_path"]).stem + "_vis.png"
        save_prediction_visualization(
            entry["image_path"],
            entry["pred_map"],
            normal_dir / fname,
            gt_mask=entry.get("gt_mask"),
        )

    # Save individual visualizations for anomalous predictions
    for entry in anomalous_entries:
        fname = Path(entry["image_path"]).stem + "_vis.png"
        save_prediction_visualization(
            entry["image_path"],
            entry["pred_map"],
            anomalous_dir / fname,
            gt_mask=entry.get("gt_mask"),
        )

    # Top-5 and Bottom-5 for anomalous (ranked by AP)
    if anomalous_entries:
        create_topk_summary(
            anomalous_entries,
            f"{class_name} - {algorithm} - Top-5 Anomalous (Highest AP)",
            output_path / "top5_anomalous.png",
            k=5,
            mode="top",
        )
        create_topk_summary(
            anomalous_entries,
            f"{class_name} - {algorithm} - Bottom-5 Anomalous (Lowest AP)",
            output_path / "bottom5_anomalous.png",
            k=5,
            mode="bottom",
        )

    # Top-5 and Bottom-5 for normal (ranked by activity)
    # For normal: low activity = good prediction, high activity = bad prediction
    if normal_entries:
        create_topk_summary(
            normal_entries,
            f"{class_name} - {algorithm} - Top-5 Normal (Least Active - Best)",
            output_path / "top5_normal.png",
            k=5,
            mode="bottom",  # Lowest activity = best prediction for normal
        )
        create_topk_summary(
            normal_entries,
            f"{class_name} - {algorithm} - Bottom-5 Normal (Most Active - Worst)",
            output_path / "bottom5_normal.png",
            k=5,
            mode="top",  # Highest activity = worst prediction for normal
        )

    # Save stats text file
    stats_path = output_path / "validation_stats.txt"
    with open(stats_path, "w") as f:
        f.write(f"Validation Results - {class_name}\n")
        f.write(f"Algorithm: {algorithm}\n")
        f.write("=" * 50 + "\n\n")
        f.write("Pixel-Level Metrics (Global):\n")
        for metric_name, value in metrics.items():
            f.write(f"  {metric_name}: {value:.6f}\n")
        f.write(f"\nNumber of normal validation samples: {len(normal_entries)}\n")
        f.write(f"Number of anomalous validation samples: {len(anomalous_entries)}\n")
