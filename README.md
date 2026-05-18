# Anomalib-Industrial: Multi-View Anomaly Detection Framework

A framework for training and evaluating pixel-level anomaly detection models on multi-view industrial image datasets using [Anomalib](https://github.com/open-edge-platform/anomalib).

## Overview

This project implements a pipeline for the Spacepresso industrial anomaly detection challenge. Each sample consists of **5 multi-view images**, and the goal is to produce pixel-level anomaly segmentation maps.

### Supported Algorithms

| Algorithm | Type | Epochs | Description |
|-----------|------|--------|-------------|
| **PatchCore** | Memory Bank | 1 | Coreset-based patch feature matching |
| **PaDIM** | Statistical | 1 | Gaussian modeling of patch embeddings |
| **FastFlow** | Normalizing Flow | 50 | 2D normalizing flow on feature maps |
| **DRAEM** | Reconstruction | 50 | Discriminatively trained reconstruction with synthetic anomalies |

## Installation

```bash
pip install -r requirements.txt
```

For GPU support (recommended):
```bash
pip install "anomalib[cu126]"  # For CUDA 12.6
```

## Dataset Structure

```
dataset/
├── class_XX/
│   ├── train/
│   │   ├── good/                    # Normal training images (5 views per sample)
│   │   ├── anomaly_01/             # Anomalous sample type 1 (1 sample, 5 views)
│   │   └── anomaly_YY/             # Anomalous sample type Y
│   ├── ground_truth_train/
│   │   ├── anomaly_01/             # Masks for anomaly type 1
│   │   └── anomaly_YY/             # Masks for anomaly type Y
│   └── test/                        # Unlabeled test images (5 views per sample)
```

Image naming: `img_<hash>_view<1-5>.png`

## Pipeline Flow

```
1. Data Preparation
   ├── Discover all classes in dataset
   ├── Group images by sample ID (hash) to keep all 5 views together
   └── Split: 80% normal → train, 20% normal + all anomalous → validation

2. Training (per class)
   ├── Prepare directory with symlinks (Anomalib-compatible structure)
   ├── Create Anomalib Folder datamodule
   ├── Train model using Anomalib Engine
   └── Save checkpoint

3. Validation
   ├── Run inference on validation images
   ├── Compute pixel-level metrics (AP, AUPRC, AUROC)
   ├── Generate overlay visualizations
   └── Create top-5 / bottom-5 summary images

4. Prediction
   ├── Load trained checkpoint
   ├── Run inference on test images
   ├── Resize anomaly maps to original resolution
   ├── Encode with q8rle format
   └── Generate submission CSV
```

## Usage

### Training

```bash
# Train PatchCore on all classes
python train.py \
    --algorithm patchcore \
    --data-dir ./dataset \
    --img-size 224 \
    --model-output-path ./checkpoints \
    --output-dir ./results/validation

# Train PaDIM on specific classes
python train.py \
    --algorithm padim \
    --data-dir ./dataset \
    --img-size 224 \
    --classes class_01 class_02 \
    --model-output-path ./checkpoints \
    --output-dir ./results/validation

# Train FastFlow with custom parameters
python train.py \
    --algorithm fastflow \
    --data-dir ./dataset \
    --img-size 224 \
    --backbone resnet18 \
    --flow-steps 8 \
    --max-epochs 50 \
    --model-output-path ./checkpoints \
    --output-dir ./results/validation

# Train DRAEM with external anomaly textures
python train.py \
    --algorithm draem \
    --data-dir ./dataset \
    --img-size 224 \
    --anomaly-source-path ./textures \
    --max-epochs 50 \
    --model-output-path ./checkpoints \
    --output-dir ./results/validation

# PatchCore with custom coreset parameters
python train.py \
    --algorithm patchcore \
    --data-dir ./dataset \
    --img-size 224 \
    --backbone wide_resnet50_2 \
    --coreset-sampling-ratio 0.1 \
    --num-neighbors 9 \
    --model-output-path ./checkpoints \
    --output-dir ./results/validation
```

### Prediction

```bash
# Generate submission for all classes
python predict.py \
    --algorithm patchcore \
    --data-dir ./dataset \
    --img-size 224 \
    --model-input-path ./checkpoints \
    --submission-path ./submission.csv

# With visualizations
python predict.py \
    --algorithm patchcore \
    --data-dir ./dataset \
    --img-size 224 \
    --model-input-path ./checkpoints \
    --submission-path ./submission.csv \
    --output-dir ./results/test_predictions

# Predict specific classes
python predict.py \
    --algorithm padim \
    --data-dir ./dataset \
    --img-size 224 \
    --model-input-path ./checkpoints \
    --classes class_01 class_03 \
    --submission-path ./submission.csv
```

## Command-Line Arguments

### Training Script (`train.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algorithm` | str | *required* | Algorithm: patchcore, padim, fastflow, draem |
| `--data-dir` | str | *required* | Root directory of dataset |
| `--img-size` | int | 224 | Image size (square, no center crop) |
| `--model-output-path` | str | None | Directory to save model checkpoints |
| `--output-dir` | str | None | Directory for validation visualizations |
| `--classes` | str[] | all | Specific classes to train |
| `--val-ratio` | float | 0.2 | Fraction of normal samples for validation |
| `--seed` | int | 42 | Random seed |
| `--max-epochs` | int | auto | Training epochs (auto: 1 for PatchCore/PaDIM, 50 for others) |
| `--batch-size` | int | 32 | Training batch size |
| `--backbone` | str | algo-specific | Feature extractor backbone |
| `--coreset-sampling-ratio` | float | 0.1 | PatchCore: fraction of features in coreset |
| `--num-neighbors` | int | 9 | PatchCore: KNN neighbors for scoring |
| `--n-features` | int | None | PaDIM: dimensionality reduction target |
| `--flow-steps` | int | 8 | FastFlow: normalizing flow steps |
| `--anomaly-source-path` | str | None | DRAEM: external texture dataset path |

### Prediction Script (`predict.py`)

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--algorithm` | str | *required* | Algorithm (must match training) |
| `--data-dir` | str | *required* | Root directory of dataset |
| `--model-input-path` | str | *required* | Directory with trained checkpoints |
| `--img-size` | int | 224 | Image size (must match training) |
| `--output-dir` | str | None | Directory for prediction visualizations |
| `--submission-path` | str | submission.csv | Output CSV path |
| `--classes` | str[] | all | Specific classes to predict |
| `--batch-size` | int | 32 | Inference batch size |

Algorithm-specific arguments (must match training configuration) are also available.

## Output Structure

### Validation Output (`--output-dir`)

```
output_dir/
├── class_XX/
│   ├── normal/                      # Normal sample visualizations
│   │   └── img_xxx_viewN_vis.png   # [Original | GT (zeros) | Prediction]
│   ├── anomalous/                   # Anomalous sample visualizations
│   │   └── img_xxx_viewN_vis.png   # [Original | GT mask | Prediction]
│   ├── top5_anomalous.png          # Best 5 anomalous predictions (highest AP)
│   ├── bottom5_anomalous.png       # Worst 5 anomalous predictions (lowest AP)
│   ├── top5_normal.png             # Best 5 normal predictions (least active)
│   ├── bottom5_normal.png          # Worst 5 normal predictions (most active)
│   └── validation_stats.txt        # Metrics summary
```

### Prediction Output

```
output_dir/
├── class_XX/
│   └── test/
│       └── img_xxx_viewN_vis.png   # [Original | Prediction heatmap]
```

### Submission Format

```csv
ID,Label
img_000001_view1,q8rle 224 224 0 50176
img_000001_view2,q8rle 224 224 0 120 255 10 0 50046
```

## Validation Split Strategy

To prevent data leakage from multi-view samples:

1. Images are grouped by sample ID (the hash in `img_<hash>_viewN.png`)
2. All 5 views of a sample are assigned to the same split
3. 80% of normal samples (all views) go to training
4. 20% of normal samples + ALL anomalous samples go to validation
5. Split is deterministic (controlled by `--seed`)

## Metrics

- **Average Precision (AP)**: Primary competition metric. Measures ranking quality of anomalous pixels.
- **AUPRC**: Area under the precision-recall curve (equivalent to AP for binary).
- **AUROC**: Area under the ROC curve. Measures discrimination ability.

All metrics are computed at the **pixel level** by pooling predictions across all validation images.

## Notes

- All training and inference is designed to be reproducible on Google Colab
- Models are saved as Lightning checkpoints (`.ckpt`) and can be loaded for inference
- The framework handles per-class training automatically
- Anomaly maps are normalized to [0, 1] before q8rle encoding
- For submission, maps are resized back to original image resolution
