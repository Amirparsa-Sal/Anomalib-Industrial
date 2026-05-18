# Mission: Spacepresso
I'm participating in an industrial anomaly detection challenge. Below, you can find the information about the challenge and at the end I'll tell you what I want to implement exactly.

## Challenge Theme

Milan Space Center, 14 days before the rocket launch. The Italian Space Agency is ready to go, but the logistics department has made a catastrophic mess.

We’ve just received our final shipment of supplies for the mission to Mars:

* Mechanical gears
* Electronic components
* Coffee beans
* Bronte pistachios for the onboard gelato machine

The problem? The manufacturing facility sent us a batch riddled with defects. If even one bad coffee bean hits the grinder, the crew will revolt.

The launch is in 14 days.

Your task is to build an anomaly detection system to filter out every single faulty piece before we blast off.

Identify the broken components, save the gelato, and ensure the espresso is impeccable.

---

# Project Description

In this assignment, you will be provided with a collection of multi-view image sets across several object categories.

Each sample consists of **five images** captured from different perspectives.

## Objective

Perform **pixel-level anomaly detection**, accurately identifying all anomalous pixels within each provided image.

---

# Leaderboard Metrics

The models will be evaluated using **Pixel-Level Average Precision (AP)**.

Average Precision is computed as:

AP = \sum_{n=1}^{N} (R_n - R_{n-1}) \cdot P_n

Where:

* ( P_n ) = precision at the ( n )-th threshold
* ( R_n ) = recall at the ( n )-th threshold

This metric measures how well the model ranks anomalous pixels above normal pixels across all possible thresholds.

## Additional Notes

* Pretrained models are allowed.
* All training and inference must be fully reproducible on Google Colab.
* Refer to the Rules page for more details.

---

# Submission File Format

You must submit a `.csv` file with the following format:

```csv
ID,Label
img_000001_view1,q8rle 224 224 0 50176
img_000001_view2,q8rle 224 224 0 120 255 10 0 50046
```

## Fields

* **ID**: Image identifier
* **Label**: Anomaly score map encoded as a `q8rle` string

To reduce upload size, the CSV may also be submitted as a `.zip` file.

---

# q8rle Encoding

The project uses a quantized 8-bit run-length encoding format called **q8rle**.

## Encoding Procedure

A mask is first represented as a 2D array of floating-point anomaly scores in the range `[0, 1]`.

These values are then:

1. Quantized to integers in `[0, 255]`
2. Flattened column-wise
3. Run-length encoded as `(value, length)` pairs

## Final Format

```text
q8rle <height> <width> <value_1> <runlen_1> <value_2> <runlen_2> ...
```

## Example

```text
q8rle 224 224 0 50176
```

This means that all pixels have score `0`.

---

# Minimal q8rle Python Implementation

```python
import numpy as np

def float_matrix_to_q8rle(x: np.ndarray) -> str:
    q = np.clip(np.rint(np.asarray(x, dtype=np.float32) * 255), 0, 255).astype(np.uint8)
    h, w = q.shape
    flat = q.T.reshape(-1)  # column-wise flattening

    if flat.size == 0:
        return f"q8rle {h} {w}"

    cuts = np.flatnonzero(flat[1:] != flat[:-1]) + 1
    starts = np.r_[0, cuts]
    ends = np.r_[cuts, flat.size]

    parts = ["q8rle", str(h), str(w)]

    for v, n in zip(flat[starts], ends - starts):
        parts += [str(int(v)), str(int(n))]

    return " ".join(parts)

def q8rle_to_float_matrix(s: str) -> np.ndarray:
    t = s.split()

    h, w = int(t[1]), int(t[2])

    vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, t[4::2])), dtype=np.int64)

    flat = np.repeat(vals, lens).reshape(w, h).T

    return flat.astype(np.float32) / 255.0
```

---

# Dataset Description

Dataset root folder:

```text
./dataset
```

## Folder Structure

### Training Data

```text
class_XX/train/good/
```

Contains clean training images. Each sample has 5 different views:
sample_id_view1.png
sample_id_view2.png
sample_id_view3.png
sample_id_view4.png
sample_id_view5.png
 
### Labeled Anomalous Samples

```text
class_XX/train/anomaly_YY/
```

Each anomaly_YY folder is a specific type of anomaly for class_XX. As above, it contains only one sample with 5 different views.


### Ground Truth Masks

```text
class_XX/ground_truth_train/anomaly_YY/
```

Contains masks for the labeled anomaly examples. Again only one sample per defect type and with 5 different views.

### Test Set

```text
class_XX/test/
```

Contains unlabeled leaderboard images, including both clean and anomalous cases. Here we also have 5 different views.

# What you should implement

I want you to implement a general framework to train different anomaly detection models on this data using Anomalib framework in python. 

Here is the documentation: https://anomalib.readthedocs.io/en/lib-v2.4.2/markdown/get_started/anomalib.html

What steps you should implement? 
For each class:
1) Creating a validation set using training data. The validation set should include 20% of normal samples (all different views of sample should be moved to prevent data leakage) and all anomalous samples.
2) Training the anomaly detection model on the training set
3) Validating the results for the validation set, using the Average Precision, AUPRC, and AUROC metrics.
4) Predicting the outputs for the test set and creating the submission file as described above. If the trained models can be saved and later loaded, the prediction script could be a different script than the training one.

I'm not sure but different algorithms might need different directory structure and the code should be able to handle this. Also, here is also another documentation about using custom datasets with this framework:
https://anomalib.readthedocs.io/en/stable/markdown/guides/how_to/data/custom_data.html

For now, I want you to only implement the following algorithms:
- PatchCore
- Padim
- Fast-flow
- Dream

if any of these algorithms require anomalous data in the training data, ask me before implementation.

A list of command-line arguments for the training script:
- algorithm: The type of algorithm to use
- img-size: the image size (no need to center crop because the images are square)
- data-dir: root directory of dataset
- model-output-path: (if the save and load functionality is available)
- algorithm specific parameters: like core-set p in patch core
- output-dir: the path to directory to store the segmentation results (if specified). For each validation image I want the original image along with its ground truth and the predicted segmentation mask. The folder for anomalous and normal samples should be separated. Also store a text file in the directory containing the model stats on validation set. Also create 2 separate images to show the top-5 and bottom-5 predictions based on AP (both for anomalous and normal data). 

A list of command-line arguments for the prediction script:
- algorithm: The type of algorithm to use
- img-size: the image size (no need to center crop because the images are square)
- data-dir: root directory of dataset
- model-input-path: (if the save and load functionality is available)
- output-dir: the path to directory to store the segmentation results (if specified). For each test image I want the original image along with its ground truth and the predicted segmentation mask. The folder for anomalous and normal samples should be separated.


At the end write a readme.md file to explain the flow of the system and also the example usage including all parameters.