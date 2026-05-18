"""q8rle encoding/decoding for anomaly score maps.

The q8rle format quantizes float anomaly maps to 8-bit integers,
flattens them column-wise, and run-length encodes the result.
"""

import numpy as np


def float_matrix_to_q8rle(x: np.ndarray) -> str:
    """Convert a 2D float anomaly map [0,1] to q8rle string.

    Args:
        x: 2D array of anomaly scores in range [0, 1].

    Returns:
        q8rle encoded string.
    """
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
    """Decode a q8rle string back to a 2D float anomaly map.

    Args:
        s: q8rle encoded string.

    Returns:
        2D array of anomaly scores in range [0, 1].
    """
    t = s.split()
    h, w = int(t[1]), int(t[2])

    if len(t) <= 3:
        return np.zeros((h, w), dtype=np.float32)

    vals = np.array(list(map(int, t[3::2])), dtype=np.uint8)
    lens = np.array(list(map(int, t[4::2])), dtype=np.int64)

    flat = np.repeat(vals, lens).reshape(w, h).T
    return flat.astype(np.float32) / 255.0
