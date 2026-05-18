"""Model factory and algorithm-specific configurations.

Provides a unified interface to create and configure anomaly detection models
from the Anomalib framework: PatchCore, PaDIM, FastFlow, and DRAEM.
"""

from anomalib.models import Draem, Fastflow, Padim, Patchcore


SUPPORTED_ALGORITHMS = ["patchcore", "padim", "fastflow", "draem"]


def get_model(algorithm: str, **kwargs):
    """Create and return an Anomalib model instance.

    Args:
        algorithm: Algorithm name (patchcore, padim, fastflow, draem).
        **kwargs: Algorithm-specific parameters.

    Returns:
        Instantiated Anomalib model.

    Raises:
        ValueError: If algorithm is not supported.
    """
    algorithm = algorithm.lower()

    if algorithm == "patchcore":
        return _create_patchcore(**kwargs)
    elif algorithm == "padim":
        return _create_padim(**kwargs)
    elif algorithm == "fastflow":
        return _create_fastflow(**kwargs)
    elif algorithm == "draem":
        return _create_draem(**kwargs)
    else:
        raise ValueError(
            f"Unsupported algorithm: '{algorithm}'. "
            f"Supported: {SUPPORTED_ALGORITHMS}"
        )


def _create_patchcore(
    backbone: str = "wide_resnet50_2",
    layers: list[str] | None = None,
    coreset_sampling_ratio: float = 0.1,
    num_neighbors: int = 9,
    **kwargs,
) -> Patchcore:
    """Create a PatchCore model.

    Args:
        backbone: Feature extractor backbone network.
        layers: Which layers to extract features from.
        coreset_sampling_ratio: Fraction of patch features to keep in coreset.
        num_neighbors: Number of nearest neighbors for scoring.
    """
    params = {
        "backbone": backbone,
        "coreset_sampling_ratio": coreset_sampling_ratio,
        "num_neighbors": num_neighbors,
    }
    if layers is not None:
        params["layers"] = layers

    return Patchcore(**params)


def _create_padim(
    backbone: str = "resnet18",
    layers: list[str] | None = None,
    n_features: int | None = None,
    **kwargs,
) -> Padim:
    """Create a PaDIM model.

    Args:
        backbone: Feature extractor backbone network.
        layers: Which layers to extract features from.
        n_features: Number of features to retain (dimensionality reduction).
    """
    params = {"backbone": backbone}
    if layers is not None:
        params["layers"] = layers
    if n_features is not None:
        params["n_features"] = n_features

    return Padim(**params)


def _create_fastflow(
    backbone: str = "resnet18",
    flow_steps: int = 8,
    **kwargs,
) -> Fastflow:
    """Create a FastFlow model.

    Args:
        backbone: Feature extractor backbone network.
        flow_steps: Number of normalizing flow steps.
    """
    return Fastflow(
        backbone=backbone,
        flow_steps=flow_steps,
    )


def _create_draem(
    anomaly_source_path: str | None = None,
    **kwargs,
) -> Draem:
    """Create a DRAEM model.

    DRAEM generates synthetic anomalies using Perlin noise during training.
    Optionally, an external texture dataset can be used as anomaly source.

    Args:
        anomaly_source_path: Path to external texture images for synthetic
            anomaly generation. If None, uses Perlin noise only.
    """
    params = {}
    if anomaly_source_path is not None:
        params["anomaly_source_path"] = anomaly_source_path

    return Draem(**params)


def get_default_max_epochs(algorithm: str) -> int:
    """Get default training epochs for each algorithm.

    PatchCore and PaDIM are memory-bank methods (1 epoch),
    while FastFlow and DRAEM are trainable networks.

    Args:
        algorithm: Algorithm name.

    Returns:
        Default number of training epochs.
    """
    defaults = {
        "patchcore": 1,
        "padim": 1,
        "fastflow": 50,
        "draem": 50,
    }
    return defaults.get(algorithm.lower(), 1)
