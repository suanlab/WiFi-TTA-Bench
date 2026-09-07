"""Evaluation metrics for model assessment."""

import math

import torch
from torch import Tensor


def accuracy(predictions: Tensor, targets: Tensor) -> float:
    """Compute classification accuracy.

    Args:
        predictions: Model predictions. Shape: (batch,) or (batch, num_classes).
                     If 2D, argmax is taken along dim 1.
        targets: Ground truth labels. Shape: (batch,).

    Returns:
        Accuracy as a float in [0, 1].
    """
    if predictions.dim() == 2:
        predictions = torch.argmax(predictions, dim=1)
    correct = (predictions == targets).sum().item()
    total = targets.shape[0]
    return correct / total


def nmse(predictions: Tensor, targets: Tensor) -> float:
    """Compute Normalized Mean Squared Error (NMSE).

    Args:
        predictions: Model predictions. Shape: (batch, ...).
        targets: Ground truth values. Shape: (batch, ...).

    Returns:
        NMSE as a float. Lower is better.
    """
    mse = torch.mean((predictions - targets) ** 2).item()
    target_power = torch.mean(targets**2).item()
    if target_power == 0:
        return 0.0
    return mse / target_power


def f1_score(predictions: Tensor, targets: Tensor) -> float:
    """Compute F1 score for binary classification.

    Args:
        predictions: Model predictions. Shape: (batch,).
                     Values > 0.5 are treated as class 1.
        targets: Ground truth labels. Shape: (batch,). Values in {0, 1}.

    Returns:
        F1 score as a float in [0, 1].
    """
    pred_binary = (predictions > 0.5).long()
    tp = ((pred_binary == 1) & (targets == 1)).sum().item()
    fp = ((pred_binary == 1) & (targets == 0)).sum().item()
    fn = ((pred_binary == 0) & (targets == 1)).sum().item()

    if tp + fp == 0 or tp + fn == 0:
        return 0.0

    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    if precision + recall == 0:
        return 0.0
    return 2 * (precision * recall) / (precision + recall)


def cohens_d(group_a: Tensor, group_b: Tensor) -> float:
    """Compute Cohen's d effect size between two groups.

    Args:
        group_a: First group values. Shape: (n,).
        group_b: Second group values. Shape: (m,).

    Returns:
        Cohen's d (positive means group_a > group_b).
    """
    a = group_a.float()
    b = group_b.float()
    n_a, n_b = a.shape[0], b.shape[0]
    if n_a < 2 or n_b < 2:
        return 0.0
    mean_a = a.mean()
    mean_b = b.mean()
    var_a = a.var(unbiased=True)
    var_b = b.var(unbiased=True)
    pooled_std = torch.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))
    if pooled_std.item() < 1e-12:
        return 0.0
    return float(((mean_a - mean_b) / pooled_std).item())


def bootstrap_ci(
    values: Tensor,
    confidence: float = 0.95,
    num_samples: int = 10000,
) -> tuple[float, float, float]:
    """Compute bootstrap confidence interval for the mean.

    Args:
        values: 1D tensor of observed values. Shape: (n,).
        confidence: Confidence level (e.g. 0.95).
        num_samples: Number of bootstrap resamples.

    Returns:
        Tuple of (point_estimate, ci_lower, ci_upper).
    """
    v = values.float()
    n = v.shape[0]
    point_estimate = float(v.mean().item())
    if n < 2:
        return (point_estimate, point_estimate, point_estimate)
    indices = torch.randint(0, n, (num_samples, n), device=v.device)
    resampled_means = v[indices].mean(dim=1)
    alpha = 1.0 - confidence
    lower = float(torch.quantile(resampled_means, alpha / 2).item())
    upper = float(torch.quantile(resampled_means, 1.0 - alpha / 2).item())
    return (point_estimate, lower, upper)


def bonferroni_corrected_alpha(
    alpha: float = 0.05,
    num_comparisons: int = 1,
) -> float:
    """Compute Bonferroni-corrected significance threshold.

    Args:
        alpha: Family-wise error rate.
        num_comparisons: Number of simultaneous hypothesis tests.

    Returns:
        Corrected per-comparison alpha.
    """
    if num_comparisons < 1:
        return alpha
    return alpha / num_comparisons


def paired_cohens_d(differences: Tensor) -> float:
    """Compute Cohen's d for paired differences (one-sample effect size).

    Args:
        differences: 1D tensor of paired differences. Shape: (n,).

    Returns:
        Cohen's d_z (mean difference / std of differences).
    """
    d = differences.float()
    n = d.shape[0]
    if n < 2:
        return 0.0
    std = d.std(unbiased=True)
    if std.item() < 1e-12:
        return 0.0
    return float((d.mean() / std).item())


def prediction_diversity(logits: Tensor) -> float:
    """Normalized entropy of the batch-mean predictive distribution.

    Collapse indicator for test-time adaptation: 1.0 means every class is used
    equally, 0.0 means every sample is assigned the same class. This is
    deliberately *not* the mean per-sample entropy that entropy-minimizing
    objectives drive down; it measures whether marginal class usage is
    degenerating, which per-sample confidence cannot reveal.

    Args:
        logits: Unnormalized class scores. Shape: (batch, num_classes).

    Returns:
        Diversity in [0, 1]. 0.0 for an empty batch or a single class.

    Raises:
        ValueError: If logits is not 2D.
    """
    if logits.dim() != 2:
        raise ValueError(
            f"logits must be 2D (batch, num_classes), got {tuple(logits.shape)}"
        )
    num_samples, num_classes = logits.shape
    if num_samples == 0 or num_classes < 2:
        return 0.0
    mean_probs = torch.softmax(logits.float(), dim=1).mean(dim=0)
    safe_probs = mean_probs.clamp_min(1e-12)
    entropy = -(safe_probs * safe_probs.log()).sum()
    return float((entropy / math.log(num_classes)).item())


def _flatten_features(features: Tensor) -> Tensor:
    """Flatten per-sample features to 2D float. Shape: (batch, flattened)."""
    if features.dim() == 0:
        raise ValueError("features must have at least 1 dimension, got a scalar")
    if features.dim() == 1:
        return features.reshape(-1, 1).float()
    # Infer the trailing size explicitly: reshape(0, -1) is ambiguous when the
    # tensor is empty, which happens for degenerate splits.
    trailing = features.shape[1:].numel()
    return features.reshape(features.shape[0], trailing).float()


def feature_scale(features: Tensor) -> float:
    """Compute the typical magnitude of a feature vector.

    Used to turn absolute nearest-neighbour distances into a scale-free
    duplicate criterion.

    Args:
        features: Sample features. Shape: (batch, ...).

    Returns:
        Root-mean-square L2 norm across samples. 0.0 if there are no samples.
    """
    flat = _flatten_features(features)
    if flat.shape[0] == 0:
        return 0.0
    return float(torch.sqrt((flat**2).sum(dim=1).mean()).item())


def nearest_neighbour_distances(
    query: Tensor,
    reference: Tensor,
    chunk_size: int = 512,
) -> Tensor:
    """Compute each query sample's distance to its closest reference sample.

    Args:
        query: Query features. Shape: (num_query, ...).
        reference: Reference features. Shape: (num_reference, ...).
        chunk_size: Number of query rows compared per block, to bound memory.

    Returns:
        Distance to the nearest reference sample. Shape: (num_query,).
        Empty if either input has no samples.

    Raises:
        ValueError: If chunk_size < 1 or the flattened dimensions disagree.
    """
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}")
    q = _flatten_features(query)
    r = _flatten_features(reference)
    if q.shape[0] == 0 or r.shape[0] == 0:
        return torch.empty(0, dtype=q.dtype)
    if q.shape[1] != r.shape[1]:
        raise ValueError(
            f"Feature dimensions differ: query {q.shape[1]} vs reference {r.shape[1]}"
        )
    distances = torch.empty(q.shape[0], dtype=q.dtype)
    for start in range(0, q.shape[0], chunk_size):
        block = q[start : start + chunk_size]
        block_distances = torch.cdist(block, r).min(dim=1).values
        distances[start : start + block.shape[0]] = block_distances
    return distances


def duplicate_leakage_rate(
    train_features: Tensor,
    test_features: Tensor,
    relative_threshold: float = 0.01,
    chunk_size: int = 512,
) -> float:
    """Fraction of test samples that have a near-duplicate in the train split.

    Sliding-window CSI pipelines can place overlapping windows on both sides of
    a random split, which inflates reported accuracy. A test sample counts as
    leaked when its nearest training neighbour sits within
    ``relative_threshold`` of the typical feature magnitude.

    Args:
        train_features: Training-split features. Shape: (num_train, ...).
        test_features: Test-split features. Shape: (num_test, ...).
        relative_threshold: Duplicate cutoff as a fraction of feature scale.
        chunk_size: Number of test rows compared per block.

    Returns:
        Leakage rate in [0, 1]. 0.0 when either split is empty.

    Raises:
        ValueError: If relative_threshold is negative.
    """
    if relative_threshold < 0.0:
        raise ValueError(f"relative_threshold must be >= 0, got {relative_threshold}")
    distances = nearest_neighbour_distances(
        test_features, train_features, chunk_size=chunk_size
    )
    if distances.numel() == 0:
        return 0.0
    scale = feature_scale(test_features)
    cutoff = relative_threshold * scale
    return float((distances <= cutoff).float().mean().item())


def environment_leakage_rate(
    train_environments: Tensor,
    test_environments: Tensor,
) -> float:
    """Fraction of test samples whose environment also appears in training.

    Must be 0.0 under a leave-one-environment-out protocol; any larger value
    means the split does not isolate environments, so cross-environment gains
    computed from it are optimistic.

    Args:
        train_environments: Training environment IDs. Shape: (num_train,).
        test_environments: Test environment IDs. Shape: (num_test,).

    Returns:
        Leakage rate in [0, 1]. 0.0 when either split is empty.
    """
    if test_environments.numel() == 0 or train_environments.numel() == 0:
        return 0.0
    seen = torch.unique(train_environments)
    overlap = torch.isin(test_environments, seen)
    return float(overlap.float().mean().item())


def split_leakage_report(
    train_features: Tensor,
    test_features: Tensor,
    train_environments: Tensor | None = None,
    test_environments: Tensor | None = None,
    relative_threshold: float = 0.01,
    chunk_size: int = 512,
) -> dict[str, float]:
    """Audit one train/test split for duplicate and environment leakage.

    Args:
        train_features: Training-split features. Shape: (num_train, ...).
        test_features: Test-split features. Shape: (num_test, ...).
        train_environments: Optional training environment IDs. Shape: (num_train,).
        test_environments: Optional test environment IDs. Shape: (num_test,).
        relative_threshold: Duplicate cutoff as a fraction of feature scale.
        chunk_size: Number of test rows compared per block.

    Returns:
        Mapping with ``duplicate_leakage_rate``, ``min_nn_distance``,
        ``median_nn_distance``, and ``feature_scale``. Includes
        ``environment_leakage_rate`` when both environment tensors are given.
    """
    distances = nearest_neighbour_distances(
        test_features, train_features, chunk_size=chunk_size
    )
    report: dict[str, float] = {
        "duplicate_leakage_rate": duplicate_leakage_rate(
            train_features,
            test_features,
            relative_threshold=relative_threshold,
            chunk_size=chunk_size,
        ),
        "feature_scale": feature_scale(test_features),
        "min_nn_distance": (
            float(distances.min().item()) if distances.numel() > 0 else 0.0
        ),
        "median_nn_distance": (
            float(distances.median().item()) if distances.numel() > 0 else 0.0
        ),
    }
    if train_environments is not None and test_environments is not None:
        report["environment_leakage_rate"] = environment_leakage_rate(
            train_environments, test_environments
        )
    return report
