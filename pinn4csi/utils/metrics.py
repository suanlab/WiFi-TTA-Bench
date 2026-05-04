"""Evaluation metrics for model assessment."""

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
