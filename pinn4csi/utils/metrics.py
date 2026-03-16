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
