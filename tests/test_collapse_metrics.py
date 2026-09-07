"""Tests for the prediction-diversity collapse indicator."""

import pytest
import torch

from pinn4csi.utils.metrics import prediction_diversity


def test_uniform_predictions_are_maximally_diverse() -> None:
    logits = torch.zeros(16, 4)
    assert prediction_diversity(logits) == pytest.approx(1.0, abs=1e-5)


def test_total_collapse_to_one_class_is_zero() -> None:
    logits = torch.full((16, 4), -20.0)
    logits[:, 2] = 20.0
    assert prediction_diversity(logits) == pytest.approx(0.0, abs=1e-4)


def test_confident_but_balanced_predictions_stay_diverse() -> None:
    """Confidence alone must not look like collapse.

    Every sample is near-one-hot, but the classes are used evenly, so marginal
    diversity stays maximal even though mean per-sample entropy is ~0.
    """
    num_classes = 4
    logits = torch.full((num_classes * 8, num_classes), -20.0)
    for i in range(logits.shape[0]):
        logits[i, i % num_classes] = 20.0
    assert prediction_diversity(logits) == pytest.approx(1.0, abs=1e-4)


def test_diversity_separates_collapse_from_confidence() -> None:
    """The collapsed batch scores far lower than the equally-confident one."""
    num_classes = 4
    confident_balanced = torch.full((32, num_classes), -20.0)
    for i in range(32):
        confident_balanced[i, i % num_classes] = 20.0
    confident_collapsed = torch.full((32, num_classes), -20.0)
    confident_collapsed[:, 0] = 20.0
    assert prediction_diversity(confident_balanced) > 0.99
    assert prediction_diversity(confident_collapsed) < 0.01


def test_partial_collapse_is_between_extremes() -> None:
    logits = torch.full((32, 4), -20.0)
    logits[:16, 0] = 20.0
    logits[16:, 1] = 20.0
    diversity = prediction_diversity(logits)
    assert 0.4 < diversity < 0.6


def test_degenerate_inputs_return_zero() -> None:
    assert prediction_diversity(torch.empty(0, 4)) == 0.0
    assert prediction_diversity(torch.randn(8, 1)) == 0.0


def test_non_2d_logits_rejected() -> None:
    with pytest.raises(ValueError, match="2D"):
        prediction_diversity(torch.randn(8))
    with pytest.raises(ValueError, match="2D"):
        prediction_diversity(torch.randn(2, 3, 4))
