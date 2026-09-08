"""Tests for the expected-calibration-error metric."""

import pytest
import torch

from pinn4csi.utils.metrics import expected_calibration_error


def test_confident_and_always_correct_is_well_calibrated() -> None:
    logits = torch.full((32, 4), -20.0)
    targets = torch.arange(32) % 4
    logits[torch.arange(32), targets] = 20.0
    assert expected_calibration_error(logits, targets) == pytest.approx(0.0, abs=1e-4)


def test_confident_and_always_wrong_is_maximally_miscalibrated() -> None:
    logits = torch.full((32, 4), -20.0)
    targets = torch.arange(32) % 4
    logits[torch.arange(32), (targets + 1) % 4] = 20.0
    assert expected_calibration_error(logits, targets) == pytest.approx(1.0, abs=1e-4)


def test_uniform_predictions_gap_equals_confidence_minus_accuracy() -> None:
    """Uniform logits over C classes give confidence 1/C and accuracy 1/C."""
    logits = torch.zeros(40, 4)
    targets = torch.arange(40) % 4
    assert expected_calibration_error(logits, targets) == pytest.approx(0.0, abs=1e-4)


def test_over_and_under_confident_regions_do_not_cancel() -> None:
    """A raw confidence-minus-accuracy gap would cancel here; ECE must not."""
    logits = torch.full((4, 2), 0.0)
    # Two near-certain correct samples and two near-certain wrong ones.
    logits[0] = torch.tensor([8.0, -8.0])
    logits[1] = torch.tensor([8.0, -8.0])
    logits[2] = torch.tensor([8.0, -8.0])
    logits[3] = torch.tensor([8.0, -8.0])
    targets = torch.tensor([0, 0, 1, 1])
    # Confidence ~1.0 everywhere, accuracy 0.5 => ECE ~0.5, not 0.
    assert expected_calibration_error(logits, targets) == pytest.approx(0.5, abs=1e-3)


def test_empty_batch_returns_zero() -> None:
    assert expected_calibration_error(torch.zeros(0, 3), torch.zeros(0).long()) == 0.0


def test_rejects_non_2d_logits() -> None:
    with pytest.raises(ValueError, match="must be 2D"):
        expected_calibration_error(torch.zeros(8), torch.zeros(8).long())


def test_rejects_mismatched_targets() -> None:
    with pytest.raises(ValueError, match="must be 1D of length"):
        expected_calibration_error(torch.zeros(8, 3), torch.zeros(5).long())


def test_rejects_non_positive_bin_count() -> None:
    with pytest.raises(ValueError, match="num_bins must be"):
        expected_calibration_error(torch.zeros(8, 3), torch.zeros(8).long(), num_bins=0)
