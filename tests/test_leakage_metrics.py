"""Tests for split-leakage audit metrics."""

import pytest
import torch

from pinn4csi.utils.metrics import (
    duplicate_leakage_rate,
    environment_leakage_rate,
    feature_scale,
    nearest_neighbour_distances,
    split_leakage_report,
)


def test_nearest_neighbour_distances_exact_duplicates_are_zero() -> None:
    reference = torch.randn(8, 4)
    query = reference[:3].clone()
    distances = nearest_neighbour_distances(query, reference)
    assert distances.shape == (3,)
    assert torch.allclose(distances, torch.zeros(3), atol=1e-5)


def test_nearest_neighbour_distances_matches_bruteforce() -> None:
    query = torch.randn(5, 3)
    reference = torch.randn(7, 3)
    expected = torch.cdist(query, reference).min(dim=1).values
    got = nearest_neighbour_distances(query, reference, chunk_size=2)
    assert torch.allclose(got, expected, atol=1e-5)


def test_nearest_neighbour_distances_flattens_multidim_features() -> None:
    reference = torch.randn(6, 4, 2)
    query = reference[:2].clone()
    distances = nearest_neighbour_distances(query, reference)
    assert distances.shape == (2,)
    assert torch.allclose(distances, torch.zeros(2), atol=1e-5)


def test_nearest_neighbour_distances_empty_inputs() -> None:
    assert (
        nearest_neighbour_distances(torch.empty(0, 3), torch.randn(4, 3)).numel() == 0
    )
    assert (
        nearest_neighbour_distances(torch.randn(4, 3), torch.empty(0, 3)).numel() == 0
    )


def test_nearest_neighbour_distances_rejects_bad_arguments() -> None:
    with pytest.raises(ValueError, match="chunk_size"):
        nearest_neighbour_distances(torch.randn(2, 3), torch.randn(2, 3), chunk_size=0)
    with pytest.raises(ValueError, match="Feature dimensions differ"):
        nearest_neighbour_distances(torch.randn(2, 3), torch.randn(2, 5))


def test_duplicate_leakage_rate_detects_full_leakage() -> None:
    train = torch.randn(10, 4)
    test = train[:5].clone()
    assert duplicate_leakage_rate(train, test) == pytest.approx(1.0)


def test_duplicate_leakage_rate_clean_split_is_zero() -> None:
    torch.manual_seed(0)
    train = torch.randn(64, 8)
    test = torch.randn(32, 8) + 50.0
    assert duplicate_leakage_rate(train, test) == pytest.approx(0.0)


def test_duplicate_leakage_rate_partial_leakage() -> None:
    torch.manual_seed(0)
    train = torch.randn(40, 6)
    leaked = train[:4].clone()
    clean = torch.randn(4, 6) + 100.0
    test = torch.cat([leaked, clean], dim=0)
    assert duplicate_leakage_rate(train, test) == pytest.approx(0.5)


def test_duplicate_leakage_rate_rejects_negative_threshold() -> None:
    with pytest.raises(ValueError, match="relative_threshold"):
        duplicate_leakage_rate(
            torch.randn(2, 3), torch.randn(2, 3), relative_threshold=-0.1
        )


def test_environment_leakage_rate_leave_one_out_is_clean() -> None:
    train_envs = torch.tensor([0, 0, 1, 1])
    test_envs = torch.tensor([2, 2, 2])
    assert environment_leakage_rate(train_envs, test_envs) == pytest.approx(0.0)


def test_environment_leakage_rate_detects_overlap() -> None:
    train_envs = torch.tensor([0, 1])
    test_envs = torch.tensor([0, 2, 1, 2])
    assert environment_leakage_rate(train_envs, test_envs) == pytest.approx(0.5)


def test_environment_leakage_rate_empty_inputs() -> None:
    assert environment_leakage_rate(torch.empty(0), torch.tensor([1])) == 0.0
    assert environment_leakage_rate(torch.tensor([1]), torch.empty(0)) == 0.0


def test_feature_scale_is_rms_l2_norm() -> None:
    features = torch.tensor([[3.0, 4.0], [3.0, 4.0]])
    assert feature_scale(features) == pytest.approx(5.0)
    assert feature_scale(torch.empty(0, 3)) == 0.0


def test_split_leakage_report_flags_a_leaky_split() -> None:
    train = torch.randn(20, 5)
    test = train[:6].clone()
    train_envs = torch.zeros(20, dtype=torch.long)
    test_envs = torch.zeros(6, dtype=torch.long)
    report = split_leakage_report(train, test, train_envs, test_envs)
    assert report["duplicate_leakage_rate"] == pytest.approx(1.0)
    assert report["environment_leakage_rate"] == pytest.approx(1.0)
    assert report["min_nn_distance"] == pytest.approx(0.0, abs=1e-5)


def test_split_leakage_report_omits_environment_key_when_unavailable() -> None:
    report = split_leakage_report(torch.randn(8, 3), torch.randn(4, 3))
    assert "environment_leakage_rate" not in report
    assert set(report) == {
        "duplicate_leakage_rate",
        "feature_scale",
        "min_nn_distance",
        "median_nn_distance",
    }
