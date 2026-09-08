# pyright: basic, reportUnknownVariableType=false

"""Shared utilities and helpers."""

from pinn4csi.utils.device import get_device
from pinn4csi.utils.experiment import (
    parse_csv_floats,
    parse_csv_ints,
    parse_csv_items,
    save_dataclass_rows_csv,
    save_json_file,
)
from pinn4csi.utils.metrics import (
    accuracy,
    bonferroni_corrected_alpha,
    bootstrap_ci,
    cohens_d,
    duplicate_leakage_rate,
    environment_leakage_rate,
    expected_calibration_error,
    f1_score,
    feature_scale,
    nearest_neighbour_distances,
    nmse,
    paired_cohens_d,
    prediction_diversity,
    split_leakage_report,
)

__all__ = [
    "get_device",
    "accuracy",
    "nmse",
    "f1_score",
    "cohens_d",
    "paired_cohens_d",
    "bootstrap_ci",
    "bonferroni_corrected_alpha",
    "duplicate_leakage_rate",
    "environment_leakage_rate",
    "expected_calibration_error",
    "feature_scale",
    "nearest_neighbour_distances",
    "prediction_diversity",
    "split_leakage_report",
    "parse_csv_items",
    "parse_csv_ints",
    "parse_csv_floats",
    "save_dataclass_rows_csv",
    "save_json_file",
]
