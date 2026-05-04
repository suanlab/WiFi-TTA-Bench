# pyright: basic, reportMissingImports=false

"""Tests for readiness audit tool."""

from __future__ import annotations

import importlib.util
import json
import tempfile
from collections.abc import Generator
from pathlib import Path
from typing import Any

import numpy as np
import pytest  # noqa: F401


def _load_audit_module() -> Any:
    """Load audit_readiness module dynamically via importlib."""
    import sys

    script_path = Path(__file__).parent.parent / "scripts" / "audit_readiness.py"
    spec = importlib.util.spec_from_file_location("audit_readiness", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["audit_readiness"] = module
    spec.loader.exec_module(module)
    return module


# Load module once at import time
_audit_module = _load_audit_module()
AuditReport = _audit_module.AuditReport
AuditResult = _audit_module.AuditResult
ReadinessAuditor = _audit_module.ReadinessAuditor
format_cli_report = _audit_module.format_cli_report


@pytest.fixture
def temp_prepared_root() -> Generator[Path, None, None]:
    """Create a temporary prepared data root directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


def test_audit_result_creation() -> None:
    """Test AuditResult dataclass creation."""
    result = AuditResult(
        category="Test Category",
        check_name="test_check",
        status="ready",
        message="Test message",
        details={"key": "value"},
    )
    assert result.category == "Test Category"
    assert result.status == "ready"
    assert result.details == {"key": "value"}


def test_audit_report_is_ready() -> None:
    """Test AuditReport.is_ready property."""
    # All passed
    report = AuditReport(
        timestamp="2026-03-16T00:00:00",
        total_checks=3,
        passed_checks=3,
        failed_checks=0,
        partial_checks=0,
        results=[],
    )
    assert report.is_ready is True

    # Some failed
    report = AuditReport(
        timestamp="2026-03-16T00:00:00",
        total_checks=3,
        passed_checks=2,
        failed_checks=1,
        partial_checks=0,
        results=[],
    )
    assert report.is_ready is False


def test_audit_report_to_dict() -> None:
    """Test AuditReport.to_dict() serialization."""
    result = AuditResult(
        category="Test",
        check_name="test",
        status="ready",
        message="OK",
    )
    report = AuditReport(
        timestamp="2026-03-16T00:00:00",
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        partial_checks=0,
        results=[result],
    )

    data = report.to_dict()
    assert data["timestamp"] == "2026-03-16T00:00:00"
    assert data["total_checks"] == 1
    assert data["is_ready"] is True
    assert len(data["results"]) == 1
    assert data["results"][0]["status"] == "ready"


def test_auditor_initialization(temp_prepared_root: Path) -> None:
    """Test ReadinessAuditor initialization."""
    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    assert auditor.prepared_root == temp_prepared_root


def test_auditor_paper1_missing(temp_prepared_root: Path) -> None:
    """Test Paper 1 audit when datasets are missing."""
    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_paper1_prepared_data()

    assert len(auditor.results) == 2  # signfi + ut_har
    for result in auditor.results:
        assert result.status == "missing"
        assert "not found" in result.message.lower()


def test_auditor_paper1_valid(temp_prepared_root: Path) -> None:
    """Test Paper 1 audit with valid data."""
    # Create valid Paper 1 data
    for dataset_name in ["signfi", "ut_har"]:
        dataset_dir = temp_prepared_root / dataset_name
        dataset_dir.mkdir(parents=True)

        # Create valid CSI and labels
        csi = np.random.randn(10, 16, 2).astype(np.float32)
        labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)

        np.save(dataset_dir / "csi.npy", csi)
        np.save(dataset_dir / "labels.npy", labels)

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_paper1_prepared_data()

    assert len(auditor.results) == 2
    for result in auditor.results:
        assert result.status == "ready"
        assert result.details is not None
        assert "csi_shape" in result.details
        assert "labels_shape" in result.details


def test_auditor_paper1_invalid_shape(temp_prepared_root: Path) -> None:
    """Test Paper 1 audit with invalid shapes."""
    dataset_dir = temp_prepared_root / "signfi"
    dataset_dir.mkdir(parents=True)

    # Create invalid CSI (2D instead of 3D)
    csi = np.random.randn(10, 16).astype(np.float32)
    labels = np.array([0, 1, 2, 0, 1, 2, 0, 1, 2, 0], dtype=np.int64)

    np.save(dataset_dir / "csi.npy", csi)
    np.save(dataset_dir / "labels.npy", labels)

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_paper1_prepared_data()

    result = next(r for r in auditor.results if "signfi" in r.check_name.lower())
    assert result.status == "invalid"
    assert "shape" in result.message.lower()


def test_auditor_t12_missing(temp_prepared_root: Path) -> None:
    """Test T12 audit when captures are missing."""
    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t12_esp32_wifi6()

    assert len(auditor.results) == 2  # esp32 + wifi6
    for result in auditor.results:
        assert result.status == "missing"


def test_auditor_t12_partial(temp_prepared_root: Path) -> None:
    """Test T12 audit with partial data (< 3 environments)."""
    for hardware in ["esp32_captures", "wifi6_captures"]:
        hardware_dir = temp_prepared_root / hardware
        hardware_dir.mkdir(parents=True)

        # Create only 2 environments
        for env_id in range(2):
            env_dir = hardware_dir / f"environment_{env_id}"
            env_dir.mkdir(parents=True)
            (env_dir / "csi.npy").touch()
            (env_dir / "labels.npy").touch()
            (env_dir / "metadata.json").touch()

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t12_esp32_wifi6()

    for result in auditor.results:
        assert result.status == "partial"
        assert result.details is not None
        assert result.details["environments_found"] == 2


def test_auditor_t12_ready(temp_prepared_root: Path) -> None:
    """Test T12 audit with valid data (≥3 environments)."""
    for hardware in ["esp32_captures", "wifi6_captures"]:
        hardware_dir = temp_prepared_root / hardware
        hardware_dir.mkdir(parents=True)

        # Create 3 valid environments
        for env_id in range(3):
            env_dir = hardware_dir / f"environment_{env_id}"
            env_dir.mkdir(parents=True)

            # Create valid files
            csi = np.random.randn(5, 16, 2).astype(np.float32)
            labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)
            np.save(env_dir / "csi.npy", csi)
            np.save(env_dir / "labels.npy", labels)

            metadata = {"environment": f"env_{env_id}"}
            (env_dir / "metadata.json").write_text(json.dumps(metadata))

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t12_esp32_wifi6()

    for result in auditor.results:
        assert result.status == "ready"
        assert result.details is not None
        assert result.details["environments_found"] == 3


def test_auditor_t15_missing(temp_prepared_root: Path) -> None:
    """Test T15 audit when multi-environment data is missing."""
    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t15_multi_environment()

    assert len(auditor.results) == 1
    result = auditor.results[0]
    assert result.status == "missing"


def test_auditor_t15_partial(temp_prepared_root: Path) -> None:
    """Test T15 audit with partial data (< 3 environments per dataset)."""
    multi_env_dir = temp_prepared_root / "multi_environment"
    multi_env_dir.mkdir(parents=True)

    # Create dataset with only 2 environments
    dataset_dir = multi_env_dir / "test_dataset"
    dataset_dir.mkdir(parents=True)

    for env_id in range(2):
        env_dir = dataset_dir / f"environment_{env_id}"
        env_dir.mkdir(parents=True)
        csi = np.random.randn(5, 16, 2).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        np.save(env_dir / "csi.npy", csi)
        np.save(env_dir / "labels.npy", labels)

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t15_multi_environment()

    result = auditor.results[0]
    assert result.status == "partial"


def test_auditor_t15_ready(temp_prepared_root: Path) -> None:
    """Test T15 audit with valid multi-environment data."""
    multi_env_dir = temp_prepared_root / "multi_environment"
    multi_env_dir.mkdir(parents=True)

    # Create dataset with 3 environments
    dataset_dir = multi_env_dir / "test_dataset"
    dataset_dir.mkdir(parents=True)

    for env_id in range(3):
        env_dir = dataset_dir / f"environment_{env_id}"
        env_dir.mkdir(parents=True)
        csi = np.random.randn(5, 16, 2).astype(np.float32)
        labels = np.array([0, 1, 0, 1, 0], dtype=np.int64)
        np.save(env_dir / "csi.npy", csi)
        np.save(env_dir / "labels.npy", labels)

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t15_multi_environment()

    result = auditor.results[0]
    assert result.status == "ready"
    assert result.details is not None
    assert len(result.details["valid_datasets"]) == 1


def test_auditor_t22_missing(temp_prepared_root: Path) -> None:
    """Test T22 audit when baseline artifacts are missing."""
    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t22_baseline_artifacts()

    result = auditor.results[0]
    assert result.status == "missing"


def test_auditor_t22_partial(temp_prepared_root: Path) -> None:
    """Test T22 audit with partial baseline artifacts."""
    baseline_dir = temp_prepared_root / "baseline_artifacts"
    baseline_dir.mkdir(parents=True)

    # Create only NeRF results
    newrf_dir = baseline_dir / "newrf"
    newrf_dir.mkdir(parents=True)
    (newrf_dir / "results.json").write_text('{"status": "ok"}')

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t22_baseline_artifacts()

    result = auditor.results[0]
    assert result.status == "partial"
    assert result.details is not None
    assert len(result.details["found"]) == 1


def test_auditor_t22_ready(temp_prepared_root: Path) -> None:
    """Test T22 audit with both baseline artifacts."""
    baseline_dir = temp_prepared_root / "baseline_artifacts"
    baseline_dir.mkdir(parents=True)

    # Create both NeRF and 3DGS results
    for baseline_name in ["newrf", "gsrf"]:
        baseline_path = baseline_dir / baseline_name
        baseline_path.mkdir(parents=True)
        (baseline_path / "results.json").write_text('{"status": "ok"}')

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t22_baseline_artifacts()

    result = auditor.results[0]
    assert result.status == "ready"
    assert result.details is not None
    assert len(result.details["baselines"]) == 2


def test_auditor_t22_scaffold_placeholder(temp_prepared_root: Path) -> None:
    """Test T22 audit rejects scaffold placeholder results."""
    baseline_dir = temp_prepared_root / "baseline_artifacts"
    baseline_dir.mkdir(parents=True)

    # Create scaffold placeholder results (with TODO strings)
    for baseline_name in ["newrf", "gsrf"]:
        baseline_path = baseline_dir / baseline_name
        baseline_path.mkdir(parents=True)
        placeholder = {
            "model": baseline_name,
            "dataset": "TODO: fill in",
            "status": "TODO: fill in (pending, in_progress, completed)",
            "metrics": {
                "mse": "TODO: fill in",
                "nmse": "TODO: fill in",
            },
            "notes": "Placeholder template. Replace with actual results.",
        }
        (baseline_path / "results.json").write_text(json.dumps(placeholder))

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t22_baseline_artifacts()

    result = auditor.results[0]
    assert result.status == "missing"
    assert result.details is not None
    assert len(result.details["missing"]) == 2
    assert all(
        "placeholder" in item["issue"].lower() for item in result.details["missing"]
    )


def test_auditor_t22_mixed_real_and_placeholder(temp_prepared_root: Path) -> None:
    """Test T22 audit with one real and one placeholder artifact."""
    baseline_dir = temp_prepared_root / "baseline_artifacts"
    baseline_dir.mkdir(parents=True)

    # Create real NeRF results
    newrf_path = baseline_dir / "newrf"
    newrf_path.mkdir(parents=True)
    real_results = {
        "model": "newrf",
        "dataset": "test_dataset",
        "status": "completed",
        "metrics": {"mse": 0.123, "nmse": 0.456, "mae": 0.789},
        "timestamp": "2026-03-16T10:00:00",
    }
    (newrf_path / "results.json").write_text(json.dumps(real_results))

    # Create placeholder 3DGS results
    gsrf_path = baseline_dir / "gsrf"
    gsrf_path.mkdir(parents=True)
    placeholder = {
        "model": "gsrf",
        "dataset": "TODO: fill in",
        "status": "pending",
        "metrics": {"mse": "TODO: fill in"},
    }
    (gsrf_path / "results.json").write_text(json.dumps(placeholder))

    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    auditor._audit_t22_baseline_artifacts()

    result = auditor.results[0]
    assert result.status == "partial"
    assert len(result.details["found"]) == 1
    assert result.details["found"][0]["baseline"] == "newrf"
    assert len(result.details["missing"]) == 1
    assert "placeholder" in result.details["missing"][0]["issue"].lower()


def test_auditor_audit_all(temp_prepared_root: Path) -> None:
    """Test complete audit run."""
    auditor = ReadinessAuditor(prepared_root=temp_prepared_root)
    report = auditor.audit_all()

    assert isinstance(report, AuditReport)
    assert report.total_checks > 0
    total = report.passed_checks + report.failed_checks + report.partial_checks
    assert total == report.total_checks
    assert report.failed_checks > 0


def test_format_cli_report() -> None:
    """Test CLI report formatting."""
    result = AuditResult(
        category="Test Category",
        check_name="test_check",
        status="ready",
        message="Test passed",
    )
    report = AuditReport(
        timestamp="2026-03-16T00:00:00",
        total_checks=1,
        passed_checks=1,
        failed_checks=0,
        partial_checks=0,
        results=[result],
    )

    cli_report = format_cli_report(report)
    assert "READINESS AUDIT REPORT" in cli_report
    assert "READY" in cli_report
    assert "test_check" in cli_report
    assert "Test passed" in cli_report


def test_format_cli_report_with_failures() -> None:
    """Test CLI report formatting with failures."""
    result = AuditResult(
        category="T12: Self-Collected Captures",
        check_name="T12_esp32_captures",
        status="missing",
        message="Directory not found",
    )
    report = AuditReport(
        timestamp="2026-03-16T00:00:00",
        total_checks=1,
        passed_checks=0,
        failed_checks=1,
        partial_checks=0,
        results=[result],
    )

    cli_report = format_cli_report(report)
    assert "BLOCKERS REMAIN" in cli_report
    assert "T12" in cli_report
