# pyright: basic, reportMissingImports=false

"""Readiness audit tool for remaining external blockers (T12, T15, T22).

Checks whether prepared datasets, self-collected captures, and baseline artifacts
are present and valid according to the repo's data contracts.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Profile definitions
# ---------------------------------------------------------------------------

# Categories required for each profile.  Each entry is a check_name prefix
# that must be present AND valid (status == "ready") for the profile to pass.
PROFILE_REQUIRED: dict[str, list[str]] = {
    "paper-only": [
        # Manuscript-cited artifacts only: the three real WiFi CSI datasets
        # that feed the decision table and per-method tables in paper2.
        # T15_MultiEnvironment (self-collected WiFi-6) is NOT cited in the
        # paper-only submission and is optional.
        "Paper2_WIDAR_BVP",
        "Paper2_NTUFI_HAR",
        "Paper2_SIGNFI_TOP10",
    ],
    "artifact-ready": [
        # All canonical outputs referenced in submission checklist + legacy
        # Paper 1 baselines.
        "Paper2_WIDAR_BVP",
        "Paper2_NTUFI_HAR",
        "Paper2_SIGNFI_TOP10",
        "Paper1_SIGNFI",
        "Paper1_UT_HAR",
        "T22_BaselineArtifacts",
    ],
    "extended-external-ready": [
        # Everything in artifact-ready plus self-collected / external hardware.
        "Paper2_WIDAR_BVP",
        "Paper2_NTUFI_HAR",
        "Paper2_SIGNFI_TOP10",
        "Paper1_SIGNFI",
        "Paper1_UT_HAR",
        "T12_esp32_captures",
        "T12_wifi6_captures",
        "T15_MultiEnvironment",
        "T22_BaselineArtifacts",
    ],
}

# Optional categories (present in audit but not required for profile pass).
PROFILE_OPTIONAL: dict[str, list[str]] = {
    "paper-only": [
        "Paper1_SIGNFI",
        "Paper1_UT_HAR",
        "T12_esp32_captures",
        "T12_wifi6_captures",
        "T15_MultiEnvironment",
        "T22_BaselineArtifacts",
    ],
    "artifact-ready": [
        "T12_esp32_captures",
        "T12_wifi6_captures",
        "T15_MultiEnvironment",
    ],
    "extended-external-ready": [],
}


@dataclass
class AuditResult:
    """Single audit check result."""

    category: str
    check_name: str
    status: str  # "ready", "missing", "invalid", "partial"
    message: str
    details: dict[str, Any] | None = None


@dataclass
class AuditReport:
    """Complete audit report."""

    timestamp: str
    total_checks: int
    passed_checks: int
    failed_checks: int
    partial_checks: int
    results: list[AuditResult]
    invalid_checks: int = 0

    @property
    def is_ready(self) -> bool:
        """True if all critical checks passed (no missing, no invalid)."""
        return self.failed_checks == 0 and self.invalid_checks == 0

    def is_ready_for_profile(self, profile: str) -> bool:
        """True if all required checks for the given profile are ready."""
        required_prefixes = PROFILE_REQUIRED.get(profile, [])
        result_map = {r.check_name: r for r in self.results}
        for prefix in required_prefixes:
            result = result_map.get(prefix)
            if result is None or result.status != "ready":
                return False
        return True

    def to_dict(self) -> dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "timestamp": self.timestamp,
            "total_checks": self.total_checks,
            "passed_checks": self.passed_checks,
            "failed_checks": self.failed_checks,
            "partial_checks": self.partial_checks,
            "invalid_checks": self.invalid_checks,
            "is_ready": self.is_ready,
            "results": [asdict(r) for r in self.results],
        }


class ReadinessAuditor:
    """Audit tool for external blocker readiness."""

    def __init__(self, prepared_root: Path | None = None):
        """Initialize auditor.

        Args:
            prepared_root: Root directory for prepared datasets.
                If None, uses environment variable or defaults to ./data/prepared.
        """
        if prepared_root is None:
            prepared_root = Path("data/prepared")
        self.prepared_root = prepared_root
        self.results: list[AuditResult] = []

    @staticmethod
    def _is_scaffold_placeholder(results_file: Path) -> bool:
        """Check if a results.json file is a scaffold placeholder.

        Scaffold placeholders contain TODO strings or "pending" status.
        Real artifacts contain actual metrics and completed status.

        Args:
            results_file: Path to results.json file.

        Returns:
            True if file is a scaffold placeholder, False if real artifact.
        """
        try:
            with results_file.open("r") as f:
                content = f.read()

            # Check for placeholder markers
            placeholder_markers = [
                "TODO: fill in",
                '"status": "TODO',
                '"status": "pending"',
                "Placeholder template",
            ]

            for marker in placeholder_markers:
                if marker in content:
                    return True

            # Try to parse as JSON and check for TODO values
            data = json.loads(content)
            if isinstance(data, dict):
                # Check if any value is a TODO string
                def has_todo(obj: Any) -> bool:
                    if isinstance(obj, str):
                        return "TODO" in obj
                    elif isinstance(obj, dict):
                        return any(has_todo(v) for v in obj.values())
                    elif isinstance(obj, list):
                        return any(has_todo(v) for v in obj)
                    return False

                if has_todo(data):
                    return True

            return False
        except (json.JSONDecodeError, OSError):
            # If we can't read/parse, assume it's not a valid placeholder
            # (might be corrupted or in a different format)
            return False

    def audit_all(self) -> AuditReport:
        """Run all audit checks.

        Returns:
            Complete audit report with all check results.
        """
        from datetime import datetime

        self.results = []

        # T12: Self-collected ESP32 / WiFi6 prepared captures
        self._audit_t12_esp32_wifi6()

        # T15: Multi-environment prepared datasets
        self._audit_t15_multi_environment()

        # T22: External baseline result artifacts (NeRF/3DGS)
        self._audit_t22_baseline_artifacts()

        # Paper 1 prepared datasets (legacy baselines)
        self._audit_paper1_prepared_data()

        # Paper 2 (WiFi-TTA-Bench) prepared datasets: the three real datasets
        # cited in the ED-track paper-only submission.
        self._audit_paper2_prepared_data()

        # Compute summary — invalid counts as a blocker
        passed = sum(1 for r in self.results if r.status == "ready")
        failed = sum(1 for r in self.results if r.status == "missing")
        partial = sum(1 for r in self.results if r.status == "partial")
        invalid = sum(1 for r in self.results if r.status == "invalid")

        return AuditReport(
            timestamp=datetime.now().isoformat(),
            total_checks=len(self.results),
            passed_checks=passed,
            failed_checks=failed,
            partial_checks=partial,
            invalid_checks=invalid,
            results=self.results,
        )

    def _audit_paper1_prepared_data(self) -> None:
        """Audit Paper 1 prepared datasets (SignFi, UT_HAR)."""
        datasets = ["signfi", "ut_har"]

        for dataset_name in datasets:
            dataset_dir = self.prepared_root / dataset_name
            status = "ready"
            message = f"Paper 1 {dataset_name.upper()} prepared data"
            details: dict[str, Any] = {}

            if not dataset_dir.exists():
                status = "missing"
                message = f"Directory not found: {dataset_dir}"
            else:
                # Check required files
                required_files = ["csi.npy", "labels.npy"]
                missing_files = []
                for fname in required_files:
                    fpath = dataset_dir / fname
                    if not fpath.exists():
                        missing_files.append(fname)

                if missing_files:
                    status = "missing"
                    message = f"Missing files: {', '.join(missing_files)}"
                    details["missing_files"] = missing_files
                else:
                    # Validate shapes
                    try:
                        csi = np.load(dataset_dir / "csi.npy")
                        labels = np.load(dataset_dir / "labels.npy")

                        if csi.ndim != 3:
                            status = "invalid"
                            message = f"CSI shape invalid: expected 3D, got {csi.ndim}D"
                            details["csi_shape"] = csi.shape
                        elif labels.ndim != 1:
                            status = "invalid"
                            message = (
                                f"Labels shape invalid: expected 1D, got {labels.ndim}D"
                            )
                            details["labels_shape"] = labels.shape
                        elif csi.shape[0] != labels.shape[0]:
                            status = "invalid"
                            message = (
                                f"Shape mismatch: CSI has {csi.shape[0]} samples, "
                                f"labels has {labels.shape[0]}"
                            )
                            details["csi_samples"] = int(csi.shape[0])
                            details["label_samples"] = int(labels.shape[0])
                        else:
                            details["csi_shape"] = csi.shape
                            details["labels_shape"] = labels.shape
                            details["num_classes"] = int(np.max(labels)) + 1
                    except Exception as e:
                        status = "invalid"
                        message = f"Error loading data: {e}"
                        details["error"] = str(e)

            self.results.append(
                AuditResult(
                    category="Paper 1 Prepared Data",
                    check_name=f"Paper1_{dataset_name.upper()}",
                    status=status,
                    message=message,
                    details=details,
                )
            )

    def _audit_paper2_prepared_data(self) -> None:
        """Audit Paper 2 prepared datasets (WiFi-TTA-Bench)."""
        datasets = {
            "widar_bvp": "Paper2_WIDAR_BVP",
            "ntufi_har": "Paper2_NTUFI_HAR",
            "signfi_top10": "Paper2_SIGNFI_TOP10",
        }
        for dataset_name, check_name in datasets.items():
            dataset_dir = self.prepared_root / dataset_name
            status = "ready"
            message = f"Paper 2 {dataset_name} prepared data"
            details: dict[str, Any] = {}

            if not dataset_dir.exists():
                status = "missing"
                message = f"Directory not found: {dataset_dir}"
            else:
                required = ["csi.npy", "labels.npy", "environments.npy"]
                missing = [f for f in required if not (dataset_dir / f).exists()]
                if missing:
                    status = "missing"
                    message = f"Missing files: {', '.join(missing)}"
                    details["missing_files"] = missing
                else:
                    try:
                        csi = np.load(dataset_dir / "csi.npy")
                        labels = np.load(dataset_dir / "labels.npy")
                        envs = np.load(dataset_dir / "environments.npy")
                        if csi.ndim < 2:
                            status = "invalid"
                            message = f"CSI shape invalid: {csi.shape}"
                        elif labels.ndim != 1 or envs.ndim != 1:
                            status = "invalid"
                            message = (
                                f"1D labels/envs expected; got "
                                f"{labels.shape}/{envs.shape}"
                            )
                        elif not (csi.shape[0] == labels.shape[0] == envs.shape[0]):
                            status = "invalid"
                            message = (
                                f"Sample count mismatch: csi={csi.shape[0]} "
                                f"labels={labels.shape[0]} envs={envs.shape[0]}"
                            )
                        else:
                            details["csi_shape"] = csi.shape
                            details["num_classes"] = int(np.max(labels)) + 1
                            details["num_environments"] = int(np.max(envs)) + 1
                    except Exception as e:
                        status = "invalid"
                        message = f"Error loading data: {e}"
                        details["error"] = str(e)

            self.results.append(
                AuditResult(
                    category="Paper 2 Prepared Data",
                    check_name=check_name,
                    status=status,
                    message=message,
                    details=details,
                )
            )

    def _audit_t12_esp32_wifi6(self) -> None:
        """Audit T12: Self-collected ESP32 / WiFi6 prepared captures.

        Expected structure:
        - data/prepared/esp32_captures/
          - environment_1/
            - csi.npy, labels.npy, metadata.json
          - environment_2/
            - csi.npy, labels.npy, metadata.json
          - environment_3/
            - csi.npy, labels.npy, metadata.json
        - data/prepared/wifi6_captures/
          - environment_1/
            - csi.npy, labels.npy, metadata.json
          - ...
        """
        for hardware in ["esp32_captures", "wifi6_captures"]:
            hardware_dir = self.prepared_root / hardware
            status = "missing"
            message = f"T12 {hardware} not found"
            details: dict[str, Any] = {}

            if hardware_dir.exists():
                # Check for at least 3 environments
                env_dirs = [
                    d
                    for d in hardware_dir.iterdir()
                    if d.is_dir() and not d.name.startswith(".")
                ]
                num_envs = len(env_dirs)

                if num_envs == 0:
                    status = "missing"
                    message = f"No environment directories found in {hardware_dir}"
                elif num_envs < 3:
                    status = "partial"
                    message = f"Only {num_envs} environment(s) found, need >=3"
                    details["environments_found"] = num_envs
                    details["environments"] = [d.name for d in env_dirs]
                else:
                    # Validate each environment
                    all_valid = True
                    invalid_envs = []

                    for env_dir in env_dirs:
                        required = ["csi.npy", "labels.npy", "metadata.json"]
                        missing = [f for f in required if not (env_dir / f).exists()]

                        if missing:
                            all_valid = False
                            invalid_envs.append(
                                {
                                    "environment": env_dir.name,
                                    "missing_files": missing,
                                }
                            )

                    if all_valid:
                        status = "ready"
                        message = f"T12 {hardware}: {num_envs} valid environments"
                        details["environments_found"] = num_envs
                        details["environments"] = [d.name for d in env_dirs]
                    else:
                        status = "partial"
                        message = (
                            f"T12 {hardware}: {len(invalid_envs)} "
                            "environment(s) incomplete"
                        )
                        details["invalid_environments"] = invalid_envs

            self.results.append(
                AuditResult(
                    category="T12: Self-Collected Captures",
                    check_name=f"T12_{hardware}",
                    status=status,
                    message=message,
                    details=details,
                )
            )

    def _audit_t15_multi_environment(self) -> None:
        """Audit T15: Multi-environment prepared datasets.

        Expected structure:
        - data/prepared/multi_environment/
          - dataset_name/
            - environment_1/
              - csi.npy, labels.npy, metadata.json
            - environment_2/
              - csi.npy, labels.npy, metadata.json
            - environment_3/
              - csi.npy, labels.npy, metadata.json
        """
        multi_env_dir = self.prepared_root / "multi_environment"
        status = "missing"
        message = "T15 multi-environment datasets not found"
        details: dict[str, Any] = {}

        if multi_env_dir.exists():
            # Check for dataset directories
            dataset_dirs = [
                d
                for d in multi_env_dir.iterdir()
                if d.is_dir() and not d.name.startswith(".")
            ]

            if len(dataset_dirs) == 0:
                status = "missing"
                message = "No dataset directories found in multi_environment/"
            else:
                # Validate each dataset
                all_valid = True
                incomplete_datasets = []
                valid_datasets = []

                for dataset_dir in dataset_dirs:
                    env_dirs = [
                        d
                        for d in dataset_dir.iterdir()
                        if d.is_dir() and not d.name.startswith(".")
                    ]

                    if len(env_dirs) < 3:
                        # Mark as incomplete but not invalid
                        incomplete_datasets.append(
                            {
                                "dataset": dataset_dir.name,
                                "environments_found": len(env_dirs),
                            }
                        )
                    else:
                        # Check files in each environment
                        dataset_valid = True
                        for env_dir in env_dirs:
                            required = ["csi.npy", "labels.npy"]
                            if not all((env_dir / f).exists() for f in required):
                                dataset_valid = False
                                break

                        if dataset_valid:
                            valid_datasets.append(
                                {
                                    "dataset": dataset_dir.name,
                                    "environments": len(env_dirs),
                                }
                            )
                        else:
                            incomplete_datasets.append(
                                {
                                    "dataset": dataset_dir.name,
                                    "issue": "incomplete environment files",
                                }
                            )

                if all_valid and len(valid_datasets) > 0:
                    status = "ready"
                    message = (
                        f"T15: {len(valid_datasets)} valid multi-environment dataset(s)"
                    )
                    details["valid_datasets"] = valid_datasets
                elif len(valid_datasets) > 0:
                    status = "partial"
                    message = (
                        f"T15: {len(valid_datasets)} valid, "
                        f"{len(incomplete_datasets)} incomplete"
                    )
                    details["valid_datasets"] = valid_datasets
                    details["incomplete_datasets"] = incomplete_datasets
                elif len(incomplete_datasets) > 0:
                    status = "partial"
                    message = (
                        f"T15: {len(incomplete_datasets)} dataset(s) "
                        "with <3 environments"
                    )
                    details["incomplete_datasets"] = incomplete_datasets
                else:
                    status = "missing"
                    message = "T15: No multi-environment datasets found"

        self.results.append(
            AuditResult(
                category="T15: Multi-Environment Data",
                check_name="T15_MultiEnvironment",
                status=status,
                message=message,
                details=details,
            )
        )

    def _audit_t22_baseline_artifacts(self) -> None:
        """Audit T22: External baseline result artifacts (NeRF/3DGS).

        Expected structure:
        - data/prepared/baseline_artifacts/
          - newrf/
            - results.json or results.npy (must contain real data, not scaffold)
          - gsrf/
            - results.json or results.npy (must contain real data, not scaffold)

        Scaffold placeholders (containing TODO strings or pending status) are
        not counted as ready artifacts.
        """
        baseline_dir = self.prepared_root / "baseline_artifacts"
        status = "missing"
        message = "T22 baseline artifacts not found"
        details: dict[str, Any] = {}

        if baseline_dir.exists():
            baselines = ["newrf", "gsrf"]
            found_baselines = []
            missing_baselines = []

            for baseline_name in baselines:
                baseline_path = baseline_dir / baseline_name
                if baseline_path.exists():
                    result_files = list(baseline_path.glob("results.*"))
                    if result_files:
                        real_artifacts = []
                        for result_file in result_files:
                            if not self._is_scaffold_placeholder(result_file):
                                real_artifacts.append(result_file.name)

                        if real_artifacts:
                            found_baselines.append(
                                {
                                    "baseline": baseline_name,
                                    "result_files": real_artifacts,
                                }
                            )
                        else:
                            missing_baselines.append(
                                {
                                    "baseline": baseline_name,
                                    "issue": "result files are scaffold placeholders",
                                }
                            )
                    else:
                        missing_baselines.append(
                            {
                                "baseline": baseline_name,
                                "issue": "no result files found",
                            }
                        )
                else:
                    missing_baselines.append(
                        {
                            "baseline": baseline_name,
                            "issue": "directory not found",
                        }
                    )

            if len(found_baselines) == 2:
                status = "ready"
                message = "T22: Both NeRF and 3DGS baseline artifacts present"
                details["baselines"] = found_baselines
            elif len(found_baselines) > 0:
                status = "partial"
                message = f"T22: {len(found_baselines)}/2 baseline artifacts present"
                details["found"] = found_baselines
                details["missing"] = missing_baselines
            else:
                status = "missing"
                message = "T22: No baseline artifacts found"
                details["missing"] = missing_baselines
        else:
            details["baseline_dir"] = str(baseline_dir)

        self.results.append(
            AuditResult(
                category="T22: Baseline Artifacts",
                check_name="T22_BaselineArtifacts",
                status=status,
                message=message,
                details=details,
            )
        )


def format_cli_report(report: AuditReport, profile: str | None = None) -> str:
    """Format audit report for CLI output.

    Args:
        report: Audit report to format.
        profile: Active readiness profile name, or None for default.

    Returns:
        Formatted string for console display.
    """
    # Determine profile-level readiness
    if profile is not None:
        profile_ready = report.is_ready_for_profile(profile)
        profile_label = profile
        overall_label = "READY" if profile_ready else "BLOCKERS REMAIN"
        overall_icon = "GREEN" if profile_ready else "RED"
    else:
        profile_label = "default"
        profile_ready = report.is_ready
        overall_label = "READY" if profile_ready else "BLOCKERS REMAIN"
        overall_icon = "GREEN" if profile_ready else "RED"

    lines = [
        "=" * 70,
        "PINN4CSI READINESS AUDIT REPORT",
        "=" * 70,
        f"Timestamp:        {report.timestamp}",
        f"Readiness Profile: {profile_label}",
        "",
        f"Summary: {report.passed_checks}/{report.total_checks} checks passed",
        f"  [+] Ready:   {report.passed_checks}",
        f"  [~] Partial: {report.partial_checks}",
        f"  [!] Invalid: {report.invalid_checks}",
        f"  [-] Missing: {report.failed_checks}",
        "",
        f"Overall Status [{profile_label}]: [{overall_icon}] {overall_label}",
        "",
        "-" * 70,
        "DETAILED RESULTS",
        "-" * 70,
    ]

    # Separate required vs optional for the active profile
    required_names: set[str] = set(PROFILE_REQUIRED.get(profile or "", []))
    optional_names: set[str] = set(PROFILE_OPTIONAL.get(profile or "", []))

    # Group by category
    by_category: dict[str, list[AuditResult]] = {}
    for result in report.results:
        if result.category not in by_category:
            by_category[result.category] = []
        by_category[result.category].append(result)

    for category in sorted(by_category.keys()):
        lines.append(f"\n{category}:")
        for result in by_category[category]:
            status_icon = {
                "ready": "[+]",
                "missing": "[-]",
                "partial": "[~]",
                "invalid": "[!]",
            }.get(result.status, "[?]")

            # Annotate required/optional when a profile is active
            if profile is not None:
                if result.check_name in required_names:
                    req_tag = " [REQUIRED]"
                elif result.check_name in optional_names:
                    req_tag = " [optional]"
                else:
                    req_tag = ""
            else:
                req_tag = ""

            lines.append(
                f"  {status_icon} {result.check_name}{req_tag}: {result.message}"
            )

            if result.details:
                for key, value in result.details.items():
                    if isinstance(value, list):
                        lines.append(f"      {key}:")
                        for item in value:
                            if isinstance(item, dict):
                                lines.append(f"        - {item}")
                            else:
                                lines.append(f"        - {item}")
                    else:
                        lines.append(f"      {key}: {value}")

    lines.extend(
        [
            "",
            "=" * 70,
            "RECOMMENDATIONS",
            "=" * 70,
        ]
    )

    # Blockers = missing OR invalid
    blocker_results = [r for r in report.results if r.status in ("missing", "invalid")]

    if blocker_results:
        lines.append(
            "Blockers detected (missing or invalid). To complete external work:"
        )
        seen_recommendations: set[str] = set()
        for result in blocker_results:
            if "T12" in result.check_name:
                rec_key = "T12"
                if rec_key not in seen_recommendations:
                    lines.append(
                        "  * T12: Collect ESP32/WiFi6 CSI data in >=3 environments"
                    )
                    seen_recommendations.add(rec_key)
            elif "T15" in result.check_name:
                rec_key = "T15"
                if rec_key not in seen_recommendations:
                    lines.append(
                        "  * T15: Prepare multi-environment datasets "
                        "with >=3 environments"
                    )
                    seen_recommendations.add(rec_key)
            elif "T22" in result.check_name:
                rec_key = "T22"
                if rec_key not in seen_recommendations:
                    lines.append(
                        "  * T22: Generate NeRF/3DGS baseline artifacts for comparison"
                    )
                    seen_recommendations.add(rec_key)
            elif "Paper1" in result.check_name:
                dataset_name = result.check_name.replace("Paper1_", "")
                rec_key = f"Paper1_{dataset_name}"
                if rec_key not in seen_recommendations:
                    lines.append(
                        f"  * Prepare {dataset_name} dataset using "
                        "scripts/prepare_data.py"
                    )
                    seen_recommendations.add(rec_key)
    else:
        lines.append("[+] All external blockers are ready!")

    lines.append("")
    return "\n".join(lines)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Audit readiness of external blockers (T12, T15, T22)"
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=None,
        help="Root directory for prepared datasets (default: data/prepared)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write JSON report to this file",
    )
    parser.add_argument(
        "--output-cli",
        type=Path,
        default=None,
        help="Write CLI report to this file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )
    parser.add_argument(
        "--profile",
        choices=["paper-only", "artifact-ready", "extended-external-ready"],
        default=None,
        help=(
            "Readiness profile to evaluate against. "
            "paper-only: only manuscript-cited artifacts. "
            "artifact-ready: all canonical submission outputs. "
            "extended-external-ready: includes self-collected/external artifacts."
        ),
    )

    args = parser.parse_args()

    # Run audit
    auditor = ReadinessAuditor(prepared_root=args.prepared_root)
    report = auditor.audit_all()

    # Output JSON if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(report.to_dict(), f, indent=2)
        if not args.quiet:
            print(f"JSON report written to: {args.output_json}")

    # Output CLI report
    cli_report = format_cli_report(report, profile=args.profile)

    if args.output_cli:
        args.output_cli.parent.mkdir(parents=True, exist_ok=True)
        with args.output_cli.open("w") as f:
            f.write(cli_report)
        if not args.quiet:
            print(f"CLI report written to: {args.output_cli}")

    # Print to console unless quiet
    if not args.quiet:
        print(cli_report)

    # Exit code: 0 only if all required artifacts present & valid for the
    # active profile; non-zero otherwise.
    if args.profile is not None:
        ready = report.is_ready_for_profile(args.profile)
    else:
        ready = report.is_ready

    raise SystemExit(0 if ready else 1)


if __name__ == "__main__":
    main()
