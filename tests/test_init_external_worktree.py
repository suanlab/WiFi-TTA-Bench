# pyright: basic, reportMissingImports=false

"""Tests for external worktree scaffold generator.

Verifies that the scaffold generator creates the correct directory structure
and placeholder files expected by audit_readiness.py.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import pytest


def _load_scaffold_module() -> Any:
    """Load init_external_worktree module dynamically via importlib."""
    import sys

    script_path = Path(__file__).parent.parent / "scripts" / "init_external_worktree.py"
    spec = importlib.util.spec_from_file_location("init_external_worktree", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["init_external_worktree"] = module
    spec.loader.exec_module(module)
    return module


# Load module once at import time
_scaffold_module = _load_scaffold_module()
ExternalWorktreeScaffold = _scaffold_module.ExternalWorktreeScaffold


@pytest.fixture
def temp_prepared_root(tmp_path: Path) -> Path:
    """Create a temporary prepared root directory."""
    return tmp_path / "prepared"


class TestExternalWorktreeScaffold:
    """Test suite for ExternalWorktreeScaffold."""

    def test_scaffold_creates_prepared_root(self, temp_prepared_root: Path) -> None:
        """Test that scaffold creates the prepared root directory."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        summary = scaffold.create_all_scaffolds()

        assert temp_prepared_root.exists()
        assert summary["prepared_root"] == str(temp_prepared_root)

    def test_paper1_scaffolds_created(self, temp_prepared_root: Path) -> None:
        """Test that Paper 1 dataset scaffolds are created."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        # Check directories exist
        assert (temp_prepared_root / "signfi").exists()
        assert (temp_prepared_root / "ut_har").exists()

        # Check metadata files exist
        assert (temp_prepared_root / "signfi" / "metadata.json").exists()
        assert (temp_prepared_root / "ut_har" / "metadata.json").exists()

        # Check README files exist
        assert (temp_prepared_root / "signfi" / "README.txt").exists()
        assert (temp_prepared_root / "ut_har" / "README.txt").exists()

        # Verify metadata is valid JSON
        with (temp_prepared_root / "signfi" / "metadata.json").open() as f:
            metadata = json.load(f)
            assert metadata["dataset"] == "signfi"
            assert "description" in metadata

    def test_t12_esp32_scaffolds_created(self, temp_prepared_root: Path) -> None:
        """Test that T12 ESP32 capture scaffolds are created."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        esp32_dir = temp_prepared_root / "esp32_captures"
        assert esp32_dir.exists()

        # Check 3 environments exist
        for env_num in range(1, 4):
            env_dir = esp32_dir / f"environment_{env_num}"
            assert env_dir.exists()
            assert (env_dir / "metadata.json").exists()
            assert (env_dir / "README.txt").exists()

        # Check root README
        assert (esp32_dir / "README.txt").exists()

        # Verify metadata is valid JSON
        with (esp32_dir / "environment_1" / "metadata.json").open() as f:
            metadata = json.load(f)
            assert metadata["hardware"] == "ESP32"
            assert metadata["environment"] == "environment_1"

    def test_t12_wifi6_scaffolds_created(self, temp_prepared_root: Path) -> None:
        """Test that T12 WiFi6 capture scaffolds are created."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        wifi6_dir = temp_prepared_root / "wifi6_captures"
        assert wifi6_dir.exists()

        # Check 3 environments exist
        for env_num in range(1, 4):
            env_dir = wifi6_dir / f"environment_{env_num}"
            assert env_dir.exists()
            assert (env_dir / "metadata.json").exists()
            assert (env_dir / "README.txt").exists()

        # Check root README
        assert (wifi6_dir / "README.txt").exists()

    def test_t15_multi_environment_scaffolds_created(
        self, temp_prepared_root: Path
    ) -> None:
        """Test that T15 multi-environment dataset scaffolds are created."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        multi_env_dir = temp_prepared_root / "multi_environment"
        assert multi_env_dir.exists()

        # Check root README
        assert (multi_env_dir / "README.txt").exists()

        # Check template dataset exists
        template_dir = multi_env_dir / "dataset_template"
        assert template_dir.exists()

        # Check 3 environments in template
        for env_num in range(1, 4):
            env_dir = template_dir / f"environment_{env_num}"
            assert env_dir.exists()
            assert (env_dir / "metadata.json").exists()

        # Check template README
        assert (template_dir / "README.txt").exists()

        # Verify metadata is valid JSON
        with (template_dir / "environment_1" / "metadata.json").open() as f:
            metadata = json.load(f)
            assert metadata["dataset"] == "dataset_template"
            assert metadata["environment"] == "environment_1"

    def test_t22_baseline_artifacts_scaffolds_created(
        self, temp_prepared_root: Path
    ) -> None:
        """Test that T22 baseline artifact scaffolds are created."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        baseline_dir = temp_prepared_root / "baseline_artifacts"
        assert baseline_dir.exists()

        # Check root README
        assert (baseline_dir / "README.txt").exists()

        # Check baseline directories
        for baseline_name in ["newrf", "gsrf"]:
            baseline_path = baseline_dir / baseline_name
            assert baseline_path.exists()
            assert (baseline_path / "README.txt").exists()
            assert (baseline_path / "results.json").exists()

            # Verify results.json is valid JSON
            with (baseline_path / "results.json").open() as f:
                results = json.load(f)
                assert results["model"] == baseline_name
                assert "metrics" in results

    def test_summary_contains_created_paths(self, temp_prepared_root: Path) -> None:
        """Test that summary contains all created paths."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        summary = scaffold.create_all_scaffolds()

        # Check summary structure
        assert "created_paths" in summary
        assert "created_files" in summary
        assert len(summary["created_paths"]) > 0
        assert len(summary["created_files"]) > 0

        # Verify all paths exist
        for path_str in summary["created_paths"]:
            assert Path(path_str).exists()

        # Verify all files exist
        for file_str in summary["created_files"]:
            assert Path(file_str).exists()

    def test_metadata_templates_are_valid_json(self, temp_prepared_root: Path) -> None:
        """Test that all metadata.json files are valid JSON."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        # Find all metadata.json files
        metadata_files = list(temp_prepared_root.glob("**/metadata.json"))
        assert len(metadata_files) > 0

        # Verify each is valid JSON
        for metadata_file in metadata_files:
            with metadata_file.open() as f:
                data = json.load(f)
                assert isinstance(data, dict)
                assert "description" in data or "dataset" in data

    def test_readme_files_are_readable(self, temp_prepared_root: Path) -> None:
        """Test that all README.txt files are readable."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        # Find all README.txt files
        readme_files = list(temp_prepared_root.glob("**/README.txt"))
        assert len(readme_files) > 0

        # Verify each is readable
        for readme_file in readme_files:
            content = readme_file.read_text()
            assert len(content) > 0
            assert "TODO" in content or "Expected" in content

    def test_scaffold_idempotent(self, temp_prepared_root: Path) -> None:
        """Test that running scaffold twice is idempotent."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)

        # First run
        summary1 = scaffold.create_all_scaffolds()
        count1 = len(summary1["created_files"])

        # Second run (should not fail)
        summary2 = scaffold.create_all_scaffolds()
        count2 = len(summary2["created_files"])

        # Both should create the same number of files
        assert count1 == count2

    def test_all_required_directories_exist(self, temp_prepared_root: Path) -> None:
        """Test that all required top-level directories are created."""
        scaffold = ExternalWorktreeScaffold(temp_prepared_root)
        scaffold.create_all_scaffolds()

        required_dirs = [
            "signfi",
            "ut_har",
            "esp32_captures",
            "wifi6_captures",
            "multi_environment",
            "baseline_artifacts",
        ]

        for dir_name in required_dirs:
            assert (temp_prepared_root / dir_name).exists(), (
                f"Required directory {dir_name} not created"
            )
