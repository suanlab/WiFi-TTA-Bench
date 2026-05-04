"""Scaffold generator for external blocker directory structure.

Creates empty-but-correct directory layouts for:
- Paper 1 prepared datasets (signfi, ut_har)
- T12 self-collected captures (esp32_captures, wifi6_captures) with 3 environments each
- T15 multi-environment dataset roots
- T22 baseline_artifacts/{newrf,gsrf} roots

Includes placeholder metadata templates and README notes, but no fake data files.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


class ExternalWorktreeScaffold:
    """Generator for external blocker directory scaffolds."""

    def __init__(self, prepared_root: Path):
        """Initialize scaffold generator.

        Args:
            prepared_root: Root directory where scaffolds will be created.
        """
        self.prepared_root = prepared_root

    def create_all_scaffolds(self) -> dict[str, Any]:
        """Create all scaffold structures.

        Returns:
            Summary dict with created paths and counts.
        """
        summary: dict[str, Any] = {
            "prepared_root": str(self.prepared_root),
            "created_paths": [],
            "created_files": [],
            "skipped": [],
        }

        # Paper 1 prepared datasets
        paper1_result = self._create_paper1_scaffolds()
        summary["paper1"] = paper1_result
        summary["created_paths"].extend(paper1_result["created_paths"])
        summary["created_files"].extend(paper1_result["created_files"])

        # T12: Self-collected captures
        t12_result = self._create_t12_scaffolds()
        summary["t12"] = t12_result
        summary["created_paths"].extend(t12_result["created_paths"])
        summary["created_files"].extend(t12_result["created_files"])

        # T15: Multi-environment datasets
        t15_result = self._create_t15_scaffolds()
        summary["t15"] = t15_result
        summary["created_paths"].extend(t15_result["created_paths"])
        summary["created_files"].extend(t15_result["created_files"])

        # T22: Baseline artifacts
        t22_result = self._create_t22_scaffolds()
        summary["t22"] = t22_result
        summary["created_paths"].extend(t22_result["created_paths"])
        summary["created_files"].extend(t22_result["created_files"])

        return summary

    def _create_paper1_scaffolds(self) -> dict[str, Any]:
        """Create Paper 1 prepared dataset scaffolds (signfi, ut_har).

        Returns:
            Summary of created paths and files.
        """
        result: dict[str, Any] = {"created_paths": [], "created_files": []}
        datasets = ["signfi", "ut_har"]

        for dataset_name in datasets:
            dataset_dir = self.prepared_root / dataset_name
            dataset_dir.mkdir(parents=True, exist_ok=True)
            result["created_paths"].append(str(dataset_dir))

            # Create placeholder metadata template
            metadata_template = {
                "dataset": dataset_name,
                "description": f"Prepared {dataset_name.upper()} dataset for Paper 1",
                "num_samples": "TODO: fill in",
                "num_subcarriers": "TODO: fill in",
                "num_antennas": "TODO: fill in",
                "num_classes": "TODO: fill in",
                "split": "full",
                "notes": (
                    "CSI shape: (num_samples, num_subcarriers, num_antennas) "
                    "complex-valued"
                ),
            }

            metadata_file = dataset_dir / "metadata.json"
            with metadata_file.open("w") as f:
                json.dump(metadata_template, f, indent=2)
            result["created_files"].append(str(metadata_file))

            # Create README
            readme_file = dataset_dir / "README.txt"
            readme_content = f"""Paper 1 Prepared Dataset: {dataset_name.upper()}

Expected files:
  - csi.npy: Complex-valued CSI array,
    shape (num_samples, num_subcarriers, num_antennas)
  - labels.npy: Integer class labels, shape (num_samples,)
  - metadata.json: Dataset metadata (this file)

To populate:
  1. Prepare your {dataset_name.upper()} data locally
  2. Use scripts/prepare_data.py to convert to the expected format
  3. Place csi.npy and labels.npy in this directory

See README.md for data preparation instructions.
"""
            with readme_file.open("w") as f:
                f.write(readme_content)
            result["created_files"].append(str(readme_file))

        return result

    def _create_t12_scaffolds(self) -> dict[str, Any]:
        """Create T12 self-collected capture scaffolds (esp32, wifi6).

        Expected structure:
        - esp32_captures/
          - environment_1/
            - csi.npy, labels.npy, metadata.json
          - environment_2/
          - environment_3/
        - wifi6_captures/
          - environment_1/
          - environment_2/
          - environment_3/

        Returns:
            Summary of created paths and files.
        """
        result: dict[str, Any] = {"created_paths": [], "created_files": []}
        hardware_types = ["esp32_captures", "wifi6_captures"]

        for hardware in hardware_types:
            hardware_dir = self.prepared_root / hardware
            hardware_dir.mkdir(parents=True, exist_ok=True)
            result["created_paths"].append(str(hardware_dir))

            # Create README for hardware root
            readme_file = hardware_dir / "README.txt"
            hw_name = hardware.replace("_", " ").title()
            readme_content = f"""T12 Self-Collected {hw_name} Dataset

Expected structure:
  - environment_1/
    - csi.npy: Complex-valued CSI array
    - labels.npy: Integer class labels
    - metadata.json: Environment metadata
  - environment_2/
  - environment_3/

Each environment should contain CSI captures from a distinct physical location.

To populate:
  1. Collect CSI data using the appropriate hardware (ESP32 or WiFi6)
  2. Process captures into csi.npy and labels.npy arrays
  3. Create metadata.json with environment details
  4. Place in the corresponding environment_N/ directory

See README.md for hardware setup and collection instructions.
"""
            with readme_file.open("w") as f:
                f.write(readme_content)
            result["created_files"].append(str(readme_file))

            # Create 3 environment directories
            for env_num in range(1, 4):
                env_dir = hardware_dir / f"environment_{env_num}"
                env_dir.mkdir(parents=True, exist_ok=True)
                result["created_paths"].append(str(env_dir))

                # Create metadata template
                metadata_template = {
                    "hardware": hardware.replace("_captures", "").upper(),
                    "environment": f"environment_{env_num}",
                    "description": (
                        f"CSI captures from {hardware} in environment {env_num}"
                    ),
                    "location": "TODO: fill in (e.g., 'Lab Room A')",
                    "num_samples": "TODO: fill in",
                    "num_subcarriers": "TODO: fill in",
                    "num_antennas": "TODO: fill in",
                    "num_classes": "TODO: fill in",
                    "collection_date": "TODO: fill in (YYYY-MM-DD)",
                    "notes": (
                        "CSI shape: (num_samples, num_subcarriers, num_antennas) "
                        "complex-valued"
                    ),
                }

                metadata_file = env_dir / "metadata.json"
                with metadata_file.open("w") as f:
                    json.dump(metadata_template, f, indent=2)
                result["created_files"].append(str(metadata_file))

                # Create environment README
                env_readme = env_dir / "README.txt"
                hw_name = hardware.replace("_", " ").title()
                env_readme_content = f"""Environment {env_num} - {hw_name}

Location: TODO: fill in

Expected files:
  - csi.npy: Complex-valued CSI array,
    shape (num_samples, num_subcarriers, num_antennas)
  - labels.npy: Integer class labels, shape (num_samples,)
  - metadata.json: Environment metadata (this file)

Collection notes:
  - Record the exact location and date
  - Document any special conditions (e.g., furniture, obstacles)
  - Ensure consistent hardware setup across environments
"""
                with env_readme.open("w") as f:
                    f.write(env_readme_content)
                result["created_files"].append(str(env_readme))

        return result

    def _create_t15_scaffolds(self) -> dict[str, Any]:
        """Create T15 multi-environment dataset scaffolds.

        Expected structure:
        - multi_environment/
          - dataset_name_1/
            - environment_1/
              - csi.npy, labels.npy, metadata.json
            - environment_2/
            - environment_3/
          - dataset_name_2/
            - ...

        Returns:
            Summary of created paths and files.
        """
        result: dict[str, Any] = {"created_paths": [], "created_files": []}

        multi_env_dir = self.prepared_root / "multi_environment"
        multi_env_dir.mkdir(parents=True, exist_ok=True)
        result["created_paths"].append(str(multi_env_dir))

        # Create root README
        readme_file = multi_env_dir / "README.txt"
        readme_content = """T15 Multi-Environment Prepared Datasets

Expected structure:
  - dataset_name_1/
    - environment_1/
      - csi.npy, labels.npy, metadata.json
    - environment_2/
    - environment_3/
  - dataset_name_2/
    - ...

Each dataset should contain CSI captures from at least 3 distinct environments.
This enables cross-environment generalization studies.

To populate:
  1. Create a directory for each multi-environment dataset
  2. Within each dataset, create environment_1/, environment_2/, environment_3/
  3. Place csi.npy, labels.npy, and metadata.json in each environment directory
  4. Ensure consistent feature dimensions across environments

See README.md for multi-environment dataset preparation instructions.
"""
        with readme_file.open("w") as f:
            f.write(readme_content)
        result["created_files"].append(str(readme_file))

        # Create a template dataset directory (users can copy/rename)
        template_dir = multi_env_dir / "dataset_template"
        template_dir.mkdir(parents=True, exist_ok=True)
        result["created_paths"].append(str(template_dir))

        # Create 3 environment directories in template
        for env_num in range(1, 4):
            env_dir = template_dir / f"environment_{env_num}"
            env_dir.mkdir(parents=True, exist_ok=True)
            result["created_paths"].append(str(env_dir))

            # Create metadata template
            metadata_template = {
                "dataset": "dataset_template",
                "environment": f"environment_{env_num}",
                "description": (
                    f"Template environment {env_num} for multi-environment dataset"
                ),
                "location": "TODO: fill in",
                "num_samples": "TODO: fill in",
                "num_subcarriers": "TODO: fill in",
                "num_antennas": "TODO: fill in",
                "num_classes": "TODO: fill in",
                "notes": (
                    "CSI shape: (num_samples, num_subcarriers, num_antennas) "
                    "complex-valued"
                ),
            }

            metadata_file = env_dir / "metadata.json"
            with metadata_file.open("w") as f:
                json.dump(metadata_template, f, indent=2)
            result["created_files"].append(str(metadata_file))

        # Create template README
        template_readme = template_dir / "README.txt"
        template_readme_content = """Multi-Environment Dataset Template

This is a template directory. To create a new multi-environment dataset:

1. Copy this entire directory to a new name:
   cp -r dataset_template my_dataset_name

2. Update metadata.json in each environment_N/ directory

3. Place your csi.npy and labels.npy files in each environment directory

4. Ensure all environments have the same feature dimensions
   (num_subcarriers, num_antennas)

Expected files in each environment:
  - csi.npy: Complex-valued CSI array,
    shape (num_samples, num_subcarriers, num_antennas)
  - labels.npy: Integer class labels, shape (num_samples,)
  - metadata.json: Environment metadata
"""
        with template_readme.open("w") as f:
            f.write(template_readme_content)
        result["created_files"].append(str(template_readme))

        return result

    def _create_t22_scaffolds(self) -> dict[str, Any]:
        """Create T22 baseline artifacts scaffolds (NeRF, 3DGS).

        Expected structure:
        - baseline_artifacts/
          - newrf/
            - results.json or results.npy (placeholder)
          - gsrf/
            - results.json or results.npy (placeholder)

        Returns:
            Summary of created paths and files.
        """
        result: dict[str, Any] = {"created_paths": [], "created_files": []}

        baseline_dir = self.prepared_root / "baseline_artifacts"
        baseline_dir.mkdir(parents=True, exist_ok=True)
        result["created_paths"].append(str(baseline_dir))

        # Create root README
        readme_file = baseline_dir / "README.txt"
        readme_content = """T22 Baseline Artifacts (NeRF / 3DGS)

Expected structure:
  - newrf/
    - results.json or results.npy
  - gsrf/
    - results.json or results.npy

These directories should contain pre-computed baseline results for comparison
with PINN4CSI models.

To populate:
  1. Run NeRF and 3DGS baselines on your datasets
  2. Save results as results.json (recommended) or results.npy
  3. Place in the corresponding baseline directory

Results format (JSON):
  {
    "model": "newrf" or "gsrf",
    "dataset": "dataset_name",
    "metrics": {
      "mse": float,
      "nmse": float,
      "mae": float,
      ...
    },
    "timestamp": "ISO 8601 datetime"
  }

See README.md for baseline implementation and evaluation instructions.
"""
        with readme_file.open("w") as f:
            f.write(readme_content)
        result["created_files"].append(str(readme_file))

        # Create baseline directories
        for baseline_name in ["newrf", "gsrf"]:
            baseline_path = baseline_dir / baseline_name
            baseline_path.mkdir(parents=True, exist_ok=True)
            result["created_paths"].append(str(baseline_path))

            # Create baseline README
            baseline_readme = baseline_path / "README.txt"
            baseline_readme_content = f"""{baseline_name.upper()} Baseline Results

Expected files:
  - results.json: JSON file with results and metrics
  - results.npy: NumPy file with results (alternative to JSON)

Results should include:
  - Model name and version
  - Dataset name
  - Evaluation metrics (MSE, NMSE, MAE, etc.)
  - Timestamp of evaluation

To populate:
  1. Train {baseline_name.upper()} baseline on your dataset
  2. Evaluate on test set
  3. Save results in results.json or results.npy
  4. Place in this directory

See parent README.txt for format specifications.
"""
            with baseline_readme.open("w") as f:
                f.write(baseline_readme_content)
            result["created_files"].append(str(baseline_readme))

            # Create placeholder results template
            results_template = {
                "model": baseline_name,
                "dataset": "TODO: fill in",
                "status": "TODO: fill in (pending, in_progress, completed)",
                "metrics": {
                    "mse": "TODO: fill in",
                    "nmse": "TODO: fill in",
                    "mae": "TODO: fill in",
                },
                "timestamp": "TODO: fill in (ISO 8601)",
                "notes": "Placeholder template. Replace with actual results.",
            }

            results_file = baseline_path / "results.json"
            with results_file.open("w") as f:
                json.dump(results_template, f, indent=2)
            result["created_files"].append(str(results_file))

        return result


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Initialize external blocker directory scaffolds"
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=None,
        help="Root directory for scaffolds (default: data/prepared)",
    )
    parser.add_argument(
        "--output-json",
        type=Path,
        default=None,
        help="Write summary to JSON file",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress console output",
    )

    args = parser.parse_args()

    # Default to data/prepared if not specified
    prepared_root = args.prepared_root or Path("data/prepared")

    # Create scaffolds
    scaffold = ExternalWorktreeScaffold(prepared_root)
    summary = scaffold.create_all_scaffolds()

    # Output summary
    if not args.quiet:
        print(f"✓ Scaffolds created in: {prepared_root}")
        print(f"  Created {len(summary['created_paths'])} directories")
        print(f"  Created {len(summary['created_files'])} files")
        print("\nStructure:")
        print("  - signfi/")
        print("  - ut_har/")
        print("  - esp32_captures/")
        print("    - environment_1/")
        print("    - environment_2/")
        print("    - environment_3/")
        print("  - wifi6_captures/")
        print("    - environment_1/")
        print("    - environment_2/")
        print("    - environment_3/")
        print("  - multi_environment/")
        print("    - dataset_template/")
        print("      - environment_1/")
        print("      - environment_2/")
        print("      - environment_3/")
        print("  - baseline_artifacts/")
        print("    - newrf/")
        print("    - gsrf/")

    # Write JSON summary if requested
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        with args.output_json.open("w") as f:
            json.dump(summary, f, indent=2)
        if not args.quiet:
            print(f"\nSummary written to: {args.output_json}")


if __name__ == "__main__":
    main()
