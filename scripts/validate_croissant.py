# pyright: basic, reportMissingImports=false

"""Local Croissant 1.0 metadata validator.

Two modes:
    1. mlcroissant available  -> use the official validator (recommended).
    2. mlcroissant unavailable -> fall back to a lightweight structural check
       against the MLCommons Croissant 1.0 spec (core + RAI fields). This is
       NOT a full JSON-LD validator; it only catches missing keys, malformed
       distribution / recordSet entries, and obvious type mistakes.

Both modes mirror the checks performed by the online tool
`huggingface.co/spaces/JoaquinVanschoren/croissant-checker`. After this passes
locally, upload the JSON to the online tool for the official ED-track
verification.

Usage:
    python scripts/validate_croissant.py
    python scripts/validate_croissant.py \\
        --path outputs/croissant/wifi_tta_bench_metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

DEFAULT_PATH = Path("outputs/croissant/wifi_tta_bench_metadata.json")

CORE_REQUIRED = [
    "@context",
    "@type",
    "name",
    "description",
    "conformsTo",
    "version",
    "license",
    "distribution",
]
CORE_RECOMMENDED = [
    "keywords",
    "citeAs",
    "url",
    "creator",
    "datePublished",
    "recordSet",
]
RAI_REQUIRED = [
    "rai:dataCollection",
    "rai:dataCollectionType",
    "rai:dataUseCases",
    "rai:dataBiases",
    "rai:personalSensitiveInformation",
]
RAI_RECOMMENDED = [
    "rai:dataCollectionRawData",
    "rai:dataPreprocessingProtocol",
    "rai:dataAnnotationProtocol",
    "rai:dataReleaseMaintenancePlan",
    "rai:dataSocialImpact",
]
DISTRIBUTION_REQUIRED = ["@type", "@id", "name", "contentUrl", "encodingFormat"]
RECORDSET_REQUIRED = ["@type", "@id", "name", "field"]


def lightweight_check(d: dict) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    for key in CORE_REQUIRED:
        if key not in d:
            errors.append(f"missing core field: {key}")
    for key in CORE_RECOMMENDED:
        if key not in d:
            warnings.append(f"missing recommended core field: {key}")
    for key in RAI_REQUIRED:
        if key not in d:
            errors.append(f"missing required RAI field: {key}")
    for key in RAI_RECOMMENDED:
        if key not in d:
            warnings.append(f"missing recommended RAI field: {key}")

    conforms = d.get("conformsTo", "")
    if "croissant" not in str(conforms).lower():
        errors.append(f"conformsTo does not reference Croissant: {conforms!r}")
    elif "1.0" not in str(conforms):
        warnings.append(f"conformsTo does not pin spec version 1.0: {conforms!r}")

    typ = d.get("@type")
    if typ not in {"sc:Dataset", "schema:Dataset", "Dataset"}:
        errors.append(f"@type must be sc:Dataset (got {typ!r})")

    distribution = d.get("distribution") or []
    if not isinstance(distribution, list) or not distribution:
        errors.append("distribution must be a non-empty list of cr:FileObject")
    else:
        for i, entry in enumerate(distribution):
            if not isinstance(entry, dict):
                errors.append(f"distribution[{i}] is not an object")
                continue
            for k in DISTRIBUTION_REQUIRED:
                if k not in entry:
                    errors.append(f"distribution[{i}] missing required field: {k}")
            etype = entry.get("@type")
            if etype not in {"cr:FileObject", "cr:FileSet"}:
                errors.append(
                    f"distribution[{i}].@type must be "
                    f"cr:FileObject or cr:FileSet (got {etype!r})"
                )

    record_sets = d.get("recordSet") or []
    if record_sets and isinstance(record_sets, list):
        for i, rs in enumerate(record_sets):
            if not isinstance(rs, dict):
                errors.append(f"recordSet[{i}] is not an object")
                continue
            for k in RECORDSET_REQUIRED:
                if k not in rs:
                    errors.append(f"recordSet[{i}] missing required field: {k}")
            fields = rs.get("field") or []
            if not isinstance(fields, list) or not fields:
                errors.append(f"recordSet[{i}].field must be a non-empty list")

    keywords = d.get("keywords") or []
    if isinstance(keywords, list) and len(keywords) < 3:
        warnings.append(
            f"keywords list has only {len(keywords)} entries (>=3 recommended)"
        )

    return errors, warnings


def try_mlcroissant(path: Path) -> tuple[bool, list[str]]:
    """Attempt to use the official mlcroissant validator if installed."""
    try:
        from mlcroissant import Dataset  # type: ignore[import]
    except ImportError:
        return False, ["(mlcroissant not installed; skipping official check)"]
    try:
        ds = Dataset(jsonld=str(path))
        ds.metadata.issues.report()
        return True, []
    except Exception as e:  # noqa: BLE001
        return True, [f"mlcroissant raised: {e}"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument("--path", type=Path, default=DEFAULT_PATH)
    args = parser.parse_args()

    if not args.path.exists():
        print(f"ERROR: file not found: {args.path}")
        return 2

    try:
        d = json.loads(args.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        print(f"ERROR: invalid JSON: {e}")
        return 2

    print(f"Validating {args.path}")
    print("-" * 70)

    errors, warnings = lightweight_check(d)
    used_official, official_messages = try_mlcroissant(args.path)

    if used_official:
        print("Official mlcroissant validator:")
        if not official_messages:
            print("  OK")
        else:
            for m in official_messages:
                print(f"  {m}")

    print()
    print("Structural lightweight check:")
    if errors:
        print(f"  ERRORS ({len(errors)}):")
        for e in errors:
            print(f"    - {e}")
    else:
        print("  ERRORS: 0")

    if warnings:
        print(f"  WARNINGS ({len(warnings)}):")
        for w in warnings:
            print(f"    - {w}")
    else:
        print("  WARNINGS: 0")

    n_dist = len(d.get("distribution", []))
    n_record = len(d.get("recordSet", []))
    print()
    print(f"Summary: distribution={n_dist}, recordSet={n_record}")
    print(f"         license: {d.get('license')!r}")
    print(f"         conformsTo: {d.get('conformsTo')!r}")
    print(f"         version: {d.get('version')!r}")

    if errors:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
