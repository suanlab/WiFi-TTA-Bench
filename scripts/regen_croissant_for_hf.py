# pyright: basic, reportMissingImports=false

"""Regenerate the HF-flavored Croissant JSON.

The HF dataset has a flat layout (method_summaries/, splits/, top-level
manifest). The original Croissant referenced GitHub-style paths like
outputs/widar_full_tta/method_summary.json, which (a) do not resolve on HF
and (b) lacked SHA-256 / source bindings the official mlcroissant validator
requires. This script:

  1. Computes SHA-256 for each source file (the local bytes that were
     uploaded to HF; deterministic with the upload script).
  2. Rewrites contentUrl values to match the HF layout.
  3. Drops the RecordSet (HF auto-Croissants omit it; spec allows
     omitting recordSet for metadata-only datasets).
  4. Writes the new Croissant to outputs/croissant/wifi_tta_bench_metadata.json
     so the GitHub source-of-truth and the HF-uploaded copy match.

Usage:
    python scripts/regen_croissant_for_hf.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

# Map (FileObject @id, source-on-disk path, HF target path, encodingFormat,
#      description). The first two columns drive the file load + hash;
#      the third becomes contentUrl.
ENTRIES: list[tuple[str, str, str, str, str]] = [
    (
        "widar-full-tta-summary",
        "outputs/widar_full_tta/method_summary.json",
        "method_summaries/widar.json",
        "application/json",
        "Per-method mean gain + 95% bootstrap CI + Cohen's d + negative "
        "adaptation rate for Widar_BVP (primary MLP backbone, n=15 paired "
        "observations).",
    ),
    (
        "ntufi-tta-summary",
        "outputs/ntufi_tta/method_summary.json",
        "method_summaries/ntufi.json",
        "application/json",
        "Per-method summary for NTU-Fi HAR (primary MLP, n=15).",
    ),
    (
        "signfi-tta-summary",
        "outputs/signfi_tta/method_summary.json",
        "method_summaries/signfi.json",
        "application/json",
        "Per-method summary for SignFi-10 (primary MLP, n=15).",
    ),
    (
        "widar-lame-sar-cotta",
        "outputs/new_methods/widar_lame_sar_cotta.json",
        "method_summaries/widar_lame_sar_cotta.json",
        "application/json",
        "SAR, CoTTA, LAME on Widar_BVP (primary MLP, n=15).",
    ),
    (
        "ntufi-lame-sar-cotta",
        "outputs/new_methods/ntufi_har_lame_sar_cotta.json",
        "method_summaries/ntufi_lame_sar_cotta.json",
        "application/json",
        "SAR, CoTTA, LAME on NTU-Fi HAR (primary MLP, n=15).",
    ),
    (
        "signfi-lame-sar-cotta",
        "outputs/new_methods/signfi_top10_lame_sar_cotta.json",
        "method_summaries/signfi_lame_sar_cotta.json",
        "application/json",
        "SAR, CoTTA, LAME on SignFi-10 (primary MLP, n=15).",
    ),
    (
        "widar-mlpbn-faithful-sar-cotta",
        "outputs/new_methods/widar_mlpbn_faithful_sar_cotta.json",
        "method_summaries/widar_mlpbn_faithful_sar_cotta.json",
        "application/json",
        "Faithful BN-only SAR/CoTTA + full TENT on MLP+BN Widar_BVP backbone "
        "(n=15).",
    ),
    (
        "arch-ablation-with-ci",
        "outputs/final_ablations/arch_ablation_with_ci.json",
        "method_summaries/arch_ablation_with_ci.json",
        "application/json",
        "Three-backbone architecture ablation with 95% bootstrap CIs "
        "(MLP, MLP+BN, CNN1D+BN).",
    ),
    (
        "faithful-tent",
        "outputs/phase1/faithful_tent_results.json",
        "method_summaries/faithful_tent_results.json",
        "application/json",
        "Faithful BN-only TENT on MLP+BN Widar_BVP (n=15).",
    ),
    (
        "source-overfit",
        "outputs/source_overfit/mitigation_pilot.json",
        "method_summaries/source_overfit_pilot.json",
        "application/json",
        "Source-overfit mitigation pilot: label smoothing alpha in "
        "{0.0, 0.1, 0.2} on Widar (n=15).",
    ),
    (
        "widar-per-class",
        "outputs/overconfidence/widar_per_class.json",
        "method_summaries/widar_per_class.json",
        "application/json",
        "Per-class Widar target accuracy and mean max-softmax confidence "
        "(source-only, aggregated over n=15 runs).",
    ),
    (
        "coverage-matrix",
        "outputs/coverage/method_coverage_matrix.json",
        "method_summaries/coverage_matrix.json",
        "application/json",
        "Method x dataset coverage matrix.",
    ),
    (
        "split-widar-bvp",
        "data/prepared/widar_bvp/metadata.json",
        "splits/widar_bvp.json",
        "application/json",
        "Widar_BVP split manifest: leave-one-room-out folds and shape "
        "metadata. No raw CSI is redistributed.",
    ),
    (
        "split-ntufi-har",
        "data/prepared/ntufi_har/metadata.json",
        "splits/ntufi_har.json",
        "application/json",
        "NTU-Fi HAR split manifest. No raw CSI is redistributed.",
    ),
    (
        "split-signfi-top10",
        "data/prepared/signfi_top10/metadata.json",
        "splits/signfi_top10.json",
        "application/json",
        "SignFi-10 split manifest. No raw CSI is redistributed.",
    ),
    (
        "submission-manifest",
        "outputs/SUBMISSION_MANIFEST.md",
        "SUBMISSION_MANIFEST.md",
        "text/markdown",
        "Mapping from paper tables/figures to backing artifacts.",
    ),
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def build() -> dict:
    distribution: list[dict] = []
    for fid, src_path, hf_path, fmt, desc in ENTRIES:
        p = Path(src_path)
        if not p.exists():
            raise FileNotFoundError(p)
        distribution.append(
            {
                "@type": "cr:FileObject",
                "@id": fid,
                "name": fid,
                "contentUrl": hf_path,
                "encodingFormat": fmt,
                "sha256": sha256(p),
                "description": desc,
            }
        )

    return {
        "@context": {
            "@language": "en",
            "@vocab": "https://schema.org/",
            "citeAs": "cr:citeAs",
            "column": "cr:column",
            "conformsTo": "dct:conformsTo",
            "cr": "http://mlcommons.org/croissant/",
            "rai": "http://mlcommons.org/croissant/RAI/",
            "data": {"@id": "cr:data", "@type": "@json"},
            "dataType": {"@id": "cr:dataType", "@type": "@vocab"},
            "dct": "http://purl.org/dc/terms/",
            "examples": {"@id": "cr:examples", "@type": "@json"},
            "extract": "cr:extract",
            "field": "cr:field",
            "fileProperty": "cr:fileProperty",
            "fileObject": "cr:fileObject",
            "fileSet": "cr:fileSet",
            "format": "cr:format",
            "includes": "cr:includes",
            "isLiveDataset": "cr:isLiveDataset",
            "jsonPath": "cr:jsonPath",
            "key": "cr:key",
            "md5": "cr:md5",
            "parentField": "cr:parentField",
            "path": "cr:path",
            "recordSet": "cr:recordSet",
            "references": "cr:references",
            "regex": "cr:regex",
            "repeated": "cr:repeated",
            "replace": "cr:replace",
            "sc": "https://schema.org/",
            "separator": "cr:separator",
            "source": "cr:source",
            "subField": "cr:subField",
            "transform": "cr:transform",
        },
        "@type": "sc:Dataset",
        "name": "WiFi-TTA-Bench",
        "description": (
            "Harm-aware benchmark for test-time adaptation (TTA) on WiFi "
            "Channel State Information (CSI) under physics-structured domain "
            "shift. The benchmark evaluates up to 12 TTA methods (including "
            "TENT, SHOT/IM, T3A, SAR, CoTTA, LAME, and physics-informed "
            "variants) on three real WiFi CSI datasets and controlled "
            "synthetic data. It does not redistribute raw CSI; it provides "
            "split manifests, evaluation protocol artifacts, and canonical "
            "per-method JSON summaries."
        ),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "version": "1.0.0",
        "keywords": [
            "test-time adaptation",
            "domain adaptation",
            "WiFi sensing",
            "channel state information",
            "harm-aware evaluation",
            "physics-structured shift",
            "benchmark",
            "negative results",
        ],
        "license": (
            "CC-BY-4.0 (benchmark artifacts and code; raw datasets retain "
            "original academic licences)"
        ),
        "citeAs": "To be assigned upon acceptance.",
        "url": "https://huggingface.co/datasets/WiFi-TTA-Bench/wifi-tta-bench",
        "creator": {
            "@type": "Organization",
            "name": "WiFi-TTA-Bench",
        },
        "datePublished": "2026-04-18",
        "rai:dataCollection": (
            "Benchmark re-uses existing academic-licensed WiFi CSI datasets "
            "(Widar_BVP, NTU-Fi HAR, SignFi-10). No new human-subject data "
            "collected by the benchmark authors."
        ),
        "rai:dataCollectionType": (
            "secondary_reuse_of_existing_academic_datasets"
        ),
        "rai:dataCollectionRawData": (
            "Raw CSI recordings were collected by the original dataset "
            "authors under academic research protocols. WiFi-TTA-Bench does "
            "not redistribute the raw recordings; it releases only split "
            "manifests and evaluation artifacts."
        ),
        "rai:dataImputationProtocol": (
            "None. Missing samples are dropped at prepare-time."
        ),
        "rai:dataPreprocessingProtocol": (
            "Per-dataset preparation scripts in pinn4csi/data and "
            "scripts/prepare_*. Documented shapes: Widar_BVP (22,400), "
            "NTU-Fi (114,500), SignFi-10 (200,90). Cross-environment splits "
            "are leave-one-environment-out with 3 folds x 5 seeds = 15 "
            "paired observations per (dataset, method) cell."
        ),
        "rai:dataAnnotationProtocol": (
            "Labels inherited unchanged from upstream datasets."
        ),
        "rai:dataAnnotationPlatform": "N/A",
        "rai:dataAnnotationAnalysis": (
            "Label integrity verified by cardinality checks in automated "
            "tests (tests/ directory)."
        ),
        "rai:dataUseCases": [
            "Evaluation of test-time adaptation methods under physics-"
            "structured domain shift.",
            "Study of overconfident misclassification mechanisms.",
            "Development of harm-aware adaptation protocols (negative-"
            "adaptation rate, source drop).",
        ],
        "rai:dataBiases": (
            "WiFi CSI datasets were collected in academic laboratory "
            "settings with volunteer subjects; demographic and device-"
            "diversity coverage is limited. Two of three datasets (NTU-Fi, "
            "SignFi-10) use temporal splits within a single laboratory "
            "rather than verified physical-room changes; only Widar_BVP is "
            "a verified cross-room split."
        ),
        "rai:personalSensitiveInformation": (
            "None. No personally identifiable information. Original dataset "
            "authors collected CSI under informed consent with non-"
            "identifying activity labels. WiFi CSI can in principle enable "
            "covert monitoring; we redistribute only evaluation artifacts, "
            "not raw captures, under an acceptable-use notice."
        ),
        "rai:dataReleaseMaintenancePlan": (
            "Benchmark artifacts maintained on HuggingFace (WiFi-TTA-Bench "
            "org) and GitHub (suanlab/WiFi-TTA-Bench). Versioned releases "
            "via Zenodo DOI on acceptance. Croissant metadata regenerated "
            "on each release."
        ),
        "rai:dataSocialImpact": (
            "Research-only benchmark. Improved WiFi-based adaptation could "
            "in principle reduce deployment cost but also lower barriers to "
            "covert monitoring; we therefore couple method releases with "
            "negative-adaptation and source-drop reporting so deployers can "
            "detect silently-degrading behaviour."
        ),
        "distribution": distribution,
    }


def main() -> int:
    out = Path("outputs/croissant/wifi_tta_bench_metadata.json")
    payload = build()
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(f"Wrote {out}")
    print(f"  distribution entries: {len(payload['distribution'])}")
    print(f"  conformsTo: {payload['conformsTo']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
