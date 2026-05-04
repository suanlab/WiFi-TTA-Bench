#!/usr/bin/env bash
# Upload the WiFi-TTA-Bench dataset card + Croissant metadata + canonical
# evaluation artifacts to a HuggingFace dataset repository.
#
# Prerequisites:
#   pip install -U "huggingface_hub[cli]"   # provides the `hf` CLI
#   hf auth login                            # paste a write token
#
# This script is idempotent: re-running after edits will push only the
# changed files.

set -euo pipefail

# === CONFIGURE ===
HF_NAMESPACE="${HF_NAMESPACE:-WiFi-TTA-Bench}"
HF_REPO="${HF_REPO:-wifi-tta-bench}"
HF_TYPE="${HF_TYPE:-dataset}"   # dataset (not model)
HF_BIN="${HF_BIN:-hf}"           # override e.g. HF_BIN=venv/bin/hf
# =================

REPO_ID="$HF_NAMESPACE/$HF_REPO"

if ! command -v "$HF_BIN" >/dev/null 2>&1; then
  echo "ERROR: '$HF_BIN' not found. Install with:"
  echo "  pip install -U 'huggingface_hub[cli]'"
  echo "  hf auth login"
  exit 1
fi

# 1. Create the repo (no-op if it already exists)
"$HF_BIN" repo create "$REPO_ID" --repo-type "$HF_TYPE" -y || true

# 2. Stage upload contents under a temp dir so we control exactly what HF sees
STAGING="$(mktemp -d)"
trap 'rm -rf "$STAGING"' EXIT

cp outputs/croissant/HF_DATASET_CARD.md "$STAGING/README.md"
cp outputs/croissant/wifi_tta_bench_metadata.json "$STAGING/croissant.json"
cp outputs/SUBMISSION_MANIFEST.md "$STAGING/SUBMISSION_MANIFEST.md"

# Per-method canonical summaries are small JSON; mirror them for reviewer access.
mkdir -p "$STAGING/method_summaries"
cp outputs/widar_full_tta/method_summary.json   "$STAGING/method_summaries/widar.json"
cp outputs/ntufi_tta/method_summary.json        "$STAGING/method_summaries/ntufi.json"
cp outputs/signfi_tta/method_summary.json       "$STAGING/method_summaries/signfi.json"
cp outputs/new_methods/widar_lame_sar_cotta.json            "$STAGING/method_summaries/widar_lame_sar_cotta.json"
cp outputs/new_methods/widar_mlpbn_faithful_sar_cotta.json  "$STAGING/method_summaries/widar_mlpbn_faithful_sar_cotta.json"
cp outputs/new_methods/ntufi_har_lame_sar_cotta.json        "$STAGING/method_summaries/ntufi_lame_sar_cotta.json"
cp outputs/new_methods/signfi_top10_lame_sar_cotta.json     "$STAGING/method_summaries/signfi_lame_sar_cotta.json"
cp outputs/coverage/method_coverage_matrix.json             "$STAGING/method_summaries/coverage_matrix.json"
cp outputs/overconfidence/widar_per_class.json              "$STAGING/method_summaries/widar_per_class.json"
cp outputs/source_overfit/mitigation_pilot.json             "$STAGING/method_summaries/source_overfit_pilot.json"
cp outputs/final_ablations/arch_ablation_with_ci.json       "$STAGING/method_summaries/arch_ablation_with_ci.json"
cp outputs/phase1/faithful_tent_results.json                "$STAGING/method_summaries/faithful_tent_results.json"

# Split manifests (no raw CSI; just the metadata.json per dataset)
mkdir -p "$STAGING/splits"
cp data/prepared/widar_bvp/metadata.json    "$STAGING/splits/widar_bvp.json"
cp data/prepared/ntufi_har/metadata.json    "$STAGING/splits/ntufi_har.json"
cp data/prepared/signfi_top10/metadata.json "$STAGING/splits/signfi_top10.json"

# 3. Upload
"$HF_BIN" upload "$REPO_ID" "$STAGING" . \
  --repo-type "$HF_TYPE" \
  --commit-message "Initial WiFi-TTA-Bench dataset card + Croissant metadata + evaluation artifacts"

echo ""
echo "Uploaded to https://huggingface.co/datasets/$REPO_ID"
echo ""
echo "NEXT:"
echo "  - Verify the Croissant file passes https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker"
echo "  - Optional: tag the dataset version with 'hf tag $REPO_ID v0.1.0 --repo-type=$HF_TYPE'"
