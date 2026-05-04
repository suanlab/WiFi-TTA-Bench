#!/usr/bin/env bash
# Template for uploading the WiFi-TTA-Bench dataset card + Croissant metadata
# to a HuggingFace anonymous namespace for double-blind NeurIPS ED review.
#
# Prerequisites:
#   pip install -U "huggingface_hub[cli]"
#   huggingface-cli login   # use a freshly created anon account; do NOT use your real account
#
# This script is idempotent: re-running after edits will push only the changed files.

set -euo pipefail

# === CONFIGURE ===
# Choose an anon namespace and dataset name. After acceptance you will rename.
HF_NAMESPACE="${HF_NAMESPACE:-anonymous-ed2026}"
HF_REPO="${HF_REPO:-wifi-tta-bench}"
HF_TYPE="${HF_TYPE:-dataset}"   # dataset (not model)
# =================

REPO_ID="$HF_NAMESPACE/$HF_REPO"

if ! command -v huggingface-cli >/dev/null 2>&1; then
  echo "ERROR: huggingface-cli not found. Install with:"
  echo "  pip install -U 'huggingface_hub[cli]'"
  exit 1
fi

# 1. Create the repo (no-op if it already exists)
huggingface-cli repo create "$HF_REPO" \
  --type "$HF_TYPE" \
  --organization "$HF_NAMESPACE" \
  -y || true

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
huggingface-cli upload "$REPO_ID" "$STAGING" . \
  --repo-type "$HF_TYPE" \
  --commit-message "Initial WiFi-TTA-Bench dataset card + Croissant metadata (NeurIPS 2026 ED, anonymous)"

echo ""
echo "Uploaded to https://huggingface.co/datasets/$REPO_ID"
echo ""
echo "REMEMBER:"
echo "  - Update introduction.tex / checklist.tex with the actual URL if it differs."
echo "  - Keep this account anonymous until the camera-ready period."
echo "  - Verify the Croissant file passes https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker"
