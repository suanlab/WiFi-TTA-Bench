#!/usr/bin/env bash
# Build the anonymous submission bundle for NeurIPS 2026 ED track.
#
# Produces two artifacts under build/submission/:
#   - wifi-tta-bench-code.zip     : code + scripts + manuscript + Croissant
#                                   (drop into anonymous.4open.science)
#   - wifi-tta-bench-supp.zip     : same minus large data/prepared/*.npy,
#                                   plus the compiled main.pdf as supplementary
#
# Excludes:
#   - .git, .archive, .sisyphus, .omc, .coverage
#   - venv, node_modules, __pycache__
#   - outputs/* binary heavy / per-run logs
#   - data/prepared/*.npy (large prepared CSI arrays are NOT redistributed)
#
# Usage: bash scripts/prepare_submission_bundle.sh

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="$ROOT/build/submission"
mkdir -p "$BUILD"

cd "$ROOT"

# 1. Code bundle (everything reviewers need)
CODE_ZIP="$BUILD/wifi-tta-bench-code.zip"
rm -f "$CODE_ZIP"

zip -q -r "$CODE_ZIP" \
  pinn4csi \
  scripts \
  tests \
  manuscript/paper2 \
  outputs/croissant \
  outputs/widar_full_tta \
  outputs/ntufi_tta \
  outputs/signfi_tta \
  outputs/new_methods \
  outputs/final_ablations \
  outputs/phase1 \
  outputs/source_overfit \
  outputs/overconfidence \
  outputs/coverage \
  outputs/SUBMISSION_MANIFEST.md \
  data/prepared/*/metadata.json \
  README.md \
  SUBMISSION_README.md \
  Makefile \
  pyproject.toml \
  requirements-lock.txt \
  -x \
  '*/__pycache__/*' \
  '*/.pytest_cache/*' \
  'manuscript/paper2/.archive/*' \
  'manuscript/paper2/main.aux' \
  'manuscript/paper2/main.bbl' \
  'manuscript/paper2/main.blg' \
  'manuscript/paper2/main.fdb_latexmk' \
  'manuscript/paper2/main.fls' \
  'manuscript/paper2/main.log' \
  'manuscript/paper2/main.out' \
  'manuscript/paper2/*.aux' \
  'manuscript/paper2/*.bbl' \
  'manuscript/paper2/*.blg' \
  'manuscript/paper2/*.fdb_latexmk' \
  'manuscript/paper2/*.fls' \
  'manuscript/paper2/*.log' \
  'manuscript/paper2/*.out' \
  'manuscript/paper2/main_korean.*' \
  'manuscript/paper2/korean.*' \
  'manuscript/paper2/sections_ko/*' \
  'manuscript/paper2/checklist_korean.tex' \
  'manuscript/paper2/supplementary_korean.tex' \
  'manuscript/paper2/figures/*.png' \
  '*.pyc'

echo "Built code bundle: $CODE_ZIP"
ls -la "$CODE_ZIP"

# 2. Supplementary bundle (compiled PDF + Croissant + scripts only, smaller)
SUPP_ZIP="$BUILD/wifi-tta-bench-supp.zip"
rm -f "$SUPP_ZIP"

# Re-compile to make sure main.pdf is current
( cd manuscript/paper2 && pdflatex -interaction=batchmode main.tex >/dev/null 2>&1 && pdflatex -interaction=batchmode main.tex >/dev/null 2>&1 ) || true

zip -q -r "$SUPP_ZIP" \
  manuscript/paper2/main.pdf \
  outputs/croissant \
  outputs/SUBMISSION_MANIFEST.md \
  manuscript/paper2/SUBMISSION_MANIFEST.md \
  manuscript/paper2/ARTIFACT_MANIFEST.md \
  manuscript/paper2/ARTIFACT_README.md \
  SUBMISSION_README.md

echo "Built supplementary bundle: $SUPP_ZIP"
ls -la "$SUPP_ZIP"

# 3. Sanity print
echo ""
echo "Bundle contents (top entries):"
CODE_LISTING="$BUILD/wifi-tta-bench-code-listing.txt"
unzip -l "$CODE_ZIP" > "$CODE_LISTING"
sed -n '1,25p' "$CODE_LISTING"
echo "..."
echo ""
echo "Total entries in code bundle: $(tail -1 "$CODE_LISTING" | awk '{print $2}')"
echo ""
echo "NEXT STEPS:"
echo "  1. Run scripts/anonymization_audit.py and resolve any findings."
echo "  2. Run scripts/validate_croissant.py to lightweight-check the JSON."
echo "  3. Upload Croissant JSON to https://huggingface.co/spaces/JoaquinVanschoren/croissant-checker"
echo "  4. Upload $CODE_ZIP to https://anonymous.4open.science (creates the anon mirror)."
echo "  5. Update the anon URL in introduction.tex + checklist.tex (3 places) once anonymous.4open.science returns the final path."
echo "  6. Push outputs/croissant/HF_DATASET_CARD.md to a HuggingFace anon dataset namespace (see scripts/upload_huggingface.sh)."
echo "  7. Re-compile manuscript/paper2/main.tex and resubmit."
