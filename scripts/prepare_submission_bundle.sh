#!/usr/bin/env bash
# Build the anonymous supplementary-code bundle for a double-blind submission.
#
# The bundle is made with `git archive` from a commit, so it contains exactly
# the tracked files -- the same files as the public repository -- and nothing
# local-only (notes, drafts, manuscript sources, review correspondence) can leak
# into it by accident. The script then unpacks the bundle and runs the
# anonymisation audit on what was actually packed, failing if anything is found.
#
# Usage: bash scripts/prepare_submission_bundle.sh [REV]
#   REV defaults to HEAD. Uncommitted changes are NOT included, so commit first.
#
# Output: build/submission/wifi-tta-bench-code.zip (+ a file listing)

set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
REV="${1:-HEAD}"
BUILD="$ROOT/build/submission"
CODE_ZIP="$BUILD/wifi-tta-bench-code.zip"
PYTHON="${PYTHON:-$ROOT/venv/bin/python}"

cd "$ROOT"
mkdir -p "$BUILD"

if [ -n "$(git status --porcelain --untracked-files=no)" ]; then
  echo "WARNING: tracked files have uncommitted changes; they will not be bundled." >&2
fi

git archive --format=zip --prefix=wifi-tta-bench/ -o "$CODE_ZIP" "$REV"
unzip -l "$CODE_ZIP" > "$BUILD/wifi-tta-bench-code-listing.txt"
echo "Built $CODE_ZIP from $(git rev-parse --short "$REV")"
echo "Entries: $(tail -1 "$BUILD/wifi-tta-bench-code-listing.txt" | awk '{print $2}')"

# Audit the packed files, not the working tree.
CHECK_DIR="$(mktemp -d "$BUILD/audit.XXXXXX")"
unzip -q "$CODE_ZIP" -d "$CHECK_DIR"
if (cd "$CHECK_DIR/wifi-tta-bench" && "$PYTHON" scripts/anonymization_audit.py --paths .); then
  echo "Anonymisation audit passed on the bundle."
else
  echo "Anonymisation audit FAILED on the bundle; see findings above." >&2
  echo "Unpacked copy kept for inspection: $CHECK_DIR" >&2
  exit 1
fi
echo "Unpacked audit copy: $CHECK_DIR (safe to delete)"
