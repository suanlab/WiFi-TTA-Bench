# PINN4CSI: Physics-Informed Neural Networks for WiFi CSI Analysis

Physics-Informed Neural Networks (PINNs) applied to WiFi Channel State Information (CSI) for signal analysis, localization, and sensing applications.

> **Status (2026-04-08):** The WiFi-TTA-Bench benchmark is complete with 3 real datasets (Widar\_BVP, NTU-Fi, SignFi-10), 10+ TTA methods, and n=15 paired evaluations per method. Paper-quality results are in `outputs/` and `.sisyphus/evidence/`. Self-collected hardware CSI collection remains pending.

---

## Environment Snapshot

A pinned snapshot of the verified environment is in [`requirements-lock.txt`](requirements-lock.txt).

- Python 3.12, CPU-only PyTorch 2.10.0
- All dev tools: ruff 0.15.6, mypy 1.19.1, pytest 9.0.2

---

## 1. Setup & Install

```bash
# Clone / enter the repo
cd PINN4CSI

# Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

# Install the package in editable mode with dev dependencies
pip install -e ".[dev]"

# Optional: Jupyter / notebook extras
pip install -e ".[dev,notebook]"

# Verify the package is importable
python -c "import pinn4csi; print(f'pinn4csi v{pinn4csi.__version__}')"
```

---

## 2. Full Verification (ruff · mypy · pytest)

Run all three checks from the repo root with the venv active.

```bash
# Lint — must report "All checks passed!"
ruff check .

# Format check — must report no changes needed
ruff format --check .

# Type checking — must report "Success: no issues found in N source files"
mypy pinn4csi/

# Test suite (excludes slow/gpu tests by default)
# Expected: 248 passed (as of 2026-04-08)
pytest --tb=short -q

# Collect-only sanity check (no execution)
pytest --co -q
```

All four commands must exit 0 before any code change is considered verified.

---

## 3. Feasibility Rerun

The feasibility script trains small OFDM-PINN and baseline models on **synthetic** data
and writes a CSV summary to `outputs/feasibility_ofdm_results.csv`.

```bash
python scripts/feasibility.py
```

Expected tail output (values are synthetic, not paper-quality):

```
Results written to: outputs/feasibility_ofdm_results.csv
Baseline mean test NMSE: ~1.33
Best OFDM-PINN mean test NMSE: ~1.16
Best OFDM-residual mean test NMSE: ~0.42
```

> **Note:** These numbers come from randomly generated synthetic CSI tensors.
> They demonstrate that the training loop runs end-to-end, not that the model
> achieves publication-level performance.

---

## 4. Paper 1 — Mock / Prepared-Data Harness

`scripts/run_paper1_experiments.py` runs the full Paper 1 experiment grid
(autoencoder, CNN, MLP, residual-prior models × SignFi / UT-HAR datasets ×
train-val-test / held-out benchmark-split evaluations).

### 4a. Quick smoke-test with auto-generated mock data

```bash
python scripts/run_paper1_experiments.py \
    --prepared-root /tmp/mock_p1 \
    --create-mock-data
```

Outputs:
- `outputs/paper1_results.csv` — per-run metrics
- `outputs/paper1_analysis.json` — aggregated analysis

### 4b. With real prepared data

```bash
# Convert your own local arrays/files into the Paper 1 prepared-data contract.
# The script does not download SignFi / UT-HAR for you.
python scripts/prepare_data.py \
    --dataset signfi \
    --features /data/signfi/csi.npy \
    --labels /data/signfi/labels.npy \
    --metadata /data/signfi/metadata.json \
    --output-dir /data/prepared

python scripts/prepare_data.py \
    --dataset ut_har \
    --features /data/ut_har/csi.npy \
    --labels /data/ut_har/labels.npy \
    --environments /data/ut_har/environments.npy \
    --metadata /data/ut_har/metadata.json \
    --output-dir /data/prepared

# SenseFi-style UT_HAR layout adapter (local extraction/manual download only):
# expects /data/UT_HAR/data/*.csv and /data/UT_HAR/label/*.csv,
# where each .csv contains a NumPy array saved with np.save.
python scripts/prepare_data.py \
    --dataset ut_har \
    --source-format ut_har_layout \
    --source-root /data/UT_HAR \
    --output-dir /data/prepared

# SignFi MAT adapter (best effort): requires scipy for MAT parsing.
# If scipy is unavailable or the MAT flavor is unsupported, convert manually
# and fall back to --source-format generic.
python scripts/prepare_data.py \
    --dataset signfi \
    --source-format signfi_mat \
    --mat-file /data/signfi/dataset_lab_276_dl.mat \
    --features-key features \
    --labels-key labels \
    --output-dir /data/prepared

python scripts/run_paper1_experiments.py \
    --prepared-root /data/prepared \
    --datasets signfi,ut_har \
    --seeds 0,1,2 \
    --epochs 50
```

> **Status:** `scripts/prepare_data.py` prepares the repo's expected bundle
> layout from user-supplied local files plus metadata. Raw public-dataset
> download/extraction is still manual, so from a clean clone the only fully
> self-contained end-to-end path remains `--create-mock-data`. This workspace
> may additionally contain prepared real-data bundles under `data/prepared/`.

### 4c. Unit-test harness (always runnable)

```bash
pytest tests/test_paper1_experiments.py -v
# Expected: 5 passed
```

---

## 5. Paper 2 — Synthetic / Multi-Environment Harness

Paper 2 tests domain-invariant and multi-environment generalisation using
**fully synthetic** CSI tensors (no real data required).

```bash
# Run the Paper 2 test harness
pytest tests/test_paper2_baselines.py -v
# Expected: 4 passed
```

Key scenarios covered:
- Single-environment train/test split
- Cross-environment generalisation (synthetic source → synthetic target)
- Domain-invariant feature learning baselines

> **Status:** WiFi-TTA-Bench evaluated on Widar\_BVP (3 rooms), NTU-Fi (3 sessions), SignFi-10 (3 splits), and controlled synthetic data. Self-collected hardware CSI remains pending.

---

## Project Structure

```
pinn4csi/
├── models/      # Neural network architectures (PINN, baselines)
├── physics/     # Physics equations and PDE residuals
├── data/        # Data loading and preprocessing
├── training/    # Training loops and optimization
├── utils/       # Shared utilities
└── configs/     # Hydra configuration files
scripts/
├── feasibility.py              # Synthetic feasibility sweep
├── run_paper1_experiments.py   # Paper 1 experiment grid
└── train.py                    # General Hydra training entry-point
tests/
├── test_paper1_experiments.py  # Paper 1 mock harness
├── test_paper2_baselines.py    # Paper 2 synthetic harness
└── ...                         # Unit tests for models, physics, data
```

---

## 6. External Work Execution

> **Contract vs. execution:** The repo ships the directory contracts, helper scripts,
> and tooling described below. **Actual data collection and external baseline runs
> are manual steps** — nothing here downloads data or runs hardware automatically.

### 6a. Initialize the expected directory tree

```bash
python scripts/init_external_worktree.py
```

Creates the full skeleton of `data/`, `outputs/`, and external-artifact directories
so that downstream scripts can assume the layout exists before any real data arrives.
Safe to re-run; existing files are never overwritten.

### 6b. Prepare real SignFi / UT_HAR data

After manually downloading the raw datasets, convert them into the repo's
prepared-data contract:

```bash
# SignFi (MAT file from the SignFi project page)
python scripts/prepare_data.py \
    --dataset signfi \
    --source-format signfi_mat \
    --mat-file /data/signfi/dataset_lab_276_dl.mat \
    --output-dir /data/prepared

# UT_HAR (SenseFi-style CSV layout)
python scripts/prepare_data.py \
    --dataset ut_har \
    --source-format ut_har_layout \
    --source-root /data/UT_HAR \
    --output-dir /data/prepared
```

See §4b above for the full set of `prepare_data.py` flags.

### 6c. Check readiness before a run

```bash
python scripts/audit_readiness.py
```

Prints a checklist of which prepared datasets, hardware captures, and external
baseline artifacts are present or missing. Exit code is non-zero if any required
item is absent, making it suitable as a pre-flight gate in CI or shell scripts.

### 6d. Where hardware captures go

| Source | Drop directory |
|--------|---------------|
| ESP32 CSI captures | `data/esp32/` |
| WiFi 6 (802.11ax) prepared captures | `data/wifi6/` |

Place raw `.pcap` / `.npy` / `.csv` files in the appropriate subdirectory.
`scripts/prepare_data.py` and the Paper harnesses will look there by default.

### 6e. Where external baseline artifacts go

| Baseline | Artifact directory |
|----------|--------------------|
| NeWRF results (metrics, checkpoints) | `outputs/newrf/` |
| GSRF results (metrics, checkpoints) | `outputs/gsrf/` |

After running NeWRF or GSRF externally, copy their output JSON / CSV / checkpoint
files into the corresponding directory. `scripts/audit_readiness.py` checks for
at least one result file in each location.

---

## Development

```bash
ruff check --fix .   # Lint with autofix
ruff format .        # Format (Black-compatible)
mypy pinn4csi/       # Type checking (strict)
pytest               # Full test suite
pytest -m "not slow and not gpu"  # Fast subset (default)
```

See [`AGENTS.md`](AGENTS.md) for detailed code style and conventions.
