# Learnings — PINN4CSI Project Scaffold (Task 1)

## Environment Setup

### Python & Virtual Environment
- System Python 3.12.3 available but lacks pip in system environment
- Solution: Created local venv with `python3 -m venv venv`
- Torch installation requires CPU-specific wheel from PyTorch index to avoid timeout
- Use `--index-url https://download.pytorch.org/whl/cpu` for faster torch installation

### Dependency Installation Order
1. Core build tools: setuptools, wheel, pip (via venv)
2. Dev tools: pytest, ruff, mypy, hydra-core, omegaconf, numpy
3. Heavy dependencies: torch (CPU version, separate step)
4. Package itself: `pip install -e .`

## Project Structure

### Package Layout
- Minimal scaffold approach: 7 empty `__init__.py` files for submodules
- Module docstrings (one-liner) sufficient for empty packages
- No implementation code in Task 1 — structure only

### Configuration Files
- `pyproject.toml`: Single source of truth for build, dependencies, tool config
- Ruff config: rules E,F,W,I,UP,N,B,SIM with line-length 88
- Mypy config: strict mode enabled for `pinn4csi/` package
- Pytest config: markers `slow` and `gpu` with default `-m "not slow and not gpu"`

## Testing & Verification

### Deterministic Seed Fixture
- `tests/conftest.py` uses `autouse=True` fixture to reset seeds before each test
- Resets: `np.random.seed(42)`, `torch.manual_seed(42)`, `torch.cuda.manual_seed(42)`
- Ensures reproducibility across test runs

### Verification Checklist
All acceptance criteria passed:
- ✅ `pip install -e ".[dev]"` succeeds
- ✅ `python -c "import pinn4csi"` succeeds
- ✅ `ruff check .` → 0 errors
- ✅ `mypy pinn4csi/` → 0 errors
- ✅ `pytest --co` → collected 0 items (no tests yet, but framework ready)

## Code Style Observations

### Docstrings
- Module-level docstrings are necessary for public API documentation
- Fixture docstrings explain purpose and behavior
- Keep docstrings concise for empty modules

### Type Annotations
- Empty `__init__.py` files require no type annotations
- `conftest.py` uses proper type hints: `None` return type, `torch.cuda.is_available()` check

### Imports
- Proper import order: stdlib → third-party → local (handled by ruff isort)
- No wildcard imports
- Absolute imports within package

## Next Steps (Task 2-3)

### Data Module (Task 2)
- Will implement `pinn4csi/data/csi_dataset.py` with PyTorch Dataset
- Shape convention: `(num_packets, num_subcarriers, num_antennas)` — complex-valued
- Amplitude/phase separation for real-valued processing

### Training Module (Task 3)
- Will implement `pinn4csi/training/trainer.py` with base training loop
- Hydra config structure in `pinn4csi/configs/`
- Entry point: `scripts/train.py`

### Physics Module (Task 4)
- TDD approach: write tests first, then implementation
- Start with log-distance path loss model
- Gradient flow verification with `create_graph=True`

## Post-Verification Fixes (Task 1 Refinement)

### Fix 1: Python random seed in conftest.py
- **Issue**: Task 1 required resetting Python, NumPy, and Torch seeds; only NumPy and Torch were reset
- **Fix**: Added `import random` and `random.seed(seed)` to deterministic_seed fixture
- **Impact**: Ensures full reproducibility across all random sources

### Fix 2: Subpackage inclusion in pyproject.toml
- **Issue**: `packages = ["pinn4csi"]` only includes root package; subpackages excluded from distribution
- **Fix**: Explicitly listed all subpackages: `packages = ["pinn4csi", "pinn4csi.models", "pinn4csi.physics", "pinn4csi.data", "pinn4csi.training", "pinn4csi.utils", "pinn4csi.configs"]`
- **Impact**: Ensures all submodules are included in built distributions

### Fix 3: README.md creation
- **Issue**: `pyproject.toml` referenced non-existent README.md
- **Fix**: Created minimal README.md with quick start, project structure, and development commands
- **Impact**: Resolves build warning; provides user-facing documentation

### Fix 4: Cache cleanup
- **Issue**: Generated `__pycache__`, `.mypy_cache`, `.ruff_cache` directories polluted repo
- **Fix**: Removed all cache directories with `find . -type d -name __pycache__ -exec rm -rf {} +`
- **Impact**: Clean scaffold state for Tasks 2-3

### Final Verification (Post-Fixes)
All acceptance criteria re-verified:
- ✅ `pip install -e ".[dev]"` succeeds with updated config
- ✅ All subpackage imports work: `import pinn4csi.{data,models,physics,training,utils,configs}`
- ✅ `ruff check .` → 0 errors
- ✅ `mypy pinn4csi/` → 0 errors (7 source files)
- ✅ `pytest --co` → collected 0 items (framework ready)

## Task 3: Base Training Loop + Hydra Configuration

### Trainer Architecture
- **BaseTrainer class**: Minimal, concrete implementation with train/eval loops
- **Loss component logging**: Separate tracking of `loss_data`, `loss_physics`, `loss_total`
- **Checkpoint management**: Save/load with epoch and optimizer state
- **Device handling**: Automatic CUDA/CPU selection via `get_device()` utility

### Metrics Module
- **accuracy()**: Handles both binary (1D) and multiclass (2D logits) predictions
- **nmse()**: Normalized Mean Squared Error with zero-division guard
- **f1_score()**: Binary classification F1 with threshold at 0.5

### Hydra Configuration Structure
- **Defaults pattern**: Use `defaults:` section to compose configs
  - `defaults: [model: pinn, data: default]` enables `model=pinn data=default` overrides
- **Config files**:
  - `config.yaml`: Root config with defaults + trainer/device/logging settings
  - `model/pinn.yaml`: Model-specific hyperparameters
  - `data/default.yaml`: Data loading and preprocessing settings
- **Synthetic data fallback**: Built into `train.py` for smoke testing without real datasets

### Training Script Design
- **Synthetic dataset creation**: `create_synthetic_dataset()` generates random CSI-like data
- **Simple MLP model**: `create_simple_model()` creates baseline architecture
- **Dynamic input dimension**: Adjusts model input based on `synthetic_num_subcarriers * synthetic_num_features`
- **Logging integration**: Uses Python logging with Hydra-compatible format

### Type Annotations
- **DataLoader typing**: `DataLoader[tuple[Tensor, Tensor]]` for proper generic typing
- **Return types**: All public functions have explicit return type annotations
- **Device handling**: `torch.device | None` union syntax (Python 3.10+)

### Testing Strategy
- **Unit tests**: Device selection, metrics computation, trainer initialization
- **Integration tests**: Full train/eval epoch with synthetic data
- **Checkpoint tests**: Save/load roundtrip verification
- **Smoke tests**: Marked with `@pytest.mark.slow` for full pipeline validation

### Verification Results
- ✅ `mypy pinn4csi/training/ pinn4csi/utils/ --strict` → 0 errors
- ✅ `ruff check pinn4csi/training/ pinn4csi/utils/ scripts/train.py` → 0 errors
- ✅ `pytest tests/test_training.py -v` → 13/13 passed
- ✅ `python scripts/train.py trainer.max_epochs=1 data.batch_size=16` → 1 epoch completed
- ✅ `python scripts/train.py model=pinn data=default trainer.max_epochs=1` → Hydra overrides work
- ✅ Checkpoints saved to `outputs/checkpoints/` (epoch_1.pt, final.pt)
- ✅ Loss components logged separately (loss_data, loss_physics, loss_total)

### Key Design Decisions
1. **Minimal trainer**: No W&B, TensorBoard, or distributed training yet
2. **Synthetic data by default**: Allows smoke tests without real datasets
3. **Separate loss components**: Enables future physics loss integration
4. **Hydra defaults pattern**: Cleaner config composition than manual overrides
5. **Type-strict mypy**: Catches errors early; no `Any` types

### Lessons Learned
- Hydra requires `defaults:` section for config composition to work
- DataLoader generic typing requires explicit type parameters in strict mypy
- Synthetic data shape must match model input dimension (subcarriers × features)
- Loss component logging is essential for debugging physics loss integration
