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

## Task 2: CSI Data Loader Implementation

### Dataset Design

#### Shape Convention
- Raw CSI: `(num_packets, num_subcarriers, num_antennas)` — complex-valued
- After amplitude/phase split: `(num_packets, num_subcarriers, 2*num_antennas)` — real-valued
- Per-sample output: `(num_subcarriers, features)` where features = 2*num_antennas (amplitude+phase)

#### Amplitude/Phase Separation
- `amplitude_phase_split()` converts complex CSI to real-valued amplitude and phase channels
- Stacks amplitude and phase as separate channels: `[amp_ch1, phase_ch1, amp_ch2, phase_ch2, ...]`
- Supports both 2D (single sample) and 3D (batch) tensors
- Enables real-valued neural network processing without complex-valued layers

### Dataset Class Features

#### CSIDataset
- Flexible file format support: `.npy`, `.pt`, `.npz`
- Optional amplitude/phase conversion (configurable)
- Per-sample metadata: labels (integer class IDs) and environments (for cross-env splits)
- Default labels/environments if not provided (single-class, single-env datasets)
- Transform pipeline support for preprocessing

#### Split Utilities
- `train_val_test_split()`: Stratified split with configurable ratios (default 70/15/15)
- `cross_environment_split()`: Environment-aware split with no overlap guarantee
- Both utilities use `RandomState` for reproducible shuffling
- Subset creation via `object.__new__()` to avoid file reloading

### Testing Strategy

#### Test Coverage (19 tests, 100% pass)
1. **Amplitude/Phase Transform** (3 tests)
   - 2D and 3D tensor shapes
   - Numerical correctness (amplitude and phase values)

2. **Normalization** (2 tests)
   - Output range [0, 1]
   - Shape preservation

3. **CSIDataset** (7 tests)
   - Initialization and shape validation
   - Label/environment handling (default and custom)
   - Complex vs. real-valued CSI modes
   - DataLoader iteration (one epoch)

4. **Split Utilities** (7 tests)
   - Train/val/test ratio validation
   - No sample overlap
   - Reproducibility with seed
   - Cross-environment split non-overlap guarantee
   - Multiple environment ID support

### Code Quality

#### Linting & Type Checking
- `ruff check`: All checks passed (0 errors)
- `mypy`: Strict mode, 0 errors in 3 source files
- Docstring line length: Fixed to 88 char limit
- Type annotations: Full coverage for public API

#### Design Decisions
1. **Subset Creation**: Used `object.__new__()` instead of reloading from file
   - Avoids redundant I/O
   - Preserves metadata (labels, environments)
   - Efficient memory usage

2. **Environment Metadata**: Separate from labels
   - Enables cross-environment evaluation
   - Supports multi-environment datasets
   - Flexible for future domain adaptation tasks

3. **Transform Pipeline**: Optional callable
   - Allows custom preprocessing (normalization, augmentation)
   - Decoupled from dataset class
   - Composable with other transforms

### Integration Points

#### Unblocks
- Task 6 (Feasibility experiment): Can now load real CSI data
- Task 10 (Paper 1 experiments): Multi-dataset support ready
- Task 15 (Paper 2 multi-environment): Cross-env split utilities in place

#### Dependencies
- Task 1 (Project scaffold): Provides package structure ✓
- No external dependencies beyond PyTorch/NumPy ✓


## Hydra Configuration Fix (Post-Task 3)

### Issue: Missing `_self_` in defaults list
- **Problem**: Hydra warning "In 'config': Defaults list is missing _self_"
- **Root cause**: Hydra 1.3+ requires explicit `_self_` placement in defaults
- **Solution**: Added `_self_` at end of defaults list in config.yaml
  ```yaml
  defaults:
    - model: pinn
    - data: default
    - _self_
  ```
- **Effect**: `_self_` at end ensures root config values are merged after includes, allowing command-line overrides to take precedence
- **Verification**: Both smoke tests run warning-free

### Hydra Composition Order
- **Before `_self_`**: Included configs (model/pinn.yaml, data/default.yaml) are merged
- **At `_self_`**: Root config values are merged (can override includes)
- **After `_self_`**: Command-line overrides applied (highest priority)
- **Result**: Clean composition with proper override precedence

## Task 4: Log-Distance Path Loss Model (TDD)

### Implementation Strategy

#### TDD Approach (RED → GREEN → REFACTOR)
1. **RED Phase**: Wrote 14 comprehensive tests covering:
   - Analytical validation against Friis free-space equation
   - Multiple distances and frequencies
   - Batched and scalar tensor inputs
   - Gradient flow (first and second-order)
   - Edge cases (zero/negative distances, reference distances)

2. **GREEN Phase**: Implemented `compute_path_loss()` function:
   - Formula: `PL(d) = PL(d0) + 10*n*log10(d/d0)`
   - Reference level anchored to Friis: `PL(d0) = 20*log10(4*pi*d0*f/c)`
   - Full type annotations: `Tensor | float` input, `Tensor` output
   - Autograd-compatible: gradients flow through distance parameter

3. **REFACTOR Phase**:
   - Added comprehensive Google-style docstring with examples
   - Implemented input validation (positive distance/reference_distance)
   - Exported from `pinn4csi.physics.__init__.py`
   - Fixed code style (ruff, mypy strict mode)

### Physics Model Details

#### Log-Distance Path Loss
- **Formula**: `PL(d) = PL(d0) + 10*n*log10(d/d0)` (dB)
- **Parameters**:
  - `n`: path loss exponent (2.0 for free space, 1.6-3.3 for indoor)
  - `d0`: reference distance (default 1.0m)
  - `f`: frequency (Hz)
  - `c`: speed of light (3e8 m/s)

#### Friis Anchoring
- Reference level computed from Friis free-space equation
- `PL(d0) = 20*log10(4*pi*d0*f/c)`
- Ensures physical consistency with electromagnetic theory
- At d=d0, path loss matches Friis prediction exactly

### Verification Results

#### Pytest (14/14 PASS)
- Analytical tests: Friis agreement at 1m, 10m, 100m, 1000m
- Gradient tests: First-order and second-order (create_graph=True)
- Edge cases: Zero/negative distance/reference_distance validation
- Batched inputs: Shape preservation and finite values

#### Type Checking (mypy strict)
- 0 errors in `pinn4csi/physics/path_loss.py`
- 0 errors in `pinn4csi/physics/__init__.py`
- Full type coverage: no `Any` types

#### Code Style (ruff)
- 0 errors in path_loss.py and test_physics.py
- Line length: 88 char limit
- Import order: stdlib → third-party → local

#### Analytical Verification (Python script)
- Friis equation: 5.99e-07 dB error at d=1m
- Path loss slope: 20.0 dB/decade (exact for n=2.0)
- Frequency dependence: 2.4GHz < 5.0GHz < 6.0GHz (correct ordering)
- Gradient flow: 3.94e-08 dB/m error vs analytical
- Second-order gradient: Finite and non-zero (PDE residual ready)

### Key Design Decisions

1. **Tensor Input Handling**:
   - Accept both `Tensor` and `float` for distance
   - Convert to float32 for consistency
   - Preserve input shape in output

2. **Validation Strategy**:
   - Check distance > 0 (raises ValueError)
   - Check reference_distance > 0 (raises ValueError)
   - Use `.any()` for batched validation

3. **Friis Anchoring**:
   - Compute reference level once per call
   - Use `torch.tensor()` for scalar constants
   - Ensures frequency-aware reference (not hardcoded)

4. **Autograd Support**:
   - All operations use torch functions (log10, tensor operations)
   - No numpy operations on distance tensor
   - Supports `create_graph=True` for PDE residuals

### Integration Points

#### Unblocks
- Task 5 (PINN model): Can now integrate path loss as physics constraint
- Task 6 (Feasibility experiment): Physics loss term ready
- Task 8 (OFDM model): Foundation for more complex physics

#### Dependencies
- Task 1 (Project scaffold): Package structure ✓
- Task 3 (Training loop): Trainer ready for physics loss ✓

### Lessons Learned

1. **Friis Equation Anchoring**: Using Friis as reference ensures physical consistency
   - Path loss exponent n=2.0 matches free-space theory
   - Frequency-dependent reference level (not constant)
   - Enables validation against known electromagnetic laws

2. **Gradient Flow for PDEs**: `create_graph=True` essential for physics constraints
   - Second-order gradients needed for PDE residuals
   - Test with both first and second-order to verify

3. **Type Annotations in Physics**: Full type coverage catches errors early
   - Union types (`Tensor | float`) for flexible inputs
   - Explicit return types for clarity
   - Mypy strict mode enforces consistency

4. **Test-Driven Physics**: TDD works well for physics modules
   - Analytical tests validate against known solutions
   - Gradient tests verify autograd compatibility
   - Edge case tests catch validation bugs early


### Device Safety in Physics Modules
- **Key Lesson**: Always create constant tensors on the same device/dtype as input tensors
- **Pattern**: `torch.tensor(value, dtype=input.dtype, device=input.device)`
- **Why**: Prevents device mismatch errors when inputs are on CUDA or have different dtypes
- **Testing**: Add explicit tests for dtype and device preservation (not just CPU float32)
- **Impact**: Enables seamless CUDA support and dtype flexibility without code changes

