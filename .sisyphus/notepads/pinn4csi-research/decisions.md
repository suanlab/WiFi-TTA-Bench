# Decisions — PINN4CSI Project Scaffold (Task 1)

## Build System & Packaging

### Decision: setuptools + pyproject.toml
- **Rationale**: Standard Python packaging, no external build system needed
- **Alternative considered**: Poetry (rejected — adds complexity for greenfield project)
- **Config location**: Single `pyproject.toml` file with all tool configs

### Decision: Minimal scaffold (empty __init__.py files)
- **Rationale**: No implementation code in Task 1; structure only
- **Benefit**: Allows Tasks 2-3 to run in parallel without blocking on Task 1
- **Module docstrings**: One-liner per module for clarity

## Development Environment

### Decision: Local venv instead of system Python
- **Rationale**: System Python lacks pip; venv provides isolated environment
- **Setup**: `python3 -m venv venv` + activate before pip operations
- **Torch installation**: CPU-specific wheel to avoid timeout on large downloads

### Decision: Ruff + Mypy + Pytest (no additional tools)
- **Rationale**: Minimal, focused tooling per AGENTS.md
- **Ruff rules**: E,F,W,I,UP,N,B,SIM (no additional rules)
- **Mypy**: Strict mode for `pinn4csi/` package only
- **Pytest**: Markers `slow` and `gpu` for test categorization

## Testing & Reproducibility

### Decision: Deterministic seed fixture with autouse=True
- **Rationale**: Ensures reproducibility across all tests without explicit calls
- **Seed value**: 42 (arbitrary but fixed)
- **Scope**: Python, NumPy, PyTorch (CPU + CUDA if available)
- **Alternative considered**: Manual seed setting in each test (rejected — error-prone)

### Decision: Pytest markers for slow/GPU tests
- **Rationale**: CI/local development can skip expensive tests
- **Default behavior**: `-m "not slow and not gpu"` in pytest.ini_options
- **Usage**: `@pytest.mark.slow` and `@pytest.mark.gpu` decorators

## Code Style

### Decision: Google-style docstrings (per AGENTS.md)
- **Rationale**: Consistency with project conventions
- **Module docstrings**: One-liner for empty packages
- **Function docstrings**: Full Args/Returns/Raises (when implemented)

### Decision: Double quotes for strings (per AGENTS.md)
- **Rationale**: Black-compatible, consistent with ruff formatter
- **Applied to**: All docstrings and string literals

### Decision: Type annotations for public functions (per AGENTS.md)
- **Rationale**: Strict mypy mode requires full annotations
- **Applied to**: conftest.py fixture (return type `None`)

## .gitignore Strategy

### Decision: Comprehensive .gitignore
- **Rationale**: Prevent accidental commits of large/sensitive files
- **Entries**:
  - `data/` — raw/processed datasets
  - `outputs/` — experiment checkpoints, logs
  - `*.pt`, `*.pth` — PyTorch model files
  - `__pycache__/`, `*.pyc` — Python cache
  - `.pytest_cache/`, `.mypy_cache/`, `.ruff_cache/` — tool caches
  - `.venv/`, `venv/` — virtual environments
  - IDE files: `.idea/`, `.vscode/`

## Dependency Management

### Decision: Separate dev extras in pyproject.toml
- **Rationale**: Allows `pip install -e .` for production, `pip install -e ".[dev]"` for development
- **Dev extras**: pytest, pytest-cov, ruff, mypy, types-PyYAML
- **Optional extras**: notebook (jupyter, matplotlib, seaborn)

### Decision: Minimal core dependencies
- **Core**: torch, numpy, hydra-core, omegaconf
- **Rationale**: Only essentials for PINN framework; data loaders/models added in Tasks 2-3

## Next Phase Decisions (Deferred)

### Task 2 (Data Module)
- Will decide on CSI dataset format (complex vs. amplitude+phase)
- Will decide on public dataset priority (SignFi vs. UT-HAR vs. Widar3.0)

### Task 3 (Training Module)
- Will decide on Hydra config structure (flat vs. hierarchical)
- Will decide on checkpoint format (PyTorch native vs. custom)

### Task 4 (Physics Module)
- Will decide on TDD test structure (analytical solutions vs. numerical)
- Will decide on gradient computation strategy (autograd.grad vs. backward)

## Post-Verification Decisions (Task 1 Refinement)

### Decision: Explicit subpackage listing in pyproject.toml
- **Rationale**: Ensures all submodules are included in built distributions
- **Alternative considered**: Using `find_packages()` from setuptools (rejected — explicit is clearer for small project)
- **Benefit**: Clear visibility of package structure in config file

### Decision: Minimal README.md
- **Rationale**: Satisfies build requirement; provides quick start for users
- **Content**: Setup instructions, project structure, development commands
- **Scope**: Minimal — detailed docs deferred to later phases

### Decision: Cache cleanup strategy
- **Rationale**: Keep scaffold clean; caches regenerated on first run
- **Cleanup**: `__pycache__`, `.mypy_cache`, `.ruff_cache` removed
- **Benefit**: Clean state for Tasks 2-3; .gitignore prevents future pollution

## Task 3: Base Training Loop + Hydra Configuration

### Architectural Decisions

#### Decision: Separate loss components in trainer
- **Rationale**: Physics loss integration (Task 5) requires tracking individual loss terms
- **Implementation**: `loss_data`, `loss_physics`, `loss_total` logged separately
- **Trade-off**: Slightly more verbose logging, but essential for debugging
- **Future**: Enables adaptive lambda weighting and ablation studies

#### Decision: Synthetic data fallback in train.py
- **Rationale**: Allows smoke tests without real datasets (Task 2 may be parallel)
- **Implementation**: `create_synthetic_dataset()` generates random CSI-like tensors
- **Trade-off**: Synthetic data doesn't reflect real CSI properties
- **Future**: Real datasets integrated in Task 6

#### Decision: Hydra defaults pattern for config composition
- **Rationale**: Cleaner than manual config merging; enables `model=pinn data=default` syntax
- **Implementation**: `defaults: [model: pinn, data: default]` in root config
- **Trade-off**: Requires understanding Hydra's composition semantics
- **Future**: Enables easy experiment configuration in Task 6

#### Decision: Minimal trainer without W&B/TensorBoard
- **Rationale**: Follows "no premature abstractions" principle; logging via Python logger
- **Implementation**: `BaseTrainer` uses standard logging module
- **Trade-off**: No experiment tracking yet; can be added later
- **Future**: W&B integration in Phase 2 if needed

#### Decision: Type-strict mypy for training/utils modules
- **Rationale**: Catches errors early; aligns with AGENTS.md strict mode requirement
- **Implementation**: `mypy pinn4csi/training/ pinn4csi/utils/ --strict`
- **Trade-off**: More verbose type annotations
- **Future**: Maintains code quality as physics loss logic grows

### Configuration Design

#### Decision: Separate config files for model and data
- **Rationale**: Enables independent model/data experimentation
- **Files**: `model/pinn.yaml`, `data/default.yaml`
- **Trade-off**: More files to manage
- **Future**: Easy to add `model/baseline.yaml`, `data/signfi.yaml`, etc.

#### Decision: Synthetic data configuration in data/default.yaml
- **Rationale**: Allows tuning synthetic data properties without code changes
- **Parameters**: `synthetic_samples`, `synthetic_num_subcarriers`, `synthetic_num_features`
- **Trade-off**: Config duplication with code defaults
- **Future**: Replaced by real dataset configs in Task 2

### Testing Strategy

#### Decision: Separate test classes for device, metrics, trainer
- **Rationale**: Clear organization; each class tests one concern
- **Implementation**: `TestDeviceManagement`, `TestMetrics`, `TestBaseTrainer`, `TestTrainingSmoke`
- **Trade-off**: More test files
- **Future**: Easy to add physics loss tests in Task 5

#### Decision: Checkpoint save/load roundtrip test
- **Rationale**: Validates reproducibility; essential for long training runs
- **Implementation**: `test_checkpoint_save_load()` creates new trainer and loads state
- **Trade-off**: Requires temporary directory
- **Future**: Enables resuming training from checkpoints

## Task 2: CSI Data Loader Implementation

### Architectural Decisions

#### 1. Amplitude/Phase Split as Primary Representation
- **Decision**: Provide amplitude/phase split as default, complex mode as optional
- **Rationale**: 
  - PyTorch has limited complex tensor support (no complex convolution)
  - Amplitude/phase is standard in signal processing literature
  - Enables real-valued neural networks (MLP, CNN, etc.)
- **Trade-off**: Loses phase coherence information (mitigated by phase channel)
- **Reversibility**: Can reconstruct complex CSI from amplitude+phase if needed

#### 2. Separate Environment Metadata
- **Decision**: Store environment IDs separately from labels
- **Rationale**:
  - Enables cross-environment evaluation (train env A → test env B)
  - Supports multi-environment datasets without label collision
  - Aligns with domain adaptation literature
- **Trade-off**: Requires two metadata arrays instead of one
- **Future**: Can extend to other metadata (time, location, etc.)

#### 3. Flexible File Format Support
- **Decision**: Support `.npy`, `.pt`, `.npz` with auto-detection
- **Rationale**:
  - Covers most public CSI datasets (SignFi, UT-HAR, Widar3.0)
  - Simple format detection via file extension
  - Extensible for future formats
- **Trade-off**: No support for `.h5`, `.mat`, `.csv` yet
- **Future**: Add loaders incrementally as needed

#### 4. Subset Creation via `object.__new__()`
- **Decision**: Bypass `__init__()` when creating subsets
- **Rationale**:
  - Avoids redundant file I/O
  - Preserves all metadata (labels, environments, transforms)
  - Efficient for large datasets
- **Trade-off**: Less conventional than calling constructor
- **Lesson**: Learned from initial implementation that reloaded files

#### 5. Transform Pipeline Support
- **Decision**: Optional callable transform in `__getitem__()`
- **Rationale**:
  - Decouples preprocessing from dataset class
  - Composable with other transforms (torchvision.transforms style)
  - Enables custom augmentation strategies
- **Trade-off**: Transform applied per-sample (slower than batch preprocessing)
- **Future**: Can add batch-level transforms if needed

### Data Format Conventions

#### CSI Shape Convention (from AGENTS.md)
- Raw: `(num_packets, num_subcarriers, num_antennas)` — complex-valued
- After split: `(num_packets, num_subcarriers, 2*num_antennas)` — real-valued
- Per-sample: `(num_subcarriers, features)` where features = 2*num_antennas

#### Label Convention
- Integer class IDs (0-indexed)
- No one-hot encoding (handled by loss function)
- Supports multi-class classification

#### Environment Convention
- Integer environment IDs (0-indexed)
- Used for cross-environment split
- Enables leave-one-environment-out evaluation


## Task 4: Path Loss Model Design Decisions

### Physics Model Choice: Log-Distance Path Loss
- **Decision**: Start with log-distance model (simplest useful model)
- **Rationale**: 
  - Captures distance-dependent attenuation (primary effect)
  - Frequency-aware via Friis anchoring
  - Analytically tractable (easy to validate)
  - Foundation for more complex models (OFDM, Helmholtz)
- **Alternative Considered**: OFDM H(f) model
  - Deferred to Task 8 (more complex, requires multipath parameters)

### Reference Distance Anchoring: Friis Equation
- **Decision**: Use Friis free-space equation for reference level
- **Rationale**:
  - Physically grounded in electromagnetic theory
  - Frequency-dependent (not hardcoded constant)
  - Enables validation against known solutions
  - Matches free-space behavior at n=2.0
- **Formula**: `PL(d0) = 20*log10(4*pi*d0*f/c)`
- **Alternative Considered**: Fixed reference level (e.g., -30 dB at 1m)
  - Would lose frequency dependence
  - Harder to validate against theory

### Input Type Flexibility: `Tensor | float`
- **Decision**: Accept both Tensor and float for distance
- **Rationale**:
  - Convenience for scalar inputs (e.g., single distance)
  - Flexibility for batched inputs (e.g., multiple distances)
  - Automatic conversion to float32 for consistency
- **Implementation**: Check `isinstance(distance, Tensor)` and convert if needed

### Validation Strategy: Explicit Error Messages
- **Decision**: Raise ValueError with specific error messages
- **Rationale**:
  - Catches invalid inputs early (distance ≤ 0)
  - Clear error messages for debugging
  - Prevents silent NaN/Inf propagation
- **Errors Raised**:
  - `ValueError` if distance contains non-positive values
  - `ValueError` if reference_distance ≤ 0

### Autograd Support: Full Gradient Flow
- **Decision**: Ensure gradients flow through distance parameter
- **Rationale**:
  - Required for PINN physics loss computation
  - Enables second-order gradients (create_graph=True)
  - Necessary for PDE residual calculations
- **Implementation**: Use only torch operations (no numpy on distance tensor)

### Export Strategy: Clean Public API
- **Decision**: Export only `compute_path_loss` from `pinn4csi.physics`
- **Rationale**:
  - Single entry point for users
  - Hides implementation details
  - Enables future refactoring without breaking API
- **Alternative Considered**: Export module directly
  - Would expose internal structure
  - Harder to maintain backward compatibility

