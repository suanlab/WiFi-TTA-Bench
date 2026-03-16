# Issues & Blockers — PINN4CSI Project Scaffold (Task 1)

## Resolved Issues

### Issue: System Python lacks pip
- **Status**: RESOLVED
- **Problem**: `python3 -m pip` failed with "No module named pip"
- **Root cause**: Debian/Ubuntu system Python has ensurepip disabled
- **Solution**: Created local venv with `python3 -m venv venv`
- **Lesson**: Always use venv for development; system Python is locked down

### Issue: Torch installation timeout
- **Status**: RESOLVED
- **Problem**: `pip install torch` timed out after 120s
- **Root cause**: Default PyPI index downloads large torch wheel (>500MB)
- **Solution**: Used CPU-specific wheel from PyTorch index: `--index-url https://download.pytorch.org/whl/cpu`
- **Lesson**: For large packages, use vendor-specific indices

### Issue: LSP diagnostics show unresolved imports in conftest.py
- **Status**: EXPECTED (not a blocker)
- **Problem**: LSP reports "Import numpy/pytest/torch could not be resolved"
- **Root cause**: LSP runs in system Python context, not venv
- **Impact**: None — tests run correctly in venv
- **Workaround**: Ignore LSP errors; rely on mypy in venv for type checking

## Potential Future Issues

### Issue: Torch CPU wheel size
- **Status**: MONITORING
- **Problem**: CPU wheel is ~190MB; may cause issues on slow networks
- **Mitigation**: Document torch installation in README
- **Alternative**: Consider torch as optional dependency for data-only workflows

### Issue: Mypy strict mode may be too strict
- **Status**: MONITORING
- **Problem**: Strict mode requires full type annotations everywhere
- **Mitigation**: Can relax with `[[tool.mypy.overrides]]` if needed
- **Decision**: Keep strict for now; relax only if it blocks development

### Issue: Pytest markers not yet used
- **Status**: INFORMATIONAL
- **Problem**: No tests exist yet; markers defined but unused
- **Impact**: None — markers will be used in Tasks 4-6
- **Action**: Document marker usage in test files when tests are added

## Non-Issues (Clarifications)

### "Empty __init__.py files"
- **Clarification**: This is intentional for Task 1 (scaffold only)
- **Not a problem**: Implementation code added in Tasks 2-3

### "No tests collected"
- **Clarification**: Expected — conftest.py exists but no test files yet
- **Not a problem**: Test files added in Tasks 4-6

### "LSP errors in conftest.py"
- **Clarification**: LSP runs in system Python, not venv
- **Not a problem**: Mypy in venv reports "Success: no issues found"

## Post-Verification Issues (Task 1 Refinement)

### Issue: Incomplete seed reset in conftest.py
- **Status**: RESOLVED
- **Problem**: Python's `random` module not reset; only NumPy and Torch
- **Root cause**: Oversight in initial implementation
- **Solution**: Added `import random` and `random.seed(42)` to fixture
- **Lesson**: Always reset all random sources for full reproducibility

### Issue: Subpackages not included in distribution
- **Status**: RESOLVED
- **Problem**: `packages = ["pinn4csi"]` excludes subpackages from built wheel
- **Root cause**: Setuptools requires explicit subpackage listing (or find_packages())
- **Solution**: Explicitly listed all 7 subpackages in pyproject.toml
- **Lesson**: Test distribution build, not just editable install

### Issue: Missing README.md
- **Status**: RESOLVED
- **Problem**: `pyproject.toml` referenced non-existent README.md
- **Root cause**: Oversight in initial config
- **Solution**: Created minimal README.md with quick start and structure
- **Lesson**: Validate all file references in config before committing

## Task 3: Base Training Loop + Hydra Configuration

### Resolved Issues

#### Issue: Model input dimension mismatch
- **Status**: RESOLVED
- **Problem**: Synthetic data shape (batch, 104) but model expected (batch, 52)
- **Root cause**: Synthetic data has 52 subcarriers × 2 features = 104 total features
- **Solution**: Dynamically calculate `actual_in_features = num_subcarriers * num_features` in train.py
- **Lesson**: Always match model input dimension to actual data shape

#### Issue: Hydra config composition not working
- **Status**: RESOLVED
- **Problem**: `python scripts/train.py model=pinn data=default` failed with "No match in defaults list"
- **Root cause**: Missing `defaults:` section in root config.yaml
- **Solution**: Added `defaults: [model: pinn, data: default]` to config.yaml
- **Lesson**: Hydra requires explicit defaults for config composition

#### Issue: Line length violations in trainer.py
- **Status**: RESOLVED
- **Problem**: Type annotations exceeded 88 character line limit
- **Solution**: Split long function signatures across multiple lines
- **Lesson**: Type annotations can be verbose; use line breaks for readability

### Potential Future Issues

#### Issue: Loss component logging is placeholder
- **Status**: MONITORING
- **Problem**: `loss_physics` is always 0.0 (no physics loss implemented yet)
- **Impact**: None for Task 3; will be addressed in Task 5
- **Mitigation**: Trainer structure supports physics loss integration

#### Issue: Synthetic data doesn't reflect real CSI properties
- **Status**: INFORMATIONAL
- **Problem**: Random Gaussian data doesn't capture CSI structure (frequency correlation, phase coherence)
- **Impact**: Smoke tests pass but don't validate physics constraints
- **Mitigation**: Real datasets used in Task 6 (feasibility experiment)

#### Issue: No learning rate scheduling
- **Status**: INFORMATIONAL
- **Problem**: Fixed learning rate throughout training
- **Impact**: None for Task 3; can be added later if needed
- **Mitigation**: Adam optimizer with default settings works for smoke tests

## Task 2: CSI Data Loader Implementation

### Resolved Issues

#### Issue: Subset creation reloading from file
- **Status**: RESOLVED
- **Problem**: `_create_subset()` was calling `CSIDataset.__init__()` with original file path, causing full data reload
- **Root cause**: Attempted to reuse constructor for subset creation
- **Solution**: Used `object.__new__()` to create instance without calling `__init__`, then set attributes directly
- **Impact**: Efficient subset creation without redundant I/O
- **Lesson**: For dataset subsets, bypass constructor to avoid reloading

#### Issue: Type annotation for Dataset generic
- **Status**: RESOLVED
- **Problem**: `class CSIDataset(Dataset)` missing type parameter
- **Root cause**: PyTorch Dataset is generic type `Dataset[T]`
- **Solution**: Changed to `class CSIDataset(Dataset[tuple[Tensor, int]])`
- **Impact**: Full type safety for mypy strict mode
- **Lesson**: Always specify generic type parameters for PyTorch classes

#### Issue: Line length in docstrings
- **Status**: RESOLVED
- **Problem**: Docstring lines exceeded 88 char limit (ruff E501)
- **Root cause**: Long parameter descriptions and shape specifications
- **Solution**: Wrapped long lines across multiple lines in docstrings
- **Impact**: All ruff checks pass
- **Lesson**: Docstrings count toward line length; break early

### Potential Future Issues

#### Issue: File format detection
- **Status**: MONITORING
- **Problem**: Only `.npy`, `.pt`, `.npz` supported; no `.h5`, `.mat`, `.csv`
- **Mitigation**: Can add loaders incrementally as needed
- **Decision**: Keep minimal for now; extend when real datasets require it

#### Issue: Memory efficiency for large datasets
- **Status**: MONITORING
- **Problem**: Entire CSI array loaded into memory in `__init__`
- **Mitigation**: Could implement lazy loading with memory mapping
- **Decision**: Current approach fine for public datasets (typically <1GB); revisit if needed

#### Issue: Complex-valued tensor handling
- **Status**: INFORMATIONAL
- **Problem**: PyTorch has limited complex tensor support (no complex convolution, etc.)
- **Mitigation**: Amplitude/phase split provides real-valued alternative
- **Decision**: Amplitude/phase split is primary path; complex mode for reference only


## Hydra Configuration Fix (Post-Task 3)

### Issue: Hydra warning about missing `_self_`
- **Status**: RESOLVED
- **Problem**: Running train.py emitted "Defaults list is missing _self_" warning
- **Root cause**: Hydra 1.3+ requires explicit `_self_` in defaults for proper composition
- **Solution**: Added `_self_` at end of defaults list in pinn4csi/configs/config.yaml
- **Verification**: Both smoke tests run without warnings
- **Lesson**: Always include `_self_` in Hydra defaults for explicit composition control
