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
