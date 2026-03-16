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
