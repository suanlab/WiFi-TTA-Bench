# Problems & Technical Debt — PINN4CSI Project Scaffold (Task 1)

## Current Technical Debt

### Problem: Virtual environment not in repo
- **Description**: venv/ directory created locally but not tracked
- **Impact**: Each developer must create their own venv
- **Mitigation**: Document venv setup in README (future task)
- **Priority**: LOW — standard practice

### Problem: No README or setup documentation
- **Description**: No instructions for setting up development environment
- **Impact**: Future developers may struggle with venv/pip setup
- **Mitigation**: Create README.md in Task 2 or 3
- **Priority**: MEDIUM — needed before sharing code

### Problem: No CI/CD pipeline
- **Description**: No GitHub Actions or similar for automated testing
- **Impact**: Manual verification required for each commit
- **Mitigation**: Defer to later phase (not in Phase 0 scope)
- **Priority**: LOW — out of scope for Task 1

## Unresolved Design Questions

### Question: Should torch be optional?
- **Current state**: torch is a core dependency
- **Consideration**: Data-only workflows might not need torch
- **Decision needed**: Keep as core or make optional?
- **Impact**: Affects installation size and complexity
- **Deferred to**: Task 2 (data module design)

### Question: Should configs be in pinn4csi/configs/ or separate?
- **Current state**: Directory structure created but no YAML files
- **Consideration**: Hydra can load from package or external directory
- **Decision needed**: Package-embedded vs. external configs?
- **Impact**: Affects reproducibility and customization
- **Deferred to**: Task 3 (training module design)

### Question: How to handle complex-valued tensors?
- **Current state**: No decision made
- **Consideration**: PyTorch lacks native complex support; need amplitude+phase or 2-channel real
- **Decision needed**: Representation strategy for CSI data?
- **Impact**: Affects data loader design and physics module
- **Deferred to**: Task 2 (data module design)

## Known Limitations

### Limitation: No type stubs for third-party packages
- **Description**: mypy may complain about untyped third-party imports
- **Mitigation**: Use `types-*` packages (e.g., types-PyYAML)
- **Status**: Partially addressed (types-PyYAML in dev extras)
- **Action**: Add more type stubs as needed

### Limitation: Pytest markers not enforced
- **Description**: Markers are defined but not validated
- **Mitigation**: CI/CD will enforce marker usage (future)
- **Status**: Acceptable for now
- **Action**: Document marker usage in test files

### Limitation: No pre-commit hooks
- **Description**: No automatic linting/formatting before commits
- **Mitigation**: Developers must run ruff/mypy manually
- **Status**: Acceptable for small team
- **Action**: Add pre-commit hooks in future (optional)

## Deferred Decisions

### Defer: Docker setup
- **Reason**: Not needed for Phase 0 (local development only)
- **Timeline**: Consider in Phase 2+ if needed

### Defer: W&B integration
- **Reason**: Not needed until experiments start (Task 6)
- **Timeline**: Add in Task 3 or later

### Defer: Multi-GPU support
- **Reason**: Not needed for initial development
- **Timeline**: Add in Phase 2+ if needed

### Defer: Distributed training
- **Reason**: Out of scope for PhD research
- **Timeline**: Unlikely to be needed
