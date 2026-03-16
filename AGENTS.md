# AGENTS.md — PINN4CSI

> Physics-Informed Neural Networks for WiFi CSI (Channel State Information) analysis.
> PyTorch-based deep learning research project.

## Project Overview

This project applies Physics-Informed Neural Networks (PINNs) to WiFi Channel State
Information (CSI) data for signal analysis, localization, and sensing applications.
PINNs embed physical laws (e.g., electromagnetic wave propagation, Friis equation,
multipath fading models) as soft constraints in the neural network loss function.

## Directory Structure

```
PINN4CSI/
├── pinn4csi/              # Main package
│   ├── models/            # Neural network architectures (PINN, baselines)
│   ├── physics/           # Physics equations, PDE residuals, loss terms
│   ├── data/              # Data loading, CSI parsing, preprocessing
│   ├── training/          # Training loops, optimizers, schedulers
│   ├── utils/             # Shared helpers (device, logging, metrics)
│   └── configs/           # Hydra/OmegaConf YAML configs
├── scripts/               # Entry-point scripts (train.py, evaluate.py)
├── notebooks/             # Jupyter notebooks for exploration & visualization
├── tests/                 # pytest test suite
│   ├── test_models.py
│   ├── test_physics.py
│   └── test_data.py
├── data/                  # Raw & processed datasets (gitignored)
├── outputs/               # Experiment outputs, checkpoints (gitignored)
├── pyproject.toml
└── AGENTS.md
```

## Build / Lint / Test Commands

```bash
# --- Environment ---
pip install -e ".[dev]"              # Install package in editable mode with dev deps
pip install -e ".[dev,notebook]"     # Include Jupyter extras

# --- Linting & Formatting ---
ruff check .                         # Lint (errors + warnings)
ruff check --fix .                   # Lint with autofix
ruff format .                        # Format (Black-compatible)
ruff format --check .                # Check formatting without changing files
mypy pinn4csi/                       # Type checking

# --- Testing ---
pytest                               # Run full test suite
pytest tests/test_models.py          # Run single test file
pytest tests/test_models.py::test_pinn_forward  # Run single test
pytest -k "test_physics"             # Run tests matching keyword
pytest -m "not slow"                 # Skip slow tests (default in CI)
pytest -m "not gpu"                  # Skip GPU-requiring tests
pytest -x                            # Stop on first failure
pytest --tb=short                    # Short traceback

# --- Training ---
python scripts/train.py              # Train with default config
python scripts/train.py model=pinn data=csi_indoor  # Hydra overrides
```

## Code Style

### Formatting & Linting
- **Formatter**: `ruff format` (Black-compatible, 88 char line width)
- **Linter**: `ruff check` with rules: E, F, W, I (isort), UP, N, B, SIM
- **Type checker**: `mypy` in strict mode for `pinn4csi/`
- **Line length**: 88 characters (Black default)
- **Quote style**: double quotes (`"string"`)

### Imports
```python
# 1. stdlib
import os
from pathlib import Path

# 2. third-party
import numpy as np
import torch
import torch.nn as nn
from torch import Tensor

# 3. local
from pinn4csi.models.pinn import PINN
from pinn4csi.physics.wave import wave_equation_residual
```
- Use `isort` profile (handled by `ruff`): stdlib → third-party → local, separated by blank lines
- Prefer `from torch import Tensor` for frequent use in type annotations
- Never use wildcard imports (`from module import *`)
- Use absolute imports within the package

### Type Annotations
```python
def train_step(
    model: nn.Module,
    x: Tensor,                          # Not torch.Tensor — use imported alias
    y: Tensor,
    device: torch.device | None = None, # Use union syntax (Python 3.10+)
) -> dict[str, float]:                  # Return type always specified
    ...
```
- **All public functions** must have full type annotations (params + return)
- Use `Tensor` (imported from torch) not `torch.Tensor` in annotations
- Use `torch.device | None` not `Optional[torch.device]`
- Use `dict`, `list`, `tuple` (lowercase) not `Dict`, `List`, `Tuple`
- Private helpers: annotations encouraged but not mandatory

### Naming Conventions
| Element          | Convention         | Example                            |
|------------------|--------------------|------------------------------------|
| Module/file      | `snake_case`       | `wave_propagation.py`              |
| Class            | `PascalCase`       | `PINNModel`, `CSIDataset`          |
| Function/method  | `snake_case`       | `compute_physics_loss`             |
| Constant         | `UPPER_SNAKE`      | `SPEED_OF_LIGHT`, `NUM_SUBCARRIERS`|
| Variable         | `snake_case`       | `csi_amplitude`, `path_loss`       |
| Tensor variable  | Short descriptive  | `x`, `h_pred`, `csi_complex`       |
| Private          | `_leading_under`   | `_parse_raw_csi`                   |
| Acronyms in class| Keep uppercase     | `CSIDataLoader`, `PINNTrainer`     |

### Error Handling
```python
# Good — specific exception, informative message
if csi_data.shape[-1] != num_subcarriers:
    raise ValueError(
        f"Expected {num_subcarriers} subcarriers, got {csi_data.shape[-1]}"
    )

# Bad — never use bare except or empty catch
try: ...
except: pass           # NEVER
except Exception: ...  # NEVER (too broad without re-raise)
```
- Raise specific exceptions: `ValueError`, `TypeError`, `FileNotFoundError`
- Never suppress exceptions with empty `except`/`catch` blocks
- Use `logging` module, never `print()` for diagnostic output

### Docstrings
```python
def compute_path_loss(
    distance: Tensor,
    frequency: float,
    n: float = 2.0,
) -> Tensor:
    """Compute free-space path loss using the log-distance model.

    Args:
        distance: Tx-Rx distances in meters. Shape: (batch,).
        frequency: Carrier frequency in Hz.
        n: Path loss exponent (default: 2.0 for free space).

    Returns:
        Path loss in dB. Shape: (batch,).
    """
```
- Google-style docstrings (Args/Returns/Raises sections)
- All public functions and classes must have docstrings
- Include tensor shapes in docstrings: `Shape: (batch, num_subcarriers)`

## PINN-Specific Conventions

### Model Architecture
```python
class PINN(nn.Module):
    """Physics-Informed Neural Network base class."""

    def __init__(self, in_features: int, out_features: int, hidden_dim: int = 64, num_layers: int = 4):
        super().__init__()
        # Build MLP layers with nn.ModuleList for flexibility
        ...

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass — data prediction only."""
        ...
```
- Separate **data prediction** (`forward`) from **physics computation**
- Physics residual computation lives in `pinn4csi/physics/`, not inside the model

### Loss Structure
```python
loss_data = mse_loss(pred, target)          # Data fidelity term
loss_physics = pde_residual(pred, coords)   # Physics constraint
loss_bc = boundary_loss(pred, bc_coords)    # Boundary conditions
loss = loss_data + lambda_phys * loss_physics + lambda_bc * loss_bc
```
- Always keep loss terms separate and logged individually
- Use configurable weighting coefficients (`lambda_phys`, `lambda_bc`)
- Log each loss component to experiment tracker (W&B / TensorBoard)

### Automatic Differentiation for PDEs
```python
# Use torch.autograd.grad with create_graph=True for higher-order derivatives
u = model(x)
du_dx = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                             create_graph=True, retain_graph=True)[0]
```
- Always use `create_graph=True` when computing PDE residuals during training
- Input tensors for PDE residual must have `requires_grad=True`

## CSI Data Conventions

- CSI data shape convention: `(num_packets, num_subcarriers, num_antennas)` — complex-valued
- Amplitude: `torch.abs(csi_complex)`, Phase: `torch.angle(csi_complex)`
- Always sanitize phase (unwrap or use amplitude-only when phase is noisy)
- Subcarrier indices are 0-based; document any mapping to actual subcarrier numbers
- Use `csiread` for Intel 5300 / Atheros / ESP32 raw CSI parsing

## Device Management
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
data = data.to(device)  # Move data in DataLoader or training loop, not in Dataset
```
- Never hardcode `"cuda"` — always check availability
- Move data to device in training loop, not in `Dataset.__getitem__`
- Support `CUDA_VISIBLE_DEVICES` for multi-GPU selection

## Testing Conventions
- Use `@pytest.mark.slow` for tests > 5 seconds (training loops, large data)
- Use `@pytest.mark.gpu` for tests requiring CUDA
- Prefer small synthetic data for unit tests, not real CSI datasets
- Test physics residual functions with known analytical solutions
- Test model forward/backward pass shapes with `torch.randn` inputs
