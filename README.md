# PINN4CSI: Physics-Informed Neural Networks for WiFi CSI Analysis

Physics-Informed Neural Networks (PINNs) applied to WiFi Channel State Information (CSI) for signal analysis, localization, and sensing applications.

## Quick Start

### Setup

```bash
python3 -m venv venv
source venv/bin/activate
python -m pip install -e ".[dev]"
```

### Verify Installation

```bash
python -c "import pinn4csi; print(f'pinn4csi v{pinn4csi.__version__}')"
ruff check .
mypy pinn4csi/
pytest --co
```

## Project Structure

```
pinn4csi/
├── models/      # Neural network architectures
├── physics/     # Physics equations and PDE residuals
├── data/        # Data loading and preprocessing
├── training/    # Training loops and optimization
├── utils/       # Shared utilities
└── configs/     # Hydra configuration files
```

## Development

- **Linting**: `ruff check .` and `ruff format .`
- **Type checking**: `mypy pinn4csi/`
- **Testing**: `pytest` (markers: `slow`, `gpu`)

See `AGENTS.md` for detailed code style and conventions.
