# pyright: basic, reportMissingImports=false, reportUnknownVariableType=false

"""Physics equations and PDE residuals."""

from pinn4csi.physics.helmholtz import (
    helmholtz_residual,
    helmholtz_residual_loss,
    helmholtz_wavenumber,
)
from pinn4csi.physics.ofdm_channel import (
    ofdm_channel_response,
    ofdm_residual,
    subcarrier_correlation_loss,
)
from pinn4csi.physics.path_loss import compute_path_loss

__all__ = [
    "compute_path_loss",
    "helmholtz_wavenumber",
    "helmholtz_residual",
    "helmholtz_residual_loss",
    "ofdm_channel_response",
    "ofdm_residual",
    "subcarrier_correlation_loss",
]
