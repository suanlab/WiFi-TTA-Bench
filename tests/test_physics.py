"""Tests for physics modules (path loss, OFDM, etc.)."""

# pyright: reportMissingImports=false

import numpy as np
import pytest
import torch
from torch import Tensor

from pinn4csi.physics.ofdm_channel import (
    ofdm_channel_response,
    ofdm_residual,
    subcarrier_correlation_loss,
)
from pinn4csi.physics.path_loss import compute_path_loss


class TestPathLossAnalytical:
    """Validate path loss against analytical solutions (Friis/free-space)."""

    def test_path_loss_analytical_single_distance(self) -> None:
        """Test path loss at a single distance matches Friis formula.

        Friis free-space path loss:
            PL(d) = 20*log10(4*pi*d*f/c)

        Log-distance model with n=2.0 (free-space):
            PL(d) = PL(d0) + 10*n*log10(d/d0)

        At d0=1m, f=2.4GHz, n=2.0:
            PL(1m) = 20*log10(4*pi*1*2.4e9/3e8) ≈ 40.04 dB
        """
        distance = torch.tensor([1.0], dtype=torch.float32)
        frequency = 2.4e9  # 2.4 GHz
        n = 2.0

        # Compute path loss
        pl = compute_path_loss(distance, frequency, n=n, reference_distance=1.0)

        # Friis formula: PL(d) = 20*log10(4*pi*d*f/c)
        c = 3e8  # speed of light
        friis_pl = 20 * np.log10(4 * np.pi * 1.0 * frequency / c)

        # At d0=1m with n=2.0, path loss should match Friis
        assert torch.allclose(
            pl,
            torch.tensor([friis_pl], dtype=torch.float32),
            atol=1e-5,
        ), f"Path loss at d=1m: {pl.item():.6f} dB, Friis: {friis_pl:.6f} dB"

    def test_path_loss_analytical_multiple_distances(self) -> None:
        """Test path loss at multiple distances.

        At d0=1m, f=2.4GHz, n=2.0:
            PL(d) = PL(1m) + 20*log10(d)
        """
        distances = torch.tensor([1.0, 10.0, 100.0], dtype=torch.float32)
        frequency = 2.4e9
        n = 2.0

        pl = compute_path_loss(distances, frequency, n=n, reference_distance=1.0)

        # Friis at d=1m
        c = 3e8
        pl_ref = 20 * np.log10(4 * np.pi * 1.0 * frequency / c)

        # Expected: PL(d) = PL(1m) + 20*log10(d)
        expected = torch.tensor(
            [
                pl_ref,
                pl_ref + 20 * np.log10(10.0),
                pl_ref + 20 * np.log10(100.0),
            ],
            dtype=torch.float32,
        )

        assert torch.allclose(pl, expected, atol=1e-5), (
            f"Path loss: {pl}, Expected: {expected}"
        )

    def test_path_loss_analytical_different_frequencies(self) -> None:
        """Test path loss at different frequencies.

        At d=10m, d0=1m, n=2.0:
            PL(d) = PL(d0) + 20*log10(d)

        Frequency affects PL(d0) but not the slope.
        """
        distance = torch.tensor([10.0])
        frequencies = [2.4e9, 5.0e9, 6.0e9]  # 2.4, 5, 6 GHz
        n = 2.0
        d0 = 1.0

        c = 3e8
        pls = []
        for freq in frequencies:
            pl = compute_path_loss(distance, freq, n=n, reference_distance=d0)
            pls.append(pl.item())

            # Verify: PL(d) = PL(d0) + 20*log10(d/d0)
            pl_d0 = 20 * np.log10(4 * np.pi * d0 * freq / c)
            expected = pl_d0 + 20 * np.log10(distance.item() / d0)

            assert abs(pl.item() - expected) < 1e-5, (
                f"At f={freq / 1e9:.1f}GHz: {pl.item():.6f} dB, "
                f"Expected: {expected:.6f} dB"
            )

        # Higher frequency → higher path loss (at same distance)
        assert pls[0] < pls[1] < pls[2], (
            f"Path loss should increase with frequency: {pls}"
        )

    def test_path_loss_batched_input(self) -> None:
        """Test path loss with batched distance tensor."""
        distances = torch.tensor([[1.0, 10.0], [100.0, 1000.0]])
        frequency = 2.4e9
        n = 2.0

        pl = compute_path_loss(distances, frequency, n=n, reference_distance=1.0)

        # Shape should match input
        assert pl.shape == distances.shape, (
            f"Output shape {pl.shape} != input shape {distances.shape}"
        )

        # All values should be finite
        assert torch.isfinite(pl).all(), "Path loss contains NaN or Inf"

    def test_path_loss_scalar_input(self) -> None:
        """Test path loss with scalar distance."""
        distance = 10.0
        frequency = 2.4e9
        n = 2.0

        pl = compute_path_loss(distance, frequency, n=n, reference_distance=1.0)

        # Should return scalar tensor
        assert isinstance(pl, Tensor), f"Expected Tensor, got {type(pl)}"
        assert pl.shape == torch.Size([]), (
            f"Expected scalar tensor, got shape {pl.shape}"
        )


class TestPathLossGradientFlow:
    """Validate gradient flow through path loss computation."""

    def test_path_loss_gradient_basic(self) -> None:
        """Test that gradients flow through distance parameter."""
        distance = torch.tensor([10.0], requires_grad=True)
        frequency = 2.4e9

        pl = compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

        # Compute gradient
        pl.backward()

        # Gradient should exist and be non-zero
        assert distance.grad is not None, "Gradient is None"
        assert distance.grad.item() != 0.0, "Gradient is zero"
        assert torch.isfinite(distance.grad).all(), "Gradient contains NaN or Inf"

    def test_path_loss_gradient_with_create_graph(self) -> None:
        """Test second-order gradients (for PDE residuals)."""
        distance = torch.tensor([10.0], requires_grad=True)
        frequency = 2.4e9

        pl = compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

        # First-order gradient with create_graph=True
        grad_pl = torch.autograd.grad(
            pl.sum(),
            distance,
            create_graph=True,
            retain_graph=True,
        )[0]

        assert grad_pl is not None, "First-order gradient is None"
        assert torch.isfinite(grad_pl).all(), "First-order gradient contains NaN or Inf"

        # Second-order gradient (for PDE residuals)
        grad2_pl = torch.autograd.grad(
            grad_pl.sum(),
            distance,
            create_graph=True,
        )[0]

        assert grad2_pl is not None, "Second-order gradient is None"
        assert torch.isfinite(grad2_pl).all(), (
            "Second-order gradient contains NaN or Inf"
        )

    def test_path_loss_gradient_batched(self) -> None:
        """Test gradients with batched input."""
        distances = torch.tensor(
            [[1.0, 10.0], [100.0, 1000.0]],
            requires_grad=True,
        )
        frequency = 2.4e9

        pl = compute_path_loss(distances, frequency, n=2.0, reference_distance=1.0)
        loss = pl.sum()
        loss.backward()

        # All gradients should exist and be finite
        assert distances.grad is not None, "Gradient is None"
        assert torch.isfinite(distances.grad).all(), "Gradient contains NaN or Inf"
        assert (distances.grad != 0.0).any(), "All gradients are zero"

    def test_path_loss_gradient_analytical(self) -> None:
        """Verify gradient matches analytical derivative.

        d/dd [PL(d)] = d/dd [PL(d0) + 10*n*log10(d/d0)]
                     = 10*n / (d * ln(10))
        """
        distance = torch.tensor([10.0], requires_grad=True)
        frequency = 2.4e9
        n = 2.0
        d0 = 1.0

        pl = compute_path_loss(distance, frequency, n=n, reference_distance=d0)
        pl.backward()

        # Analytical gradient
        grad_analytical = 10 * n / (distance.item() * np.log(10))

        assert abs(distance.grad.item() - grad_analytical) < 1e-5, (
            f"Gradient {distance.grad.item():.6f} != analytical {grad_analytical:.6f}"
        )


class TestPathLossEdgeCases:
    """Test edge cases and error handling."""

    def test_path_loss_zero_distance_raises(self) -> None:
        """Test that zero distance raises ValueError."""
        distance = torch.tensor([0.0])
        frequency = 2.4e9

        with pytest.raises(ValueError, match="distance.*positive"):
            compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

    def test_path_loss_negative_distance_raises(self) -> None:
        """Test that negative distance raises ValueError."""
        distance = torch.tensor([-10.0])
        frequency = 2.4e9

        with pytest.raises(ValueError, match="distance.*positive"):
            compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

    def test_path_loss_zero_reference_distance_raises(self) -> None:
        """Test that zero reference distance raises ValueError."""
        distance = torch.tensor([10.0])
        frequency = 2.4e9

        with pytest.raises(ValueError, match="reference_distance.*positive"):
            compute_path_loss(distance, frequency, n=2.0, reference_distance=0.0)

    def test_path_loss_negative_reference_distance_raises(self) -> None:
        """Test that negative reference distance raises ValueError."""
        distance = torch.tensor([10.0])
        frequency = 2.4e9

        with pytest.raises(ValueError, match="reference_distance.*positive"):
            compute_path_loss(distance, frequency, n=2.0, reference_distance=-1.0)

    def test_path_loss_different_reference_distances(self) -> None:
        """Test path loss with different reference distances.

        PL(d) = PL(d0) + 10*n*log10(d/d0)

        The absolute path loss value depends on d0, but the physical
        relationship should be consistent. At d=d0, the log term is 0.
        """
        distance = torch.tensor([10.0])
        frequency = 2.4e9
        n = 2.0

        pl_d0_1 = compute_path_loss(distance, frequency, n=n, reference_distance=1.0)
        pl_d0_10 = compute_path_loss(distance, frequency, n=n, reference_distance=10.0)

        # Both should give the same physical path loss (same distance, same frequency)
        # because the reference distance only affects the reference level, not the
        # actual propagation physics
        assert torch.allclose(pl_d0_1, pl_d0_10, atol=1e-5), (
            f"Path loss should be independent of reference distance choice: "
            f"d0=1m: {pl_d0_1.item():.6f} dB, "
            f"d0=10m: {pl_d0_10.item():.6f} dB"
        )


class TestPathLossDeviceSafety:
    """Verify device and dtype consistency."""

    def test_path_loss_preserves_dtype_float32(self) -> None:
        """Test that output dtype matches input dtype (float32)."""
        distance = torch.tensor([10.0], dtype=torch.float32)
        frequency = 2.4e9

        pl = compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

        assert pl.dtype == torch.float32, (
            f"Output dtype {pl.dtype} != input dtype {distance.dtype}"
        )

    def test_path_loss_preserves_dtype_float64(self) -> None:
        """Test that output dtype matches input dtype (float64)."""
        distance = torch.tensor([10.0], dtype=torch.float64)
        frequency = 2.4e9

        pl = compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

        assert pl.dtype == torch.float64, (
            f"Output dtype {pl.dtype} != input dtype {distance.dtype}"
        )

    def test_path_loss_preserves_device_cpu(self) -> None:
        """Test that output device matches input device (CPU)."""
        distance = torch.tensor([10.0], device="cpu")
        frequency = 2.4e9

        pl = compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

        assert pl.device == distance.device, (
            f"Output device {pl.device} != input device {distance.device}"
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="CUDA not available",
    )
    def test_path_loss_preserves_device_cuda(self) -> None:
        """Test that output device matches input device (CUDA)."""
        distance = torch.tensor([10.0], device="cuda")
        frequency = 2.4e9

        pl = compute_path_loss(distance, frequency, n=2.0, reference_distance=1.0)

        assert pl.device == distance.device, (
            f"Output device {pl.device} != input device {distance.device}"
        )

    def test_path_loss_batched_dtype_consistency(self) -> None:
        """Test dtype consistency with batched input."""
        distances = torch.tensor(
            [[1.0, 10.0], [100.0, 1000.0]],
            dtype=torch.float64,
        )
        frequency = 2.4e9

        pl = compute_path_loss(distances, frequency, n=2.0, reference_distance=1.0)

        assert pl.dtype == distances.dtype, (
            f"Output dtype {pl.dtype} != input dtype {distances.dtype}"
        )
        assert pl.device == distances.device, (
            f"Output device {pl.device} != input device {distances.device}"
        )


class TestOFDMChannelModel:
    """Validate OFDM channel response and residual constraints."""

    def test_ofdm_channel_model_two_path_analytical(self) -> None:
        """Test two-path OFDM response against a hand-computed result."""
        h_l = torch.tensor([1.0 + 0.0j, 0.5 - 0.25j], dtype=torch.complex64)
        tau_l = torch.tensor([0.0, 2.5e-7], dtype=torch.float32)
        f_k = torch.tensor([0.0, 1.0e6, 2.0e6], dtype=torch.float32)

        response = ofdm_channel_response(h_l=h_l, tau_l=tau_l, f_k=f_k)

        expected_values = []
        for frequency in f_k.tolist():
            exponent = np.exp(-1j * 2.0 * np.pi * frequency * tau_l.numpy())
            expected_values.append(np.sum(h_l.numpy() * exponent))
        expected = torch.tensor(expected_values, dtype=torch.complex64)

        assert response.shape == f_k.shape
        assert response.dtype == torch.complex64
        assert torch.allclose(response, expected, atol=1e-5)
        assert torch.isfinite(response.real).all()
        assert torch.isfinite(response.imag).all()

    def test_ofdm_channel_model_batched_paths(self) -> None:
        """Test batched multipath inputs produce batched subcarrier responses."""
        h_l = torch.tensor(
            [[1.0 + 0.0j, 0.4 + 0.2j], [0.7 - 0.1j, 0.2 + 0.3j]],
            dtype=torch.complex64,
        )
        tau_l = torch.tensor(
            [[0.0, 1.0e-7], [0.0, 1.5e-7]],
            dtype=torch.float32,
        )
        f_k = torch.tensor([0.0, 3.0e5, 6.0e5], dtype=torch.float32)

        response = ofdm_channel_response(h_l=h_l, tau_l=tau_l, f_k=f_k)

        assert response.shape == (2, 3)
        assert response.dtype == torch.complex64
        assert torch.isfinite(response.real).all()
        assert torch.isfinite(response.imag).all()

    def test_ofdm_channel_model_gradient_flow_path_gains_and_delays(self) -> None:
        """Test gradients flow through path gains and delays with create_graph."""
        h_real = torch.tensor([1.0, 0.35], dtype=torch.float64, requires_grad=True)
        h_imag = torch.tensor([0.1, -0.2], dtype=torch.float64, requires_grad=True)
        h_l = torch.complex(h_real, h_imag)
        tau_l = torch.tensor([1.0e-8, 2.0e-7], dtype=torch.float64, requires_grad=True)
        f_k = torch.tensor([1.0e6, 2.0e6, 3.0e6], dtype=torch.float64)

        response = ofdm_channel_response(h_l=h_l, tau_l=tau_l, f_k=f_k)
        objective = response.real.sum() + response.imag.sum()

        grad_h_real, grad_h_imag, grad_tau = torch.autograd.grad(
            objective,
            (h_real, h_imag, tau_l),
            create_graph=True,
        )

        assert grad_h_real is not None
        assert grad_h_imag is not None
        assert grad_tau is not None
        assert torch.isfinite(grad_h_real).all()
        assert torch.isfinite(grad_h_imag).all()
        assert torch.isfinite(grad_tau).all()

    def test_ofdm_residual_matches_analytical_response(self) -> None:
        """Test residual is near zero when prediction equals analytical OFDM CSI."""
        path_gains = torch.tensor([1.0 + 0.0j, 0.25 + 0.5j], dtype=torch.complex64)
        path_delays = torch.tensor([0.0, 1.25e-7], dtype=torch.float32)
        subcarrier_frequencies = torch.tensor(
            [-1.0e6, -5.0e5, 0.0, 5.0e5, 1.0e6],
            dtype=torch.float32,
        )
        predicted_csi = ofdm_channel_response(
            h_l=path_gains,
            tau_l=path_delays,
            f_k=subcarrier_frequencies,
        )

        residual = ofdm_residual(
            predicted_csi=predicted_csi,
            path_gains=path_gains,
            path_delays=path_delays,
            subcarrier_frequencies=subcarrier_frequencies,
        )

        assert residual.shape == torch.Size([])
        assert residual.dtype in {torch.float32, torch.float64}
        assert torch.isfinite(residual)
        assert residual.item() < 1e-8

    def test_subcarrier_correlation_loss_prefers_smooth_frequency_response(
        self,
    ) -> None:
        """Test neighboring-subcarrier smoothness loss behavior."""
        smooth_csi = torch.tensor([1 + 0j, 1.01 + 0j, 1.02 + 0j], dtype=torch.complex64)
        rough_csi = torch.tensor([1 + 0j, -2 + 0j, 3 + 0j], dtype=torch.complex64)

        smooth_loss = subcarrier_correlation_loss(smooth_csi)
        rough_loss = subcarrier_correlation_loss(rough_csi)

        assert smooth_loss.shape == torch.Size([])
        assert rough_loss.shape == torch.Size([])
        assert torch.isfinite(smooth_loss)
        assert torch.isfinite(rough_loss)
        assert smooth_loss.item() < rough_loss.item()

    def test_ofdm_residual_gradient_flow_path_parameters(self) -> None:
        """Test OFDM residual keeps gradient flow through gains and delays."""
        h_real = torch.tensor([1.0, 0.45], dtype=torch.float64, requires_grad=True)
        h_imag = torch.tensor([0.0, -0.25], dtype=torch.float64, requires_grad=True)
        path_gains = torch.complex(h_real, h_imag)
        path_delays = torch.tensor(
            [0.0, 1.3e-7],
            dtype=torch.float64,
            requires_grad=True,
        )
        subcarrier_frequencies = torch.tensor(
            [-1.0e6, 0.0, 1.0e6],
            dtype=torch.float64,
        )

        target = ofdm_channel_response(
            h_l=path_gains,
            tau_l=path_delays,
            f_k=subcarrier_frequencies,
        )
        predicted = target + torch.tensor(
            [0.01 + 0.0j, -0.02 + 0.0j, 0.01 + 0.0j],
            dtype=torch.complex128,
        )

        residual = ofdm_residual(
            predicted_csi=predicted,
            path_gains=path_gains,
            path_delays=path_delays,
            subcarrier_frequencies=subcarrier_frequencies,
        )

        grad_h_real, grad_h_imag, grad_tau = torch.autograd.grad(
            residual,
            (h_real, h_imag, path_delays),
            create_graph=True,
        )

        assert torch.isfinite(grad_h_real).all()
        assert torch.isfinite(grad_h_imag).all()
        assert torch.isfinite(grad_tau).all()
