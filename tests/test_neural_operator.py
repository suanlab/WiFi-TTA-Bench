# pyright: basic, reportMissingImports=false

import torch

from pinn4csi.models import PhysicsInformedDeepONet, create_physics_informed_deeponet
from pinn4csi.physics import ofdm_channel_response


def _complex_to_stacked(csi: torch.Tensor) -> torch.Tensor:
    return torch.stack([csi.real, csi.imag], dim=-1)


def test_deeponet_forward_shape_for_shared_and_batched_queries() -> None:
    batch_size = 5
    num_queries = 12
    model = create_physics_informed_deeponet(
        environment_dim=6,
        query_dim=1,
        hidden_dim=24,
        latent_dim=16,
        num_layers=2,
        output_channels=2,
    )
    environment = torch.randn(batch_size, 6)
    shared_queries = torch.linspace(-1.0, 1.0, num_queries).unsqueeze(-1)
    batched_queries = shared_queries.unsqueeze(0).repeat(batch_size, 1, 1)

    shared_output = model(environment, shared_queries)
    batched_output = model(environment, batched_queries)

    assert shared_output.shape == (batch_size, num_queries, 2)
    assert batched_output.shape == (batch_size, num_queries, 2)


def test_deeponet_backward_has_finite_gradients() -> None:
    batch_size = 4
    num_queries = 10
    num_paths = 3
    model = PhysicsInformedDeepONet(
        environment_dim=7,
        query_dim=1,
        hidden_dim=20,
        latent_dim=14,
        num_layers=2,
        output_channels=2,
    )

    environment = torch.randn(batch_size, 7)
    query_coordinates = torch.linspace(-0.5, 0.5, num_queries).unsqueeze(-1)
    path_gains = torch.complex(
        torch.randn(batch_size, num_paths),
        torch.randn(batch_size, num_paths),
    )
    path_delays = 10e-9 + 90e-9 * torch.rand(batch_size, num_paths)
    subcarrier_frequencies = torch.linspace(-1.5e6, 1.5e6, num_queries)
    target_complex = ofdm_channel_response(
        h_l=path_gains,
        tau_l=path_delays,
        f_k=subcarrier_frequencies,
    )
    target = _complex_to_stacked(target_complex)

    prediction = model(environment, query_coordinates)
    losses = model.compute_losses(
        predicted_response=prediction,
        target_response=target,
        physics={
            "path_gains_real": path_gains.real,
            "path_gains_imag": path_gains.imag,
            "path_delays": path_delays,
            "subcarrier_frequencies": subcarrier_frequencies,
        },
        correlation_weight=0.2,
    )
    losses["loss_total"].backward()  # type: ignore[no-untyped-call]

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_deeponet_ofdm_loss_is_lower_for_analytical_prediction() -> None:
    batch_size = 3
    num_queries = 9
    num_paths = 2

    path_gains = torch.complex(
        torch.randn(batch_size, num_paths),
        torch.randn(batch_size, num_paths),
    )
    path_delays = 25e-9 + 75e-9 * torch.rand(batch_size, num_paths)
    subcarrier_frequencies = torch.linspace(-2.0e6, 2.0e6, num_queries)
    analytical = ofdm_channel_response(
        h_l=path_gains,
        tau_l=path_delays,
        f_k=subcarrier_frequencies,
    )
    perfect_prediction = _complex_to_stacked(analytical)
    noisy_prediction = perfect_prediction + 0.05 * torch.randn_like(perfect_prediction)

    physics = {
        "path_gains_real": path_gains.real,
        "path_gains_imag": path_gains.imag,
        "path_delays": path_delays,
        "subcarrier_frequencies": subcarrier_frequencies,
    }
    helper = PhysicsInformedDeepONet(
        environment_dim=4,
        query_dim=1,
        output_channels=2,
    )
    perfect_losses = helper.compute_losses(
        predicted_response=perfect_prediction,
        target_response=perfect_prediction,
        physics=physics,
        data_weight=0.0,
        correlation_weight=0.0,
    )
    noisy_losses = helper.compute_losses(
        predicted_response=noisy_prediction,
        target_response=perfect_prediction,
        physics=physics,
        data_weight=0.0,
        correlation_weight=0.0,
    )

    assert perfect_losses["loss_ofdm"].item() < 1e-8
    assert noisy_losses["loss_ofdm"] > perfect_losses["loss_ofdm"]
