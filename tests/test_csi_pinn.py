# pyright: basic, reportMissingImports=false

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from pinn4csi.models import CSIPhysicsAutoencoder, LossToggles
from pinn4csi.physics import ofdm_channel_response


def _complex_to_stacked(csi: Tensor) -> Tensor:
    return torch.cat([csi.real, csi.imag], dim=-1)


def _make_synthetic_batch(
    batch_size: int = 8,
    in_features: int = 12,
    num_subcarriers: int = 16,
    num_paths: int = 3,
) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]:
    x = torch.randn(batch_size, in_features)
    path_gains = torch.complex(
        torch.randn(batch_size, num_paths),
        torch.randn(batch_size, num_paths),
    )
    path_delays = 20e-9 + 180e-9 * torch.rand(batch_size, num_paths)

    center = 5.32e9
    spacing = 312.5e3
    base_freq = center + spacing * (
        torch.arange(num_subcarriers, dtype=torch.float32) - float(num_subcarriers // 2)
    )
    subcarrier_frequencies = base_freq.unsqueeze(0).repeat(batch_size, 1)

    prior_complex = ofdm_channel_response(
        h_l=path_gains,
        tau_l=path_delays,
        f_k=subcarrier_frequencies,
    )
    prior_stacked = _complex_to_stacked(prior_complex)

    target_reconstruction = prior_stacked + 0.05 * torch.randn_like(prior_stacked)
    task_target = torch.randn(batch_size, 2)

    physics = {
        "path_gains_real": path_gains.real,
        "path_gains_imag": path_gains.imag,
        "path_delays": path_delays,
        "subcarrier_frequencies": subcarrier_frequencies,
        "distance": 1.0 + 9.0 * torch.rand(batch_size),
        "frequency": torch.full((batch_size,), 2.4e9),
        "tx_power_dbm": torch.full((batch_size,), 20.0),
        "path_loss_exponent": torch.full((batch_size,), 2.0),
    }
    return x, target_reconstruction, task_target, physics, prior_stacked


def _active_sum(losses: dict[str, Tensor], toggles: LossToggles) -> Tensor:
    total = torch.zeros_like(losses["loss_total"])
    if toggles.loss_reconstruction:
        total = total + losses["loss_reconstruction"]
    if toggles.loss_task:
        total = total + losses["loss_task"]
    if toggles.loss_ofdm:
        total = total + losses["loss_ofdm"]
    if toggles.loss_path:
        total = total + losses["loss_path"]
    return total


def test_csi_autoencoder_forward_shapes() -> None:
    batch_size = 10
    in_features = 12
    latent_dim = 6
    num_subcarriers = 16

    model = CSIPhysicsAutoencoder(
        in_features=in_features,
        latent_dim=latent_dim,
        hidden_dim=32,
        num_subcarriers=num_subcarriers,
        task_output_dim=2,
        use_residual_prior=True,
    )

    x, _, _, _, prior = _make_synthetic_batch(
        batch_size=batch_size,
        in_features=in_features,
        num_subcarriers=num_subcarriers,
    )
    outputs = model(x, prior_reconstruction=prior)

    assert outputs["latent"].shape == (batch_size, latent_dim)
    assert outputs["reconstruction"].shape == (batch_size, 2 * num_subcarriers)
    assert outputs["reconstruction_residual"].shape == (batch_size, 2 * num_subcarriers)
    assert outputs["task_prediction"].shape == (batch_size, 2)
    expected_reconstruction = prior + outputs["reconstruction_residual"]
    assert torch.allclose(outputs["reconstruction"], expected_reconstruction)


def test_csi_autoencoder_backward_has_finite_gradients() -> None:
    model = CSIPhysicsAutoencoder(
        in_features=12,
        latent_dim=8,
        hidden_dim=32,
        num_subcarriers=16,
        task_output_dim=2,
        use_fourier_features=True,
    )

    x, target, task_target, physics, prior = _make_synthetic_batch()
    outputs = model(x, prior_reconstruction=prior)
    losses = model.compute_losses(
        outputs=outputs,
        target_reconstruction=target,
        task_target=task_target,
        physics=physics,
        toggles=LossToggles(),
    )
    losses["loss_total"].backward()  # type: ignore[no-untyped-call]

    for parameter in model.parameters():
        if parameter.requires_grad:
            assert parameter.grad is not None
            assert torch.all(torch.isfinite(parameter.grad))


def test_csi_autoencoder_amplitude_phase_reconstruction_mode() -> None:
    batch_size = 4
    num_subcarriers = 10
    num_antennas = 3
    in_features = num_subcarriers * (2 * num_antennas)
    model = CSIPhysicsAutoencoder(
        in_features=in_features,
        latent_dim=8,
        hidden_dim=32,
        num_subcarriers=num_subcarriers * num_antennas,
        reconstruction_dim=in_features,
        task_output_dim=2,
        reconstruction_representation="amplitude_phase",
    )

    x = torch.randn(batch_size, in_features)
    outputs = model(x)
    physics = {
        "path_gains_real": torch.randn(batch_size, 1),
        "path_gains_imag": torch.randn(batch_size, 1),
        "path_delays": torch.zeros(batch_size, 1),
        "subcarrier_frequencies": torch.linspace(
            5.0e9,
            5.1e9,
            num_subcarriers * num_antennas,
        )
        .unsqueeze(0)
        .repeat(batch_size, 1),
        "distance": torch.full((batch_size,), 2.0),
        "frequency": torch.full((batch_size,), 2.4e9),
        "tx_power_dbm": torch.full((batch_size,), 20.0),
        "path_loss_exponent": torch.full((batch_size,), 2.0),
    }
    losses = model.compute_losses(
        outputs=outputs,
        target_reconstruction=x,
        task_target=torch.randint(0, 2, (batch_size,)),
        physics=physics,
        toggles=LossToggles(True, True, True, True),
        task_loss_fn=torch.nn.functional.cross_entropy,
    )

    assert torch.isfinite(losses["loss_reconstruction"])
    assert torch.isfinite(losses["loss_task"])
    assert torch.isfinite(losses["loss_ofdm"])
    assert torch.isfinite(losses["loss_path"])
    assert torch.isfinite(losses["loss_total"])
    losses["loss_total"].backward()  # type: ignore[no-untyped-call]


def test_fourier_feature_toggle_behavior() -> None:
    in_features = 12
    x = torch.randn(4, in_features)

    model_no_fourier = CSIPhysicsAutoencoder(
        in_features=in_features,
        latent_dim=8,
        use_fourier_features=False,
    )
    plain_features = model_no_fourier.apply_feature_embedding(x)

    model_with_fourier = CSIPhysicsAutoencoder(
        in_features=in_features,
        latent_dim=8,
        use_fourier_features=True,
        fourier_num_frequencies=10,
    )
    embedded_features = model_with_fourier.apply_feature_embedding(x)

    assert model_no_fourier.fourier_embedding is None
    assert model_with_fourier.fourier_embedding is not None
    assert plain_features.shape[-1] == in_features
    assert embedded_features.shape[-1] == in_features + 20


def test_loss_component_toggles_are_ablation_ready() -> None:
    model = CSIPhysicsAutoencoder(
        in_features=12,
        latent_dim=8,
        hidden_dim=32,
        num_subcarriers=16,
        task_output_dim=2,
    )
    x, target, task_target, physics, prior = _make_synthetic_batch()
    outputs = model(x, prior_reconstruction=prior)

    full_toggles = LossToggles(True, True, True, True)
    full_losses = model.compute_losses(
        outputs=outputs,
        target_reconstruction=target,
        task_target=task_target,
        physics=physics,
        toggles=full_toggles,
    )
    assert torch.isclose(
        full_losses["loss_total"], _active_sum(full_losses, full_toggles)
    )

    toggle_variants = [
        LossToggles(False, True, True, True),
        LossToggles(True, False, True, True),
        LossToggles(True, True, False, True),
        LossToggles(True, True, True, False),
    ]
    for toggles in toggle_variants:
        losses = model.compute_losses(
            outputs=outputs,
            target_reconstruction=target,
            task_target=task_target,
            physics=physics,
            toggles=toggles,
        )
        assert torch.isclose(losses["loss_total"], _active_sum(losses, toggles))


class _SyntheticCSITrainDataset(
    Dataset[tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]]
):
    def __init__(self, num_samples: int = 32) -> None:
        self.samples = [_make_synthetic_batch(batch_size=1) for _ in range(num_samples)]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[Tensor, Tensor, Tensor, dict[str, Tensor], Tensor]:
        x, target, task_target, physics, prior = self.samples[index]
        return (
            x.squeeze(0),
            target.squeeze(0),
            task_target.squeeze(0),
            {key: value.squeeze(0) for key, value in physics.items()},
            prior.squeeze(0),
        )


def test_csi_autoencoder_one_epoch_smoke_interaction() -> None:
    dataset = _SyntheticCSITrainDataset(num_samples=24)
    loader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = CSIPhysicsAutoencoder(
        in_features=12,
        latent_dim=8,
        hidden_dim=32,
        num_subcarriers=16,
        task_output_dim=2,
        use_fourier_features=True,
        use_residual_prior=True,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    model.train()
    observed_losses: list[float] = []
    for x, target, task_target, physics, prior in loader:
        optimizer.zero_grad()
        outputs = model(x, prior_reconstruction=prior)
        losses = model.compute_losses(
            outputs=outputs,
            target_reconstruction=target,
            task_target=task_target,
            physics=physics,
            toggles=LossToggles(True, True, True, True),
        )
        losses["loss_total"].backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        observed_losses.append(float(losses["loss_total"].item()))

    assert observed_losses
    assert all(
        torch.isfinite(torch.tensor(loss_value)) for loss_value in observed_losses
    )
