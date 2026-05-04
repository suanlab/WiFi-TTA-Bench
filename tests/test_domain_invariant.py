# pyright: basic, reportMissingImports=false

import torch

from pinn4csi.models import (
    DomainInvariantLossToggles,
    PhysicsDomainInvariantModule,
    coral_loss,
    residual_moment_alignment_loss,
)


def test_domain_invariant_forward_shapes() -> None:
    module = PhysicsDomainInvariantModule(
        in_features=12,
        latent_dim=8,
        hidden_dim=24,
        num_layers=2,
        task_output_dim=4,
        use_residual_prior=True,
    )
    x = torch.randn(7, 12)
    prior = torch.randn(7, 12)

    outputs = module(x, prior)

    assert outputs["features"].shape == (7, 8)
    assert outputs["invariant_features"].shape == (7, 8)
    assert outputs["physics_residual"].shape == (7, 12)
    assert outputs["task_logits"].shape == (7, 4)


def test_domain_losses_backward_has_finite_gradients() -> None:
    module = PhysicsDomainInvariantModule(
        in_features=10,
        latent_dim=6,
        hidden_dim=20,
        num_layers=2,
        task_output_dim=3,
        use_residual_prior=True,
    )

    source_x = torch.randn(8, 10)
    target_x = torch.randn(9, 10)
    source_prior = 0.5 * source_x + 0.1 * torch.randn_like(source_x)
    target_prior = 0.5 * target_x + 0.1 * torch.randn_like(target_x)
    source_labels = torch.randint(0, 3, (8,))

    source_outputs = module(source_x, source_prior)
    target_outputs = module(target_x, target_prior)
    losses = module.compute_domain_losses(
        source_outputs=source_outputs,
        target_outputs=target_outputs,
        source_labels=source_labels,
        toggles=DomainInvariantLossToggles(True, True, True),
    )
    losses["loss_total"].backward()

    for parameter in module.parameters():
        assert parameter.grad is not None
        assert torch.all(torch.isfinite(parameter.grad))


def test_coral_and_residual_alignment_are_zero_for_identical_inputs() -> None:
    features = torch.randn(6, 5)
    residual = torch.randn(6, 5)

    assert torch.isclose(coral_loss(features, features), torch.tensor(0.0), atol=1e-7)
    assert torch.isclose(
        residual_moment_alignment_loss(residual, residual),
        torch.tensor(0.0),
        atol=1e-7,
    )


def test_invariance_loss_increases_with_distribution_shift() -> None:
    base = torch.randn(12, 6)
    shifted = 2.5 * base + 1.7
    near = base + 0.01 * torch.randn_like(base)

    shifted_loss = coral_loss(base, shifted)
    near_loss = coral_loss(base, near)
    assert shifted_loss > near_loss


def test_toggles_control_total_loss_sum() -> None:
    module = PhysicsDomainInvariantModule(
        in_features=8,
        latent_dim=4,
        hidden_dim=16,
        num_layers=2,
        task_output_dim=2,
    )
    source = module(torch.randn(5, 8), torch.randn(5, 8))
    target = module(torch.randn(6, 8), torch.randn(6, 8))
    labels = torch.randint(0, 2, (5,))

    toggles = DomainInvariantLossToggles(
        loss_task=True,
        loss_invariance=False,
        loss_physics_residual=True,
    )
    losses = module.compute_domain_losses(source, target, labels, toggles=toggles)
    expected = losses["loss_task"] + losses["loss_physics_residual"]
    assert torch.isclose(losses["loss_total"], expected)
