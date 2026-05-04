# pyright: basic, reportMissingImports=false

import importlib.util
import sys
from pathlib import Path

import torch

from pinn4csi.models import WiFiImagingPINN


def _load_evaluate_wifi_imager_module():
    """Load evaluate_wifi_imager.py via importlib for pytest compatibility."""
    script_path = Path(__file__).parent.parent / "scripts" / "evaluate_wifi_imager.py"
    spec = importlib.util.spec_from_file_location("evaluate_wifi_imager", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load module spec from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["evaluate_wifi_imager"] = module
    spec.loader.exec_module(module)
    return module


_evaluate_module = _load_evaluate_wifi_imager_module()
load_checkpoint = _evaluate_module.load_checkpoint


def _make_inverse_batch(
    batch_size: int = 5,
    num_points: int = 24,
    csi_feature_dim: int = 10,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    csi_features = torch.randn(batch_size, csi_feature_dim)
    coordinates = 2.0 * torch.rand(batch_size, num_points, 2) - 1.0

    x_coord = coordinates[..., 0]
    y_coord = coordinates[..., 1]
    context = csi_features[:, 0].unsqueeze(1)
    field_target = (
        torch.sin(2.5 * x_coord) + 0.25 * torch.cos(1.5 * y_coord) + 0.1 * context
    )
    permittivity_target = 2.0 + 0.4 * torch.sigmoid(x_coord + y_coord)

    return csi_features, coordinates, field_target, permittivity_target


def test_wifi_imager_forward_shapes() -> None:
    model = WiFiImagingPINN(
        csi_feature_dim=10,
        coordinate_dim=2,
        hidden_dim=32,
        latent_dim=16,
        num_layers=2,
    )
    csi_features, coordinates, _, _ = _make_inverse_batch()

    outputs = model(csi_features, coordinates)

    assert outputs["latent"].shape == (5, 16)
    assert outputs["field"].shape == (5, 24)
    assert outputs["permittivity"].shape == (5, 24)
    assert torch.all(outputs["permittivity"] > 0.0)


def test_wifi_imager_loss_path_backward_has_finite_gradients() -> None:
    model = WiFiImagingPINN(
        csi_feature_dim=10,
        coordinate_dim=2,
        hidden_dim=32,
        latent_dim=16,
        num_layers=2,
    )
    csi_features, coordinates, field_target, permittivity_target = _make_inverse_batch()

    losses = model.compute_losses(
        csi_features=csi_features,
        coordinates=coordinates,
        frequency=2.4e9,
        field_target=field_target,
        permittivity_target=permittivity_target,
        lambda_field=1.0,
        lambda_permittivity=0.5,
        lambda_physics=0.2,
    )
    losses["loss_total"].backward()  # type: ignore[no-untyped-call]

    assert torch.isfinite(losses["loss_total"])
    assert torch.isfinite(losses["loss_field"])
    assert torch.isfinite(losses["loss_permittivity"])
    assert torch.isfinite(losses["loss_physics"])
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()


def test_wifi_imager_supports_physics_only_training_path() -> None:
    model = WiFiImagingPINN(csi_feature_dim=10, coordinate_dim=2)
    csi_features, coordinates, _, _ = _make_inverse_batch()

    losses = model.compute_losses(
        csi_features=csi_features,
        coordinates=coordinates,
        frequency=5.32e9,
    )

    assert torch.isfinite(losses["loss_total"])
    assert losses["loss_field"].item() == 0.0
    assert losses["loss_permittivity"].item() == 0.0
    assert torch.isclose(losses["loss_total"], losses["loss_physics"])


def test_load_checkpoint_recovers_wifi_imager_dimensions(tmp_path: Path) -> None:
    model = WiFiImagingPINN(
        csi_feature_dim=12,
        coordinate_dim=2,
        hidden_dim=16,
        latent_dim=8,
        num_layers=2,
    )
    checkpoint_path = tmp_path / "wifi_imager_checkpoint.pt"
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    torch.save(
        {
            "epoch": 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": 0.25,
        },
        checkpoint_path,
    )

    loaded_model, checkpoint = load_checkpoint(checkpoint_path, torch.device("cpu"))
    outputs = loaded_model(
        torch.randn(2, 12),
        torch.randn(2, 5, 2),
    )

    assert checkpoint["epoch"] == 1
    assert loaded_model.coordinate_dim == 2
    assert outputs["latent"].shape == (2, 8)
    assert outputs["field"].shape == (2, 5)
