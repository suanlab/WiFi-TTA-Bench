# pyright: basic, reportMissingImports=false

"""Evaluation entry point for WiFi imaging PINN branch.

Evaluates a trained WiFiImagingPINN model on test data with:
- Checkpoint loading and model restoration
- Per-sample field and permittivity reconstruction metrics
- Structured CSV/JSON output with honest metric claims
- Support for smoke evaluation on mock/prepared data
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pinn4csi.data.wifi_imaging_dataset import (
    create_wifi_imaging_splits,
    load_wifi_imaging_prepared_dataset,
)
from pinn4csi.models.wifi_imager import WiFiImagingPINN
from pinn4csi.utils.device import get_device
from pinn4csi.utils.metrics import nmse

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalConfig:
    """Configuration for WiFi imaging PINN evaluation."""

    prepared_root: Path
    checkpoint_path: Path
    batch_size: int = 8
    output_dir: Path = Path("outputs/wifi_imager_eval")


@dataclass(frozen=True)
class PerSampleMetrics:
    """Per-sample evaluation metrics."""

    sample_index: int
    environment_id: int
    field_mse: float
    field_nmse: float
    permittivity_mse: float
    permittivity_nmse: float
    physics_loss: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Aggregate evaluation metrics across test set."""

    num_samples: int
    num_environments: int
    field_mse_mean: float
    field_mse_std: float
    field_nmse_mean: float
    field_nmse_std: float
    permittivity_mse_mean: float
    permittivity_mse_std: float
    permittivity_nmse_mean: float
    permittivity_nmse_std: float
    physics_loss_mean: float
    physics_loss_std: float


def load_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple[WiFiImagingPINN, dict[str, object]]:
    """Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file.
        device: Device to load model onto.

    Returns:
        Tuple of (model, checkpoint_dict).

    Raises:
        FileNotFoundError: If checkpoint does not exist.
        RuntimeError: If checkpoint format is invalid.
    """
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"Invalid checkpoint format: {type(checkpoint)}")

    model_state = checkpoint.get("model_state_dict")
    if model_state is None:
        raise RuntimeError("Checkpoint missing 'model_state_dict' key")

    encoder_weight_indices = _get_linear_weight_indices(model_state, prefix="encoder")
    if not encoder_weight_indices:
        raise RuntimeError("Cannot infer model dimensions from checkpoint")

    first_encoder_weight = model_state[f"encoder.{encoder_weight_indices[0]}.weight"]
    final_encoder_weight = model_state[f"encoder.{encoder_weight_indices[-1]}.weight"]
    csi_feature_dim = int(first_encoder_weight.shape[1])
    hidden_dim = int(first_encoder_weight.shape[0])
    latent_dim = int(final_encoder_weight.shape[0])

    field_decoder_indices = _get_linear_weight_indices(
        model_state,
        prefix="field_decoder",
    )
    if not field_decoder_indices:
        raise RuntimeError("Cannot infer coordinate_dim from checkpoint")
    field_decoder_weight = model_state[
        f"field_decoder.{field_decoder_indices[0]}.weight"
    ]
    coordinate_dim = int(field_decoder_weight.shape[1]) - latent_dim

    num_layers = len(encoder_weight_indices) - 1

    model = WiFiImagingPINN(
        csi_feature_dim=csi_feature_dim,
        coordinate_dim=coordinate_dim,
        hidden_dim=hidden_dim,
        latent_dim=latent_dim,
        num_layers=num_layers,
    )
    model.to(device)
    model.load_state_dict(model_state)
    model.eval()

    logger.info(
        f"Loaded checkpoint from {checkpoint_path} "
        f"(epoch {checkpoint.get('epoch', '?')}, "
        f"val_loss {checkpoint.get('val_loss', '?'):.4f})"
    )
    return model, checkpoint


def _get_linear_weight_indices(
    state_dict: dict[str, object],
    prefix: str,
) -> tuple[int, ...]:
    indices: list[int] = []
    for key, value in state_dict.items():
        if not isinstance(value, Tensor):
            continue
        if not key.startswith(f"{prefix}.") or not key.endswith(".weight"):
            continue
        parts = key.split(".")
        if len(parts) >= 3 and parts[1].isdigit():
            indices.append(int(parts[1]))
    return tuple(sorted(indices))


def compute_per_sample_metrics(
    model: WiFiImagingPINN,
    batch: dict[str, Tensor],
    device: torch.device,
) -> list[PerSampleMetrics]:
    """Compute per-sample metrics for a batch.

    Args:
        model: Trained WiFiImagingPINN model.
        batch: Batch dictionary from DataLoader.
        device: Device to use for computation.

    Returns:
        List of PerSampleMetrics for each sample in batch.
    """
    csi_features = batch["csi_features"].to(device)
    query_coordinates = batch["query_coordinates"].to(device)
    field_target = batch["field_target"].to(device)
    permittivity_target = batch["permittivity_target"].to(device)
    sample_indices = batch["sample_index"].to(device)
    environment_ids = batch["environment_id"].to(device)

    with torch.no_grad():
        outputs = model.forward(
            csi_features=csi_features,
            coordinates=query_coordinates,
        )

    field_pred = outputs["field"]
    permittivity_pred = outputs["permittivity"]

    batch_size = csi_features.shape[0]
    metrics_list: list[PerSampleMetrics] = []

    for i in range(batch_size):
        field_pred_i = field_pred[i : i + 1]
        field_target_i = field_target[i : i + 1]
        perm_pred_i = permittivity_pred[i : i + 1]
        perm_target_i = permittivity_target[i : i + 1]

        field_mse = float(torch.mean((field_pred_i - field_target_i) ** 2).item())
        field_nmse_val = nmse(field_pred_i, field_target_i)

        perm_mse = float(torch.mean((perm_pred_i - perm_target_i) ** 2).item())
        perm_nmse_val = nmse(perm_pred_i, perm_target_i)

        # Compute physics loss for this sample
        coordinates_for_physics = query_coordinates[i : i + 1].detach().clone()
        coordinates_for_physics.requires_grad_(True)
        losses = model.compute_losses(
            csi_features=csi_features[i : i + 1],
            coordinates=coordinates_for_physics,
            frequency=batch["frequency_hz"][i],
            field_target=field_target[i : i + 1],
            permittivity_target=permittivity_target[i : i + 1],
            lambda_field=0.0,
            lambda_permittivity=0.0,
            lambda_physics=1.0,
        )
        physics_loss = float(losses["loss_physics"].item())

        metrics_list.append(
            PerSampleMetrics(
                sample_index=int(sample_indices[i].item()),
                environment_id=int(environment_ids[i].item()),
                field_mse=field_mse,
                field_nmse=field_nmse_val,
                permittivity_mse=perm_mse,
                permittivity_nmse=perm_nmse_val,
                physics_loss=physics_loss,
            )
        )

    return metrics_list


def evaluate_test_set(
    model: WiFiImagingPINN,
    test_loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> tuple[list[PerSampleMetrics], AggregateMetrics]:
    """Evaluate model on full test set.

    Args:
        model: Trained WiFiImagingPINN model.
        test_loader: Test data loader.
        device: Device to use for computation.

    Returns:
        Tuple of (per_sample_metrics, aggregate_metrics).
    """
    all_metrics: list[PerSampleMetrics] = []

    for batch_idx, batch in enumerate(test_loader):
        batch_metrics = compute_per_sample_metrics(model, batch, device)
        all_metrics.extend(batch_metrics)
        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Evaluated {len(all_metrics)} samples")

    # Compute aggregate statistics
    field_mse_values = [m.field_mse for m in all_metrics]
    field_nmse_values = [m.field_nmse for m in all_metrics]
    perm_mse_values = [m.permittivity_mse for m in all_metrics]
    perm_nmse_values = [m.permittivity_nmse for m in all_metrics]
    physics_loss_values = [m.physics_loss for m in all_metrics]

    field_mse_tensor = torch.tensor(field_mse_values, dtype=torch.float32)
    field_nmse_tensor = torch.tensor(field_nmse_values, dtype=torch.float32)
    perm_mse_tensor = torch.tensor(perm_mse_values, dtype=torch.float32)
    perm_nmse_tensor = torch.tensor(perm_nmse_values, dtype=torch.float32)
    physics_loss_tensor = torch.tensor(physics_loss_values, dtype=torch.float32)

    num_environments = len(set(m.environment_id for m in all_metrics))

    aggregate = AggregateMetrics(
        num_samples=len(all_metrics),
        num_environments=num_environments,
        field_mse_mean=float(field_mse_tensor.mean().item()),
        field_mse_std=float(field_mse_tensor.std().item()),
        field_nmse_mean=float(field_nmse_tensor.mean().item()),
        field_nmse_std=float(field_nmse_tensor.std().item()),
        permittivity_mse_mean=float(perm_mse_tensor.mean().item()),
        permittivity_mse_std=float(perm_mse_tensor.std().item()),
        permittivity_nmse_mean=float(perm_nmse_tensor.mean().item()),
        permittivity_nmse_std=float(perm_nmse_tensor.std().item()),
        physics_loss_mean=float(physics_loss_tensor.mean().item()),
        physics_loss_std=float(physics_loss_tensor.std().item()),
    )

    return all_metrics, aggregate


def write_results(
    per_sample_metrics: list[PerSampleMetrics],
    aggregate_metrics: AggregateMetrics,
    output_dir: Path,
) -> tuple[Path, Path]:
    """Write evaluation results to CSV and JSON.

    Args:
        per_sample_metrics: Per-sample metrics list.
        aggregate_metrics: Aggregate metrics.
        output_dir: Output directory.

    Returns:
        Tuple of (csv_path, json_path).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Write per-sample CSV
    csv_path = output_dir / "wifi_imager_eval_per_sample.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(per_sample_metrics[0]).keys())
        writer.writeheader()
        for metrics in per_sample_metrics:
            writer.writerow(asdict(metrics))
    logger.info(f"Per-sample metrics written to {csv_path}")

    # Write aggregate JSON
    json_path = output_dir / "wifi_imager_eval_aggregate.json"
    aggregate_dict = asdict(aggregate_metrics)
    with open(json_path, "w") as f:
        json.dump(aggregate_dict, f, indent=2)
    logger.info(f"Aggregate metrics written to {json_path}")

    return csv_path, json_path


def main() -> None:
    """Main evaluation entry point."""
    parser = argparse.ArgumentParser(
        description="Evaluate WiFi imaging PINN on test data"
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        required=True,
        help="Path to prepared WiFi imaging data directory",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        required=True,
        help="Path to trained model checkpoint",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size for evaluation (default: 8)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/wifi_imager_eval"),
        help="Output directory for results (default: outputs/wifi_imager_eval)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    config = EvalConfig(
        prepared_root=args.prepared_root,
        checkpoint_path=args.checkpoint_path,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
    )

    logger.info(f"Evaluation config: {config}")

    device = get_device()
    logger.info(f"Using device: {device}")

    # Load checkpoint
    model, _checkpoint_info = load_checkpoint(config.checkpoint_path, device)

    # Load prepared data
    logger.info(f"Loading prepared data from {config.prepared_root}")
    bundle = load_wifi_imaging_prepared_dataset(config.prepared_root)
    logger.info(
        f"Loaded {bundle.num_samples} samples with "
        f"{bundle.csi_feature_dim} CSI features"
    )

    # Create splits
    splits = create_wifi_imaging_splits(bundle, seed=42)
    logger.info(
        f"Test set: {len(splits.test)} samples from "
        f"environments {splits.test_environment_ids}"
    )

    # Create test loader
    test_loader = DataLoader(
        splits.test,
        batch_size=config.batch_size,
        shuffle=False,
    )

    # Evaluate
    logger.info("Starting evaluation on test set...")
    per_sample_metrics, aggregate_metrics = evaluate_test_set(
        model, test_loader, device
    )

    # Write results
    csv_path, json_path = write_results(
        per_sample_metrics, aggregate_metrics, config.output_dir
    )

    # Log summary
    logger.info(
        f"Evaluation complete. "
        f"Test samples: {aggregate_metrics.num_samples}, "
        f"Environments: {aggregate_metrics.num_environments}"
    )
    logger.info(
        f"Field MSE: {aggregate_metrics.field_mse_mean:.6f} "
        f"± {aggregate_metrics.field_mse_std:.6f}"
    )
    logger.info(
        f"Field NMSE: {aggregate_metrics.field_nmse_mean:.6f} "
        f"± {aggregate_metrics.field_nmse_std:.6f}"
    )
    logger.info(
        f"Permittivity MSE: {aggregate_metrics.permittivity_mse_mean:.6f} "
        f"± {aggregate_metrics.permittivity_mse_std:.6f}"
    )
    logger.info(
        f"Permittivity NMSE: {aggregate_metrics.permittivity_nmse_mean:.6f} "
        f"± {aggregate_metrics.permittivity_nmse_std:.6f}"
    )
    logger.info(
        f"Physics Loss: {aggregate_metrics.physics_loss_mean:.6f} "
        f"± {aggregate_metrics.physics_loss_std:.6f}"
    )
    logger.info(f"Per-sample CSV written to {csv_path}")
    logger.info(f"Aggregate JSON written to {json_path}")


if __name__ == "__main__":
    main()
