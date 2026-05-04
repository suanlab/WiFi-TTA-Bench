# pyright: basic, reportMissingImports=false

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pinn4csi.data.paper1 import (
    PreparedCSIBundle,
    PreparedCSIDataset,
    create_mock_paper1_prepared_data,
    get_paper1_dataset_config,
)
from pinn4csi.models import (
    DomainAdaptationBaselineConfig,
    create_domain_adaptation_baseline,
    list_domain_adaptation_baselines,
)
from pinn4csi.training import (
    Paper2BaselineExperimentConfig,
    Paper2MultiEnvironmentExperimentConfig,
    Paper2PreparedDataConfig,
    build_paper2_leave_one_environment_out_matrix,
    run_domain_adaptation_baselines,
    run_paper2_leave_one_environment_out_matrix,
)


def test_domain_adaptation_baselines_expose_shared_factory_and_losses() -> None:
    source_batch, target_batch, input_shape, num_classes = (
        _make_synthetic_domain_batches()
    )
    config = DomainAdaptationBaselineConfig(
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        invariance_weight=0.7,
        physics_weight=0.4,
        domain_weight=0.6,
        meta_weight=0.5,
        meta_inner_lr=0.2,
    )

    assert list_domain_adaptation_baselines() == (
        "residual_source_only",
        "coral",
        "dann",
        "maml",
    )

    for baseline_name in list_domain_adaptation_baselines():
        model = create_domain_adaptation_baseline(
            baseline_name=baseline_name,
            input_shape=input_shape,
            num_classes=num_classes,
            config=config,
        )
        outputs = model(
            source_batch["x"],
            prior=source_batch["prior"],
            has_prior=source_batch["has_prior"],
        )
        losses = model.compute_batch_losses(
            source_x=source_batch["x"],
            source_labels=source_batch["label"],
            target_x=target_batch["x"],
            source_prior=source_batch["prior"],
            target_prior=target_batch["prior"],
            source_has_prior=source_batch["has_prior"],
            target_has_prior=target_batch["has_prior"],
        )

        assert outputs["task_logits"].shape == (8, num_classes)
        assert outputs["invariant_features"].shape == (8, config.latent_dim)
        assert torch.isfinite(losses["loss_total"])
        assert torch.isfinite(losses["loss_task"])
        assert torch.isfinite(losses["loss_invariance"])
        assert torch.isfinite(losses["loss_physics_residual"])
        if baseline_name == "residual_source_only":
            assert torch.isclose(losses["loss_total"], losses["loss_task"])
        elif baseline_name == "coral":
            assert losses["loss_invariance"] > 0
        elif baseline_name == "dann":
            assert losses["loss_domain"] > 0
        elif baseline_name == "maml":
            assert losses["loss_meta"] > 0


def test_domain_adaptation_harness_runs_on_synthetic_cross_domain_data(
    tmp_path: Path,
) -> None:
    del tmp_path
    source_loader, target_loader, val_loader, test_loader, input_shape = (
        _make_synthetic_cross_domain_loaders()
    )
    config = Paper2BaselineExperimentConfig(
        epochs=3,
        learning_rate=1e-2,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        invariance_weight=0.5,
        physics_weight=0.3,
        domain_weight=0.3,
        meta_weight=0.4,
        meta_inner_lr=0.15,
        domain_hidden_dim=16,
    )

    results = run_domain_adaptation_baselines(
        source_train_loader=source_loader,
        target_train_loader=target_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        input_shape=input_shape,
        num_classes=3,
        config=config,
    )

    assert len(results) == 4
    assert {result.baseline_name for result in results} == set(
        list_domain_adaptation_baselines()
    )
    for result in results:
        assert result.best_epoch >= 0
        assert result.train_loss >= 0.0
        assert 0.0 <= result.val_accuracy <= 1.0
        assert 0.0 <= result.test_accuracy <= 1.0
        assert result.val_loss >= 0.0
        assert result.test_loss >= 0.0


def _make_synthetic_domain_batches() -> tuple[
    dict[str, Tensor],
    dict[str, Tensor],
    tuple[int, int],
    int,
]:
    generator = torch.Generator().manual_seed(23)
    num_classes = 3
    input_shape = (6, 2)
    source_x, source_labels, source_prior = _make_domain_samples(
        num_samples=8,
        num_classes=num_classes,
        input_shape=input_shape,
        generator=generator,
        domain_shift=0.0,
    )
    target_x, _, target_prior = _make_domain_samples(
        num_samples=8,
        num_classes=num_classes,
        input_shape=input_shape,
        generator=generator,
        domain_shift=0.45,
    )
    batch_template = {
        "has_prior": torch.ones(8, dtype=torch.bool),
    }
    return (
        {
            "x": source_x,
            "label": source_labels,
            "prior": source_prior,
            **batch_template,
        },
        {
            "x": target_x,
            "label": source_labels.clone(),
            "prior": target_prior,
            **batch_template,
        },
        input_shape,
        num_classes,
    )


def _make_synthetic_cross_domain_loaders() -> tuple[
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    DataLoader[dict[str, Tensor]],
    tuple[int, int],
]:
    input_shape = (8, 2)
    generator = torch.Generator().manual_seed(11)
    source_x, source_labels, source_prior = _make_domain_samples(
        num_samples=36,
        num_classes=3,
        input_shape=input_shape,
        generator=generator,
        domain_shift=0.0,
    )
    target_x, target_labels, target_prior = _make_domain_samples(
        num_samples=36,
        num_classes=3,
        input_shape=input_shape,
        generator=generator,
        domain_shift=0.55,
    )
    config = get_paper1_dataset_config("ut_har")
    source_bundle = PreparedCSIBundle(
        config=config,
        features=source_x,
        labels=source_labels,
        environments=torch.zeros(36, dtype=torch.long),
        priors=source_prior,
    )
    target_bundle = PreparedCSIBundle(
        config=config,
        features=target_x,
        labels=target_labels,
        environments=torch.ones(36, dtype=torch.long),
        priors=target_prior,
    )
    val_bundle = PreparedCSIBundle(
        config=config,
        features=target_x[:18],
        labels=target_labels[:18],
        environments=torch.ones(18, dtype=torch.long),
        priors=target_prior[:18],
    )
    test_bundle = PreparedCSIBundle(
        config=config,
        features=target_x[18:],
        labels=target_labels[18:],
        environments=torch.ones(18, dtype=torch.long),
        priors=target_prior[18:],
    )
    return (
        DataLoader(PreparedCSIDataset(source_bundle), batch_size=12, shuffle=True),
        DataLoader(PreparedCSIDataset(target_bundle), batch_size=12, shuffle=True),
        DataLoader(PreparedCSIDataset(val_bundle), batch_size=9, shuffle=False),
        DataLoader(PreparedCSIDataset(test_bundle), batch_size=9, shuffle=False),
        input_shape,
    )


def _make_domain_samples(
    num_samples: int,
    num_classes: int,
    input_shape: tuple[int, int],
    generator: torch.Generator,
    domain_shift: float,
) -> tuple[Tensor, Tensor, Tensor]:
    labels = torch.arange(num_samples, dtype=torch.long) % num_classes
    class_centers = torch.stack(
        [
            torch.linspace(
                0.2 * (idx + 1),
                0.6 * (idx + 1),
                input_shape[0] * input_shape[1],
            )
            for idx in range(num_classes)
        ],
        dim=0,
    ).reshape(num_classes, *input_shape)
    base = class_centers[labels]
    nuisance = 0.05 * torch.randn((num_samples, *input_shape), generator=generator)
    domain_pattern = domain_shift * torch.sin(
        torch.linspace(0.0, 3.14159, input_shape[0] * input_shape[1])
    ).reshape(*input_shape)
    features = base + domain_pattern + nuisance
    priors = base + 0.03 * torch.randn((num_samples, *input_shape), generator=generator)
    return features.float(), labels, priors.float()


def test_paper2_matrix_builds_leave_one_environment_out_entries(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "prepared"
    create_mock_paper1_prepared_data(
        prepared_root=prepared_root,
        samples_per_class=6,
        num_subcarriers=10,
        num_classes=3,
    )
    config = Paper2MultiEnvironmentExperimentConfig(
        data=Paper2PreparedDataConfig(
            prepared_root=prepared_root,
            paper1_dataset_names=("ut_har",),
            include_individual_bundles=True,
            include_combined_bundle=False,
        ),
        baseline=Paper2BaselineExperimentConfig(
            baseline_names=("residual_source_only",),
            epochs=1,
            hidden_dim=16,
            num_layers=2,
            latent_dim=8,
        ),
        seeds=(0, 1),
        batch_size=8,
    )

    entries = build_paper2_leave_one_environment_out_matrix(config)

    assert len(entries) == 4
    assert {entry.held_out_environment_id for entry in entries} == {0, 1}
    assert {entry.seed for entry in entries} == {0, 1}
    for entry in entries:
        assert entry.source_train_examples > 0
        assert entry.target_train_examples > 0
        assert entry.target_val_examples > 0
        assert entry.target_test_examples > 0


def test_paper2_matrix_runner_combines_available_prepared_sources(
    tmp_path: Path,
) -> None:
    prepared_root = tmp_path / "prepared"
    create_mock_paper1_prepared_data(
        prepared_root=prepared_root,
        samples_per_class=5,
        num_subcarriers=10,
        num_classes=3,
    )
    wifi6_dir = _write_mock_wifi6_capture(
        tmp_path / "wifi6_capture",
        num_samples=30,
        num_subcarriers=10,
        num_antennas=1,
        num_classes=3,
    )

    config = Paper2MultiEnvironmentExperimentConfig(
        data=Paper2PreparedDataConfig(
            prepared_root=prepared_root,
            paper1_dataset_names=("ut_har",),
            wifi6_prepared_dirs=(wifi6_dir,),
            include_individual_bundles=False,
            include_combined_bundle=True,
        ),
        baseline=Paper2BaselineExperimentConfig(
            baseline_names=("residual_source_only",),
            epochs=1,
            hidden_dim=16,
            num_layers=2,
            latent_dim=8,
        ),
        seeds=(0,),
        batch_size=10,
    )

    results = run_paper2_leave_one_environment_out_matrix(config)

    assert results
    dataset_names = {row.dataset_name for row in results}
    assert any(name.startswith("combined:") for name in dataset_names)
    assert {row.baseline_name for row in results} == {"residual_source_only"}
    for row in results:
        assert row.best_epoch >= 0
        assert 0.0 <= row.val_accuracy <= 1.0
        assert 0.0 <= row.test_accuracy <= 1.0
        assert row.source_train_examples > 0
        assert row.target_train_examples > 0
        assert row.target_val_examples > 0
        assert row.target_test_examples > 0


def _write_mock_wifi6_capture(
    session_dir: Path,
    num_samples: int,
    num_subcarriers: int,
    num_antennas: int,
    num_classes: int,
) -> Path:
    rng = np.random.default_rng(7)
    labels = np.arange(num_samples, dtype=np.int64) % num_classes
    environments = np.arange(num_samples, dtype=np.int64) % 3
    real = rng.standard_normal((num_samples, num_subcarriers, num_antennas))
    imag = rng.standard_normal((num_samples, num_subcarriers, num_antennas))
    csi = (real + 1j * imag).astype(np.complex64)

    metadata = {
        "source": "wifi6",
        "protocol_version": "1.0",
        "capture_id": "mock-paper2",
        "task_name": "classification",
        "environment_names": {"0": "lab", "1": "corridor", "2": "office"},
        "label_names": {str(idx): f"class_{idx}" for idx in range(num_classes)},
        "num_subcarriers": num_subcarriers,
        "num_rx_antennas": num_antennas,
        "center_frequency_hz": 5.32e9,
        "bandwidth_mhz": 80.0,
        "receiver": "intel-ax210",
        "chipset": "AX210",
        "standard": "802.11ax",
        "num_tx_streams": 2,
    }

    session_dir.mkdir(parents=True, exist_ok=True)
    (session_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    np.save(session_dir / "csi.npy", csi)
    np.save(session_dir / "labels.npy", labels)
    np.save(session_dir / "environments.npy", environments)
    return session_dir
