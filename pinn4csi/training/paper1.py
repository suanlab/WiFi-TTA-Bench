# pyright: basic, reportMissingImports=false

"""Reusable multi-dataset Paper 1 experiment harness."""

from __future__ import annotations

import logging
import math
import random
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor
from torch.optim import Adam
from torch.utils.data import DataLoader, Dataset

from pinn4csi.data import paper1 as paper1_data
from pinn4csi.models import (
    Paper1Model,
    Paper1ModelFactoryConfig,
    Paper1ModelSpec,
    create_paper1_model,
    expand_paper1_model_specs,
)
from pinn4csi.utils import save_dataclass_rows_csv, save_json_file

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Paper1ExperimentConfig:
    prepared_root: Path
    dataset_names: tuple[str, ...] = ("signfi", "ut_har")
    model_names: tuple[str, ...] = (
        "mlp",
        "cnn",
        "cnn_gru",
        "dgsense_lite",
        "autoencoder",
        "residual_prior",
    )
    seeds: tuple[int, ...] = tuple(range(10))
    batch_size: int = 32
    epochs: int = 12
    learning_rate: float = 1e-3
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    test_ratio: float = 0.15
    hidden_dim: int = 64
    num_layers: int = 3
    latent_dim: int = 32
    reconstruction_weight: float = 0.2
    ofdm_weight: float = 0.05
    path_weight: float = 0.05
    num_workers: int = 0
    output_csv: Path = Path("outputs/paper1_results.csv")
    analysis_json: Path = Path("outputs/paper1_analysis.json")
    summary_json: Path = Path("outputs/paper1_summary.json")
    summary_latex: Path = Path("outputs/paper1_summary.tex")
    embeddings_dir: Path | None = None
    summary_split: str = "test"
    include_loss_ablation_variants: bool = True
    include_fixed_weight_variant: bool = True
    include_fourier_variant: bool = True
    include_train_val_test_reference_for_cross_environment: bool = True


@dataclass(frozen=True)
class Paper1ResultRow:
    dataset_name: str
    model_name: str
    variant_name: str
    comparison_name: str
    seed: int
    eval_mode: str
    split: str
    train_environment_ids: str
    test_environment_ids: str
    uses_reconstruction_loss: bool
    uses_ofdm_loss: bool
    uses_path_loss: bool
    uses_fourier_features: bool
    uses_adaptive_physics_weighting: bool
    physics_weighting_mode: str
    requires_prior: bool
    accuracy: float
    macro_f1: float
    loss_total: float
    loss_task: float
    loss_reconstruction: float
    loss_ofdm: float
    loss_path: float
    best_epoch: int
    num_examples: int


@dataclass(frozen=True)
class Paper1SummaryRow:
    dataset_name: str
    model_name: str
    variant_name: str
    comparison_name: str
    eval_mode: str
    split: str
    num_runs: int
    seeds: str
    accuracy_mean: float
    accuracy_std: float
    macro_f1_mean: float
    macro_f1_std: float
    loss_total_mean: float
    loss_total_std: float
    loss_task_mean: float
    loss_task_std: float
    loss_reconstruction_mean: float
    loss_reconstruction_std: float
    loss_ofdm_mean: float
    loss_ofdm_std: float
    loss_path_mean: float
    loss_path_std: float
    reference_comparison_name: str
    paired_seed_count: int
    paired_accuracy_mean_delta: float | None
    paired_accuracy_std_delta: float | None
    paired_accuracy_wins: int
    paired_accuracy_losses: int
    paired_accuracy_ties: int
    paired_accuracy_sign_test_pvalue: float | None
    paired_macro_f1_mean_delta: float | None
    paired_loss_total_mean_delta: float | None


def run_paper1_experiments(config: Paper1ExperimentConfig) -> list[Paper1ResultRow]:
    """Execute the Paper 1 harness across datasets, models, and seeds."""
    results: list[Paper1ResultRow] = []
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    factory_config = Paper1ModelFactoryConfig(
        hidden_dim=config.hidden_dim,
        num_layers=config.num_layers,
        latent_dim=config.latent_dim,
        reconstruction_weight=config.reconstruction_weight,
        ofdm_weight=config.ofdm_weight,
        path_weight=config.path_weight,
    )
    model_specs = expand_paper1_model_specs(
        model_names=config.model_names,
        include_loss_ablation_variants=config.include_loss_ablation_variants,
        include_fixed_weight_variant=config.include_fixed_weight_variant,
        include_fourier_variant=config.include_fourier_variant,
    )

    for dataset_name in config.dataset_names:
        bundle = paper1_data.load_prepared_paper1_dataset(
            dataset_name=dataset_name,
            prepared_root=config.prepared_root,
        )
        # Detect whether this dataset has meaningful phase information.
        # Features are amplitude/phase-expanded: [..., amp0, phase0, amp1, ...]
        # If phase channels are all near-zero, phase-dependent losses are harmful.
        phase_channels = bundle.features[..., 1::2]
        phase_energy = float(torch.mean(phase_channels**2).item())
        dataset_has_phase = phase_energy > 1e-4
        if not dataset_has_phase:
            logger.warning(
                "Phase absent in %s (phase energy=%.2e) — "
                "phase-dependent physics losses will be disabled.",
                dataset_name,
                phase_energy,
            )
        for seed in config.seeds:
            set_experiment_seed(seed)
            for eval_mode in _resolve_eval_modes(
                bundle.config.evaluation_mode,
                config.include_train_val_test_reference_for_cross_environment,
            ):
                splits = paper1_data.create_paper1_splits(
                    bundle=bundle,
                    seed=seed,
                    train_ratio=config.train_ratio,
                    val_ratio=config.val_ratio,
                    test_ratio=config.test_ratio,
                    evaluation_mode=eval_mode,
                )
                train_loader = _make_loader(
                    splits.train,
                    batch_size=config.batch_size,
                    shuffle=True,
                    num_workers=config.num_workers,
                )
                val_loader = _make_loader(
                    splits.val,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                )
                test_loader = _make_loader(
                    splits.test,
                    batch_size=config.batch_size,
                    shuffle=False,
                    num_workers=config.num_workers,
                )

                for raw_spec in model_specs:
                    model_spec = raw_spec
                    if not dataset_has_phase and (
                        raw_spec.use_ofdm_loss or raw_spec.use_path_loss
                    ):
                        model_spec = replace(
                            raw_spec,
                            use_ofdm_loss=False,
                            use_path_loss=False,
                        )
                    set_experiment_seed(seed)
                    model = create_paper1_model(
                        model_name=model_spec,
                        input_shape=bundle.input_shape,
                        num_classes=bundle.num_classes,
                        config=factory_config,
                    ).to(device)
                    optimizer = Adam(model.parameters(), lr=config.learning_rate)
                    best_epoch, best_state = _train_model(
                        model=model,
                        optimizer=optimizer,
                        train_loader=train_loader,
                        val_loader=val_loader,
                        device=device,
                        epochs=config.epochs,
                    )
                    model.load_state_dict(best_state)

                    val_metrics = evaluate_paper1_model(model, val_loader, device)
                    test_metrics = evaluate_paper1_model(model, test_loader, device)
                    train_env_ids = _format_environment_ids(
                        splits.train_environment_ids
                    )
                    test_env_ids = _format_environment_ids(splits.test_environment_ids)
                    results.extend(
                        [
                            _build_result_row(
                                dataset_name=bundle.config.name,
                                model_spec=model_spec,
                                model=model,
                                seed=seed,
                                eval_mode=splits.eval_mode,
                                split="val",
                                metrics=val_metrics,
                                best_epoch=best_epoch,
                                train_environment_ids=train_env_ids,
                                test_environment_ids=test_env_ids,
                            ),
                            _build_result_row(
                                dataset_name=bundle.config.name,
                                model_spec=model_spec,
                                model=model,
                                seed=seed,
                                eval_mode=splits.eval_mode,
                                split="test",
                                metrics=test_metrics,
                                best_epoch=best_epoch,
                                train_environment_ids=train_env_ids,
                                test_environment_ids=test_env_ids,
                            ),
                        ]
                    )
                    if config.embeddings_dir is not None:
                        _export_latent_embeddings(
                            model=model,
                            loader=test_loader,
                            device=device,
                            output_path=_build_embedding_output_path(
                                config.embeddings_dir,
                                dataset_name=bundle.config.name,
                                model_spec=model_spec,
                                eval_mode=splits.eval_mode,
                                seed=seed,
                                split="test",
                            ),
                            dataset_name=bundle.config.name,
                            model_spec=model_spec,
                            eval_mode=splits.eval_mode,
                            split="test",
                            seed=seed,
                        )
    return results


def save_paper1_results_csv(
    results: list[Paper1ResultRow], output_csv: str | Path
) -> None:
    save_dataclass_rows_csv(
        results,
        output_csv,
        fieldnames=[field.name for field in fields(Paper1ResultRow)],
    )


def summarize_paper1_results(
    results: list[Paper1ResultRow], split: str = "test"
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for row in build_paper1_summary_rows(results, split=split):
        summary[
            f"{row.dataset_name}:{row.model_name}:{row.variant_name}:{row.eval_mode}"
        ] = {
            "accuracy": row.accuracy_mean,
            "accuracy_std": row.accuracy_std,
            "macro_f1": row.macro_f1_mean,
            "macro_f1_std": row.macro_f1_std,
            "loss_total": row.loss_total_mean,
            "loss_total_std": row.loss_total_std,
            "loss_task": row.loss_task_mean,
            "loss_task_std": row.loss_task_std,
            "loss_reconstruction": row.loss_reconstruction_mean,
            "loss_reconstruction_std": row.loss_reconstruction_std,
            "loss_ofdm": row.loss_ofdm_mean,
            "loss_ofdm_std": row.loss_ofdm_std,
            "loss_path": row.loss_path_mean,
            "loss_path_std": row.loss_path_std,
            "num_runs": row.num_runs,
            "seeds": row.seeds,
            "reference_comparison_name": row.reference_comparison_name,
            "paired_seed_count": row.paired_seed_count,
            "paired_accuracy_mean_delta": row.paired_accuracy_mean_delta,
            "paired_accuracy_std_delta": row.paired_accuracy_std_delta,
            "paired_accuracy_wins": row.paired_accuracy_wins,
            "paired_accuracy_losses": row.paired_accuracy_losses,
            "paired_accuracy_ties": row.paired_accuracy_ties,
            "paired_accuracy_sign_test_pvalue": (row.paired_accuracy_sign_test_pvalue),
            "paired_macro_f1_mean_delta": row.paired_macro_f1_mean_delta,
            "paired_loss_total_mean_delta": row.paired_loss_total_mean_delta,
        }
    return summary


def analyze_paper1_results(
    results: list[Paper1ResultRow], split: str = "test"
) -> dict[str, Any]:
    filtered = [row for row in results if row.split == split]
    summary_rows = build_paper1_summary_rows(results, split=split)
    return {
        "split": split,
        "num_rows": len(filtered),
        "summary_row_count": len(summary_rows),
        "summary_rows": [asdict(row) for row in summary_rows],
        "paired_variant_comparisons": _build_paired_variant_comparisons(summary_rows),
        "per_model_summary": _aggregate_rows(
            filtered,
            group_fields=("model_name", "variant_name", "eval_mode"),
        ),
        "per_dataset_summary": _summarize_by_dataset(filtered),
        "ut_har_cross_environment": _build_ut_har_cross_environment(filtered),
    }


def save_paper1_analysis_json(
    analysis: dict[str, Any], output_json: str | Path
) -> None:
    save_json_file(analysis, output_json)


def build_paper1_summary_rows(
    results: list[Paper1ResultRow], split: str = "test"
) -> list[Paper1SummaryRow]:
    filtered = [row for row in results if row.split == split]
    grouped: dict[tuple[str, str, str, str], list[Paper1ResultRow]] = {}
    for row in filtered:
        grouped.setdefault(
            (row.dataset_name, row.model_name, row.variant_name, row.eval_mode),
            [],
        ).append(row)

    summary_rows: list[Paper1SummaryRow] = []
    for dataset_name, model_name, variant_name, eval_mode in sorted(grouped):
        rows = sorted(
            grouped[(dataset_name, model_name, variant_name, eval_mode)],
            key=lambda row: row.seed,
        )
        paired_stats = _build_paired_seed_statistics(
            rows=rows,
            grouped=grouped,
            dataset_name=dataset_name,
            model_name=model_name,
            variant_name=variant_name,
            eval_mode=eval_mode,
        )
        summary_rows.append(
            Paper1SummaryRow(
                dataset_name=dataset_name,
                model_name=model_name,
                variant_name=variant_name,
                comparison_name=f"{model_name}:{variant_name}",
                eval_mode=eval_mode,
                split=split,
                num_runs=len(rows),
                seeds=",".join(str(row.seed) for row in rows),
                accuracy_mean=float(np.mean([row.accuracy for row in rows])),
                accuracy_std=float(np.std([row.accuracy for row in rows])),
                macro_f1_mean=float(np.mean([row.macro_f1 for row in rows])),
                macro_f1_std=float(np.std([row.macro_f1 for row in rows])),
                loss_total_mean=float(np.mean([row.loss_total for row in rows])),
                loss_total_std=float(np.std([row.loss_total for row in rows])),
                loss_task_mean=float(np.mean([row.loss_task for row in rows])),
                loss_task_std=float(np.std([row.loss_task for row in rows])),
                loss_reconstruction_mean=float(
                    np.mean([row.loss_reconstruction for row in rows])
                ),
                loss_reconstruction_std=float(
                    np.std([row.loss_reconstruction for row in rows])
                ),
                loss_ofdm_mean=float(np.mean([row.loss_ofdm for row in rows])),
                loss_ofdm_std=float(np.std([row.loss_ofdm for row in rows])),
                loss_path_mean=float(np.mean([row.loss_path for row in rows])),
                loss_path_std=float(np.std([row.loss_path for row in rows])),
                reference_comparison_name=paired_stats["reference_comparison_name"],
                paired_seed_count=paired_stats["paired_seed_count"],
                paired_accuracy_mean_delta=paired_stats["paired_accuracy_mean_delta"],
                paired_accuracy_std_delta=paired_stats["paired_accuracy_std_delta"],
                paired_accuracy_wins=paired_stats["paired_accuracy_wins"],
                paired_accuracy_losses=paired_stats["paired_accuracy_losses"],
                paired_accuracy_ties=paired_stats["paired_accuracy_ties"],
                paired_accuracy_sign_test_pvalue=paired_stats[
                    "paired_accuracy_sign_test_pvalue"
                ],
                paired_macro_f1_mean_delta=paired_stats["paired_macro_f1_mean_delta"],
                paired_loss_total_mean_delta=paired_stats[
                    "paired_loss_total_mean_delta"
                ],
            )
        )
    return summary_rows


def save_paper1_summary_json(
    summary_rows: list[Paper1SummaryRow], output_json: str | Path
) -> None:
    save_json_file([asdict(row) for row in summary_rows], output_json)


def render_paper1_summary_latex(summary_rows: list[Paper1SummaryRow]) -> str:
    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        "\\begin{tabular}{llllllrr}",
        "\\hline",
        (
            "Dataset & Eval mode & Model & Variant & Accuracy & Macro-F1 & "
            "Delta acc. & Sign p \\\\"
        ),
        "\\hline",
    ]
    for row in summary_rows:
        delta = "--"
        if row.paired_accuracy_mean_delta is not None:
            delta = f"{row.paired_accuracy_mean_delta:+.4f}"
        pvalue = "--"
        if row.paired_accuracy_sign_test_pvalue is not None:
            pvalue = f"{row.paired_accuracy_sign_test_pvalue:.4f}"
        lines.append(
            " & ".join(
                [
                    _escape_latex_cell(row.dataset_name),
                    _escape_latex_cell(row.eval_mode),
                    _escape_latex_cell(row.model_name),
                    _escape_latex_cell(row.variant_name),
                    f"{row.accuracy_mean:.4f} $\\pm$ {row.accuracy_std:.4f}",
                    f"{row.macro_f1_mean:.4f} $\\pm$ {row.macro_f1_std:.4f}",
                    delta,
                    pvalue,
                ]
            )
            + " \\\\"
        )
    lines.extend(
        [
            "\\hline",
            "\\end{tabular}",
            (
                "\\caption{Mock Paper 1 summary on prepared-format data. "
                "Delta accuracy and sign-test p-values compare each variant "
                "against the same-model default when repeated seeds are "
                "available.}"
            ),
            "\\label{tab:paper1-mock-summary}",
            "\\end{table}",
        ]
    )
    return "\n".join(lines) + "\n"


def save_paper1_summary_latex(
    summary_rows: list[Paper1SummaryRow], output_path: str | Path
) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_paper1_summary_latex(summary_rows), encoding="utf-8")


def evaluate_paper1_model(
    model: Paper1Model,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> dict[str, float | int]:
    model.eval()
    loss_totals = {
        "loss_total": 0.0,
        "loss_task": 0.0,
        "loss_reconstruction": 0.0,
        "loss_ofdm": 0.0,
        "loss_path": 0.0,
    }
    total_examples = 0
    predictions: list[Tensor] = []
    targets: list[Tensor] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            losses = model.compute_batch_losses(
                x=moved["x"],
                labels=moved["label"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )
            logits = losses["logits"]
            predictions.append(torch.argmax(logits, dim=1).cpu())
            targets.append(moved["label"].cpu())
            batch_size = int(moved["label"].shape[0])
            for key in loss_totals:
                loss_totals[key] += float(losses[key].item()) * batch_size
            total_examples += batch_size

    predictions_tensor = torch.cat(predictions, dim=0)
    targets_tensor = torch.cat(targets, dim=0)
    return {
        "accuracy": _accuracy(predictions_tensor, targets_tensor),
        "macro_f1": _macro_f1(predictions_tensor, targets_tensor),
        "loss_total": loss_totals["loss_total"] / max(total_examples, 1),
        "loss_task": loss_totals["loss_task"] / max(total_examples, 1),
        "loss_reconstruction": loss_totals["loss_reconstruction"]
        / max(total_examples, 1),
        "loss_ofdm": loss_totals["loss_ofdm"] / max(total_examples, 1),
        "loss_path": loss_totals["loss_path"] / max(total_examples, 1),
        "num_examples": total_examples,
    }


def set_experiment_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def _train_model(
    model: Paper1Model,
    optimizer: Adam,
    train_loader: DataLoader[dict[str, Tensor]],
    val_loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    epochs: int,
) -> tuple[int, dict[str, Tensor]]:
    best_epoch = -1
    best_accuracy = -1.0
    best_loss = float("inf")
    best_state: dict[str, Tensor] | None = None

    for epoch_idx in range(epochs):
        model.train()
        for batch in train_loader:
            moved = _move_batch_to_device(batch, device)
            optimizer.zero_grad()
            losses = model.compute_batch_losses(
                x=moved["x"],
                labels=moved["label"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )
            losses["loss_total"].backward()  # type: ignore[no-untyped-call]
            optimizer.step()

        val_metrics = evaluate_paper1_model(model, val_loader, device)
        val_accuracy = float(val_metrics["accuracy"])
        val_loss = float(val_metrics["loss_total"])
        if val_accuracy > best_accuracy or (
            np.isclose(val_accuracy, best_accuracy) and val_loss < best_loss
        ):
            best_epoch = epoch_idx
            best_accuracy = val_accuracy
            best_loss = val_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is None:
        raise RuntimeError("No validation checkpoint selected during Paper 1 training.")
    return best_epoch, best_state


def _move_batch_to_device(
    batch: dict[str, Tensor], device: torch.device
) -> dict[str, Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


def _optional_prior(batch: dict[str, Tensor]) -> Tensor | None:
    if not bool(torch.all(batch["has_prior"]).item()):
        return None
    return batch["prior"]


def _make_loader(
    dataset: Dataset[dict[str, Tensor]],
    batch_size: int,
    shuffle: bool,
    num_workers: int,
) -> DataLoader[dict[str, Tensor]]:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
    )


def _accuracy(predictions: Tensor, targets: Tensor) -> float:
    return float((predictions == targets).float().mean().item())


def _macro_f1(predictions: Tensor, targets: Tensor) -> float:
    classes = sorted({int(value) for value in targets.tolist()})
    if not classes:
        return 0.0
    f1_values: list[float] = []
    for class_idx in classes:
        pred_positive = predictions == class_idx
        target_positive = targets == class_idx
        tp = int(torch.sum(pred_positive & target_positive).item())
        fp = int(torch.sum(pred_positive & ~target_positive).item())
        fn = int(torch.sum(~pred_positive & target_positive).item())
        precision = tp / (tp + fp) if tp + fp > 0 else 0.0
        recall = tp / (tp + fn) if tp + fn > 0 else 0.0
        if precision + recall == 0.0:
            f1_values.append(0.0)
        else:
            f1_values.append(2.0 * precision * recall / (precision + recall))
    return float(sum(f1_values) / len(f1_values))


def _resolve_eval_modes(
    evaluation_mode: paper1_data.EvaluationMode,
    include_train_val_test_reference_for_cross_environment: bool,
) -> tuple[paper1_data.EvaluationMode, ...]:
    if (
        evaluation_mode == "cross_environment"
        and include_train_val_test_reference_for_cross_environment
    ):
        return ("train_val_test", "cross_environment")
    return (evaluation_mode,)


def _format_environment_ids(environment_ids: tuple[int, ...] | None) -> str:
    if environment_ids is None:
        return "all"
    return ",".join(str(environment_id) for environment_id in environment_ids)


def _build_result_row(
    dataset_name: str,
    model_spec: Paper1ModelSpec,
    model: Paper1Model,
    seed: int,
    eval_mode: str,
    split: str,
    metrics: dict[str, float | int],
    best_epoch: int,
    train_environment_ids: str,
    test_environment_ids: str,
) -> Paper1ResultRow:
    return Paper1ResultRow(
        dataset_name=dataset_name,
        model_name=model_spec.model_name,
        variant_name=model_spec.variant_name,
        comparison_name=model_spec.comparison_name,
        seed=seed,
        eval_mode=eval_mode,
        split=split,
        train_environment_ids=train_environment_ids,
        test_environment_ids=test_environment_ids,
        uses_reconstruction_loss=model.uses_reconstruction_loss,
        uses_ofdm_loss=model.uses_ofdm_loss,
        uses_path_loss=model.uses_path_loss,
        uses_fourier_features=model.uses_fourier_features,
        uses_adaptive_physics_weighting=model.uses_adaptive_physics_weighting,
        physics_weighting_mode=model.physics_weighting_mode,
        requires_prior=model.requires_prior,
        accuracy=float(metrics["accuracy"]),
        macro_f1=float(metrics["macro_f1"]),
        loss_total=float(metrics["loss_total"]),
        loss_task=float(metrics["loss_task"]),
        loss_reconstruction=float(metrics["loss_reconstruction"]),
        loss_ofdm=float(metrics["loss_ofdm"]),
        loss_path=float(metrics["loss_path"]),
        best_epoch=best_epoch,
        num_examples=int(metrics["num_examples"]),
    )


def _aggregate_rows(
    rows: list[Paper1ResultRow],
    group_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    metric_fields = (
        "accuracy",
        "macro_f1",
        "loss_total",
        "loss_task",
        "loss_reconstruction",
        "loss_ofdm",
        "loss_path",
    )
    grouped: dict[tuple[Any, ...], list[Paper1ResultRow]] = {}
    for row in rows:
        key = tuple(getattr(row, field) for field in group_fields)
        grouped.setdefault(key, []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for key in sorted(grouped):
        group_rows = grouped[key]
        entry = {field: value for field, value in zip(group_fields, key, strict=True)}
        entry["num_runs"] = len(group_rows)
        for metric_field in metric_fields:
            values = [getattr(row, metric_field) for row in group_rows]
            entry[f"{metric_field}_mean"] = float(np.mean(values))
            entry[f"{metric_field}_std"] = float(np.std(values))
        summary_rows.append(entry)
    return summary_rows


def _build_paired_variant_comparisons(
    summary_rows: list[Paper1SummaryRow],
) -> list[dict[str, Any]]:
    comparisons: list[dict[str, Any]] = []
    for row in summary_rows:
        if not row.reference_comparison_name:
            continue
        comparisons.append(
            {
                "dataset_name": row.dataset_name,
                "eval_mode": row.eval_mode,
                "comparison_name": row.comparison_name,
                "reference_comparison_name": row.reference_comparison_name,
                "paired_seed_count": row.paired_seed_count,
                "paired_accuracy_mean_delta": row.paired_accuracy_mean_delta,
                "paired_accuracy_std_delta": row.paired_accuracy_std_delta,
                "paired_accuracy_wins": row.paired_accuracy_wins,
                "paired_accuracy_losses": row.paired_accuracy_losses,
                "paired_accuracy_ties": row.paired_accuracy_ties,
                "paired_accuracy_sign_test_pvalue": (
                    row.paired_accuracy_sign_test_pvalue
                ),
                "paired_macro_f1_mean_delta": row.paired_macro_f1_mean_delta,
                "paired_loss_total_mean_delta": row.paired_loss_total_mean_delta,
            }
        )
    return comparisons


def _build_paired_seed_statistics(
    rows: list[Paper1ResultRow],
    grouped: dict[tuple[str, str, str, str], list[Paper1ResultRow]],
    dataset_name: str,
    model_name: str,
    variant_name: str,
    eval_mode: str,
) -> dict[str, Any]:
    empty_stats: dict[str, Any] = {
        "reference_comparison_name": "",
        "paired_seed_count": 0,
        "paired_accuracy_mean_delta": None,
        "paired_accuracy_std_delta": None,
        "paired_accuracy_wins": 0,
        "paired_accuracy_losses": 0,
        "paired_accuracy_ties": 0,
        "paired_accuracy_sign_test_pvalue": None,
        "paired_macro_f1_mean_delta": None,
        "paired_loss_total_mean_delta": None,
    }
    if variant_name == "default":
        return empty_stats

    reference_rows = grouped.get((dataset_name, model_name, "default", eval_mode))
    if not reference_rows:
        return empty_stats

    rows_by_seed = {row.seed: row for row in rows}
    reference_by_seed = {row.seed: row for row in reference_rows}
    paired_seeds = sorted(set(rows_by_seed) & set(reference_by_seed))
    if not paired_seeds:
        return empty_stats

    accuracy_deltas = [
        rows_by_seed[seed].accuracy - reference_by_seed[seed].accuracy
        for seed in paired_seeds
    ]
    macro_f1_deltas = [
        rows_by_seed[seed].macro_f1 - reference_by_seed[seed].macro_f1
        for seed in paired_seeds
    ]
    loss_total_deltas = [
        rows_by_seed[seed].loss_total - reference_by_seed[seed].loss_total
        for seed in paired_seeds
    ]
    wins = sum(delta > 0.0 for delta in accuracy_deltas)
    losses = sum(delta < 0.0 for delta in accuracy_deltas)
    ties = len(accuracy_deltas) - wins - losses
    pvalue: float | None = None
    if len(paired_seeds) >= 2 and wins + losses > 0:
        pvalue = _two_sided_sign_test_pvalue(wins=wins, losses=losses)

    return {
        "reference_comparison_name": f"{model_name}:default",
        "paired_seed_count": len(paired_seeds),
        "paired_accuracy_mean_delta": float(np.mean(accuracy_deltas)),
        "paired_accuracy_std_delta": float(np.std(accuracy_deltas)),
        "paired_accuracy_wins": wins,
        "paired_accuracy_losses": losses,
        "paired_accuracy_ties": ties,
        "paired_accuracy_sign_test_pvalue": pvalue,
        "paired_macro_f1_mean_delta": float(np.mean(macro_f1_deltas)),
        "paired_loss_total_mean_delta": float(np.mean(loss_total_deltas)),
    }


def _two_sided_sign_test_pvalue(wins: int, losses: int) -> float:
    decisive = wins + losses
    tail = min(wins, losses)
    probability = sum(math.comb(decisive, k) for k in range(tail + 1)) / (2**decisive)
    return float(min(1.0, 2.0 * probability))


def _escape_latex_cell(value: str) -> str:
    return value.replace("_", "\\_")


def _summarize_by_dataset(rows: list[Paper1ResultRow]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[Paper1ResultRow]] = {}
    for row in rows:
        grouped.setdefault((row.dataset_name, row.eval_mode), []).append(row)

    summary_rows: list[dict[str, Any]] = []
    for dataset_name, eval_mode in sorted(grouped):
        group_rows = grouped[(dataset_name, eval_mode)]
        grouped_comparisons = _aggregate_rows(
            group_rows,
            group_fields=("model_name", "variant_name"),
        )
        best_entry = max(
            grouped_comparisons,
            key=lambda entry: float(entry["accuracy_mean"]),
        )
        summary_rows.append(
            {
                "dataset_name": dataset_name,
                "eval_mode": eval_mode,
                "num_model_variants": len(grouped_comparisons),
                "best_model_name": best_entry["model_name"],
                "best_variant_name": best_entry["variant_name"],
                "best_accuracy": float(best_entry["accuracy_mean"]),
                "best_macro_f1": float(best_entry["macro_f1_mean"]),
                "best_loss_total": float(best_entry["loss_total_mean"]),
            }
        )
    return summary_rows


def _build_ut_har_cross_environment(
    rows: list[Paper1ResultRow],
) -> list[dict[str, Any]]:
    ut_har_rows = [row for row in rows if row.dataset_name == "UT_HAR"]
    in_domain = _aggregate_rows(
        [row for row in ut_har_rows if row.eval_mode == "train_val_test"],
        group_fields=("model_name", "variant_name"),
    )
    cross_environment = _aggregate_rows(
        [row for row in ut_har_rows if row.eval_mode == "cross_environment"],
        group_fields=("model_name", "variant_name"),
    )
    if not in_domain or not cross_environment:
        return []

    in_domain_map = {
        (str(entry["model_name"]), str(entry["variant_name"])): entry
        for entry in in_domain
    }
    cross_environment_map = {
        (str(entry["model_name"]), str(entry["variant_name"])): entry
        for entry in cross_environment
    }
    comparison_rows: list[dict[str, Any]] = []
    for key in sorted(set(in_domain_map) & set(cross_environment_map)):
        in_domain_entry = in_domain_map[key]
        cross_entry = cross_environment_map[key]
        comparison_rows.append(
            {
                "model_name": key[0],
                "variant_name": key[1],
                "comparison_name": f"{key[0]}:{key[1]}",
                "train_val_test_accuracy": float(in_domain_entry["accuracy_mean"]),
                "cross_environment_accuracy": float(cross_entry["accuracy_mean"]),
                "accuracy_gap": float(
                    in_domain_entry["accuracy_mean"] - cross_entry["accuracy_mean"]
                ),
                "train_val_test_macro_f1": float(in_domain_entry["macro_f1_mean"]),
                "cross_environment_macro_f1": float(cross_entry["macro_f1_mean"]),
                "cross_environment_loss_reconstruction": float(
                    cross_entry["loss_reconstruction_mean"]
                ),
            }
        )
    return comparison_rows


def _build_embedding_output_path(
    embeddings_dir: Path,
    dataset_name: str,
    model_spec: Paper1ModelSpec,
    eval_mode: str,
    seed: int,
    split: str,
) -> Path:
    safe_dataset = dataset_name.lower()
    filename = (
        f"{safe_dataset}_{model_spec.model_name}_{model_spec.variant_name}_"
        f"{eval_mode}_seed{seed}_{split}.pt"
    )
    return embeddings_dir / filename


def _export_latent_embeddings(
    model: Paper1Model,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    output_path: Path,
    dataset_name: str,
    model_spec: Paper1ModelSpec,
    eval_mode: str,
    split: str,
    seed: int,
) -> None:
    model.eval()
    latents: list[Tensor] = []
    labels: list[Tensor] = []
    predictions: list[Tensor] = []
    environments: list[Tensor] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            outputs = model.forward(
                moved["x"],
                prior=_optional_prior(moved),
            )
            latent = outputs.get("latent")
            if latent is None:
                return
            latents.append(latent.cpu())
            logits = outputs.get("logits")
            if logits is not None:
                predictions.append(torch.argmax(logits, dim=1).cpu())
            labels.append(moved["label"].cpu())
            environment = moved.get("environment")
            if environment is not None:
                environments.append(environment.cpu())

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "dataset_name": dataset_name,
        "model_name": model_spec.model_name,
        "variant_name": model_spec.variant_name,
        "eval_mode": eval_mode,
        "split": split,
        "seed": seed,
        "latents": torch.cat(latents, dim=0),
        "labels": torch.cat(labels, dim=0),
    }
    if predictions:
        payload["predictions"] = torch.cat(predictions, dim=0)
    if environments:
        payload["environments"] = torch.cat(environments, dim=0)
    torch.save(payload, output_path)
