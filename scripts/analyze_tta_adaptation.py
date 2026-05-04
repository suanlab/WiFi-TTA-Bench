# pyright: basic, reportMissingImports=false

"""Post-hoc analysis of TTA adaptation: latent space visualization,
per-sample harm analysis, shift-magnitude vs gain curves, and
physics alignment diagnostics.

Generates figures for NeurIPS paper.
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pinn4csi.models import (
    Paper2DomainAdaptationBaseline,
)
from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    TTAReferenceStats,
    _move_batch_to_device,
    _optional_prior,
    adapt_model_for_tta,
    build_synthetic_tta_loaders,
    compute_tta_reference_stats,
    train_source_only_tta_model,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LatentSnapshot:
    """Embedding snapshot for visualization."""

    features: Tensor  # (N, latent_dim)
    invariant_features: Tensor  # (N, latent_dim)
    labels: Tensor  # (N,)
    predictions: Tensor  # (N,)
    confidences: Tensor  # (N,)
    domain: str  # "source" or "target"
    phase: str  # "before" or "after"


@dataclass(frozen=True)
class HarmAnalysisRow:
    """Per-sample harm analysis."""

    sample_idx: int
    label: int
    source_pred: int
    adapted_pred: int
    source_confidence: float
    adapted_confidence: float
    alignment_divergence: float
    source_correct: bool
    adapted_correct: bool
    harmed: bool
    helped: bool


@dataclass(frozen=True)
class ShiftGainPoint:
    """One point in shift-magnitude vs gain curve."""

    shift_level: str
    shift_magnitude: float
    seed: int
    method: str
    gain: float
    source_drop: float
    negative_adaptation_rate: float


def collect_latent_snapshot(
    model: Paper2DomainAdaptationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
    domain: str,
    phase: str,
) -> LatentSnapshot:
    """Collect latent embeddings from model for visualization."""
    model.eval()
    features_list: list[Tensor] = []
    invariant_list: list[Tensor] = []
    labels_list: list[Tensor] = []
    preds_list: list[Tensor] = []
    confs_list: list[Tensor] = []

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            outputs = model(
                moved["x"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )
            logits = outputs["task_logits"]
            probs = torch.softmax(logits, dim=-1)
            conf, pred = probs.max(dim=-1)

            features_list.append(outputs["features"].cpu())
            invariant_list.append(outputs["invariant_features"].cpu())
            labels_list.append(moved["label"].cpu())
            preds_list.append(pred.cpu())
            confs_list.append(conf.cpu())

    return LatentSnapshot(
        features=torch.cat(features_list, dim=0),
        invariant_features=torch.cat(invariant_list, dim=0),
        labels=torch.cat(labels_list, dim=0),
        predictions=torch.cat(preds_list, dim=0),
        confidences=torch.cat(confs_list, dim=0),
        domain=domain,
        phase=phase,
    )


def compute_tsne_embeddings(
    snapshots: list[LatentSnapshot],
) -> list[dict[str, object]]:
    """Compute 2D t-SNE embeddings from multiple snapshots.

    Returns list of dicts with 'x', 'y', 'label', 'domain', 'phase' keys.
    Falls back to PCA if sklearn is not available.
    """
    all_features = torch.cat([s.invariant_features for s in snapshots], dim=0)
    all_labels = torch.cat([s.labels for s in snapshots], dim=0)
    domains = []
    phases = []
    for s in snapshots:
        n = s.labels.shape[0]
        domains.extend([s.domain] * n)
        phases.extend([s.phase] * n)

    features_np = all_features.numpy()

    try:
        from sklearn.manifold import TSNE

        reducer = TSNE(
            n_components=2,
            perplexity=min(30, len(features_np) - 1),
            random_state=42,
        )
        coords_2d = reducer.fit_transform(features_np)
    except ImportError:
        # Fallback to PCA
        centered = features_np - features_np.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        coords_2d = centered @ vt[:2].T

    results = []
    for i in range(len(all_labels)):
        results.append(
            {
                "x": float(coords_2d[i, 0]),
                "y": float(coords_2d[i, 1]),
                "label": int(all_labels[i].item()),
                "domain": domains[i],
                "phase": phases[i],
            }
        )
    return results


def run_harm_analysis(
    source_model: Paper2DomainAdaptationBaseline,
    adapted_model: Paper2DomainAdaptationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    reference_stats: TTAReferenceStats,
    device: torch.device,
) -> list[HarmAnalysisRow]:
    """Analyze per-sample impact of TTA: which samples are helped vs harmed."""
    source_model.eval()
    adapted_model.eval()
    rows: list[HarmAnalysisRow] = []
    sample_idx = 0

    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            x = moved["x"]
            labels = moved["label"]
            prior = _optional_prior(moved)
            has_prior = moved["has_prior"]

            source_out = source_model(x, prior=prior, has_prior=has_prior)
            adapted_out = adapted_model(x, prior=prior, has_prior=has_prior)

            src_probs = torch.softmax(source_out["task_logits"], dim=-1)
            adp_probs = torch.softmax(adapted_out["task_logits"], dim=-1)
            src_conf, src_pred = src_probs.max(dim=-1)
            adp_conf, adp_pred = adp_probs.max(dim=-1)

            # Physics alignment divergence
            adp_inv = adapted_out["invariant_features"]
            ref_mean = reference_stats.invariant_mean.to(device)
            ref_var = reference_stats.invariant_var.to(device)
            alignment_div = torch.mean(
                (adp_inv - ref_mean.unsqueeze(0)) ** 2 / (ref_var.unsqueeze(0) + 1e-8),
                dim=-1,
            )

            for i in range(x.shape[0]):
                label = int(labels[i].item())
                src_c = src_pred[i].item() == label
                adp_c = adp_pred[i].item() == label
                rows.append(
                    HarmAnalysisRow(
                        sample_idx=sample_idx,
                        label=label,
                        source_pred=int(src_pred[i].item()),
                        adapted_pred=int(adp_pred[i].item()),
                        source_confidence=float(src_conf[i].item()),
                        adapted_confidence=float(adp_conf[i].item()),
                        alignment_divergence=float(alignment_div[i].item()),
                        source_correct=src_c,
                        adapted_correct=adp_c,
                        harmed=src_c and not adp_c,
                        helped=not src_c and adp_c,
                    )
                )
                sample_idx += 1

    return rows


def run_shift_gain_curve(
    seeds: tuple[int, ...] = (0, 1, 2),
    methods: tuple[str, ...] = (
        "no_adapt",
        "entropy_tta",
        "physics_tta",
        "selective_physics_tta",
    ),
) -> list[ShiftGainPoint]:
    """Compute gain as a function of shift magnitude across methods."""
    from pinn4csi.training.paper2_tta import run_tta_suite

    shift_magnitudes = {"mild": 0.25, "moderate": 0.55, "strong": 0.85}
    points: list[ShiftGainPoint] = []
    config = TTAExperimentConfig(
        methods=methods,
        source_epochs=3,
        adaptation_steps=3,
        batch_size=16,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
        selective_confidence_threshold=0.7,
        selective_alignment_threshold=5.0,
    )

    for shift_name, shift_mag in shift_magnitudes.items():
        for seed in seeds:
            (
                source_train_loader,
                source_val_loader,
                target_adapt_loader,
                target_test_loader,
                input_shape,
                num_classes,
                metadata,
            ) = build_synthetic_tta_loaders(
                shift_name=shift_name,
                seed=seed,
                batch_size=config.batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
            )

            rows = run_tta_suite(
                source_train_loader=source_train_loader,
                source_val_loader=source_val_loader,
                target_adapt_loader=target_adapt_loader,
                target_test_loader=target_test_loader,
                input_shape=input_shape,
                num_classes=num_classes,
                metadata=metadata,
                seed=seed,
                config=config,
            )

            for row in rows:
                points.append(
                    ShiftGainPoint(
                        shift_level=shift_name,
                        shift_magnitude=shift_mag,
                        seed=seed,
                        method=row.method,
                        gain=row.gain,
                        source_drop=row.source_drop,
                        negative_adaptation_rate=1.0 if row.gain < 0 else 0.0,
                    )
                )

    return points


def run_threshold_ablation(
    confidence_values: tuple[float, ...] = (0.5, 0.6, 0.7, 0.8, 0.9),
    alignment_values: tuple[float, ...] = (1.0, 2.0, 5.0, 10.0),
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, float]]:
    """Sweep selective TTA thresholds to document sensitivity."""
    from pinn4csi.training.paper2_tta import run_tta_suite

    results: list[dict[str, float]] = []
    for conf_thresh in confidence_values:
        for align_thresh in alignment_values:
            gains = []
            for seed in seeds:
                config = TTAExperimentConfig(
                    methods=("selective_physics_tta",),
                    source_epochs=3,
                    adaptation_steps=3,
                    batch_size=16,
                    hidden_dim=24,
                    num_layers=2,
                    latent_dim=12,
                    selective_confidence_threshold=conf_thresh,
                    selective_alignment_threshold=align_thresh,
                )
                (
                    source_train_loader,
                    source_val_loader,
                    target_adapt_loader,
                    target_test_loader,
                    input_shape,
                    num_classes,
                    metadata,
                ) = build_synthetic_tta_loaders(
                    shift_name="moderate",
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                )
                rows = run_tta_suite(
                    source_train_loader=source_train_loader,
                    source_val_loader=source_val_loader,
                    target_adapt_loader=target_adapt_loader,
                    target_test_loader=target_test_loader,
                    input_shape=input_shape,
                    num_classes=num_classes,
                    metadata=metadata,
                    seed=seed,
                    config=config,
                )
                for row in rows:
                    gains.append(row.gain)
            mean_gain = float(np.mean(gains))
            results.append(
                {
                    "confidence_threshold": conf_thresh,
                    "alignment_threshold": align_thresh,
                    "mean_gain": mean_gain,
                    "num_seeds": len(seeds),
                }
            )
    return results


def run_full_analysis(output_dir: Path) -> dict[str, object]:
    """Run all analyses and save results."""
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- Latent space visualization ---
    logger.info("Collecting latent snapshots...")
    config = TTAExperimentConfig(
        methods=("physics_tta",),
        source_epochs=3,
        adaptation_steps=3,
        batch_size=16,
        hidden_dim=24,
        num_layers=2,
        latent_dim=12,
    )
    (
        source_train_loader,
        source_val_loader,
        target_adapt_loader,
        target_test_loader,
        input_shape,
        num_classes,
        metadata,
    ) = build_synthetic_tta_loaders(
        shift_name="strong",
        seed=42,
        batch_size=config.batch_size,
        source_val_ratio=config.source_val_ratio,
        target_adapt_ratio=config.target_adapt_ratio,
    )

    source_model = train_source_only_tta_model(
        source_train_loader=source_train_loader,
        source_val_loader=source_val_loader,
        input_shape=input_shape,
        num_classes=num_classes,
        config=config,
    ).to(device)

    reference_stats = compute_tta_reference_stats(
        source_model, source_train_loader, device
    )

    # Snapshots before adaptation
    source_before = collect_latent_snapshot(
        source_model, source_val_loader, device, "source", "before"
    )
    target_before = collect_latent_snapshot(
        source_model, target_test_loader, device, "target", "before"
    )

    # Adapt
    adapted_model = adapt_model_for_tta(
        model=source_model,
        target_loader=target_adapt_loader,
        reference_stats=reference_stats,
        config=config,
        method="physics_tta",
        device=device,
    )

    # Snapshots after adaptation
    target_after = collect_latent_snapshot(
        adapted_model, target_test_loader, device, "target", "after"
    )

    # t-SNE
    logger.info("Computing t-SNE embeddings...")
    tsne_data = compute_tsne_embeddings([source_before, target_before, target_after])
    with open(output_dir / "tsne_embeddings.json", "w") as f:
        json.dump(tsne_data, f)

    # --- Harm analysis ---
    logger.info("Running harm analysis...")
    harm_rows = run_harm_analysis(
        source_model=source_model,
        adapted_model=adapted_model,
        loader=target_test_loader,
        reference_stats=reference_stats,
        device=device,
    )
    harm_dicts = [asdict(r) for r in harm_rows]
    with open(output_dir / "harm_analysis.json", "w") as f:
        json.dump(harm_dicts, f)

    total = len(harm_rows)
    harmed = sum(1 for r in harm_rows if r.harmed)
    helped = sum(1 for r in harm_rows if r.helped)
    harm_summary = {
        "total_samples": total,
        "harmed": harmed,
        "helped": helped,
        "unchanged": total - harmed - helped,
        "negative_adaptation_rate": harmed / max(total, 1),
        "help_rate": helped / max(total, 1),
        "mean_alignment_harmed": float(
            np.mean([r.alignment_divergence for r in harm_rows if r.harmed])
        )
        if harmed > 0
        else 0.0,
        "mean_alignment_helped": float(
            np.mean([r.alignment_divergence for r in harm_rows if r.helped])
        )
        if helped > 0
        else 0.0,
        "mean_confidence_harmed": float(
            np.mean([r.adapted_confidence for r in harm_rows if r.harmed])
        )
        if harmed > 0
        else 0.0,
        "mean_confidence_helped": float(
            np.mean([r.adapted_confidence for r in harm_rows if r.helped])
        )
        if helped > 0
        else 0.0,
    }

    # --- Shift-gain curve ---
    logger.info("Computing shift-gain curve...")
    shift_points = run_shift_gain_curve()
    shift_dicts = [asdict(p) for p in shift_points]
    with open(output_dir / "shift_gain_curve.json", "w") as f:
        json.dump(shift_dicts, f)

    # Aggregate shift-gain by method and shift level
    shift_summary: dict[str, dict[str, float]] = {}
    for p in shift_points:
        key = f"{p.method}:{p.shift_level}"
        shift_summary.setdefault(key, {"gains": [], "drops": []})
        shift_summary[key]["gains"].append(p.gain)  # type: ignore[union-attr]
        shift_summary[key]["drops"].append(p.source_drop)  # type: ignore[union-attr]

    shift_agg = {}
    for key, vals in shift_summary.items():
        shift_agg[key] = {
            "mean_gain": float(np.mean(vals["gains"])),
            "mean_source_drop": float(np.mean(vals["drops"])),
            "negative_adaptation_rate": float(
                np.mean([1.0 if g < 0 else 0.0 for g in vals["gains"]])
            ),
        }

    # Combined summary
    summary: dict[str, object] = {
        "harm_analysis": harm_summary,
        "shift_gain_curve": shift_agg,
        "tsne_points": len(tsne_data),
    }
    with open(output_dir / "analysis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("Analysis complete. Results in %s", output_dir)
    return summary


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="TTA Adaptation Analysis")
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("outputs/tta_analysis"),
    )
    args = parser.parse_args()
    summary = run_full_analysis(args.output_dir)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
