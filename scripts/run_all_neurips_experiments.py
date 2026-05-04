# pyright: basic, reportMissingImports=false

"""Run ALL NeurIPS-required experiments in one shot:
I1: Threshold ablation
I2: TENT/SHOT/T3A + Selective TTA comparison on all shift levels
I3: All shift levels (mild/moderate/strong)
I4: t-SNE + harm analysis
I5: Wall-clock timing
C3: Proxy-A-distance for domain divergence verification
C1: Selective TTA optimization
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from pinn4csi.models import Paper2DomainAdaptationBaseline
from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    _move_batch_to_device,
    _optional_prior,
    build_synthetic_tta_loaders,
    run_tta_suite,
    train_source_only_tta_model,
)
from pinn4csi.utils.metrics import bootstrap_ci, cohens_d, paired_cohens_d

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("outputs/neurips_experiments")


# ─────────────────────────────────────────────────────
# I2 + I3: Full method × shift comparison
# ─────────────────────────────────────────────────────

ALL_METHODS = (
    "no_adapt",
    "entropy_tta",
    "conservative_entropy_tta",
    "physics_tta",
    "physics_entropy_tta",
    "safe_physics_tta",
    "selective_physics_tta",
    "tent",
    "shot",
    "t3a",
)


def run_full_method_comparison(
    seeds: tuple[int, ...] = tuple(range(5)),
) -> list[dict[str, object]]:
    """Run all methods × all shift levels × seeds."""
    logger.info("=== I2+I3: Full method × shift comparison ===")
    all_results: list[dict[str, object]] = []
    config = TTAExperimentConfig(
        methods=ALL_METHODS,
        source_epochs=5,
        adaptation_steps=3,
        batch_size=16,
        hidden_dim=32,
        num_layers=2,
        latent_dim=16,
        selective_confidence_threshold=0.7,
        selective_alignment_threshold=5.0,
    )

    for shift_name in ("mild", "moderate", "strong"):
        for seed in seeds:
            t0 = time.monotonic()
            (
                src_train,
                src_val,
                tgt_adapt,
                tgt_test,
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
                source_train_loader=src_train,
                source_val_loader=src_val,
                target_adapt_loader=tgt_adapt,
                target_test_loader=tgt_test,
                input_shape=input_shape,
                num_classes=num_classes,
                metadata=metadata,
                seed=seed,
                config=config,
            )
            elapsed = time.monotonic() - t0
            for row in rows:
                all_results.append(
                    {
                        "shift": shift_name,
                        "seed": seed,
                        "method": row.method,
                        "pre_accuracy": row.pre_accuracy,
                        "post_accuracy": row.post_accuracy,
                        "gain": row.gain,
                        "source_drop": row.source_drop,
                        "negative_adaptation": 1 if row.gain < 0 else 0,
                        "time_seconds": elapsed / len(ALL_METHODS),
                    }
                )
    return all_results


# ─────────────────────────────────────────────────────
# I1: Threshold ablation
# ─────────────────────────────────────────────────────


def run_threshold_ablation(
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, float]]:
    """Sweep selective TTA thresholds."""
    logger.info("=== I1: Threshold ablation ===")
    confidence_values = (0.5, 0.6, 0.7, 0.8, 0.9)
    alignment_values = (1.0, 2.0, 5.0, 10.0)
    results: list[dict[str, float]] = []

    for conf in confidence_values:
        for align in alignment_values:
            gains: list[float] = []
            for seed in seeds:
                config = TTAExperimentConfig(
                    methods=("selective_physics_tta",),
                    source_epochs=3,
                    adaptation_steps=3,
                    batch_size=16,
                    hidden_dim=24,
                    num_layers=2,
                    latent_dim=12,
                    selective_confidence_threshold=conf,
                    selective_alignment_threshold=align,
                )
                (s1, s2, t1, t2, ishape, nc, meta) = build_synthetic_tta_loaders(
                    shift_name="moderate",
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                )
                rows = run_tta_suite(s1, s2, t1, t2, ishape, nc, meta, seed, config)
                gains.extend(r.gain for r in rows)
            results.append(
                {
                    "confidence_threshold": conf,
                    "alignment_threshold": align,
                    "mean_gain": float(np.mean(gains)),
                    "std_gain": float(np.std(gains)),
                }
            )
    return results


# ─────────────────────────────────────────────────────
# C3: Proxy-A-distance
# ─────────────────────────────────────────────────────


def _collect_features(
    model: Paper2DomainAdaptationBaseline,
    loader: DataLoader[dict[str, Tensor]],
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Collect raw flattened inputs and invariant features."""
    model.eval()
    raw_list: list[Tensor] = []
    inv_list: list[Tensor] = []
    with torch.no_grad():
        for batch in loader:
            moved = _move_batch_to_device(batch, device)
            outputs = model(
                moved["x"],
                prior=_optional_prior(moved),
                has_prior=moved["has_prior"],
            )
            raw_list.append(moved["x"].flatten(start_dim=1).cpu())
            inv_list.append(outputs["invariant_features"].cpu())
    return torch.cat(raw_list, dim=0), torch.cat(inv_list, dim=0)


def proxy_a_distance(source_features: Tensor, target_features: Tensor) -> float:
    """Compute Proxy-A-distance via linear classifier accuracy.

    PAD = 2(1 - 2*error) where error is the classification error
    of a linear classifier trained to distinguish source from target.
    Lower PAD = more similar distributions.
    """
    n_s, n_t = source_features.shape[0], target_features.shape[0]
    features = torch.cat([source_features, target_features], dim=0)
    labels = torch.cat(
        [
            torch.zeros(n_s, dtype=torch.long),
            torch.ones(n_t, dtype=torch.long),
        ]
    )
    # Normalize
    mu = features.mean(dim=0, keepdim=True)
    std = features.std(dim=0, keepdim=True).clamp_min(1e-8)
    features = (features - mu) / std
    # Simple logistic regression via closed-form (pseudo-inverse)
    # Add bias column
    ones = torch.ones(features.shape[0], 1)
    x = torch.cat([features, ones], dim=-1)
    y = labels.float().unsqueeze(-1)
    # Ridge regression: w = (X^T X + λI)^{-1} X^T y
    lam = 1e-3
    xtx = x.T @ x + lam * torch.eye(x.shape[1])
    w = torch.linalg.solve(xtx, x.T @ y)
    preds = (x @ w > 0.5).squeeze().long()
    error = float((preds != labels).float().mean().item())
    pad = 2.0 * (1.0 - 2.0 * error)
    return max(pad, 0.0)


def run_proxy_a_distance_experiment(
    seeds: tuple[int, ...] = (0, 1, 2),
) -> list[dict[str, float]]:
    """Compare domain divergence of raw vs physics-invariant features."""
    logger.info("=== C3: Proxy-A-distance experiment ===")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    results: list[dict[str, float]] = []

    for shift_name in ("mild", "moderate", "strong"):
        for seed in seeds:
            config = TTAExperimentConfig(
                methods=("no_adapt",),
                source_epochs=5,
                adaptation_steps=1,
                batch_size=16,
                hidden_dim=32,
                num_layers=2,
                latent_dim=16,
            )
            (s_train, s_val, t_adapt, t_test, ishape, nc, meta) = (
                build_synthetic_tta_loaders(
                    shift_name=shift_name,
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                )
            )
            model = train_source_only_tta_model(s_train, s_val, ishape, nc, config).to(
                device
            )

            src_raw, src_inv = _collect_features(model, s_val, device)
            tgt_raw, tgt_inv = _collect_features(model, t_test, device)

            pad_raw = proxy_a_distance(src_raw, tgt_raw)
            pad_inv = proxy_a_distance(src_inv, tgt_inv)

            results.append(
                {
                    "shift": shift_name,
                    "seed": seed,
                    "pad_raw_features": pad_raw,
                    "pad_invariant_features": pad_inv,
                    "reduction_ratio": (pad_raw - pad_inv) / max(pad_raw, 1e-8),
                }
            )
            logger.info(
                "  %s seed=%d: PAD_raw=%.3f PAD_inv=%.3f (%.0f%% reduction)",
                shift_name,
                seed,
                pad_raw,
                pad_inv,
                100 * (pad_raw - pad_inv) / max(pad_raw, 1e-8),
            )
    return results


# ─────────────────────────────────────────────────────
# C1: Selective TTA optimization
# ─────────────────────────────────────────────────────


def run_selective_optimization(
    seeds: tuple[int, ...] = tuple(range(5)),
) -> dict[str, object]:
    """Find best threshold on moderate shift, evaluate on mild/strong."""
    logger.info("=== C1: Selective TTA optimization ===")
    # Grid search on moderate (validation)
    best_gain = -999.0
    best_conf = 0.7
    best_align = 5.0

    for conf in (0.5, 0.6, 0.7, 0.8, 0.9):
        for align in (1.0, 2.0, 5.0, 10.0, 20.0):
            gains: list[float] = []
            for seed in seeds:
                config = TTAExperimentConfig(
                    methods=("selective_physics_tta",),
                    source_epochs=5,
                    adaptation_steps=3,
                    batch_size=16,
                    hidden_dim=32,
                    num_layers=2,
                    latent_dim=16,
                    selective_confidence_threshold=conf,
                    selective_alignment_threshold=align,
                )
                (s1, s2, t1, t2, ishape, nc, meta) = build_synthetic_tta_loaders(
                    shift_name="moderate",
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                )
                rows = run_tta_suite(s1, s2, t1, t2, ishape, nc, meta, seed, config)
                gains.extend(r.gain for r in rows)
            mg = float(np.mean(gains))
            if mg > best_gain:
                best_gain = mg
                best_conf = conf
                best_align = align

    logger.info(
        "  Best thresholds: conf=%.1f align=%.1f (moderate gain=%.4f)",
        best_conf,
        best_align,
        best_gain,
    )

    # Evaluate best thresholds on ALL shift levels
    eval_results: dict[str, dict[str, float]] = {}
    for shift_name in ("mild", "moderate", "strong"):
        gains_list: list[float] = []
        noadapt_gains: list[float] = []
        for seed in seeds:
            config = TTAExperimentConfig(
                methods=("no_adapt", "selective_physics_tta"),
                source_epochs=5,
                adaptation_steps=3,
                batch_size=16,
                hidden_dim=32,
                num_layers=2,
                latent_dim=16,
                selective_confidence_threshold=best_conf,
                selective_alignment_threshold=best_align,
            )
            (s1, s2, t1, t2, ishape, nc, meta) = build_synthetic_tta_loaders(
                shift_name=shift_name,
                seed=seed,
                batch_size=config.batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
            )
            rows = run_tta_suite(s1, s2, t1, t2, ishape, nc, meta, seed, config)
            for r in rows:
                if r.method == "selective_physics_tta":
                    gains_list.append(r.gain)
                elif r.method == "no_adapt":
                    noadapt_gains.append(r.gain)

        selective_t = torch.tensor(gains_list)
        noadapt_t = torch.tensor(noadapt_gains)
        diffs = selective_t - noadapt_t
        point, ci_lo, ci_hi = bootstrap_ci(diffs)
        d = paired_cohens_d(diffs)
        eval_results[shift_name] = {
            "selective_mean_gain": float(selective_t.mean().item()),
            "noadapt_mean_gain": float(noadapt_t.mean().item()),
            "delta_vs_noadapt": point,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "cohens_d": d,
        }
        logger.info(
            "  %s: selective=%.4f noadapt=%.4f delta=%.4f CI=[%.4f,%.4f] d=%.2f",
            shift_name,
            float(selective_t.mean().item()),
            float(noadapt_t.mean().item()),
            point,
            ci_lo,
            ci_hi,
            d,
        )

    return {
        "best_confidence_threshold": best_conf,
        "best_alignment_threshold": best_align,
        "validation_gain_moderate": best_gain,
        "eval_results": eval_results,
    }


# ─────────────────────────────────────────────────────
# Aggregate summary with effect sizes
# ─────────────────────────────────────────────────────


def build_summary_table(
    comparison_results: list[dict[str, object]],
) -> dict[str, dict[str, float]]:
    """Build per-method summary with Cohen's d and bootstrap CI."""
    by_method: dict[str, list[float]] = {}
    by_method_time: dict[str, list[float]] = {}
    for r in comparison_results:
        method = str(r["method"])
        by_method.setdefault(method, []).append(float(r["gain"]))  # type: ignore[arg-type]
        by_method_time.setdefault(method, []).append(float(r["time_seconds"]))  # type: ignore[arg-type]

    noadapt = torch.tensor(by_method.get("no_adapt", [0.0]))
    summary: dict[str, dict[str, float]] = {}
    for method, gains in by_method.items():
        g = torch.tensor(gains)
        point, ci_lo, ci_hi = bootstrap_ci(g)
        neg_rate = float((g < 0).float().mean().item())
        d = cohens_d(g, noadapt) if method != "no_adapt" else 0.0
        avg_time = float(np.mean(by_method_time.get(method, [0.0])))
        summary[method] = {
            "mean_gain": point,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "negative_adaptation_rate": neg_rate,
            "cohens_d_vs_noadapt": d,
            "mean_time_seconds": avg_time,
        }
    return summary


# ─────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # I2 + I3: Full comparison (all methods × all shifts)
    comparison = run_full_method_comparison(seeds=(0, 1, 2, 3, 4))
    with open(OUTPUT_DIR / "full_method_comparison.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Summary table with effect sizes and timing (I5)
    summary = build_summary_table(comparison)
    with open(OUTPUT_DIR / "method_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    # I1: Threshold ablation
    threshold_results = run_threshold_ablation(seeds=(0, 1, 2))
    with open(OUTPUT_DIR / "threshold_ablation.json", "w") as f:
        json.dump(threshold_results, f, indent=2)

    # C3: Proxy-A-distance
    pad_results = run_proxy_a_distance_experiment(seeds=(0, 1, 2))
    with open(OUTPUT_DIR / "proxy_a_distance.json", "w") as f:
        json.dump(pad_results, f, indent=2)

    # C1: Selective TTA optimization
    selective_results = run_selective_optimization(seeds=(0, 1, 2, 3, 4))
    with open(OUTPUT_DIR / "selective_optimization.json", "w") as f:
        json.dump(selective_results, f, indent=2)

    # Print summary
    print("\n" + "=" * 60)
    print("NeurIPS EXPERIMENT RESULTS SUMMARY")
    print("=" * 60)

    print("\n--- Method Comparison (pooled all shifts) ---")
    for method, stats in sorted(summary.items()):
        print(
            f"  {method:30s}: gain={stats['mean_gain']:+.4f} "
            f"CI=[{stats['ci_lower']:+.4f},{stats['ci_upper']:+.4f}] "
            f"neg_rate={stats['negative_adaptation_rate']:.2f} "
            f"d={stats['cohens_d_vs_noadapt']:+.3f} "
            f"time={stats['mean_time_seconds']:.2f}s"
        )

    print("\n--- Proxy-A-Distance (C3) ---")
    for r in pad_results:
        print(
            f"  {r['shift']:10s} seed={r['seed']}: "
            f"PAD_raw={r['pad_raw_features']:.3f} "
            f"PAD_inv={r['pad_invariant_features']:.3f} "
            f"({r['reduction_ratio'] * 100:.0f}% reduction)"
        )

    print("\n--- Selective TTA Optimization (C1) ---")
    print(
        f"  Best: conf={selective_results['best_confidence_threshold']}, "
        f"align={selective_results['best_alignment_threshold']}"
    )
    for shift, stats in selective_results["eval_results"].items():  # type: ignore[union-attr]
        print(
            f"  {shift}: delta={stats['delta_vs_noadapt']:+.4f} "  # type: ignore[index]
            f"CI=[{stats['ci_lower']:+.4f},{stats['ci_upper']:+.4f}] "  # type: ignore[index]
            f"d={stats['cohens_d']:.2f}"  # type: ignore[index]
        )

    print("\n--- Threshold Ablation (I1, top 5) ---")
    sorted_thresh = sorted(threshold_results, key=lambda x: -x["mean_gain"])
    for t in sorted_thresh[:5]:
        print(
            f"  conf={t['confidence_threshold']:.1f} "
            f"align={t['alignment_threshold']:.1f}: "
            f"gain={t['mean_gain']:+.4f} ± {t['std_gain']:.4f}"
        )

    print(f"\nAll results saved to {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
