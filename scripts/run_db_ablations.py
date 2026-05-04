# pyright: basic, reportMissingImports=false

"""NeurIPS D&B Track ablation experiments:
B1: Step-budget sweep (1, 3, 5, 10)
B3: Batch size sweep (16, 32, 64, 128)
B4: Architecture ablation (MLP vs CNN1D)
B5: Absolute baseline accuracy
B6: Per-class accuracy breakdown
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import torch
from torch import Tensor

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    _move_batch_to_device,
    _optional_prior,
    build_prepared_tta_loaders,
    build_synthetic_tta_loaders,
    evaluate_tta_classifier,
    run_tta_suite,
    train_source_only_tta_model,
)

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = Path("outputs/db_ablations")
PREPARED = Path("data/prepared")
SEEDS = (0, 1, 2)
KEY_METHODS = (
    "no_adapt",
    "entropy_tta",
    "tent",
    "physics_tta",
    "selective_physics_tta",
)


def run_step_budget_ablation() -> list[dict[str, object]]:
    """B1: Vary adaptation steps on Widar_BVP."""
    logger.info("=== B1: Step-budget ablation ===")
    results: list[dict[str, object]] = []
    for steps in (1, 3, 5, 10):
        config = TTAExperimentConfig(
            methods=KEY_METHODS,
            source_epochs=10,
            adaptation_steps=steps,
            batch_size=32,
            hidden_dim=64,
            num_layers=3,
            latent_dim=32,
            selective_confidence_threshold=0.2,
            selective_alignment_threshold=2.0,
        )
        for seed in SEEDS:
            try:
                s1, s2, t1, t2, ishape, nc, meta = build_prepared_tta_loaders(
                    "widar_bvp",
                    PREPARED,
                    seed,
                    32,
                    0.2,
                    0.5,
                    train_env_ids=(0, 1),
                    test_env_ids=(2,),
                )
                rows = run_tta_suite(s1, s2, t1, t2, ishape, nc, meta, seed, config)
                for r in rows:
                    results.append(
                        {
                            "ablation": "step_budget",
                            "steps": steps,
                            "seed": seed,
                            "method": r.method,
                            "gain": r.gain,
                            "source_drop": r.source_drop,
                        }
                    )
            except Exception as e:
                logger.error("  step=%d seed=%d FAILED: %s", steps, seed, e)
    return results


def run_batch_size_ablation() -> list[dict[str, object]]:
    """B3: Vary batch size on synthetic moderate shift."""
    logger.info("=== B3: Batch size ablation ===")
    results: list[dict[str, object]] = []
    for bs in (16, 32, 64, 128):
        config = TTAExperimentConfig(
            methods=KEY_METHODS,
            source_epochs=5,
            adaptation_steps=5,
            batch_size=bs,
            hidden_dim=32,
            num_layers=2,
            latent_dim=16,
            selective_confidence_threshold=0.2,
            selective_alignment_threshold=2.0,
        )
        for seed in SEEDS:
            s1, s2, t1, t2, ishape, nc, meta = build_synthetic_tta_loaders(
                shift_name="moderate",
                seed=seed,
                batch_size=bs,
                source_val_ratio=0.2,
                target_adapt_ratio=0.5,
            )
            rows = run_tta_suite(s1, s2, t1, t2, ishape, nc, meta, seed, config)
            for r in rows:
                results.append(
                    {
                        "ablation": "batch_size",
                        "batch_size": bs,
                        "seed": seed,
                        "method": r.method,
                        "gain": r.gain,
                    }
                )
    return results


def compute_baseline_accuracies() -> dict[str, dict[str, float]]:
    """B5: Absolute accuracy of source-only model on each split."""
    logger.info("=== B5: Baseline accuracies ===")
    device = torch.device("cpu")
    results: dict[str, dict[str, float]] = {}

    for dataset in ("widar_bvp", "ntufi_har"):
        train_envs = (0, 1)
        test_envs = (2,)
        accs: dict[str, list[float]] = {
            "source_train": [],
            "source_val": [],
            "target": [],
        }
        for seed in SEEDS:
            try:
                s1, s2, t1, t2, ishape, nc, meta = build_prepared_tta_loaders(
                    dataset,
                    PREPARED,
                    seed,
                    32,
                    0.2,
                    0.5,
                    train_env_ids=train_envs,
                    test_env_ids=test_envs,
                )
                config = TTAExperimentConfig(
                    methods=("no_adapt",),
                    source_epochs=10,
                    batch_size=32,
                    hidden_dim=64,
                    num_layers=3,
                    latent_dim=32,
                )
                model = train_source_only_tta_model(s1, s2, ishape, nc, config).to(
                    device
                )
                src_train_m = evaluate_tta_classifier(model, s1, device)
                src_val_m = evaluate_tta_classifier(model, s2, device)
                tgt_m = evaluate_tta_classifier(model, t2, device)
                accs["source_train"].append(src_train_m["accuracy"])
                accs["source_val"].append(src_val_m["accuracy"])
                accs["target"].append(tgt_m["accuracy"])
            except Exception as e:
                logger.error("  %s seed=%d FAILED: %s", dataset, seed, e)

        results[dataset] = {k: float(np.mean(v)) for k, v in accs.items() if v}
        logger.info("  %s: %s", dataset, results[dataset])
    return results


def compute_per_class_accuracy() -> dict[str, list[dict[str, object]]]:
    """B6: Per-class accuracy before and after TTA on Widar_BVP."""
    logger.info("=== B6: Per-class accuracy ===")
    device = torch.device("cpu")
    results: dict[str, list[dict[str, object]]] = {}

    config = TTAExperimentConfig(
        methods=("no_adapt", "entropy_tta", "selective_physics_tta"),
        source_epochs=10,
        adaptation_steps=5,
        batch_size=32,
        hidden_dim=64,
        num_layers=3,
        latent_dim=32,
        selective_confidence_threshold=0.2,
        selective_alignment_threshold=2.0,
    )

    s1, s2, t1, t2, ishape, nc, meta = build_prepared_tta_loaders(
        "widar_bvp",
        PREPARED,
        0,
        32,
        0.2,
        0.5,
        train_env_ids=(0, 1),
        test_env_ids=(2,),
    )

    # Collect per-class predictions for source model
    from pinn4csi.training.paper2_tta import (
        adapt_model_for_tta,
        compute_tta_reference_stats,
    )

    model = train_source_only_tta_model(s1, s2, ishape, nc, config).to(device)
    ref_stats = compute_tta_reference_stats(model, s1, device)

    for method_name in ("no_adapt", "entropy_tta", "selective_physics_tta"):
        if method_name == "no_adapt":
            eval_model = model
        else:
            eval_model = adapt_model_for_tta(
                model,
                t1,
                ref_stats,
                config,
                method_name,
                device,
            )

        all_preds: list[Tensor] = []
        all_labels: list[Tensor] = []
        eval_model.eval()
        with torch.no_grad():
            for batch in t2:
                moved = _move_batch_to_device(batch, device)
                out = eval_model(
                    moved["x"],
                    prior=_optional_prior(moved),
                    has_prior=moved["has_prior"],
                )
                preds = torch.argmax(out["task_logits"], dim=-1)
                all_preds.append(preds.cpu())
                all_labels.append(moved["label"].cpu())

        preds_t = torch.cat(all_preds, dim=0)
        labels_t = torch.cat(all_labels, dim=0)

        class_results = []
        for cls in range(nc):
            mask = labels_t == cls
            if mask.sum() == 0:
                continue
            cls_acc = float((preds_t[mask] == labels_t[mask]).float().mean().item())
            class_results.append(
                {
                    "class_id": cls,
                    "accuracy": cls_acc,
                    "n_samples": int(mask.sum().item()),
                }
            )
        results[method_name] = class_results
        logger.info("  %s: %s", method_name, [r["accuracy"] for r in class_results])

    return results


def main() -> None:
    OUTPUT.mkdir(parents=True, exist_ok=True)

    # B1
    step_results = run_step_budget_ablation()
    with open(OUTPUT / "step_budget.json", "w") as f:
        json.dump(step_results, f, indent=2)

    # Summary
    print("\n--- Step Budget Summary (Widar room2, mean over 3 seeds) ---")
    for steps in (1, 3, 5, 10):
        for m in KEY_METHODS:
            gains = [
                r["gain"]
                for r in step_results
                if r["steps"] == steps and r["method"] == m
            ]
            if gains:
                print(f"  steps={steps:2d} {m:30s}: gain={np.mean(gains):+.4f}")

    # B3
    batch_results = run_batch_size_ablation()
    with open(OUTPUT / "batch_size.json", "w") as f:
        json.dump(batch_results, f, indent=2)

    print("\n--- Batch Size Summary (synthetic moderate, mean over 3 seeds) ---")
    for bs in (16, 32, 64, 128):
        for m in KEY_METHODS:
            gains = [
                r["gain"]
                for r in batch_results
                if r["batch_size"] == bs and r["method"] == m
            ]
            if gains:
                print(f"  bs={bs:3d} {m:30s}: gain={np.mean(gains):+.4f}")

    # B5
    baselines = compute_baseline_accuracies()
    with open(OUTPUT / "baseline_accuracies.json", "w") as f:
        json.dump(baselines, f, indent=2)
    print("\n--- Baseline Accuracies ---")
    for ds, accs in baselines.items():
        print(f"  {ds}: {accs}")

    # B6
    per_class = compute_per_class_accuracy()
    with open(OUTPUT / "per_class_accuracy.json", "w") as f:
        json.dump(per_class, f, indent=2)
    print("\n--- Per-Class Accuracy (Widar room2, seed=0) ---")
    for method, classes in per_class.items():
        accs = [c["accuracy"] for c in classes]
        print(f"  {method:30s}: {[f'{a:.2f}' for a in accs]}")

    print(f"\nAll results saved to {OUTPUT}/")


if __name__ == "__main__":
    main()
