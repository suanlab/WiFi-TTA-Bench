# pyright: basic, reportMissingImports=false

"""Run Widar architecture ablation with per-seed gains and bootstrap CIs.

Trains each backbone (mlp, mlp_bn, cnn1d) with leave-one-room-out x 5 seeds
= 15 observations per (backbone, method) cell, saving per-seed gains so
95% bootstrap CIs can be reported in the paper's architecture ablation table.

Output: outputs/final_ablations/arch_ablation_with_ci.json
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import torch

from pinn4csi.training.paper2_tta import (
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    run_tta_suite,
)
from pinn4csi.utils.metrics import bootstrap_ci, cohens_d, paired_cohens_d

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_PATH = Path("outputs/final_ablations/arch_ablation_with_ci.json")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_METHODS = (
    "no_adapt",
    "entropy_tta",
    "tent",
    "physics_tta",
    "selective_physics_tta",
)

ROOM_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "room2"},
    {"train": (0, 2), "test": (1,), "name": "room1"},
    {"train": (1, 2), "test": (0,), "name": "room0"},
]


def run_backbone(
    backbone: str,
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    source_epochs: int,
) -> tuple[list[dict[str, object]], dict[str, float]]:
    rows: list[dict[str, object]] = []
    src_accs: list[float] = []
    config = TTAExperimentConfig(
        methods=methods,
        backbone=backbone,
        source_epochs=source_epochs,
        adaptation_steps=5,
        adaptation_learning_rate=5e-4,
        batch_size=32,
    )
    for split in ROOM_SPLITS:
        train_envs, test_envs, name = split["train"], split["test"], split["name"]
        logger.info(
            "[%s] split %s: train=%s test=%s", backbone, name, train_envs, test_envs
        )
        for seed in seeds:
            t0 = time.monotonic()
            try:
                (
                    src_train,
                    src_val,
                    tgt_adapt,
                    tgt_test,
                    input_shape,
                    num_classes,
                    metadata,
                ) = build_prepared_tta_loaders(
                    dataset_name="widar_bvp",
                    prepared_root=PREPARED_ROOT,
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                    train_env_ids=train_envs,
                    test_env_ids=test_envs,
                )
                run_rows = run_tta_suite(
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
                for r in run_rows:
                    rows.append(
                        {
                            "backbone": backbone,
                            "room": name,
                            "seed": seed,
                            "method": r.method,
                            "pre_accuracy": r.pre_accuracy,
                            "post_accuracy": r.post_accuracy,
                            "gain": r.gain,
                            "source_drop": r.source_drop,
                            "time_seconds": elapsed / max(len(methods), 1),
                        }
                    )
                if run_rows:
                    src_accs.append(run_rows[0].pre_accuracy)
            except Exception as e:  # noqa: BLE001
                logger.error("  FAILED: %s", e)
                continue
    src_stats: dict[str, float] = {}
    if src_accs:
        t = torch.tensor(src_accs)
        src_stats = {
            "mean": float(t.mean().item()),
            "std": float(t.std(unbiased=True).item()) if len(src_accs) > 1 else 0.0,
            "n": len(src_accs),
        }
    return rows, src_stats


def compute_summary(
    rows: list[dict[str, object]],
) -> dict[str, dict[str, dict[str, float]]]:
    by_bb_method: dict[tuple[str, str], list[float]] = {}
    for r in rows:
        bb = str(r["backbone"])
        m = str(r["method"])
        by_bb_method.setdefault((bb, m), []).append(float(r["gain"]))

    summary: dict[str, dict[str, dict[str, float]]] = {}
    for (bb, m), gains in by_bb_method.items():
        no_adapt_gains = by_bb_method.get((bb, "no_adapt"), [0.0])
        noadapt = torch.tensor(no_adapt_gains)
        g = torch.tensor(gains)
        point, ci_lo, ci_hi = bootstrap_ci(g)
        d = float(cohens_d(g, noadapt)) if m != "no_adapt" else 0.0
        nr = float((g < 0).float().mean().item())
        pd = 0.0
        if m != "no_adapt" and len(gains) == len(no_adapt_gains):
            pd = paired_cohens_d(g - noadapt)
        summary.setdefault(bb, {})[m] = {
            "mean_gain": point,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "cohens_d_vs_noadapt": d,
            "paired_d_vs_noadapt": pd,
            "negative_adaptation_rate": nr,
            "n_observations": len(gains),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--backbones",
        default="mlp,mlp_bn,cnn1d",
        help="Comma-separated backbones (default: mlp,mlp_bn,cnn1d)",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument(
        "--methods",
        default=",".join(DEFAULT_METHODS),
    )
    parser.add_argument("--source-epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    backbones = tuple(b.strip() for b in args.backbones.split(",") if b.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    all_rows: list[dict[str, object]] = []
    src_per_bb: dict[str, dict[str, float]] = {}
    t_global = time.monotonic()
    for bb in backbones:
        rows, src_stats = run_backbone(bb, seeds, methods, args.source_epochs)
        all_rows.extend(rows)
        src_per_bb[bb] = src_stats

    summary = compute_summary(all_rows)
    payload = {
        "rows": all_rows,
        "summary": summary,
        "source_accuracy": src_per_bb,
        "config": {
            "backbones": list(backbones),
            "seeds": list(seeds),
            "methods": list(methods),
            "source_epochs": args.source_epochs,
        },
        "elapsed_seconds": time.monotonic() - t_global,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 78)
    print("WIDAR ARCHITECTURE ABLATION WITH BOOTSTRAP CIs")
    print("=" * 78)
    for bb in backbones:
        s_acc = src_per_bb.get(bb, {})
        print(f"\n[{bb}] source acc (mean over seeds) = {s_acc.get('mean', 0):.4f}")
        print(
            f"  {'Method':<26s} {'Gain':>9s} {'95% CI':>24s} {'d':>6s} {'NegRate':>8s}"
        )
        for m in methods:
            row = summary.get(bb, {}).get(m)
            if row:
                ci = f"[{row['ci_lower']:+.4f}, {row['ci_upper']:+.4f}]"
                d_val = row["cohens_d_vs_noadapt"]
                nr_val = row["negative_adaptation_rate"]
                print(
                    f"  {m:<26s} {row['mean_gain']:+.4f} {ci:>24s} "
                    f"{d_val:+.3f} {nr_val:.2f}"
                )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
