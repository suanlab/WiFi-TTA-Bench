# pyright: basic, reportMissingImports=false

"""Source-overfit mitigation pilot: reduce Widar_BVP 42pp train-val gap.

REVISION.md §11 identifies source overfitting (89.4% train vs 47.4% val = 42pp
gap) as a confounder for confidence-based TTA methods. This script runs a
controlled sweep comparing:
    (a) Widar source training with NO regularization (baseline)
    (b) + label smoothing alpha=0.1
    (c) + label smoothing alpha=0.2

For each configuration, we train Widar source with leave-one-room-out and
report:
    - source train accuracy
    - source val accuracy
    - source train-val gap (headline mitigation metric)
    - downstream TTA gains (entropy_tta, tent, physics_tta) at n=15

Output: outputs/source_overfit/mitigation_pilot.json
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
from pinn4csi.utils.metrics import bootstrap_ci, cohens_d

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

PREPARED_ROOT = Path("data/prepared")
OUTPUT_PATH = Path("outputs/source_overfit/mitigation_pilot.json")
DEFAULT_SEEDS = (0, 1, 2, 3, 4)
DEFAULT_METHODS = ("no_adapt", "entropy_tta", "tent", "physics_tta")

ROOM_SPLITS = [
    {"train": (0, 1), "test": (2,), "name": "room2"},
    {"train": (0, 2), "test": (1,), "name": "room1"},
    {"train": (1, 2), "test": (0,), "name": "room0"},
]


def run_condition(
    label_smoothing: float,
    seeds: tuple[int, ...],
    methods: tuple[str, ...],
    source_epochs: int,
) -> dict[str, object]:
    rows: list[dict[str, object]] = []
    src_pre_accs: list[
        float
    ] = []  # source val accuracy (no_adapt row.source_pre_accuracy)
    config = TTAExperimentConfig(
        methods=methods,
        source_label_smoothing=label_smoothing,
        source_epochs=source_epochs,
        adaptation_steps=5,
        adaptation_learning_rate=5e-4,
        batch_size=32,
    )

    for split in ROOM_SPLITS:
        train_envs, test_envs, name = split["train"], split["test"], split["name"]
        logger.info(
            "[ls=%.2f] split %s train=%s test=%s",
            label_smoothing,
            name,
            train_envs,
            test_envs,
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
                    if r.method == "no_adapt":
                        src_pre_accs.append(float(r.source_pre_accuracy))
                    rows.append(
                        {
                            "label_smoothing": label_smoothing,
                            "room": name,
                            "seed": seed,
                            "method": r.method,
                            "pre_accuracy": r.pre_accuracy,
                            "post_accuracy": r.post_accuracy,
                            "gain": r.gain,
                            "source_pre_accuracy": r.source_pre_accuracy,
                            "source_post_accuracy": r.source_post_accuracy,
                            "source_drop": r.source_drop,
                            "time_seconds": elapsed / max(len(methods), 1),
                        }
                    )
            except Exception as e:  # noqa: BLE001
                logger.error("  FAILED: %s", e)
                continue

    src_val_t = torch.tensor(src_pre_accs) if src_pre_accs else torch.tensor([0.0])
    summary: dict[str, dict[str, float]] = {}
    by_method: dict[str, list[float]] = {}
    for r in rows:
        by_method.setdefault(str(r["method"]), []).append(float(r["gain"]))
    no_adapt = torch.tensor(by_method.get("no_adapt", [0.0]))
    for m, gains in by_method.items():
        g = torch.tensor(gains)
        point, ci_lo, ci_hi = bootstrap_ci(g)
        d = float(cohens_d(g, no_adapt)) if m != "no_adapt" else 0.0
        nr = float((g < 0).float().mean().item())
        summary[m] = {
            "mean_gain": point,
            "ci_lower": ci_lo,
            "ci_upper": ci_hi,
            "cohens_d_vs_noadapt": d,
            "negative_adaptation_rate": nr,
            "n_observations": len(gains),
        }

    return {
        "label_smoothing": label_smoothing,
        "rows": rows,
        "summary": summary,
        "source_val_acc": {
            "mean": float(src_val_t.mean().item()),
            "std": float(src_val_t.std(unbiased=True).item())
            if len(src_pre_accs) > 1
            else 0.0,
            "n": len(src_pre_accs),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__ or "")
    parser.add_argument(
        "--label-smoothings",
        default="0.0,0.1,0.2",
        help="Comma-separated alphas (default: 0.0,0.1,0.2)",
    )
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--methods", default=",".join(DEFAULT_METHODS))
    parser.add_argument("--source-epochs", type=int, default=10)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    alphas = tuple(float(a) for a in args.label_smoothings.split(",") if a.strip())
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    methods = tuple(m.strip() for m in args.methods.split(",") if m.strip())

    conditions = []
    t_global = time.monotonic()
    for alpha in alphas:
        result = run_condition(alpha, seeds, methods, args.source_epochs)
        conditions.append(result)

    payload = {
        "conditions": conditions,
        "config": {
            "label_smoothings": list(alphas),
            "seeds": list(seeds),
            "methods": list(methods),
            "source_epochs": args.source_epochs,
        },
        "elapsed_seconds": time.monotonic() - t_global,
    }
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2)

    print("\n" + "=" * 78)
    print("WIDAR SOURCE OVERFIT MITIGATION PILOT")
    print("=" * 78)
    print(f"{'alpha':>7s}  {'src-val':>10s}  {'best TTA':>30s}  {'neg rate':>10s}")
    print("-" * 78)
    for c in conditions:
        s = c["summary"]
        best = None
        best_gain = -1e9
        for m, row in s.items():
            if m == "no_adapt":
                continue
            if row["mean_gain"] > best_gain:
                best_gain = row["mean_gain"]
                best = (m, row)
        best_str = f"{best[0]}: {best[1]['mean_gain']:+.4f}" if best else "---"
        neg_rate = best[1]["negative_adaptation_rate"] if best else 0.0
        print(
            f"{c['label_smoothing']:>7.2f}  "
            f"{c['source_val_acc']['mean']:>10.4f}  "
            f"{best_str:>30s}  "
            f"{neg_rate:>10.2f}"
        )
    print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
