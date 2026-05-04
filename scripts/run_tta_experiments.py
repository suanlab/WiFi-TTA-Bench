from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from pinn4csi.training.paper2_tta import (
    CANONICAL_TTA_METHODS,
    DEFAULT_TTA_METHODS,
    SYNTHETIC_SHIFT_LEVELS,
    TTAExperimentConfig,
    build_prepared_tta_loaders,
    build_synthetic_tta_loaders,
    run_tta_suite,
    save_tta_results_csv,
)
from pinn4csi.utils.experiment import parse_csv_ints, parse_csv_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run Paper 2 physics-guided test-time adaptation experiments"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="synthetic",
        help="Dataset name: synthetic or prepared dataset name such as ut_har.",
    )
    parser.add_argument(
        "--prepared-root",
        type=Path,
        default=Path("data/prepared"),
        help="Prepared data root for non-synthetic datasets.",
    )
    parser.add_argument(
        "--shift",
        type=str,
        default="moderate",
        help="Comma-separated synthetic shifts: mild,moderate,strong.",
    )
    parser.add_argument("--seeds", type=str, default="0,1,2")
    parser.add_argument("--source-epochs", type=int, default=5)
    parser.add_argument("--adaptation-steps", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--methods",
        type=str,
        default=",".join(DEFAULT_TTA_METHODS),
        help=(
            "Comma-separated TTA methods. "
            f"Available: {', '.join(CANONICAL_TTA_METHODS)}."
        ),
    )
    parser.add_argument("--source-learning-rate", type=float, default=1e-3)
    parser.add_argument("--adaptation-learning-rate", type=float, default=5e-4)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--invariance-weight", type=float, default=1.0)
    parser.add_argument("--residual-weight", type=float, default=1.0)
    parser.add_argument("--entropy-weight", type=float, default=0.1)
    parser.add_argument("--source-val-ratio", type=float, default=0.2)
    parser.add_argument("--target-adapt-ratio", type=float, default=0.5)
    parser.add_argument(
        "--update-scope",
        type=str,
        default="encoder",
        choices=("encoder", "projection"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/paper2_tta_results.csv"),
    )
    parser.add_argument("--safe-objective-min-delta", type=float, default=0.0)
    parser.add_argument("--safe-alignment-guard-ratio", type=float, default=2.0)
    parser.add_argument(
        "--train-env-ids",
        type=str,
        default=None,
        help="Optional comma-separated source environment IDs override.",
    )
    parser.add_argument(
        "--test-env-ids",
        type=str,
        default=None,
        help="Optional comma-separated target environment IDs override.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    methods = tuple(parse_csv_items(args.methods))
    unknown_methods = sorted(set(methods) - set(CANONICAL_TTA_METHODS))
    if unknown_methods:
        raise ValueError(f"Unknown TTA methods: {', '.join(unknown_methods)}")

    config = TTAExperimentConfig(
        methods=methods,
        source_epochs=args.source_epochs,
        adaptation_steps=args.adaptation_steps,
        batch_size=args.batch_size,
        source_learning_rate=args.source_learning_rate,
        adaptation_learning_rate=args.adaptation_learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        invariance_weight=args.invariance_weight,
        residual_weight=args.residual_weight,
        entropy_weight=args.entropy_weight,
        source_val_ratio=args.source_val_ratio,
        target_adapt_ratio=args.target_adapt_ratio,
        update_scope=args.update_scope,
        safe_objective_min_delta=args.safe_objective_min_delta,
        safe_alignment_guard_ratio=args.safe_alignment_guard_ratio,
    )
    seeds = parse_csv_ints(args.seeds)
    rows = []

    if args.dataset == "synthetic":
        shifts = parse_csv_items(args.shift)
        unknown_shifts = sorted(set(shifts) - set(SYNTHETIC_SHIFT_LEVELS))
        if unknown_shifts:
            raise ValueError(f"Unknown synthetic shifts: {', '.join(unknown_shifts)}")

        total_runs = len(shifts) * len(seeds)
        run_idx = 0
        for shift_name in shifts:
            for seed in seeds:
                run_idx += 1
                print(
                    f"[{run_idx}/{total_runs}] synthetic shift={shift_name} seed={seed}"
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
                    shift_name=shift_name,
                    seed=seed,
                    batch_size=config.batch_size,
                    source_val_ratio=config.source_val_ratio,
                    target_adapt_ratio=config.target_adapt_ratio,
                )
                rows.extend(
                    run_tta_suite(
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
                )
    else:
        train_env_ids = (
            parse_csv_ints(args.train_env_ids)
            if args.train_env_ids is not None
            else None
        )
        test_env_ids = (
            parse_csv_ints(args.test_env_ids) if args.test_env_ids is not None else None
        )
        total_runs = len(seeds)
        for run_idx, seed in enumerate(seeds, start=1):
            print(f"[{run_idx}/{total_runs}] dataset={args.dataset} seed={seed}")
            (
                source_train_loader,
                source_val_loader,
                target_adapt_loader,
                target_test_loader,
                input_shape,
                num_classes,
                metadata,
            ) = build_prepared_tta_loaders(
                dataset_name=args.dataset,
                prepared_root=args.prepared_root,
                seed=seed,
                batch_size=config.batch_size,
                source_val_ratio=config.source_val_ratio,
                target_adapt_ratio=config.target_adapt_ratio,
                train_env_ids=train_env_ids,
                test_env_ids=test_env_ids,
            )
            rows.extend(
                run_tta_suite(
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
            )

    save_tta_results_csv(rows, args.output_csv)
    print(f"Saved {len(rows)} rows to {args.output_csv}")

    grouped: dict[tuple[str, str], list[float]] = {}
    for row in rows:
        grouped.setdefault((row.shift_name, row.method), []).append(row.gain)

    print("\nMean adaptation gain by method")
    for (shift_name, method), gains in sorted(grouped.items()):
        print(
            f"  {shift_name:>12s} | {method:18s} "
            f"gain={np.mean(gains):+.4f} ± {np.std(gains):.4f}"
        )


if __name__ == "__main__":
    main()
