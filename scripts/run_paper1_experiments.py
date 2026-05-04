# pyright: basic, reportMissingImports=false

"""Run reusable multi-dataset Paper 1 experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

from pinn4csi.data import create_mock_paper1_prepared_data
from pinn4csi.training import (
    Paper1ExperimentConfig,
    analyze_paper1_results,
    build_paper1_summary_rows,
    run_paper1_experiments,
    save_paper1_analysis_json,
    save_paper1_results_csv,
    save_paper1_summary_json,
    save_paper1_summary_latex,
    summarize_paper1_results,
)
from pinn4csi.utils import parse_csv_ints, parse_csv_items


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run reusable Paper 1 experiments across prepared SignFi/UT_HAR "
            "targets and baseline model families."
        )
    )
    parser.add_argument("--prepared-root", type=Path, required=True)
    parser.add_argument("--datasets", type=str, default="signfi,ut_har")
    parser.add_argument(
        "--models",
        type=str,
        default="mlp,cnn,dgsense_lite,autoencoder,residual_prior",
    )
    parser.add_argument("--seeds", type=str, default="0")
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=3)
    parser.add_argument("--latent-dim", type=int, default=32)
    parser.add_argument("--reconstruction-weight", type=float, default=0.2)
    parser.add_argument("--ofdm-weight", type=float, default=0.05)
    parser.add_argument("--path-weight", type=float, default=0.05)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/paper1_results.csv"),
    )
    parser.add_argument(
        "--analysis-json",
        type=Path,
        default=Path("outputs/paper1_analysis.json"),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=Path("outputs/paper1_summary.json"),
    )
    parser.add_argument(
        "--summary-latex",
        type=Path,
        default=Path("outputs/paper1_summary.tex"),
    )
    parser.add_argument(
        "--embeddings-dir",
        type=Path,
        default=None,
        help="Optional output directory for latent embedding dumps.",
    )
    parser.add_argument(
        "--summary-split",
        type=str,
        default="test",
        help="Split used for summary/analysis exports (default: test).",
    )
    parser.add_argument(
        "--create-mock-data",
        action="store_true",
        help="Generate tiny local-format prepared datasets before running the harness.",
    )
    parser.add_argument("--mock-samples-per-class", type=int, default=18)
    parser.add_argument("--mock-num-subcarriers", type=int, default=16)
    parser.add_argument("--mock-num-classes", type=int, default=3)
    parser.add_argument(
        "--disable-fixed-weight-variant",
        action="store_true",
        help="Disable fixed physics-weight ablation variant for autoencoder family.",
    )
    parser.add_argument(
        "--disable-fourier-variant",
        action="store_true",
        help="Disable Fourier-features ablation variant for autoencoder family.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.create_mock_data:
        create_mock_paper1_prepared_data(
            prepared_root=args.prepared_root,
            samples_per_class=args.mock_samples_per_class,
            num_subcarriers=args.mock_num_subcarriers,
            num_classes=args.mock_num_classes,
        )

    config = Paper1ExperimentConfig(
        prepared_root=args.prepared_root,
        dataset_names=parse_csv_items(args.datasets),
        model_names=parse_csv_items(args.models),
        seeds=parse_csv_ints(args.seeds),
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        latent_dim=args.latent_dim,
        reconstruction_weight=args.reconstruction_weight,
        ofdm_weight=args.ofdm_weight,
        path_weight=args.path_weight,
        output_csv=args.output_csv,
        analysis_json=args.analysis_json,
        summary_json=args.summary_json,
        summary_latex=args.summary_latex,
        embeddings_dir=args.embeddings_dir,
        summary_split=args.summary_split,
        include_fixed_weight_variant=not args.disable_fixed_weight_variant,
        include_fourier_variant=not args.disable_fourier_variant,
    )

    results = run_paper1_experiments(config)
    save_paper1_results_csv(results, config.output_csv)
    analysis = analyze_paper1_results(results, split=config.summary_split)
    save_paper1_analysis_json(analysis, config.analysis_json)
    summary_rows = build_paper1_summary_rows(results, split=config.summary_split)
    save_paper1_summary_json(summary_rows, config.summary_json)
    save_paper1_summary_latex(summary_rows, config.summary_latex)
    summary = summarize_paper1_results(results, split=config.summary_split)

    print(f"Results written to: {config.output_csv}")
    print(f"Analysis written to: {config.analysis_json}")
    print(f"Summary written to: {config.summary_json}")
    print(f"LaTeX table written to: {config.summary_latex}")
    if config.embeddings_dir is not None:
        print(f"Embeddings written to: {config.embeddings_dir}")
    for key, metrics in summary.items():
        paired_suffix = ""
        if metrics["paired_accuracy_mean_delta"] is not None:
            paired_suffix = (
                f" paired_delta_acc={metrics['paired_accuracy_mean_delta']:+.4f}"
            )
        if metrics["paired_accuracy_sign_test_pvalue"] is not None:
            paired_suffix = (
                paired_suffix
                + f" sign_p={metrics['paired_accuracy_sign_test_pvalue']:.4f}"
            )
        print(
            f"{key} | accuracy={metrics['accuracy']:.4f}±{metrics['accuracy_std']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f}±{metrics['macro_f1_std']:.4f} "
            f"loss_total={metrics['loss_total']:.4f}±{metrics['loss_total_std']:.4f} "
            f"runs={metrics['num_runs']} seeds={metrics['seeds']}"
            f"{paired_suffix}"
        )


if __name__ == "__main__":
    main()
