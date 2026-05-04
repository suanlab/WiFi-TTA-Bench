"""Synthetic classification experiment with known OFDM physics parameters.

Addresses two critical weaknesses identified in analysis:
- W2 (regression-classification mismatch): Is physics useful for classification at all?
- W3 (fabricated priors): Does genuine (non-fabricated) physics help?

Design:
- CSI generated from OFDM model H(f_k) = Σ h_l · exp(-j2πf_k τ_l) with KNOWN parameters
- Classification label = number of multipath components L ∈ {2, 3, 4, 5}
- Environment shift: different SNR + different background multipath between train/test
- Three methods: baseline, corr_physics (env-independent), amp_physics (env-dependent)
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torch import Tensor, nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, TensorDataset

sys.path.insert(0, str(Path(__file__).resolve().parent))

from run_diagnosis_experiments import (
    AmplitudePhysicsHead,
    Conv1DEncoder,
    SequenceClassifier,
    count_trainable_parameters,
)

EVAL_MODES = ("iid", "cross_env")


@dataclass(frozen=True)
class ExperimentRow:
    dataset: str
    method: str
    eval_mode: str
    seed: int
    best_epoch: int
    stopped_epoch: int
    train_examples: int
    val_examples: int
    test_examples: int
    parameter_count: int
    test_accuracy: float
    test_macro_f1: float
    test_loss: float


def ofdm_channel_response(
    gains: np.ndarray,
    delays: np.ndarray,
    n_subcarriers: int,
    fc: float = 5.8e9,
    bw: float = 20e6,
) -> np.ndarray:
    f_k = fc + np.linspace(-bw / 2, bw / 2, n_subcarriers)
    h = np.zeros(n_subcarriers, dtype=np.complex128)
    for p_idx in range(len(gains)):
        h += gains[p_idx] * np.exp(-1j * 2 * np.pi * f_k * delays[p_idx])
    return h


def generate_synthetic_dataset(
    n_samples_per_env: int = 1000,
    n_subcarriers: int = 90,
    n_time_steps: int = 250,
    snr_db_train: float = 20.0,
    snr_db_test: float = 12.0,
    n_classes: int = 4,
    n_bg_paths: int = 5,
    seed: int = 0,
) -> tuple[Tensor, Tensor, Tensor]:
    rng = np.random.RandomState(seed)
    path_counts = list(range(2, 2 + n_classes))

    csi_list: list[np.ndarray] = []
    label_list: list[int] = []
    env_list: list[int] = []

    for env in range(2):
        snr_db = snr_db_train if env == 0 else snr_db_test
        sigma = 10 ** (-snr_db / 20)

        bg_gains = 0.3 * (rng.randn(n_bg_paths) + 1j * rng.randn(n_bg_paths))
        bg_delays = np.sort(rng.uniform(50e-9, 300e-9, n_bg_paths))

        for _ in range(n_samples_per_env):
            n_paths = int(rng.choice(path_counts))

            sig_gains = rng.randn(n_paths) + 1j * rng.randn(n_paths)
            sig_gains /= np.abs(sig_gains).mean() + 1e-8
            sig_delays = np.sort(rng.uniform(10e-9, 100e-9, n_paths))

            sample = np.zeros((n_time_steps, n_subcarriers), dtype=np.float32)
            for t in range(n_time_steps):
                drift = 0.02 * np.sin(2 * np.pi * t / n_time_steps)
                perturbed = sig_gains * (1 + drift * rng.randn(n_paths) * 0.1)
                all_g = np.concatenate([perturbed, bg_gains])
                all_d = np.concatenate([sig_delays, bg_delays])
                h_resp = ofdm_channel_response(all_g, all_d, n_subcarriers)
                h_resp += sigma * (
                    rng.randn(n_subcarriers) + 1j * rng.randn(n_subcarriers)
                )
                sample[t] = np.abs(h_resp)

            csi_list.append(sample)
            label_list.append(n_paths - 2)
            env_list.append(env)

    return (
        torch.from_numpy(np.array(csi_list)),
        torch.tensor(label_list, dtype=torch.long),
        torch.tensor(env_list, dtype=torch.long),
    )


def build_splits(
    labels: Tensor,
    environments: Tensor,
    seed: int,
    eval_mode: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, Tensor]:
    if eval_mode == "cross_env":
        train_idx = torch.where(environments == 0)[0]
        test_idx = torch.where(environments == 1)[0]
        n_val = int(len(train_idx) * 0.15)
        val_idx = train_idx[:n_val]
        train_idx = train_idx[n_val:]
        return {"train": train_idx, "val": val_idx, "test": test_idx}

    n = len(labels)
    indices = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    return {
        "train": indices[:n_train],
        "val": indices[n_train : n_train + n_val],
        "test": indices[n_train + n_val :],
    }


def make_dataloaders(
    csi: Tensor,
    labels: Tensor,
    split: dict[str, Tensor],
    batch_size: int,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    for key in ("train", "val", "test"):
        idx = split[key]
        loaders[key] = DataLoader(
            TensorDataset(csi[idx], labels[idx]),
            batch_size=batch_size,
            shuffle=(key == "train"),
            drop_last=(key == "train"),
        )
    return loaders["train"], loaders["val"], loaders["test"]


def run_experiment(
    method: str,
    csi: Tensor,
    labels: Tensor,
    environments: Tensor,
    eval_mode: str,
    seed: int,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
    weight_decay: float,
    dropout: float,
    aux_weight: float,
    patience: int,
    min_delta: float,
) -> ExperimentRow:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_features = csi.shape[-1]
    num_classes = int(labels.max().item()) + 1

    split = build_splits(labels, environments, seed, eval_mode)
    train_loader, val_loader, test_loader = make_dataloaders(
        csi, labels, split, batch_size
    )

    encoder = Conv1DEncoder(
        num_features=num_features, hidden_dim=hidden_dim, dropout=dropout
    ).to(device)
    classifier_head = SequenceClassifier(
        hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout
    ).to(device)

    use_aux = "physics" in method
    use_corr = "corr" in method
    aux_head: nn.Module | None = None
    if use_aux:
        aux_head = AmplitudePhysicsHead(hidden_dim=hidden_dim).to(device)

    params = list(encoder.parameters()) + list(classifier_head.parameters())
    if aux_head is not None:
        params += list(aux_head.parameters())

    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state: dict[str, dict[str, Tensor]] = {}

    for epoch in range(epochs):
        encoder.train()
        classifier_head.train()
        if aux_head is not None:
            aux_head.train()

        epoch_loss = 0.0
        correct = 0
        total = 0

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            encoded = encoder(batch_x)
            logits = classifier_head(encoded)
            loss = functional.cross_entropy(logits, batch_y)

            if use_aux and aux_head is not None:
                aux_pred = aux_head(encoded).squeeze(-1)
                if use_corr:
                    target = batch_x[:, 1:, :] - batch_x[:, :-1, :]
                    target = target.mean(dim=-1)
                    target = functional.pad(target, (1, 0))
                else:
                    target = batch_x.mean(dim=-1)
                loss = loss + aux_weight * functional.mse_loss(aux_pred, target)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            optimizer.step()

            epoch_loss += loss.item() * batch_x.size(0)
            correct += (logits.argmax(dim=1) == batch_y).sum().item()
            total += batch_x.size(0)

        scheduler.step()

        encoder.eval()
        classifier_head.eval()
        if aux_head is not None:
            aux_head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)
                encoded = encoder(batch_x)
                logits = classifier_head(encoded)
                val_correct += (logits.argmax(dim=1) == batch_y).sum().item()
                val_total += batch_y.size(0)
        val_acc = val_correct / max(val_total, 1)

        if val_acc > best_val_acc + min_delta:
            best_val_acc = val_acc
            best_epoch = epoch
            patience_counter = 0
            best_state = {
                "encoder": {
                    k: v.cpu().clone() for k, v in encoder.state_dict().items()
                },
                "classifier": {
                    k: v.cpu().clone() for k, v in classifier_head.state_dict().items()
                },
            }
            if aux_head is not None:
                best_state["aux"] = {
                    k: v.cpu().clone() for k, v in aux_head.state_dict().items()
                }
        else:
            patience_counter += 1
            if patience_counter >= patience:
                break

    if best_state:
        encoder.load_state_dict(best_state["encoder"])
        classifier_head.load_state_dict(best_state["classifier"])
        if aux_head is not None and "aux" in best_state:
            aux_head.load_state_dict(best_state["aux"])

    encoder.eval()
    classifier_head.eval()
    if aux_head is not None:
        aux_head.eval()

    test_correct = 0
    test_total = 0
    test_loss_sum = 0.0
    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded = encoder(batch_x)
            logits = classifier_head(encoded)
            test_loss_sum += functional.cross_entropy(
                logits, batch_y
            ).item() * batch_x.size(0)
            test_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            test_total += batch_y.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())

    test_acc = test_correct / max(test_total, 1)
    test_loss = test_loss_sum / max(test_total, 1)

    preds_np = np.array(all_preds)
    labels_np = np.array(all_labels)
    f1s = []
    for c in range(num_classes):
        tp = int(((preds_np == c) & (labels_np == c)).sum())
        fp = int(((preds_np == c) & (labels_np != c)).sum())
        fn = int(((preds_np != c) & (labels_np == c)).sum())
        f1s.append(
            2
            * (tp / max(tp + fp, 1))
            * (tp / max(tp + fn, 1))
            / max(tp / max(tp + fp, 1) + tp / max(tp + fn, 1), 1e-8)
        )

    param_count = count_trainable_parameters(encoder, classifier_head)
    if aux_head is not None:
        param_count += count_trainable_parameters(aux_head)

    return ExperimentRow(
        dataset="synthetic_known_physics",
        method=method,
        eval_mode=eval_mode,
        seed=seed,
        best_epoch=best_epoch,
        stopped_epoch=epoch + 1,
        train_examples=len(split["train"]),
        val_examples=len(split["val"]),
        test_examples=len(split["test"]),
        parameter_count=param_count,
        test_accuracy=test_acc,
        test_macro_f1=float(np.mean(f1s)),
        test_loss=test_loss,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Synthetic classification with known physics"
    )
    parser.add_argument("--n-samples", type=int, default=1000, help="Per environment")
    parser.add_argument("--snr-train", type=float, default=20.0)
    parser.add_argument("--snr-test", type=float, default=12.0)
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--data-seed", type=int, default=42)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path("outputs/synthetic_classification.csv"),
    )
    args = parser.parse_args()

    print("Generating synthetic CSI with known OFDM physics...")
    csi, labels, environments = generate_synthetic_dataset(
        n_samples_per_env=args.n_samples,
        snr_db_train=args.snr_train,
        snr_db_test=args.snr_test,
        seed=args.data_seed,
    )
    n_classes = int(labels.max().item()) + 1
    print(
        f"Data: {csi.shape}, labels {labels.shape}, "
        f"envs {environments.shape}, classes {n_classes}"
    )
    print(f"  Env 0: {(environments == 0).sum()} samples (SNR={args.snr_train} dB)")
    print(f"  Env 1: {(environments == 1).sum()} samples (SNR={args.snr_test} dB)")

    methods = ["conv1d_baseline", "conv1d_corr_physics", "conv1d_amp_physics"]
    seeds = [int(s) for s in args.seeds.split(",")]
    results: list[ExperimentRow] = []
    total_runs = len(methods) * len(seeds) * len(EVAL_MODES)
    run_idx = 0

    for method in methods:
        for eval_mode in EVAL_MODES:
            for seed in seeds:
                run_idx += 1
                print(f"\n[{run_idx}/{total_runs}] {method} {eval_mode} seed={seed}")
                row = run_experiment(
                    method=method,
                    csi=csi,
                    labels=labels,
                    environments=environments,
                    eval_mode=eval_mode,
                    seed=seed,
                    epochs=args.epochs,
                    batch_size=args.batch_size,
                    hidden_dim=args.hidden_dim,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    dropout=args.dropout,
                    aux_weight=args.aux_weight,
                    patience=args.patience,
                    min_delta=args.min_delta,
                )
                results.append(row)
                print(
                    f"  acc={row.test_accuracy:.4f} "
                    f"f1={row.test_macro_f1:.4f} "
                    f"epoch={row.best_epoch}"
                )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for eval_mode in EVAL_MODES:
        print(f"\n--- {eval_mode.upper()} ---")
        for method in methods:
            method_rows = [
                r for r in results if r.method == method and r.eval_mode == eval_mode
            ]
            accs = [r.test_accuracy for r in method_rows]
            print(f"  {method:30s}: acc={np.mean(accs):.4f} ± {np.std(accs):.4f}")

        baseline = [
            r.test_accuracy
            for r in results
            if r.method == "conv1d_baseline" and r.eval_mode == eval_mode
        ]
        for physics_method in [
            "conv1d_corr_physics",
            "conv1d_amp_physics",
        ]:
            physics = [
                r.test_accuracy
                for r in results
                if r.method == physics_method and r.eval_mode == eval_mode
            ]
            deltas = [p - b for p, b in zip(physics, baseline, strict=True)]
            wins = sum(1 for d in deltas if d > 0)
            rng_perm = np.random.RandomState(42)
            obs = np.mean(deltas)
            perm = [
                np.mean(deltas * rng_perm.choice([-1, 1], size=len(deltas)))
                for _ in range(10000)
            ]
            p_val = np.mean(np.abs(perm) >= np.abs(obs))
            short_name = physics_method.replace("conv1d_", "")
            print(
                f"  {short_name:30s}: "
                f"delta={obs:+.4f} pp, wins={wins}/{len(deltas)}, p={p_val:.4f}"
            )

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
