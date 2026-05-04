"""Run NTU-Fi HAR experiments: baseline vs physics auxiliary.

Adapts the same Conv1D residual architecture from run_diagnosis_experiments.py
for NTU-Fi HAR data. Since NTU-Fi HAR has no environment labels, only IID
evaluation (random train/val/test split) is supported.

Data: data/prepared/ntufi_har/csi.npy (1200, 114, 500), labels.npy (1200,)
Shape convention: (samples, subcarriers, time) -> transpose to (samples, time, features)
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


@dataclass(frozen=True)
class ExperimentRow:
    dataset: str
    method: str
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


def load_ntufi_har(prepared_root: Path) -> tuple[Tensor, Tensor]:
    dataset_dir = prepared_root / "ntufi_har"
    csi = np.load(dataset_dir / "csi.npy")  # (1200, 114, 500)
    labels = np.load(dataset_dir / "labels.npy")  # (1200,)
    # Transpose: (samples, subcarriers, time) -> (samples, time, subcarriers)
    csi = csi.transpose(0, 2, 1).astype(np.float32)  # (1200, 500, 114)
    # Downsample time: 500 -> 125 (every 4th, matching original 2000->500 ratio)
    csi = csi[:, ::4, :]  # (1200, 125, 114)
    return torch.from_numpy(csi), torch.from_numpy(labels).long().flatten()


def build_iid_split(
    labels: Tensor,
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> dict[str, Tensor]:
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
    loaders = {}
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

    split = build_iid_split(labels, seed)
    train_loader, val_loader, test_loader = make_dataloaders(
        csi, labels, split, batch_size
    )

    num_features = csi.shape[-1]  # 114 subcarriers
    num_classes = 6

    encoder = Conv1DEncoder(
        num_features=num_features, hidden_dim=hidden_dim, dropout=dropout
    ).to(device)

    use_aux = "physics" in method
    aux_head: nn.Module | None = None
    classifier_head: nn.Module | None = None

    if use_aux:
        aux_head = AmplitudePhysicsHead(hidden_dim=hidden_dim).to(device)

    # Shared classifier head
    classifier_head = SequenceClassifier(
        hidden_dim=hidden_dim, num_classes=num_classes, dropout=dropout
    ).to(device)

    params = list(encoder.parameters()) + list(classifier_head.parameters())
    if aux_head is not None:
        params += list(aux_head.parameters())

    optimizer = AdamW(params, lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state = None

    for epoch in range(epochs):
        # Training
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
            loss_cls = functional.cross_entropy(logits, batch_y)
            loss = loss_cls

            if use_aux and aux_head is not None:
                aux_pred = aux_head(encoded).squeeze(-1)  # (B, T)
                physics_target = batch_x.mean(dim=-1)  # mean over subcarriers: (B, T)
                loss_aux = functional.mse_loss(aux_pred, physics_target)
                loss = loss_cls + aux_weight * loss_aux

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
                val_total += batch_x.size(0)
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

    # Restore best and evaluate on test
    if best_state is not None:
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
    all_preds = []
    all_labels_list = []

    with torch.no_grad():
        for batch_x, batch_y in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded = encoder(batch_x)
            logits = classifier_head(encoded)
            loss_cls = functional.cross_entropy(logits, batch_y)
            test_loss_sum += loss_cls.item() * batch_x.size(0)
            test_correct += (logits.argmax(dim=1) == batch_y).sum().item()
            test_total += batch_x.size(0)
            all_preds.extend(logits.argmax(dim=1).cpu().numpy())
            all_labels_list.extend(batch_y.cpu().numpy())

    test_acc = test_correct / max(test_total, 1)
    test_loss = test_loss_sum / max(test_total, 1)

    # Macro F1
    all_preds_np = np.array(all_preds)
    all_labels_np = np.array(all_labels_list)
    f1_scores = []
    for c in range(num_classes):
        tp = ((all_preds_np == c) & (all_labels_np == c)).sum()
        fp = ((all_preds_np == c) & (all_labels_np != c)).sum()
        fn = ((all_preds_np != c) & (all_labels_np == c)).sum()
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1_scores.append(2 * prec * rec / max(prec + rec, 1e-8))
    macro_f1 = float(np.mean(f1_scores))

    param_count = count_trainable_parameters(encoder, classifier_head)
    if aux_head is not None:
        param_count += count_trainable_parameters(aux_head)

    return ExperimentRow(
        dataset="ntufi_har",
        method=method,
        seed=seed,
        best_epoch=best_epoch,
        stopped_epoch=epoch + 1,
        train_examples=len(split["train"]),
        val_examples=len(split["val"]),
        test_examples=len(split["test"]),
        parameter_count=param_count,
        test_accuracy=test_acc,
        test_macro_f1=macro_f1,
        test_loss=test_loss,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="NTU-Fi HAR experiments")
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=5e-4)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--aux-weight", type=float, default=0.1)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument(
        "--output-csv", type=Path, default=Path("outputs/ntufi_experiments.csv")
    )
    args = parser.parse_args()

    csi, labels = load_ntufi_har(args.prepared_root)
    n_cls = len(set(labels.tolist()))
    print(f"Data: csi {csi.shape}, labels {labels.shape}, classes {n_cls}")

    methods = ["conv1d_baseline", "conv1d_amp_physics"]
    seeds = [int(s) for s in args.seeds.split(",")]

    results: list[ExperimentRow] = []
    total_runs = len(methods) * len(seeds)
    run_idx = 0

    for method in methods:
        for seed in seeds:
            run_idx += 1
            print(f"\n[{run_idx}/{total_runs}] {method} seed={seed}")
            row = run_experiment(
                method=method,
                csi=csi,
                labels=labels,
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
                f"  acc={row.test_accuracy:.4f}"
                f" f1={row.test_macro_f1:.4f}"
                f" epoch={row.best_epoch}"
            )

    # Save results
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=asdict(results[0]).keys())
        writer.writeheader()
        for r in results:
            writer.writerow(asdict(r))

    # Summary statistics
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    for method in methods:
        method_results = [r for r in results if r.method == method]
        accs = [r.test_accuracy for r in method_results]
        f1s = [r.test_macro_f1 for r in method_results]
        print(
            f"{method}: acc={np.mean(accs):.4f} +/- {np.std(accs):.4f} "
            f"f1={np.mean(f1s):.4f} +/- {np.std(f1s):.4f}"
        )

    # Paired comparison
    baseline = [r.test_accuracy for r in results if r.method == "conv1d_baseline"]
    physics = [r.test_accuracy for r in results if r.method == "conv1d_amp_physics"]
    deltas = [p - b for p, b in zip(physics, baseline, strict=True)]
    wins = sum(1 for d in deltas if d > 0)
    print(f"\nPaired delta: {np.mean(deltas):.4f} +/- {np.std(deltas):.4f} pp")
    print(f"Wins: {wins}/{len(deltas)}")

    # Paired permutation test
    n_perm = 10000
    rng = np.random.RandomState(42)
    observed_mean_delta = np.mean(deltas)
    perm_deltas = []
    for _ in range(n_perm):
        signs = rng.choice([-1, 1], size=len(deltas))
        perm_deltas.append(np.mean(deltas * signs))
    p_value = np.mean(np.abs(perm_deltas) >= np.abs(observed_mean_delta))
    print(f"Permutation test: delta={observed_mean_delta:.4f} pp, p={p_value:.4f}")

    print(f"\nResults saved to {args.output_csv}")


if __name__ == "__main__":
    main()
