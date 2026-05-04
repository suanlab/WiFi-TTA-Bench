"""Advanced physics-informed experiments: pretrain+finetune, curriculum, scaled."""

from __future__ import annotations

import argparse
import csv
import random
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F  # noqa: N812
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_prepared(root: Path, name: str) -> dict[str, torch.Tensor]:
    d = root / name
    csi = np.load(d / "csi.npy")
    labels = np.load(d / "labels.npy")
    envs = np.load(d / "environments.npy")
    amp = np.abs(csi).astype(np.float32)
    phase = np.angle(csi).astype(np.float32)
    features = np.stack([amp, phase], axis=-1).reshape(csi.shape[0], csi.shape[1], -1)
    return {
        "features": torch.from_numpy(features),
        "labels": torch.from_numpy(labels).long(),
        "environments": torch.from_numpy(envs).long(),
    }


def split_data(
    data: dict[str, torch.Tensor],
    seed: int,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
) -> tuple[dict[str, torch.Tensor], ...]:
    n = len(data["labels"])
    idx = torch.randperm(n, generator=torch.Generator().manual_seed(seed))
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)
    splits = []
    for start, end in [
        (0, n_train),
        (n_train, n_train + n_val),
        (n_train + n_val, n),
    ]:
        s_idx = idx[start:end]
        splits.append({k: v[s_idx] for k, v in data.items()})
    return tuple(splits)


class SubcarrierEncoder(nn.Module):
    def __init__(self, num_features: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(num_features, num_features, 5, padding=2, groups=num_features),
            nn.ReLU(),
            nn.Conv1d(num_features, hidden_dim, 1),
            nn.ReLU(),
            nn.Conv1d(hidden_dim, hidden_dim, 3, padding=1),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2))


class Classifier(nn.Module):
    def __init__(self, hidden_dim: int, num_classes: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        pooled_mean = torch.mean(encoded, dim=-1)
        pooled_std = torch.std(encoded, dim=-1, unbiased=False)
        pooled = torch.cat([pooled_mean, pooled_std], dim=1)
        return self.net(pooled)


class ReconstructionHead(nn.Module):
    def __init__(self, hidden_dim: int) -> None:
        super().__init__()
        self.head = nn.Conv1d(hidden_dim, 2, kernel_size=1)

    def forward(self, encoded: torch.Tensor) -> torch.Tensor:
        return self.head(encoded).transpose(1, 2)


def build_reconstruction_target(x: torch.Tensor) -> torch.Tensor:
    amp = torch.mean(x[..., 0::2], dim=-1)
    phase = torch.mean(x[..., 1::2], dim=-1)
    return torch.stack([amp, phase], dim=-1)


def evaluate(
    encoder: SubcarrierEncoder,
    classifier: Classifier,
    loader: DataLoader,  # type: ignore[type-arg]
    device: torch.device,
) -> dict[str, float]:
    encoder.eval()
    classifier.eval()
    correct = total = 0
    all_preds: list[int] = []
    all_labels: list[int] = []
    total_loss = 0.0
    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded = encoder(batch_x)
            logits = classifier(encoded)
            loss = F.cross_entropy(logits, batch_y)
            total_loss += loss.item() * len(batch_y)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == batch_y).sum().item()
            total += len(batch_y)
            all_preds.extend(preds.cpu().tolist())
            all_labels.extend(batch_y.cpu().tolist())
    acc = correct / max(total, 1)
    from sklearn.metrics import f1_score

    f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return {"accuracy": acc, "macro_f1": f1, "loss": total_loss / max(total, 1)}


def run_baseline(
    data: dict[str, torch.Tensor],
    seed: int,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    train, val, test = split_data(data, seed)
    num_features = int(train["features"].shape[-1])
    num_classes = int(train["labels"].max().item()) + 1
    encoder = SubcarrierEncoder(num_features, hidden_dim).to(device)
    classifier = Classifier(hidden_dim, num_classes).to(device)
    optimizer = Adam(list(encoder.parameters()) + list(classifier.parameters()), lr=lr)
    train_loader = DataLoader(
        TensorDataset(train["features"], train["labels"]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val["features"], val["labels"]),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(test["features"], test["labels"]),
        batch_size=batch_size,
    )
    best_val_acc = -1.0
    best_state: dict[str, dict] = {}  # type: ignore[type-arg]
    for _epoch in range(epochs):
        encoder.train()
        classifier.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded = encoder(batch_x)
            logits = classifier(encoded)
            loss = F.cross_entropy(logits, batch_y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(encoder, classifier, val_loader, device)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {
                "encoder": {
                    k: v.cpu().clone() for k, v in encoder.state_dict().items()
                },
                "classifier": {
                    k: v.cpu().clone() for k, v in classifier.state_dict().items()
                },
            }
    encoder.load_state_dict(best_state["encoder"])
    classifier.load_state_dict(best_state["classifier"])
    encoder.to(device)
    classifier.to(device)
    return evaluate(encoder, classifier, test_loader, device)


def run_pretrain_finetune(
    data: dict[str, torch.Tensor],
    seed: int,
    pretrain_epochs: int,
    finetune_epochs: int,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    train, val, test = split_data(data, seed)
    num_features = int(train["features"].shape[-1])
    num_classes = int(train["labels"].max().item()) + 1
    encoder = SubcarrierEncoder(num_features, hidden_dim).to(device)
    recon_head = ReconstructionHead(hidden_dim).to(device)
    classifier = Classifier(hidden_dim, num_classes).to(device)

    train_loader = DataLoader(
        TensorDataset(train["features"], train["labels"]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val["features"], val["labels"]),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(test["features"], test["labels"]),
        batch_size=batch_size,
    )

    # Phase 1: Pretrain encoder with reconstruction
    pretrain_opt = Adam(
        list(encoder.parameters()) + list(recon_head.parameters()), lr=lr
    )
    for _epoch in range(pretrain_epochs):
        encoder.train()
        recon_head.train()
        for batch_x, _batch_y in train_loader:
            batch_x = batch_x.to(device)
            encoded = encoder(batch_x)
            recon = recon_head(encoded)
            target = build_reconstruction_target(batch_x)
            loss = F.mse_loss(recon, target)
            pretrain_opt.zero_grad()
            loss.backward()
            pretrain_opt.step()

    # Phase 2: Fine-tune classifier (encoder frozen initially, then unfrozen)
    finetune_opt = Adam(
        list(encoder.parameters()) + list(classifier.parameters()), lr=lr * 0.1
    )
    best_val_acc = -1.0
    best_state: dict[str, dict] = {}  # type: ignore[type-arg]
    for _epoch in range(finetune_epochs):
        encoder.train()
        classifier.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded = encoder(batch_x)
            logits = classifier(encoded)
            loss = F.cross_entropy(logits, batch_y)
            finetune_opt.zero_grad()
            loss.backward()
            finetune_opt.step()
        val_metrics = evaluate(encoder, classifier, val_loader, device)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {
                "encoder": {
                    k: v.cpu().clone() for k, v in encoder.state_dict().items()
                },
                "classifier": {
                    k: v.cpu().clone() for k, v in classifier.state_dict().items()
                },
            }
    encoder.load_state_dict(best_state["encoder"])
    classifier.load_state_dict(best_state["classifier"])
    encoder.to(device)
    classifier.to(device)
    return evaluate(encoder, classifier, test_loader, device)


def run_curriculum(
    data: dict[str, torch.Tensor],
    seed: int,
    epochs: int,
    hidden_dim: int,
    batch_size: int,
    lr: float,
    device: torch.device,
) -> dict[str, float]:
    set_seed(seed)
    train, val, test = split_data(data, seed)
    num_features = int(train["features"].shape[-1])
    num_classes = int(train["labels"].max().item()) + 1
    encoder = SubcarrierEncoder(num_features, hidden_dim).to(device)
    recon_head = ReconstructionHead(hidden_dim).to(device)
    classifier = Classifier(hidden_dim, num_classes).to(device)
    all_params = (
        list(encoder.parameters())
        + list(recon_head.parameters())
        + list(classifier.parameters())
    )
    optimizer = Adam(all_params, lr=lr)

    train_loader = DataLoader(
        TensorDataset(train["features"], train["labels"]),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(val["features"], val["labels"]),
        batch_size=batch_size,
    )
    test_loader = DataLoader(
        TensorDataset(test["features"], test["labels"]),
        batch_size=batch_size,
    )

    best_val_acc = -1.0
    best_state: dict[str, dict] = {}  # type: ignore[type-arg]
    for epoch in range(epochs):
        progress = epoch / max(epochs - 1, 1)
        # Start with high recon weight, decay to near-zero
        recon_weight = 0.5 * (1.0 - progress)
        encoder.train()
        recon_head.train()
        classifier.train()
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            encoded = encoder(batch_x)
            logits = classifier(encoded)
            loss_task = F.cross_entropy(logits, batch_y)
            recon = recon_head(encoded)
            target = build_reconstruction_target(batch_x)
            loss_recon = F.mse_loss(recon, target)
            loss = loss_task + recon_weight * loss_recon
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        val_metrics = evaluate(encoder, classifier, val_loader, device)
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            best_state = {
                "encoder": {
                    k: v.cpu().clone() for k, v in encoder.state_dict().items()
                },
                "classifier": {
                    k: v.cpu().clone() for k, v in classifier.state_dict().items()
                },
            }
    encoder.load_state_dict(best_state["encoder"])
    classifier.load_state_dict(best_state["classifier"])
    encoder.to(device)
    classifier.to(device)
    return evaluate(encoder, classifier, test_loader, device)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared-root", type=Path, default=Path("data/prepared"))
    parser.add_argument("--dataset", type=str, default="ut_har")
    parser.add_argument("--seeds", type=str, default="0,1,2,3,4")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--pretrain-epochs", type=int, default=5)
    parser.add_argument("--finetune-epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument(
        "--output-csv", type=Path, default=Path("outputs/advanced_physics.csv")
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seeds = [int(s) for s in args.seeds.split(",")]
    data = load_prepared(args.prepared_root, args.dataset)

    results: list[dict[str, object]] = []
    methods = {
        "baseline": lambda seed: run_baseline(
            data, seed, args.epochs, args.hidden_dim, args.batch_size, args.lr, device
        ),
        "pretrain_finetune": lambda seed: run_pretrain_finetune(
            data,
            seed,
            args.pretrain_epochs,
            args.finetune_epochs,
            args.hidden_dim,
            args.batch_size,
            args.lr,
            device,
        ),
        "curriculum": lambda seed: run_curriculum(
            data, seed, args.epochs, args.hidden_dim, args.batch_size, args.lr, device
        ),
    }

    for method_name, method_fn in methods.items():
        accs, f1s = [], []
        for seed in seeds:
            metrics = method_fn(seed)
            accs.append(metrics["accuracy"])
            f1s.append(metrics["macro_f1"])
            results.append(
                {
                    "dataset": args.dataset,
                    "method": method_name,
                    "seed": seed,
                    "accuracy": metrics["accuracy"],
                    "macro_f1": metrics["macro_f1"],
                    "loss": metrics["loss"],
                }
            )
            sys.stdout.write(
                f"  {method_name} seed={seed} acc={metrics['accuracy']:.4f} "
                f"f1={metrics['macro_f1']:.4f}\n"
            )
            sys.stdout.flush()
        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_f1 = float(np.mean(f1s))
        std_f1 = float(np.std(f1s))
        print(
            f"{args.dataset}:{method_name} | "
            f"accuracy={mean_acc:.4f}±{std_acc:.4f} "
            f"macro_f1={mean_f1:.4f}±{std_f1:.4f} "
            f"runs={len(seeds)}"
        )

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=["dataset", "method", "seed", "accuracy", "macro_f1", "loss"]
        )
        writer.writeheader()
        writer.writerows(results)
    print(f"Results written to: {args.output_csv}")

    # Per-seed sign test: each method vs baseline
    baseline_by_seed = {
        r["seed"]: r["accuracy"] for r in results if r["method"] == "baseline"
    }
    for method_name in ["pretrain_finetune", "curriculum"]:
        method_by_seed = {
            r["seed"]: r["accuracy"] for r in results if r["method"] == method_name
        }
        wins = sum(
            1
            for s in baseline_by_seed
            if method_by_seed.get(s, 0) > baseline_by_seed[s]
        )
        losses = sum(
            1
            for s in baseline_by_seed
            if method_by_seed.get(s, 0) < baseline_by_seed[s]
        )
        ties = len(baseline_by_seed) - wins - losses
        print(f"  {method_name} vs baseline: wins={wins} losses={losses} ties={ties}")


if __name__ == "__main__":
    main()
