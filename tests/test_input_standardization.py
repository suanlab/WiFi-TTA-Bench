from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from pinn4csi.data.paper1 import get_paper1_dataset_config
from pinn4csi.training.paper2_tta import build_prepared_tta_loaders


def _write_bundle(root: Path) -> None:
    config = get_paper1_dataset_config("ntufi_har")
    rng = np.random.default_rng(0)
    environments = np.repeat(np.arange(3), 40)
    # Source environments sit near 5, the held-out one near 50.
    csi = rng.normal(5.0, 2.0, size=(120, 4, 6)).astype(np.float32)
    csi[environments == 2] += 45.0
    directory = root / config.directory_name
    directory.mkdir(parents=True)
    np.save(directory / config.csi_filename, csi)
    np.save(directory / config.labels_filename, rng.integers(0, 3, size=120))
    np.save(directory / config.environments_filename, environments)


def _features(loader: torch.utils.data.DataLoader) -> torch.Tensor:  # type: ignore[type-arg]
    return torch.cat([batch["x"] for batch in loader])


def test_standardization_uses_source_train_statistics_only(tmp_path: Path) -> None:
    _write_bundle(tmp_path)
    kwargs = dict(
        dataset_name="ntufi_har",
        prepared_root=tmp_path,
        seed=0,
        batch_size=16,
        source_val_ratio=0.2,
        target_adapt_ratio=0.5,
        train_env_ids=(0, 1),
        test_env_ids=(2,),
    )
    train, _, _, test, *_ = build_prepared_tta_loaders(**kwargs)  # type: ignore[arg-type]
    x_train, x_test = _features(train), _features(test)
    assert torch.allclose(x_train.mean(dim=0), torch.zeros_like(x_train[0]), atol=1e-5)
    # Target is scaled with source statistics, so its shift is kept, not removed.
    assert float(x_test.mean()) > 10.0

    raw_train, *_ = build_prepared_tta_loaders(  # type: ignore[arg-type]
        **kwargs, input_standardization="none"
    )
    assert float(_features(raw_train).mean()) > 4.0
