"""Prepare the SignFi-10 bundle from the public SignFi lab file.

Source: ``dataset_lab_276_dl.mat`` from the SignFi release (Ma et al., 2018), which
holds ``csid_lab`` (200 packets x 30 subcarriers x 3 antennas x 5520 instances,
complex) and ``label_lab`` (sign IDs 1-276). Instances are stored repetition-major:
instance ``r * 276 + (s - 1)`` is repetition ``r`` (0-19) of sign ``s``.

SignFi-10 keeps lab signs 1-10 (all 20 repetitions each, one user) and uses CSI
amplitude, flattened to (200 packets, 90 = 30 subcarriers x 3 antennas). The three
"environments" are ordered repetition blocks of every sign: repetitions 1-6, 7-13
and 14-20. They are partitions of one recording setting, not separate rooms.

Output (bundle contract of ``pinn4csi/data/paper1.py``): ``csi.npy`` (200, 200, 90),
``labels.npy`` (0-9 = signs 1-10), ``environments.npy`` (0-2), ``metadata.json``.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import scipy.io as sio

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

NUM_SIGNS_IN_FILE = 276
SIGN_IDS = tuple(range(1, 11))
NUM_REPETITIONS = 20
# Repetition r (0-based) -> environment: 1-6, 7-13, 14-20 in 1-based terms.
REPETITION_BLOCKS = ((0, 6), (6, 13), (13, 20))


def environment_of(repetition: int) -> int:
    for env, (start, stop) in enumerate(REPETITION_BLOCKS):
        if start <= repetition < stop:
            return env
    raise ValueError(f"repetition {repetition} outside 0-{NUM_REPETITIONS - 1}")


def build(mat_path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mat = sio.loadmat(mat_path)
    csi = mat["csid_lab"]
    labels_in_file = mat["label_lab"].ravel()
    order = [(rep, sign) for rep in range(NUM_REPETITIONS) for sign in SIGN_IDS]
    indices = [rep * NUM_SIGNS_IN_FILE + (sign - 1) for rep, sign in order]
    if any(
        int(labels_in_file[i]) != sign
        for i, (_, sign) in zip(indices, order, strict=True)
    ):
        raise ValueError(
            "label_lab does not follow the expected repetition-major order"
        )
    amplitude = np.abs(csi[:, :, :, indices])  # (200, 30, 3, N)
    features = amplitude.reshape(200, 90, -1).transpose(2, 0, 1).astype(np.float32)
    labels = np.array([sign - 1 for _, sign in order], dtype=np.int64)
    environments = np.array([environment_of(rep) for rep, _ in order], dtype=np.int64)
    return features, labels, environments


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--signfi-mat",
        type=Path,
        default=Path("data/raw/signfi/dataset_lab_276_dl.mat"),
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("data/prepared/signfi_top10")
    )
    args = parser.parse_args()
    features, labels, environments = build(args.signfi_mat)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "csi.npy", features)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "environments.npy", environments)
    metadata = {
        "source": "signfi_lab_276_dl",
        "protocol_version": "2.0",
        "task_name": "sign_language_recognition",
        "representation": "amplitude",
        "num_classes": len(SIGN_IDS),
        "num_samples": int(features.shape[0]),
        "csi_shape": list(features.shape),
        "sign_ids": list(SIGN_IDS),
        "label_names": {str(i): f"sign_{s:02d}" for i, s in enumerate(SIGN_IDS)},
        "environment_names": {
            "0": "repetitions_1-6",
            "1": "repetitions_7-13",
            "2": "repetitions_14-20",
        },
        "note": "SignFi lab signs 1-10, one user; environments are ordered "
        "repetition blocks of one recording setting",
        "reference": "SignFi (Ma et al., IMWUT 2018)",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    logger.info("Saved %s to %s", features.shape, args.output_dir)


if __name__ == "__main__":
    main()
