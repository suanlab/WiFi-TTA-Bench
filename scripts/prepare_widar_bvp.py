from __future__ import annotations

import argparse
import io
import json
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
import scipy.io as sio
from scipy.io.matlab import MatReadError

ROOM_BY_DATE = {
    "20181109": 0,
    "20181112": 0,
    "20181115": 0,
    "20181116": 0,
    "20181121": 0,
    "20181130": 0,
    "20181117": 1,
    "20181118": 1,
    "20181127": 1,
    "20181128": 1,
    "20181204": 1,
    "20181205": 1,
    "20181208": 1,
    "20181209": 1,
    "20181211": 2,
}

GESTURE_NAMES = {
    1: "push_pull",
    2: "sweep",
    3: "clap",
    4: "slide",
    5: "draw_o_h",
    6: "draw_zigzag_h",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare Widar3.0 BVP subset as a Paper1-style bundle"
    )
    parser.add_argument(
        "--bvp-zip",
        type=Path,
        default=Path("/tmp/widar_bvp/BVP.zip"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/prepared/widar_bvp"),
    )
    parser.add_argument("--target-length", type=int, default=22)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--cap-per-room-gesture",
        type=int,
        default=498,
    )
    parser.add_argument("--reserve-per-group", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    grouped_entries = collect_entries(args.bvp_zip)
    selected_entries = sample_entries(
        grouped_entries,
        cap_per_room_gesture=args.cap_per_room_gesture,
        reserve_per_group=args.reserve_per_group,
        seed=args.seed,
    )
    features, labels, environments = load_entries(
        args.bvp_zip,
        selected_entries,
        target_length=args.target_length,
        cap_per_room_gesture=args.cap_per_room_gesture,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "csi.npy", features)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "environments.npy", environments)

    metadata = {
        "source": "widar3_bvp_zip",
        "protocol_version": "1.0",
        "task_name": "gesture_recognition",
        "representation": "bvp",
        "num_classes": int(np.max(labels) + 1),
        "num_samples": int(features.shape[0]),
        "csi_shape": list(features.shape),
        "environment_names": {
            "0": "room1_classroom",
            "1": "room2_hall",
            "2": "room3_office",
        },
        "label_names": {
            str(idx): GESTURE_NAMES[gesture_id]
            for idx, gesture_id in enumerate(sorted(GESTURE_NAMES))
        },
        "shared_gesture_ids": sorted(GESTURE_NAMES),
        "cap_per_room_gesture": args.cap_per_room_gesture,
        "target_length": args.target_length,
        "normalization": "(x - 0.0025) / 0.0119",
        "reference": "Widar3.0 (Zheng et al. 2019/2021)",
        "url": "https://tns.thss.tsinghua.edu.cn/widar3.0/",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    print(f"Saved features {features.shape} to {args.output_dir / 'csi.npy'}")
    print(f"Saved labels {labels.shape} to {args.output_dir / 'labels.npy'}")
    print(
        f"Saved environments {environments.shape} to "
        f"{args.output_dir / 'environments.npy'}"
    )


def collect_entries(bvp_zip: Path) -> dict[tuple[int, int], list[str]]:
    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    with zipfile.ZipFile(bvp_zip) as archive:
        for name in archive.namelist():
            if not name.endswith(".mat") or "/6-link/" not in name:
                continue
            parts = [part for part in name.split("/") if part]
            if len(parts) < 5:
                continue
            date = parts[1].split("-")[0]
            if date not in ROOM_BY_DATE:
                continue
            stem = Path(parts[-1]).stem
            fields = stem.split("-")
            if len(fields) < 5:
                continue
            gesture_id = int(fields[1])
            if gesture_id not in GESTURE_NAMES:
                continue
            room_id = ROOM_BY_DATE[date]
            grouped[(room_id, gesture_id)].append(name)
    return grouped


def sample_entries(
    grouped_entries: dict[tuple[int, int], list[str]],
    cap_per_room_gesture: int,
    reserve_per_group: int,
    seed: int,
) -> list[tuple[str, int, int]]:
    rng = np.random.default_rng(seed)
    selected: list[tuple[str, int, int]] = []
    for room_id in sorted({room for room, _ in grouped_entries}):
        for gesture_id in sorted(GESTURE_NAMES):
            entries = grouped_entries[(room_id, gesture_id)]
            if len(entries) < cap_per_room_gesture:
                raise ValueError(
                    f"Not enough entries for room {room_id}, gesture {gesture_id}: "
                    f"{len(entries)} < {cap_per_room_gesture}"
                )
            chosen_count = min(
                len(entries),
                cap_per_room_gesture + reserve_per_group,
            )
            chosen = rng.choice(entries, size=chosen_count, replace=False)
            for name in chosen.tolist():
                selected.append((name, gesture_id - 1, room_id))
    return selected


def load_entries(
    bvp_zip: Path,
    selected_entries: list[tuple[str, int, int]],
    target_length: int,
    cap_per_room_gesture: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[int] = []
    environments: list[int] = []
    kept_counts: dict[tuple[int, int], int] = defaultdict(int)
    with zipfile.ZipFile(bvp_zip) as archive:
        for name, label, room_id in selected_entries:
            group_key = (room_id, label)
            if kept_counts[group_key] >= cap_per_room_gesture:
                continue
            with archive.open(name) as handle:
                payload = handle.read()
            try:
                mat = sio.loadmat(io.BytesIO(payload))
            except (MatReadError, OSError, ValueError):
                continue
            bvp = np.asarray(mat["velocity_spectrum_ro"], dtype=np.float32)
            bvp = reshape_bvp(bvp, target_length=target_length)
            bvp = (bvp - 0.0025) / 0.0119
            features.append(bvp.reshape(target_length, -1))
            labels.append(label)
            environments.append(room_id)
            kept_counts[group_key] += 1

    expected_groups = 3 * len(GESTURE_NAMES)
    if len(kept_counts) != expected_groups or any(
        count < cap_per_room_gesture for count in kept_counts.values()
    ):
        raise ValueError(
            "Failed to collect enough valid BVP samples for every room/gesture group."
        )
    return (
        np.stack(features).astype(np.float32),
        np.asarray(labels, dtype=np.int64),
        np.asarray(environments, dtype=np.int64),
    )


def reshape_bvp(bvp: np.ndarray, target_length: int) -> np.ndarray:
    time_major = np.transpose(bvp, (2, 0, 1))
    current_length = time_major.shape[0]
    if current_length == target_length:
        return time_major
    if current_length > target_length:
        indices = np.linspace(0, current_length - 1, target_length)
        indices = np.round(indices).astype(np.int64)
        return time_major[indices]
    padding = np.zeros(
        (target_length - current_length, time_major.shape[1], time_major.shape[2]),
        dtype=np.float32,
    )
    return np.concatenate([time_major, padding], axis=0)


if __name__ == "__main__":
    main()
