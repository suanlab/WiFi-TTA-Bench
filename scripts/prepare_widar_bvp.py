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

# Shared six-gesture task: the only gestures recorded in all three rooms (room 3,
# 20181211, has exactly these six). Label index = position in this tuple.
SHARED_GESTURES = (
    "push_pull",
    "sweep",
    "clap",
    "slide",
    "draw_o_h",
    "draw_zigzag_h",
)

_PP, _SW, _CL, _SL = "push_pull", "sweep", "clap", "slide"
_OV, _ZV, _NV = "draw_o_v", "draw_zigzag_v", "draw_n_v"
_OH, _ZH, _NH = "draw_o_h", "draw_zigzag_h", "draw_n_h"
_TH, _RH = "draw_triangle_h", "draw_rectangle_h"
_NINE_H = (_PP, _SW, _CL, _SL, _OH, _ZH, _NH, _TH, _RH)

# Gesture ID -> gesture, per recording date (and per user where the official
# README lists users separately). In Widar3.0 a gesture ID means a different
# gesture on different dates; source: the dataset README ("Data" table).
# Key None applies to every user of that date. 20181112/20181116 (digits) are
# omitted: none of their gestures is shared by all three rooms.
GESTURES_BY_DATE: dict[str, dict[str | None, tuple[str, ...]]] = {
    "20181109": {None: (_PP, _SW, _CL, _SL, _ZV, _NV)},
    "20181115": {None: (_PP, _SW, _CL, _OV, _ZV, _NV)},
    "20181117": {None: (_PP, _SW, _CL, _OV, _ZV, _NV)},
    "20181118": {None: (_PP, _SW, _CL, _OV, _ZV, _NV)},
    "20181121": {None: (_SL, _OH, _ZH, _NH, _TH, _RH)},
    "20181127": {None: (_SL, _OH, _ZH, _NH, _TH, _RH)},
    "20181128": {None: (_PP, _SW, _CL, _OH, _ZH, _NH)},
    "20181130": {None: _NINE_H},
    "20181204": {None: _NINE_H},
    "20181205": {
        "user2": (_OH, _ZH, _NH, _TH, _RH),
        "user3": (_SL, _OH, _ZH, _NH, _TH, _RH),
    },
    "20181208": {"user2": (_PP, _SW, _CL, _SL), "user3": (_PP, _SW, _CL)},
    "20181209": {"user2": (_PP,), "user6": (_PP, _SW, _CL, _SL, _OH, _ZH)},
    "20181211": {None: (_PP, _SW, _CL, _SL, _OH, _ZH)},
}


def gesture_name(date: str, user: str, gesture_id: int) -> str | None:
    """Gesture recorded under ``gesture_id`` on ``date`` by ``user``, or None."""
    by_user = GESTURES_BY_DATE.get(date)
    if by_user is None:
        return None
    sequence = by_user.get(user, by_user.get(None))
    if sequence is None or not 1 <= gesture_id <= len(sequence):
        return None
    return sequence[gesture_id - 1]


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
    features, labels, environments, sources = load_entries(
        args.bvp_zip,
        selected_entries,
        target_length=args.target_length,
        cap_per_room_gesture=args.cap_per_room_gesture,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "csi.npy", features)
    np.save(args.output_dir / "labels.npy", labels)
    np.save(args.output_dir / "environments.npy", environments)
    # Upstream file for every sample, in array order (sample-level provenance).
    (args.output_dir / "sources.txt").write_text(
        "\n".join(sources) + "\n", encoding="utf-8"
    )

    metadata = {
        "source": "widar3_bvp_zip",
        "protocol_version": "3.0",
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
        "label_names": {str(idx): name for idx, name in enumerate(SHARED_GESTURES)},
        "label_mapping": "per recording date and user (GESTURES_BY_DATE)",
        "file_filter": "BVP/<date>-VS/6-link/, plus BVP/20181130-VS/<user>/",
        "sampling": "per (room, gesture): round-robin over recording dates",
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
    """Group BVP files by (room, shared-gesture label index).

    Dates store BVP under ``<date>-VS/6-link/<user>/``, except 20181130, which
    stores it directly under ``<date>-VS/<user>/`` (same 20x20xT format).
    """
    grouped: dict[tuple[int, int], list[str]] = defaultdict(list)
    with zipfile.ZipFile(bvp_zip) as archive:
        for name in archive.namelist():
            if not name.endswith(".mat"):
                continue
            parts = [part for part in name.split("/") if part]
            if "/6-link/" not in name and not name.startswith("BVP/20181130-VS/"):
                continue
            if len(parts) < 4:
                continue
            date = parts[1].split("-")[0]
            if date not in ROOM_BY_DATE:
                continue
            stem = Path(parts[-1]).stem
            fields = stem.split("-")
            if len(fields) < 5:
                continue
            name_of_gesture = gesture_name(date, fields[0], int(fields[1]))
            if name_of_gesture not in SHARED_GESTURES:
                continue
            room_id = ROOM_BY_DATE[date]
            grouped[(room_id, SHARED_GESTURES.index(name_of_gesture))].append(name)
    return grouped


def sample_entries(
    grouped_entries: dict[tuple[int, int], list[str]],
    cap_per_room_gesture: int,
    reserve_per_group: int,
    seed: int,
) -> list[tuple[str, int, int]]:
    """Draw each (room, gesture) group evenly across its recording dates.

    Dates are visited round-robin in a seeded random order, so a gesture's
    samples are spread over every date that recorded it instead of following
    how many files each date happens to hold (which tied gesture to date).
    """
    rng = np.random.default_rng(seed)
    selected: list[tuple[str, int, int]] = []
    for room_id in sorted({room for room, _ in grouped_entries}):
        for label in range(len(SHARED_GESTURES)):
            entries = sorted(grouped_entries[(room_id, label)])
            if len(entries) < cap_per_room_gesture:
                raise ValueError(
                    f"Not enough entries for room {room_id}, gesture {label}: "
                    f"{len(entries)} < {cap_per_room_gesture}"
                )
            by_date: dict[str, list[str]] = defaultdict(list)
            for name in entries:
                by_date[name.split("/")[1].split("-")[0]].append(name)
            queues = [list(rng.permutation(by_date[date])) for date in sorted(by_date)]
            chosen_count = min(len(entries), cap_per_room_gesture + reserve_per_group)
            chosen: list[str] = []
            while len(chosen) < chosen_count:
                for queue in queues:
                    if queue and len(chosen) < chosen_count:
                        chosen.append(str(queue.pop()))
            for name in chosen:
                selected.append((name, label, room_id))
    return selected


def load_entries(
    bvp_zip: Path,
    selected_entries: list[tuple[str, int, int]],
    target_length: int,
    cap_per_room_gesture: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    sources: list[str] = []
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
            if bvp.ndim != 3:  # a few files hold a single 20x20 frame
                continue
            bvp = reshape_bvp(bvp, target_length=target_length)
            bvp = (bvp - 0.0025) / 0.0119
            features.append(bvp.reshape(target_length, -1))
            labels.append(label)
            environments.append(room_id)
            sources.append(name)
            kept_counts[group_key] += 1

    expected_groups = 3 * len(SHARED_GESTURES)
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
        sources,
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
