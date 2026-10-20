from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_script():
    path = Path(__file__).resolve().parents[1] / "scripts" / "prepare_widar_bvp.py"
    spec = importlib.util.spec_from_file_location("prepare_widar_bvp", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load prepare_widar_bvp.")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_gesture_ids_follow_the_per_date_readme_table() -> None:
    m = _load_script()
    # Same ID, different gestures on different dates (Widar3.0 README).
    assert m.gesture_name("20181211", "user3", 1) == "push_pull"
    assert m.gesture_name("20181121", "user1", 1) == "slide"
    assert m.gesture_name("20181109", "user1", 5) == "draw_zigzag_v"
    assert m.gesture_name("20181115", "user1", 4) == "draw_o_v"
    assert m.gesture_name("20181128", "user6", 4) == "draw_o_h"
    # Per-user tables on 20181205.
    assert m.gesture_name("20181205", "user2", 1) == "draw_o_h"
    assert m.gesture_name("20181205", "user3", 1) == "slide"
    # Digits and out-of-range IDs are not mapped.
    assert m.gesture_name("20181112", "user1", 1) is None
    assert m.gesture_name("20181209", "user2", 2) is None


def test_shared_gestures_are_recorded_in_every_room() -> None:
    m = _load_script()
    rooms: dict[int, set[str]] = {0: set(), 1: set(), 2: set()}
    for date, by_user in m.GESTURES_BY_DATE.items():
        for sequence in by_user.values():
            rooms[m.ROOM_BY_DATE[date]].update(sequence)
    for names in rooms.values():
        assert set(m.SHARED_GESTURES) <= names
