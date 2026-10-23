"""Re-read released per-run rows and report harm with a tolerance and per environment.

The benchmark's negative-adaptation rate is ``P(gain < 0)``. Two things make that
number hard to read on its own:

* a run that abstains (a guard that rolls back every step) scores gain exactly 0
  and so can never count as harm, while a method that perturbs the weights by a
  hair almost never lands on exactly 0;
* on small test halves one test sample moves accuracy by several points (SignFi:
  about 3 pp), so ``gain < 0`` can be a one-sample flip.

This script recomputes, from the per-run rows already released under
``outputs/``, for every (artifact, backbone, method):

* ``harm_rate``             P(gain < 0)                      (the paper's metric)
* ``harm_rate_gt1_sample``  P(gain < -1 test sample)          (loses 2+ samples)
* ``harm_rate_gt1pp``       P(gain < -1 pp)
* ``abstain_rate``          P(no accepted step and gain == 0), where recorded
* ``env_mean_pp``           mean gain per held-out environment (3 values) and how
  many of the three are negative -- the unit that generalises to a new room.
* ``hier_ci_pp``            a two-stage bootstrap interval: resample environments,
  then runs within each, so the interval reflects that only three environments
  were held out. Seeded, 10,000 resamples. With three environments it is close
  to the range of the environment means, so it is reported as descriptive.
* ``env_t_ci_pp``           a t interval (df = environments - 1) on the
  environment means -- the inferential interval for a new environment.
* ``sign_test_p_runs`` / ``sign_test_p_envs``  exact two-sided sign tests on the
  non-zero run gains and on the environment means (three environments cannot
  give p below 0.25).

No model is trained. Output: ``outputs/statistics/harm_tolerance.json``.
"""

from __future__ import annotations

import json
import logging
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT = Path("outputs/statistics/harm_tolerance.json")
PREPARED = Path("data/prepared")
TARGET_ADAPT_RATIO = 0.5
RESET_WIDAR = "outputs/reset_transfer_widar"
WIDAR_ROOM_TO_ENV = {"room0_classroom": 0, "room1_hall": 1, "room2_office": 2}

# (artifact, dataset) pairs. Every guard measurement comes from the guard
# registry (outputs/guard_coverage), plus the headline tables' sources.
EXTRA_ARTIFACTS: tuple[tuple[str, str], ...] = (
    ("outputs/widar_full_tta/full_results.json", "widar_bvp"),
    ("outputs/core_regen/ntufi_har_lame_sar_cotta_full.json", "ntufi_har"),
    ("outputs/core_regen/signfi_top10_lame_sar_cotta_full.json", "signfi_top10"),
    ("outputs/widar_mlpbn_canonical/widar_bvp_lame_sar_cotta_full.json", "widar_bvp"),
    (
        "outputs/signfi_mlpbn_canonical/signfi_top10_lame_sar_cotta_full.json",
        "signfi_top10",
    ),
    ("outputs/new_methods/widar_bvp_lame_sar_cotta_full.json", "widar_bvp"),
    ("outputs/final_ablations/arch_ablation_with_ci.json", "widar_bvp"),
    ("outputs/collapse_guard_sweep/ratio_0.9.json", "widar_bvp"),
    ("outputs/collapse_guard_sweep/ratio_0.95.json", "widar_bvp"),
    ("outputs/datta_comparison/ntufi/ntufi_har_lame_sar_cotta.json", "ntufi_har"),
    ("outputs/reset_transfer_ntufi/p0.5/ntufi_har_lame_sar_cotta.json", "ntufi_har"),
    (
        "outputs/reset_vs_guard_signfi/p0.5/signfi_top10_lame_sar_cotta.json",
        "signfi_top10",
    ),
    *(
        (f"{RESET_WIDAR}/p{p}/widar_bvp_lame_sar_cotta.json", "widar_bvp")
        for p in ("0.01", "0.05", "0.1", "0.3", "0.5")
    ),
)


def _target_test_sizes(dataset: str) -> dict[int, int]:
    """Size of the evaluated target half per held-out environment."""
    envs = np.load(PREPARED / dataset / "environments.npy")
    sizes: dict[int, int] = {}
    for env in np.unique(envs):
        n = int((envs == env).sum())
        # Mirrors _split_index_tensor: int(n * ratio) samples go to adaptation.
        n_adapt = min(max(1, int(n * TARGET_ADAPT_RATIO)), n - 1)
        sizes[int(env)] = n - n_adapt
    return sizes


def _rows_of(path: Path) -> list[dict[str, object]] | None:
    """Per-run rows for an artifact, from the file or its ``_full`` sibling."""
    candidates = [path]
    if not path.stem.endswith("_full"):
        candidates.insert(0, path.with_name(path.stem + "_full.json"))
    for candidate in candidates:
        if not candidate.exists():
            continue
        payload = json.loads(candidate.read_text())
        rows = payload.get("rows") if isinstance(payload, dict) else payload
        if isinstance(rows, list) and rows and "gain" in rows[0]:
            return rows
    return None


def _env_of(row: dict[str, object]) -> int:
    if "test_env" in row:
        return int(row["test_env"])  # type: ignore[arg-type]
    for key in ("split", "room"):
        value = row.get(key)
        if isinstance(value, str):
            if value in WIDAR_ROOM_TO_ENV:
                return WIDAR_ROOM_TO_ENV[value]
            digits = "".join(ch for ch in value if ch.isdigit())
            if digits:
                return int(digits[0])
    raise KeyError(f"No environment field in row: {sorted(row)}")


HIER_RESAMPLES = 10_000


def _hierarchical_ci(
    gains: np.ndarray, envs: np.ndarray, seed: int = 0
) -> tuple[float, float]:
    """Two-stage bootstrap of the mean: environments, then runs within them."""
    rng = np.random.default_rng(seed)
    groups = [gains[envs == e] for e in sorted(set(envs.tolist()))]
    means = np.empty(HIER_RESAMPLES)
    for b in range(HIER_RESAMPLES):
        picked = rng.integers(0, len(groups), size=len(groups))
        means[b] = np.mean(
            [rng.choice(groups[g], size=len(groups[g])).mean() for g in picked]
        )
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# Two-sided 97.5% Student-t quantiles for small df (avoids a scipy dependency).
T_975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571}


def _t_interval(values: list[float]) -> list[float]:
    """95% t interval of the mean of a few values (environment means)."""
    k = len(values)
    mean = float(np.mean(values))
    if k < 2:
        return [mean, mean]
    half = T_975[k - 1] * float(np.std(values, ddof=1)) / math.sqrt(k)
    return [mean - half, mean + half]


def _sign_test_p(values: list[float] | np.ndarray) -> float:
    """Exact two-sided sign test; zeros are dropped."""
    nonzero = [v for v in values if v != 0.0]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = min(sum(v > 0 for v in nonzero), sum(v < 0 for v in nonzero))
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2.0**n
    return float(min(1.0, 2.0 * tail))


def _summarise(
    rows: list[dict[str, object]], sizes: dict[int, int]
) -> dict[str, object]:
    gains = np.array([float(r["gain"]) for r in rows])  # type: ignore[arg-type]
    envs = np.array([_env_of(r) for r in rows])
    one_sample = np.array([1.0 / sizes[int(e)] for e in envs])
    out: dict[str, object] = {
        "n": len(rows),
        "mean_pp": 100.0 * float(gains.mean()),
        "harm_rate": float((gains < 0).mean()),
        # Losing more than one test sample, with float slack for 1/n gains.
        "harm_rate_gt1_sample": float((gains < -1.5 * one_sample).mean()),
        "harm_rate_gt1pp": float((gains < -0.01).mean()),
        "one_test_sample_pp": {
            str(e): round(100.0 / sizes[int(e)], 3) for e in sorted(set(envs))
        },
    }
    if "accepted_steps" in rows[0]:
        abstained = [
            int(r["accepted_steps"]) == 0 and float(r["gain"]) == 0.0  # type: ignore[arg-type]
            for r in rows
        ]
        out["abstain_rate"] = float(np.mean(abstained))
    env_means = {
        str(e): 100.0 * float(gains[envs == e].mean()) for e in sorted(set(envs))
    }
    out["env_mean_pp"] = env_means
    lo, hi = _hierarchical_ci(gains, envs)
    out["hier_ci_pp"] = [100.0 * lo, 100.0 * hi]
    out["envs_negative"] = int(sum(v < 0 for v in env_means.values()))
    out["env_t_ci_pp"] = _t_interval(list(env_means.values()))
    out["sign_test_p_runs"] = _sign_test_p(gains)
    out["sign_test_p_envs"] = _sign_test_p(list(env_means.values()))
    return out


DATASETS = ("widar_bvp", "ntufi_har", "signfi_top10")
GLOBBED_ARTIFACTS = (
    "outputs/faithful_ports/*/*_lame_sar_cotta_full.json",
    "outputs/bn_source_control/*/*_lame_sar_cotta_full.json",
    "outputs/bn_double_update/*/*_lame_sar_cotta_full.json",
    "outputs/tent_lr0/*/*_lame_sar_cotta_full.json",
    "outputs/published_bn_batchstats/*/*_lame_sar_cotta_full.json",
    "outputs/backbone_breadth/*.json",
)


def _artifacts() -> list[tuple[str, str]]:
    registry = json.loads(
        Path("outputs/guard_coverage/guard_coverage.json").read_text()
    )
    pairs = {(str(r["source"]), str(r["dataset"])) for r in registry["rows"]}
    pairs.update(EXTRA_ARTIFACTS)
    # Every per-run artifact the paper quotes environment-level evidence for.
    for pattern in GLOBBED_ARTIFACTS:
        for path in sorted(Path().glob(pattern)):
            if "superseded" in str(path):
                continue
            dataset = next(d for d in DATASETS if path.name.startswith(d))
            pairs.add((str(path), dataset))
    return sorted(pairs)


def main() -> None:
    sizes = {
        d: _target_test_sizes(d) for d in ("widar_bvp", "ntufi_har", "signfi_top10")
    }
    report: list[dict[str, object]] = []
    for source, dataset in _artifacts():
        if not Path(source).exists():
            logger.warning("Missing artifact %s; skipped", source)
            continue
        rows = _rows_of(Path(source))
        if rows is None:
            logger.warning("No per-run rows for %s; skipped", source)
            continue
        groups: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
        for row in rows:
            backbone = str(row.get("backbone", ""))
            groups[(backbone, str(row["method"]))].append(row)
        for (backbone, method), group in sorted(groups.items()):
            if method == "no_adapt":
                continue
            report.append(
                {
                    "source": source,
                    "dataset": dataset,
                    "backbone": backbone,
                    "method": method,
                    **_summarise(group, sizes[dataset]),
                }
            )
    # The label-smoothing pilot nests its rows per smoothing condition.
    pilot = json.loads(Path("outputs/source_overfit/mitigation_pilot.json").read_text())
    for condition in pilot["conditions"]:
        groups = defaultdict(list)
        for row in condition["rows"]:
            groups[str(row["method"])].append(row)
        for method, group in sorted(groups.items()):
            if method == "no_adapt":
                continue
            report.append(
                {
                    "source": "outputs/source_overfit/mitigation_pilot.json",
                    "dataset": "widar_bvp",
                    "backbone": "mlp",
                    "method": method,
                    "label_smoothing": condition["label_smoothing"],
                    **_summarise(group, sizes["widar_bvp"]),
                }
            )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(
        json.dumps(
            {
                "definitions": {
                    "harm_rate": "P(gain < 0)",
                    "harm_rate_gt1_sample": "P(gain < -1.5/n_test): loses 2+ samples",
                    "harm_rate_gt1pp": "P(gain < -0.01)",
                    "abstain_rate": "P(accepted_steps == 0 and gain == 0)",
                    "envs_negative": "held-out environments with mean gain < 0",
                },
                "target_test_sizes": sizes,
                "rows": report,
            },
            indent=2,
        )
    )
    logger.info("Wrote %s (%d rows)", OUTPUT, len(report))


if __name__ == "__main__":
    main()
