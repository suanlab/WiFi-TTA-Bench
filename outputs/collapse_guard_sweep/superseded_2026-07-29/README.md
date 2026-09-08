# Superseded sweep artifacts (generated 2026-07-29)

Preserved under the deprecation policy stated in the supplement: stale artifacts are not
silently overwritten.

These six files were produced by a pre-release version of the collapse guard, before the
accept/reject logic in `pinn4csi/training/paper2_tta.py` was finalised (commit 4186f4c
introduced `collapse_guard_diversity_ratio`, `reference_diversity`, the diversity floor,
and the rollback branch as new lines). They are NOT reproducible from the released code
and are retained only for provenance.

The current files one directory up were regenerated on 2026-09-09 with the released
implementation and a uniform three-method list
(`no_adapt,entropy_tta,collapse_guarded_entropy_tta`), so the unguarded baseline sits in
the same run as the guarded arm.

Differences, guarded mean gain on Widar_BVP (MLP, n=15):

| floor tau | superseded | released | delta | negative-adaptation rate |
|---|---|---|---|---|
| 0.50 | -5.84 pp | -5.84 pp | none | 0.93 -> 0.93 |
| 0.70 | -4.90 pp | -4.66 pp | +0.24 pp | 0.93 -> 0.93 |
| 0.80 | -2.67 pp | -2.67 pp | none | 0.93 -> 0.93 |
| 0.90 | -1.16 pp | -1.16 pp | none | 0.67 -> 0.67 |
| 0.95 | -0.90 pp | -0.60 pp | +0.29 pp | 0.53 -> 0.53 |
| 1.00 | +0.00 pp | +0.00 pp | none | 0.00 -> 0.00 |

Every negative-adaptation rate is unchanged; four of six floors are identical.
