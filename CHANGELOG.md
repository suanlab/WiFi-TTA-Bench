# Changelog

Corrections and additions to WiFi-TTA-Bench artifacts and reported numbers. Entries
record what changed, why, and which released artifact backs each figure, so that numbers
in the paper can be traced to the code that produced them.

Superseded artifacts are preserved rather than overwritten, per the deprecation policy in
the supplement.

## 2026-09-25

### Added

- **Regularised-source evaluation** (`outputs/regularized_source/`). Ten standard source
  regularisers leave the Widar in-domain train/val gap at 33-42 pp (unregularised n=15
  mean: 34.0 pp, not the 42 pp previously quoted from a single run); only a shorter
  training budget shrinks it (11.1 pp at one epoch). No TTA method has an interval above
  zero at any gap, and the harmful families stay harmful.
- **Backbone breadth** (`outputs/backbone_breadth/`): nine methods on ResNet-BN,
  ResNet-GN and Transformer-LN across all three datasets. 2 of 81 intervals clear zero,
  consistent with chance; how much TTA harms depends strongly on backbone and
  normaliser (Widar entropy minimisation: -6.58 pp on the MLP, +0.04 on ResNet-BN).
- Source-side regularisers (`source_weight_decay`, `source_input_noise_std`,
  `source_mixup_alpha`), `source_train_accuracy` on result rows, and TENT adaptation of
  GroupNorm/LayerNorm affines on models without BatchNorm. Existing results are
  bit-identical.

### Corrected

- **Collapse-guard scope.** The 2026-09-09 entry stated the guard improves all four
  harmful cells at tau=0.95. With eighteen dataset-by-backbone cells there are eight
  harmful ones; the guard removes 40-100% of the loss in six and at most 6% in two
  (SignFi with ResNet-BN and Transformer), where the loss arrives before marginal class
  usage falls.

## 2026-09-09

### Corrected

**Collapse-guard sweep regenerated under the released implementation.** Two tables
reported different values for the same cell (Widar_BVP, MLP, tau=0.95, n=15): -0.90 pp
from `outputs/collapse_guard_sweep/`, and -0.60 pp from `outputs/guard_backbone/`. The
sweep artifacts predated the guard's accept/reject logic reaching the released tree, so
they were not reproducible from the shipped code. All six diversity floors were re-run
with the released implementation and a uniform three-method list
(`no_adapt,entropy_tta,collapse_guarded_entropy_tta`), placing the unguarded baseline in
the same run as the guarded arm.

Two floors moved; four are unchanged, and every negative-adaptation rate is identical:

| floor tau | before | after |
|---|---|---|
| 0.70 | -4.90 pp | -4.66 pp |
| 0.95 | -0.90 pp | -0.60 pp |

The corrected figure is **91% of Widar harm removed at tau=0.95** (was 86%). Artifacts
produced at tau=0.9 -- the guard-signal ablation and the asymmetry figure -- reproduce
exactly under the released code and are unchanged. Superseded files are kept under
`outputs/collapse_guard_sweep/superseded_2026-07-29/` with a README explaining the
difference.

**Guard-by-backbone scope corrected.** The by-backbone table covered two datasets and
attributed the guard's inertness on BatchNorm backbones to the normalisation layer.
Completing the 3x3 dataset-by-backbone grid falsified that reading: the SignFi-10 x
CNN1D+BN cell pairs BatchNorm with low source-only accuracy (41.2%) and large unguarded
harm (-10.60 pp), and there the guard removes 40% of the loss and moves the
negative-adaptation rate from 0.73 to 0.53. The benefit tracks the size of the unguarded
harm, not the architecture: 40-100% removed in the four cells losing more than 6 pp,
inert in the five losing under 3 pp. The table is now nine rows.

**Guard coverage count corrected.** The abstract stated that the guard helps every one
of eight method/dataset combinations. Enumerating every guarded measurement against its
unguarded arm from the same run gives nine at tau=0.9: seven harmful pairings with 49-82%
of the harm removed, plus two where adaptation helps and the gain is preserved exactly.
At tau=0.95 the guard improves all four harmful cells of the backbone grid, removing
40-100%. `scripts/build_guard_coverage.py` now emits this enumeration as
`outputs/guard_coverage/` and is the reference for any guard number in the paper.

### Added

- **SignFi-10 backbone cells.** `entropy_tta` and `collapse_guarded_entropy_tta` at
  n=15 on the MLP, MLP+BN, and CNN1D+BN backbones, completing the grid.
  Unguarded harm is -11.43 pp (MLP) and -10.60 pp (CNN1D+BN), consistent with the
  existing finding that no method clears zero on the primary MLP backbone.
- **Pre-deployment decision-rule analysis** (`outputs/deployment_rule/`,
  `scripts/derive_deployment_rule.py`). Tests whether TTA safety can be predicted before
  deployment from label-free signals. It cannot: across the nine cells, no measured
  predictor beats a majority-class baseline under leave-one-dataset-out validation,
  because source calibration error and unlabelled-target statistics track dataset
  identity rather than harm. Reported as a bounded negative result.
- **Released artifacts the paper cites but the repository excluded**: `guard_backbone`,
  `collapse_guard_sweep`, `collapse_guard_generality`, `collapse_guard_signfi_tent`, and
  `leakage_audit`.
- `expected_calibration_error()` in `pinn4csi.utils.metrics`, with tests.
- Guard firing fields (`attempted_steps`, `accepted_steps`, `abstained`, `stop_reason`)
  are now serialised by `scripts/run_extended_tta_per_dataset.py`, matching the
  architecture-ablation runner, so artifacts from either runner are self-describing.

### Reproducibility

The strict diversity floor is not bit-reproducible across runs. Four runs of the
Widar_BVP x MLP cell at tau=0.95 under the released code gave -0.6024 pp three times and
-0.68 pp once; the method list was held fixed within one pair and varied across the
other, so the variation does not track run configuration. Two consecutive repetitions
were bit-identical, and the negative-adaptation rate was 0.53 in all four runs. Looser
floors did not vary: tau=0.90 reproduced to the second decimal in every run.

This is consistent with the guard's accept/reject test landing marginally on a small
number of runs, where a non-deterministic reduction order is enough to flip one decision.
When reproducing the sweep, compare negative-adaptation rates and intervals rather than
the strict-floor mean alone.
