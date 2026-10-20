# Changelog

Corrections and additions to WiFi-TTA-Bench artifacts and reported numbers. Entries
record what changed, why, and which released artifact backs each figure, so that numbers
in the paper can be traced to the code that produced them.

Superseded artifacts are preserved rather than overwritten, per the deprecation policy in
the supplement.

## 2026-10-17

### Corrected

- **Widar_BVP labels**: `scripts/prepare_widar_bvp.py` mapped gesture IDs 1-6 to the same
  six gestures on every recording date, but in Widar3.0 the meaning of a gesture ID changes
  between dates (and between users on 20181205/08/09). Among the candidate files, 65 % of
  room-1 and 52 % of room-2 labels were wrong; room 3 was correct; the date-to-room
  assignment was correct. Labels now follow the dataset README's per-date table
  (`GESTURES_BY_DATE`), with the same six shared gestures, 498 samples per room and gesture
  and 8,964 samples; file selection sorts the archive listing so it no longer depends on
  zip order; `sources.txt` records each sample's upstream file (bundle protocol 2.0). Every
  Widar_BVP artifact was regenerated (2026-10-20); the pre-fix artifacts are at git tag
  `pre-widar-label-fix` and the old prepared bundle is kept as superseded. Primary-MLP
  source validation / target accuracy: 73.8 % / 62.0 % (was 53.4 % / 36.0 %); CNN-GRU:
  79.5 % / 68.7 % (was 54.9 % / 37.2 %).
- **Cached results under `--force`**: `scripts/run_backbone_breadth.py` skipped any cell
  whose output existed and `scripts/run_regularized_source.py --stage search` reused arms
  already in `search.json`, so `reproduce_all_results.py --force` silently kept stale
  results for them. Both scripts take `--force` now and the reproducer passes it; the Widar
  backbone-breadth cells, the regularisation search and the TTA stage that reads its
  selection were regenerated (stale copies in `superseded_2026-10-21/`).
- **BatchNorm statistics in adaptation**: each unguarded adaptation step also scored its
  objective with a train-mode forward pass, so five steps updated BatchNorm running
  statistics ten times. `TTAExperimentConfig.freeze_bn_in_scoring_pass` (off by default, so
  released artifacts still reproduce) updates them once per step. Controls on NTU-Fi and
  SignFi-10 MLP+BN: `outputs/tent_lr0/` (TENT at learning rate 0 recovers +10.10 of
  +10.29 pp on SignFi-10) and `outputs/bn_single_update/`.
- **Environment-level inference**: `outputs/statistics/harm_tolerance.json` adds a t interval
  over the environment means (`env_t_ci_pp`) and exact sign tests over runs and over
  environments; with three environments the hierarchical bootstrap is close to the range of
  the environment means and is now described as descriptive.
- **Croissant**: `outputs/SUBMISSION_MANIFEST.md`, listed in the Croissant distribution,
  was not tracked, so the validator failed on a clean checkout; it is now tracked.

- **Overconfidence** (`outputs/overconfidence/`): the analysis trained unseeded source models,
  so they were not the benchmark's, and it labelled Widar classes with wrong names. It now
  seeds like `run_tta_suite` (all 15 Widar runs match the benchmark's source-only accuracy)
  and takes class names from the bundle metadata. Widar: confidence 61-75 %, per-class
  accuracy 25-54 % (was 69-81 % / 26-58 %). The stale `widar_per_class.*` files are
  superseded.
- **Published SAR**: its model-recovery threshold (0.2 for ImageNet) is scaled by
  ln(C)/ln(1000). With the absolute threshold, SAR reset to the source after every step in
  10 of 15 NTU-Fi MLP+BN runs. Regenerated `outputs/faithful_ports/` (previous copies in
  `superseded_2026-10-17/`); only SAR's numbers change, e.g. NTU-Fi MLP+BN +0.70 pp
  [+0.2, +1.3] (was +0.24).
- **Headline intervals at the environment level** (two-stage bootstrap over the three held-out
  environments): Widar MLP+BN BN-affine TENT -0.42 pp spans zero ([-1.09, +0.13]); SignFi
  MLP+BN TENT +10.3 pp holds ([+2.9, +17.7]); SignFi BN-reset +2.6 pp spans zero.

### Added

- **Published BatchNorm regime** (`bn_eval_stats="batch"`; `outputs/published_bn_batchstats/`):
  TENT, SAR and CoTTA evaluated with each test batch's own statistics. On MLP+BN all BN
  methods then coincide: -1.43 pp Widar, -3.03 pp NTU-Fi, +11.56 pp SignFi-10. Cumulative
  BN-reset (`outputs/bn_reset_cumulative/`): -0.99 / +0.24 / +12.95 pp.
- **Guarded published methods** (`collapse_guarded_sar_full`, `collapse_guarded_cotta_full`,
  paired same-batch diversity): stops SAR's loss on SignFi MLP (-3.84 -> 0.00 pp), raises
  SignFi MLP+BN SAR to +13.59 pp, little change elsewhere. Paired reference for the entropy
  guard (`outputs/guard_paired/`): -0.87 pp Widar, -3.93 pp NTU-Fi.
- `cnn_gru` backbone (Widar3.0-style per-frame CNN + GRU) and its Widar run
  (`outputs/cnn_gru/`).
- Unused `tent_bn_momentum` config field removed.

## 2026-10-16 (second simulated review, Tier 1)

### Corrected

- **Croissant metadata** regenerated (`outputs/croissant/`): v1.1.0 with `dateModified`, all 25
  file hashes match their artifacts, and the distribution now includes the published-method
  results, guard coverage, harm tolerance, protocol sensitivity and sample-level split
  manifests. `scripts/validate_croissant.py` now fails on any hash mismatch.
- **Reproducer**: overconfidence is registered per dataset under the file names the script
  actually writes (the old entry expected `widar_per_class.json`, a stale pre-audit file now
  in `outputs/overconfidence/superseded_2026-10-16/`); the split-leakage audit, the guard
  figures and the split manifests are registered; manifest and component labels no longer
  carry stale table numbers.
- **Public API** (`wifi_tta_bench.evaluate`): default batch size 32 as in the paper, a
  `backbone` argument, and the paper's names (`sar-lite`, `cotta-lite`, `t3a-ent`,
  `tent-proj`) accepted.
- Dataset-registry descriptions of NTU-Fi and SignFi-10 say "ordered partitions" instead of
  sessions or temporal splits; Figure 1 uses the port names and drops the LAME no-op.

### Added

- `scripts/export_split_manifests.py` -> `outputs/splits/`: per-sample environment and label
  and the exact train/val/adapt/eval indices of every fold and seed, checked against the
  benchmark loaders.
- `RESPONSIBLE_USE.md`: a non-binding responsible-use statement.

## 2026-10-15

### Corrected

- **Simplified ports renamed by what they compute.** `sar` and `cotta` are batch entropy plus
  a SAR-like low-entropy filter or a CoTTA-like squared-probability pseudo-label term; they
  update the encoder and have no sharpness-aware step, model recovery, EMA teacher or
  augmentation. `t3a` runs five entropy steps before T3A's prototype readout. They are now
  reported as `sar-lite`, `cotta-lite` and `t3a-ent` (code names unchanged), and the
  "faithful BN-only SAR/CoTTA" label on `run_widar_faithful_sar_cotta.py` is withdrawn.
  The guard result on these methods is a result across entropy variants, not across method
  families.
- **LAME** is a no-op in our port (the source model is returned unchanged); it is excluded
  from method counts.
- **Physics prior on real data.** The prepared Widar_BVP, NTU-Fi and SignFi-10 bundles carry
  no prior, so `p(x)=0` in every real-data run and the physics-prior methods reduce to
  feature-moment alignment there.
- **NTU-Fi and SignFi-10 environments** are ordered partitions of each class's recordings
  (cut into thirds), not recording sessions.
- **Label smoothing.** TENT's drop in P(gain<0) from 0.60 to 0.33 is within one test sample
  (rate of losing more than one sample: 0.13, 0.20, 0.20 at alpha 0, 0.1, 0.2) and is no
  longer reported as an effect; the entropy-minimisation loss does shrink in every room.
- **Guard.** Seven harmful method/dataset pairs on the primary MLP at tau=0.9 (not eight).
  At tau=0.95 the guard makes SignFi ResNet-BN slightly worse (-18.13 -> -18.48 pp) rather
  than removing "at most 6 %", and in low-harm cells it lowers the mean in six of ten (by up
  to 1.2 pp). tau=0.95 was chosen on the reported datasets; tau=0.9 is the pre-sweep default.
- **Guard vs weight reset.** Beyond a 1 pp loss the two are equal on Widar (0.40 each); the
  guard's lower P(gain<0) there comes from five fully rolled-back runs. On NTU-Fi the
  difference holds (0.40 vs 0.87).
- **Widar guard floor tau=0.70** (`outputs/collapse_guard_sweep/ratio_0.7.json`): the stored
  file had one run (room 2, seed 0) trained from a different source model (35.6 vs 34.5 %
  source-only accuracy), produced under CPU load with MKL dynamic threading. Regenerated
  with threading pinned: guarded -4.90 pp [-6.53, -3.24] (was -4.66); the old file is in
  `superseded_2026-10-16/`. Of 45 newly registered guard/reset/core artifacts regenerated
  and compared, the other 44 are identical.
- **Widar reset p=0.5** regenerated with per-run rows (`outputs/reset_transfer_widar/`):
  -0.66 pp, P(gain<0) 0.73 (previous summary-only file: -0.65, 0.73).

### Added

- **SAR, CoTTA and T3A as published** (`sar_full`, `cotta_full`, `t3a_full` in
  `pinn4csi/training/paper2_tta.py`; `outputs/faithful_ports/`, six dataset x backbone
  cells): SAR with E0 = 0.4 ln C filtering, the SAM two-pass update and model recovery on
  normalisation affines; CoTTA with an EMA teacher, augmentation-averaged pseudo-labels
  (gain jitter and noise in place of image augmentations) and stochastic restore; online
  gradient-free T3A with 20 supports per class. At the benchmark's optimiser and budget
  they lose at most 4.3 pp on the primary MLP where the -lite ports lose up to 18.7 pp; none
  clears zero on the primary MLP. SAR matches TENT on SignFi MLP+BN (+10.51 pp).
- **NTU-Fi MLP+BN BN-affine TENT**: +0.67 pp [+0.17, +1.31], a second positive cell.
- **Protocol sensitivity** (`scripts/run_protocol_sensitivity.py`,
  `scripts/summarize_protocol_sensitivity.py` -> `outputs/protocol_sensitivity/`):
  learning rate x step-budget sweep with source-validation selection, and class-blocked
  target streams. Selection by source accuracy picks the gentlest setting and removes most
  harm; no setting, even chosen with target labels, helps on Widar; on SignFi MLP+BN a
  smaller learning rate turns entropy minimisation from -0.89 into +9.78 pp.
- **Guard vs early stopping** (`scripts/analyze_guard_vs_early_stopping.py`): the guard's
  final model equals the unguarded k-step model; it beats a randomly timed stop with the
  same stopping distribution on Widar (p = 0.002) but not on NTU-Fi (p = 0.25).
- `scripts/analyze_collapse_trajectories.py` -> `outputs/collapse_trajectories/`: confusion
  matrices and predicted-class diversity after 0-5 entropy steps; reproduces the benchmark's
  per-cell losses exactly. Figure 5 is now built from real predictions (it previously
  filled off-diagonal cells with a synthetic confidence-gap value).
- `scripts/harm_tolerance_report.py` -> `outputs/statistics/harm_tolerance.json`: harm rate
  beyond one test sample and beyond 1 pp, abstention rate, per-environment means.
- `scripts/reproduce_all_results.py` now registers every collapse-guard and weight-reset
  run (previously none were registered) and the core NTU-Fi/SignFi per-method runs, and its
  run order is derived from the registry: the old hard-coded order listed 18 of the
  components, so a full run and `--only` silently omitted the rest.
- Determinism note: MKL's dynamic threading changes trained models under CPU load; set
  `MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE MKL_NUM_THREADS=6 OMP_NUM_THREADS=6` to reproduce.

## 2026-10-04

### Reproduction audit

Every released artifact was regenerated from a clean checkout with
`scripts/reproduce_all_results.py` and compared field by field with
`scripts/compare_artifacts.py` (report: `outputs/reproduction_audit/report.json`).
Of 89 artifacts, 70 are numerically identical, 18 differ and 1 could not be regenerated
(`final_ablations/superseded_2026-04/mlpbn_other_datasets.json`, which no reported number
uses). Differing artifacts were promoted; the previous copies are under
`superseded_2026-10-04/` next to them. Bootstrap CIs in some summaries are not seeded, so
their bounds can move by up to 0.002 between reruns; point estimates and
negative-adaptation rates do not.

### Corrected

- **SignFi MLP+BN faithful TENT** (`outputs/signfi_mlpbn_canonical/`): +10.3 pp
  [+5.3, +15.2], n=15 (source 67.3% -> 77.6%), replacing +21.1 pp [+13.1, +29.6].
  BN-reset: +2.6 pp [+0.6, +4.7].
- **Widar MLP+BN faithful TENT** (`outputs/widar_mlpbn_canonical/`): -0.42 pp
  [-0.77, -0.09], replacing -0.61 pp. The three-backbone runner
  (`final_ablations/arch_ablation_with_ci.json`, 10 source epochs) gives -0.37 pp
  [-0.75, -0.02] for the same cell.
- **SAR/CoTTA.** Faithful BN-only ports on Widar MLP+BN
  (`new_methods/widar_mlpbn_faithful_sar_cotta.json`): -0.50 / -0.49 pp, replacing
  -0.15 / -0.38. Simplified MLP ports (`new_methods/<dataset>_lame_sar_cotta.json`):
  Widar -6.13 / -6.59, NTU-Fi -18.32 / -18.68, SignFi -11.24 / -12.00.
- **Three-backbone ablation** regenerated at n=15 with 10 source epochs; the one-room
  CNN1D+BN probe's positive gains do not survive (TENT +0.13 pp [-0.32, +0.54]).
- **Overconfidence** (`overconfidence/widar_bvp_per_class.json`): Widar confidence
  69-81%, per-class accuracy 26-58%.
- **Label smoothing** (`source_overfit/mitigation_pilot.json`): TENT negative-adaptation
  rate 0.60 -> 0.33 -> 0.27 for alpha = 0 / 0.1 / 0.2.
- **Supervised upper bound** (`phase2/upper_bound.json`): now 5 seeds (15 room/seed
  rows), oracle 61.0% vs source 36.1%, +24.9 pp. The recorded protocol string is built
  from the run's arguments instead of a fixed description.
- **Selective-threshold sweep** (`final_ablations/selective_threshold_widar_main_protocol.json`):
  all 20 grid points lie between -0.30 and +0.06 pp, replacing a sweep in which every
  point was positive.
- **Step budget** (`db_ablations/step_budget.json`): 3 seeds instead of 1; entropy
  minimisation -1.2 pp at 1 step to -17.8 pp at 10.
- **Synthetic** `selective_physics_tta`: +5.6 pp (negative-adaptation rate 0.20), was 0.0.
- **Cross-dataset Spearman** (`coverage/cross_dataset_spearman.json`): mean rho 0.67,
  was 0.61.

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
