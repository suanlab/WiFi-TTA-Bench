# WiFi-TTA-Bench

A harm-aware benchmark for test-time adaptation (TTA) on WiFi channel state
information. Version 3.0.0 (see `CHANGELOG.md`; Croissant metadata in
`outputs/croissant/`).

## What is evaluated

- **Datasets** (leave-one-environment-out, 3 folds x 5 seeds = 15 paired runs per cell):
  - `widar_bvp`: Widar3.0 body-coordinate velocity profiles, six gestures shared by all
    three rooms, split by room.
  - `ntufi_har`: NTU-Fi HAR, three ordered partitions of one setting.
  - `signfi_top10`: ten SignFi signs, three ordered partitions of one setting.
- **Protocol**: source model trained on two environments; on the held-out one,
  5 Adam steps (lr 5e-4, batch 32) on an adaptation half, scored on a disjoint
  evaluation half. Inputs are standardised per feature with statistics fitted on the
  source-train split only.
- **Reporting**: mean gain over not adapting, run-level bootstrap interval, per-environment
  means and an environment-level t interval, negative-adaptation (harm) rate, harm beyond
  one evaluation sample, abstention rate and source drop
  (`outputs/statistics/harm_tolerance.json`).

## Methods and names

`wifi_tta_bench.list_methods()` returns the registry names. The paper uses fidelity labels
for the simplified ports; the API accepts both:

| Paper name   | Registry name | What it is                                    |
|--------------|---------------|-----------------------------------------------|
| `sar-lite`   | `sar`         | simplified entropy port inspired by SAR       |
| `cotta-lite` | `cotta`       | simplified entropy port inspired by CoTTA     |
| `t3a-ent`    | `t3a`         | entropy variant, not published T3A            |
| `tent-proj`  | `tent`        | TENT on the projection layer of a norm-free MLP |
| SAR / CoTTA / T3A as published | `sar_full`, `cotta_full`, `t3a_full` | the published mechanisms |

On backbones with BatchNorm, `tent` is BN-affine TENT. The physics-prior methods reduce to
feature-moment alignment on the real datasets (no channel prior is available).

## Data

Raw data are not redistributed; scripts never download data.

- **Widar_BVP**: download `BVP/BVP.zip` and `README.pdf` from the Widar3.0 project page
  (Tsinghua Disk link on the official page; registration may be required elsewhere) and run
  `python scripts/prepare_widar_bvp.py --bvp-zip <path>/BVP.zip`. Labels follow the
  dataset's per-date gesture table; `sources.txt` records each sample's upstream file.
- **NTU-Fi, SignFi-10**: the prepared bundles follow the contract in
  `pinn4csi/data/paper1.py` (`PAPER1_DATASET_REGISTRY`). Their preparation scripts are not
  part of this release; see the paper's Limitations.

Sample-level split manifests (indices into the prepared arrays, keyed by array SHA-256) are
in `outputs/splits/`.

## Quick start

```python
from wifi_tta_bench import load_dataset, evaluate

dataset = load_dataset("widar_bvp", prepared_root="data/prepared")
result = evaluate("tent", dataset, seeds=5)
print(result.mean_gain, result.ci_lower, result.ci_upper, result.negative_adaptation_rate)
```

## Reproducing the paper

```bash
export MKL_DYNAMIC=FALSE OMP_DYNAMIC=FALSE MKL_NUM_THREADS=6 OMP_NUM_THREADS=6
python scripts/reproduce_all_results.py --only <component>   # one artifact
python scripts/reproduce_all_results.py --force              # everything (about a day on CPU)
python scripts/validate_croissant.py                         # metadata hashes
pytest --tb=short -q                                         # test suite
```

`SOURCE_SCRIPTS` in `scripts/reproduce_all_results.py` maps every artifact under
`outputs/` to the command that regenerates it.
