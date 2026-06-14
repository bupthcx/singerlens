# FINDINGS: U-MSA second-target replication on CtrSVDD

## Question

The original U-MSA result was obtained on WildSVDD-T02. A reviewer could ask whether the label-efficiency gain only holds for that target domain. We therefore reran the fixed-test active-learning MSA protocol on a second target domain: CtrSVDD E1.

## Protocol

- Target domain: CtrSVDD E1 feature table (`outputs/ctrsvdd_features_e1.csv`)
- Samples: 4,800 clips, balanced real/fake = 2,400 / 2,400
- Feature set: FULL, 48 available features
- Split: 30% fixed stratified test set per trial; active learning can query labels only from the remaining 70% pool
- Trials: 10
- Strategies:
  - `random_balanced`
  - `uncertainty`
  - `kcenter`
- Label budgets: k = 6, 10, 20, 30, 50, 75, 100, 150, 200

Output files:

- `results/d3_secondtarget_ctrsvdd_fixedtest.csv`
- `results/d3_secondtarget_ctrsvdd_fixedtest.png`

## Main result

Uncertainty sampling transfers in direction, but the gain is milder than on WildSVDD-T02.

| k | random | uncertainty | kcenter |
|---:|---:|---:|---:|
| 6 | 0.585 | 0.562 | 0.562 |
| 10 | 0.581 | 0.605 | 0.599 |
| 20 | 0.612 | 0.634 | 0.650 |
| 30 | 0.643 | 0.665 | 0.646 |
| 50 | 0.644 | 0.688 | 0.643 |
| 75 | 0.707 | 0.720 | 0.668 |
| 100 | 0.734 | 0.749 | 0.691 |
| 150 | 0.759 | 0.768 | 0.724 |
| 200 | 0.778 | 0.785 | 0.747 |

## Interpretation

This is a positive but nuanced replication.

- The direction is mostly consistent: uncertainty beats random from k=10 onward, with the clearest gap around k=50 (+0.044 AUC).
- The advantage shrinks at larger budgets: by k=200, uncertainty is only +0.007 AUC over random.
- Unlike WildSVDD-T02, this CtrSVDD run does not show a clear milestone saving such as "reach 0.90 with 25% fewer labels"; neither random nor uncertainty reaches 0.80 within k<=200.
- k-center remains weaker than uncertainty at medium/high budgets, supporting the earlier finding that pure diversity is not the useful signal.

## How to write it

Use this result to soften, not inflate, the U-MSA claim:

> A second-target replication on CtrSVDD preserves the direction of the U-MSA effect but shows a smaller gain: uncertainty sampling improves AUC over random selection at most budgets, especially around k=50, but does not produce the large milestone savings observed on WildSVDD-T02. This suggests that U-MSA is a practical label-efficient exit, while its magnitude is target-dependent.

## Caveats

- This is a fast 10-trial replication, not the heavier 20-trial setting used for the original WildSVDD-T02 result.
- CtrSVDD is cleaner and harder for the FULL feature set under small target-label budgets, so the absolute AUC remains below the full-data in-domain baseline.
- The result supports external validity in direction, but not a universal fixed percentage of label saving.

