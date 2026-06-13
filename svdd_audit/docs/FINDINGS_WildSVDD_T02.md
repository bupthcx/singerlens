# WildSVDD-bilibili T02-only sanity check

## Data source

- Source: reachable bilibili subset from the official WildSVDD , enriched with SingFake metadata, URL, download status, Demucs status, and feature coverage.
- T02-only rule:  and URL contains .
- Rows with missing  were excluded from T02-only because their SingFake split cannot be verified. This removes 30 NaN rows from the bilibili-all metadata table.

## Sample statistics

| split | raw_songs | raw_real | raw_fake | download_success | demucs_success | feature_songs | feature_clips | feature_real_clips | feature_fake_clips |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| bili_all | 192 | 81 | 111 | 165 | 165 | 164 | 466 | 229 | 237 |
| bili_t02 | 162 | 80 | 82 | 136 | 136 | 135 | 387 | 226 | 161 |

## Model comparison

| subset | model | protocol | EER | AUC | F1 | n_clips | n_songs |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |
| WildSVDD-bili-T02 | AASIST | CtrSVDD->Wild | 32.32 | 0.698 | 0.1 | 136 |  |
| WildSVDD-bili-all | AASIST | CtrSVDD->Wild | 32.12 | 0.748 | 0.194 | 165 |  |
| WildSVDD-bili-all | AASIST | SingerLens->Wild | 54.54 | 0.43 | 0.176 | 165 |  |
| WildSVDD-bili-T02 | RF_FULL | CtrSVDD->Wild | 47.28 | 0.534 | 0.526 | 387 | 135 |
| WildSVDD-bili-all | RF_FULL | CtrSVDD->Wild | 48.93 | 0.52 | 0.586 | 466 | 164 |
| WildSVDD-bili-T02 | RF_FULL | SingerLens->Wild | 50.91 | 0.499 | 0 | 387 | 135 |
| WildSVDD-bili-all | RF_FULL | SingerLens->Wild | 53.65 | 0.461 | 0 | 466 | 164 |
| WildSVDD-bili-T02 | RF_FULL | Wild-domain CV | 14.22 | 0.923 | 0.797 | 387 | 135 |
| WildSVDD-bili-all | RF_FULL | Wild-domain CV | 14.81 | 0.935 | 0.859 | 466 | 164 |
| WildSVDD-bili-T02 | WavLM_RF | CtrSVDD->Wild | 52.14 | 0.455 | 0.454 | 136 | 136 |
| WildSVDD-bili-all | WavLM_RF | CtrSVDD->Wild | 50.9 | 0.499 | 0.562 | 165 | 165 |
| WildSVDD-bili-T02 | WavLM_RF | SingerLens->Wild | 48.48 | 0.583 | 0.472 | 136 | 136 |
| WildSVDD-bili-all | WavLM_RF | SingerLens->Wild | 49.1 | 0.569 | 0.509 | 165 | 165 |
| WildSVDD-bili-T02 | WavLM_RF | Wild-domain CV | 46.34 | 0.587 | 0.368 | 136 | 136 |
| WildSVDD-bili-all | WavLM_RF | Wild-domain CV | 50.31 | 0.581 | 0.526 | 165 | 165 |

## Key observations

- RF_FULL Wild-domain CV is stable: bilibili-all EER=14.81, AUC=0.935; T02-only EER=14.22, AUC=0.923.
- RF_FULL CtrSVDD->Wild remains near random under both splits: bilibili-all EER=48.93, AUC=0.520; T02-only EER=47.28, AUC=0.534.
- RF_FULL SingerLens->Wild shows the same degradation pattern: bilibili-all EER=53.65, AUC=0.461; T02-only EER=50.91, AUC=0.499.
- AASIST CtrSVDD->Wild is also consistent: bilibili-all EER=32.12, AUC=0.748; T02-only EER=32.32, AUC=0.698. No persisted AASIST SingerLens->T02 or T02 in-domain output was found.
- WavLM_RF does not overturn the conclusion: CtrSVDD->Wild is bilibili-all EER=50.90, AUC=0.499; T02-only EER=52.14, AUC=0.455.

## Conclusion

- The all-bilibili and T02-only subsets show the same core trend: in-domain Wild validation is feasible, while cross-dataset transfer from CtrSVDD or SingerLens degrades strongly.
- Therefore, the 30 bilibili rows with missing  do not dominate the WildSVDD conclusion.

## Limitation

- T02-only is a reachable WildSVDD/SingFake bilibili subset after download, Demucs, and feature-extraction filtering. It is not equivalent to the complete SingFake benchmark.

## Output files

- 
- 
- 
- 
