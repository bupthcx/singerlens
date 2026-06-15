# External Bilibili AI-cover case: BV13z4y1b7pL

Source: https://www.bilibili.com/video/BV13z4y1b7pL/

This case is treated as a known AI-cover demo sample, not as training data or aggregate evaluation data.

## Pipeline

1. Downloaded public Bilibili audio with yt-dlp.
2. Converted to WAV.
3. Separated vocals with Demucs (`--two-stems=vocals`).
4. Sliced vocals into 10s windows with 8s hop and voiced-ratio >= 0.5.
5. Ran SingerLens feature extraction and detector scoring on each valid segment.

## Result

- Valid vocal segments: 38
- Mean fake probability: 0.384
- Median fake probability: 0.375
- Min / max fake probability: 0.030 / 0.747
- Share of segments above 0.5 threshold: 28.9%
- Full mixed source fake probability: 0.183
- Full Demucs vocals fake probability: 0.157

## Interpretation

This case is useful for selecting a demo because it is longer than the previous external sample and has enough valid vocal segments to show a time-series analysis. Whether it is suitable as the main demo depends on the share of high-confidence fake segments, not only the whole-track score.

Outputs:

- `outputs/external_bilibili_BV13z4y1b7pL_singerlens_scores.csv`
- `outputs/external_bilibili_BV13z4y1b7pL_alt_scores.csv`
- `outputs/external_bilibili_BV13z4y1b7pL_timeline.png`
