# External Bilibili AI-cover case: BV1pQ4y1G7gV

Source: https://www.bilibili.com/video/BV1pQ4y1G7gV/

This case is treated as a known AI-cover demo sample, not as training data or aggregate evaluation data.

## Pipeline

1. Downloaded public Bilibili audio with yt-dlp.
2. Converted to WAV.
3. Separated vocals with Demucs (`--two-stems=vocals`).
4. Sliced vocals into 10s windows with 8s hop and voiced-ratio >= 0.5.
5. Ran SingerLens feature extraction and detector scoring on each valid segment.

## Result

- Valid vocal segments: 21
- Mean fake probability: 0.465
- Median fake probability: 0.463
- Min / max fake probability: 0.090 / 0.773
- Share of segments above 0.5 threshold: 47.6%

## Interpretation

This is a useful demo case because it comes from a real video platform and is already known as AI-generated. It should be described as an external case study. It should not be used as proof of overall accuracy because the sample is a single public video and carries platform compression, accompaniment mixing, and vocal-separation artifacts.

Outputs:

- `outputs/external_bilibili_BV1pQ4y1G7gV_singerlens_scores.csv`
- `outputs/external_bilibili_BV1pQ4y1G7gV_timeline.png`
