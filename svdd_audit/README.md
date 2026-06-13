# SVDD-Audit

**Auditing generation-family confounds in Singing Voice Deepfake Detection.**

A protocol suite + scripts to measure how much SVDD performance reflects *which generation/vocoder family produced a clip* rather than *whether it is an AI cover*. We find that detectors with near-perfect in-distribution scores collapse to near-chance under unseen vocoder/generator families and across datasets — and that this is **model-agnostic**.

> **Headline:** A detector reaching AUC ≈ 0.99 within a dataset drops to AUC ≈ 0.45–0.60 (chance) when transferred to another dataset / real wild AI covers. Holds for hand-crafted features (RF), a SOTA neural model (AASIST), and a frozen SSL embedding (wav2vec2).

---

## 1. Audit protocols
All protocols keep the bonafide split disjoint between train and test. Primary metric: **EER** (official scoring), plus AUC/F1 and **degradation ratio = shifted-EER / in-dist-EER**.

| ID | Protocol | What it isolates | Script |
|----|----------|------------------|--------|
| P0 | Standard (random/seen split, 5-fold CV) | optimistic baseline | `protocols/honest_eval.py`, `protocols/ctrsvdd_eer_analysis.py` |
| C  | Copy-synthesis control | generic "vocodedness" vs pipeline-specific artifacts | `generation/ctrsvdd_gen_bonafide_vocoded.py` + `protocols/ctrsvdd_e2_analysis.py` |
| P1 | Leave-One-Vocoder-Out (LOVO) | unseen vocoder family | `protocols/ctrsvdd_lovo.py` (RF), `models/train_aasist_lovo.py` (AASIST) |
| P2 | Leave-One-Generator-Out (LOGO) | unseen generation paradigm (SVS/SVC) | `protocols/ctrsvdd_lovo.py` adapted by `mapping/attack_map.csv` `type` |
| P3 | Leave-One-Singer-Out | singer-identity confound | `protocols/cross_singer_eval.py` |
| P3b| Leave-One-Song-Out | song-identity confound | `protocols/cross_song_eval.py` |
| P4 | Cross-dataset transfer | dataset-level distribution shift | `protocols/cross_dataset_transfer.py`, `protocols/cross_dataset_wild.py` |

Diagnostics: `protocols/attribution_analysis.py` (Cohen's d + η² label-vs-singer variance), `protocols/vocoder_control_eval.py`, mitigation `protocols/per_singer_norm_eval.py`.

## 2. attack → generator → vocoder mapping
Canonical mapping (`mapping/attack_map.csv`, derived from generation repos — see `mapping/REF_ctrsvdd_system_vocoder.md`). For CtrSVDD train/dev systems:

| vocoder family | systems |
|----------------|---------|
| HiFi-GAN | A01, A02, A05 |
| NSF / source-filter HiFi-GAN | A04, A06, A07, A08 |
| DDSP | A03 |

Generation paradigm (`type`): SVS = A01–A05; SVC = A06–A08.

## 3. Models (feature/model families)
- **RF on interpretable features** (`extract/features.py`): FULL / CLEAN / HNR / VRI.
- **AASIST** (raw-waveform graph attention; `models/train_aasist_lovo.py`, supports `--extra-tests` for cross-dataset).
- **wav2vec2 frozen SSL embedding** (`extract/wavlm_extract.py` → `protocols/wavlm_analysis.py`). *Note: frozen, not fine-tuned — a weak baseline; for a strong SSL system, fine-tune the frontend.*

## 4. Datasets
- **CtrSVDD** (controlled, 16 kHz, clean vocals; train/dev systems A01–A08) — Zenodo 10467648.
- **WildSVDD** (in-the-wild real internet AI covers; we use the **bilibili-reachable subset** of the URL annotations, Zenodo 10893604; YouTube URLs unreachable from CN). 165 songs, Demucs-separated vocals.
- **Self-collected** (SingerLens): bilibili real + Seed-VC fakes; carries a sampling-rate/production confound (illustrative).

## 5. Reproduce (high level)
```bash
# Features (CtrSVDD subset, a dir of clips, WildSVDD)
python extract/ctrsvdd_subset_extract.py --meta ctrsvdd_meta.csv --base <ctrsvdd> --out feats.csv
# Standard + grouped EER
python protocols/ctrsvdd_eer_analysis.py --features feats.csv
# LOVO (RF) / LOGO via attack_map type
python protocols/ctrsvdd_lovo.py --features feats.csv
# AASIST LOVO / cross-dataset
python models/train_aasist_lovo.py --train-csv A.csv --test-csv A_test.csv --extra-tests wild=W.csv --tag run
# Copy-synthesis control
python generation/ctrsvdd_gen_bonafide_vocoded.py --e1-features feats.csv --base <ctrsvdd> --ckpt <nsf_hifigan/model> --outdir bonavoc
python protocols/ctrsvdd_e2_analysis.py --e1-features feats.csv --voc-features bonavoc_feats.csv
# Cross-dataset
python protocols/cross_dataset_transfer.py        # CtrSVDD <-> SingerLens
python protocols/cross_dataset_wild.py            # -> WildSVDD
# SSL baseline
python extract/wavlm_extract.py --csv in.csv --out emb.csv ; python protocols/wavlm_analysis.py
```

## 6. Key results (see `results/svdd_audit_summary.csv` and `docs/`)
- **Cross-dataset collapse (model-agnostic):** within-dataset AUC 0.71–1.00 → cross-dataset 0.41–0.62.
- **LOVO/LOGO degradation:** EER ×1.1–3.0 on unseen vocoder/generation family; AASIST degrades up to ×3.0 (NSF-HiFiGAN, LOGO SVS).
- **Copy-synthesis:** detectors are *not* generic vocoder detectors (bonafide-vocoded not flagged), but key on pipeline-specific artifacts.
- **Interpretable HNR/VRI:** at chance on CtrSVDD and cross-dataset (their usefulness on web data is dataset-specific).

`results/` holds one CSV per protocol; `docs/` holds the paper skeleton (`PAPER_SVDD_Audit_skeleton.md`), the CtrSVDD experiments draft, and all FINDINGS notes.

## 7. Limitations
- WildSVDD limited to its bilibili-reachable subset (YouTube blocked from CN); songs carry background music (Demucs-separated → separation artifacts are part of the wild domain).
- wav2vec2 baseline is frozen (not fine-tuned) — a weak SSL baseline; a fine-tuned WavLM/XLSR frontend would be stronger.
- AASIST WildSVDD in-domain reference underfits (82 songs); use RF in-domain (AUC 0.935) as the "wild is solvable" reference.
- Subset sizes chosen for tractable analysis, not leaderboard scores.

## 8. Dependencies
See `requirements.txt`. External: CtrSVDD `eer.py` (official EER), CtrSVDD2024_Baseline (AASIST), openvpi NSF-HiFiGAN checkpoint (`nsf_hifigan_20221211.zip`, sha256 d86ea84b…), Seed-VC (for self-collected fake generation only).
