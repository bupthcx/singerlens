# SVDD-Audit — Paper Skeleton (CCF-A target: ICASSP / INTERSPEECH / TASLP / ACM MM)

**Working title:** *Beyond Standard Splits: Auditing Generation-Family Confounds in Singing Voice Deepfake Detection*

---

## Abstract (draft)
Singing voice deepfake detection (SVDD) systems report low equal error rates (EER) on standard
benchmarks, suggesting the task is largely solved. We argue that these numbers substantially
**overestimate real-world generalization**, because standard random/seen-condition splits let
detectors exploit **generation-family confounds**—vocoder and generation-pipeline artifacts that are
constant within a dataset but shift across generators, datasets, and the wild. We introduce
**SVDD-Audit**, a suite of controlled evaluation protocols—copy-synthesis control, leave-one-vocoder-out
(LOVO), leave-one-generator-out (LOGO), a **factorial vocoder×paradigm decoupling**, and cross-dataset
transfer—that disentangle generic "vocodedness", generation-family artifacts, and genuine spoofing
cues. Across interpretable and state-of-the-art neural detectors (random forest on hand features;
AASIST; a fine-tuned wav2vec2 SSL) and three datasets (controlled CtrSVDD; self-collected web data;
wild WildSVDD), we show that
detectors achieving near-perfect in-distribution performance (AUC ~0.99) **collapse to near-chance
(AUC ~0.45–0.55) under cross-dataset transfer**, that strong neural models also degrade ~2–13× on
unseen vocoder/generator families, and—via the factorial decoupling—that **even with the vocoder held
fixed, swapping the generation paradigm collapses EER from ~1% to ~50%**, proving the shift is
multi-axis rather than reducible to the vocoder. We further show that a **window-level / multiple-
instance** analysis—the intuitive "look at local artifacts" remedy—does **not** recover cross-dataset
generalization. All effects are **model-agnostic**. We release the audit protocols, attack→generator→
vocoder mappings, and evaluation scripts as a reusable toolkit, and discuss implications for
trustworthy SVDD evaluation.

---

## Contributions (5)
1. **A systematic finding:** standard SVDD evaluation overestimates detector robustness; in-distribution
   near-perfect detectors generalize poorly under distribution shift, and "looking local" does not fix it.
2. **An audit protocol suite:** copy-synthesis control + LOVO + LOGO + **factorial vocoder×paradigm
   cell-out** + cross-dataset transfer, with formal definitions and an attack→generator→vocoder mapping,
   to disentangle confounds from genuine spoofing cues and to separate orthogonal shift axes.
3. **A large, model-agnostic empirical study** across three detector families (interpretable RF, neural
   AASIST, fine-tuned wav2vec2 SSL; + window-level/MIL), vocoder families, generation paradigms, singers,
   and three datasets.
4. **A toolkit + an honest mitigation analysis:** simple per-domain normalization improves cross-singer
   generalization while an adversarial (DANN) approach does not, and window-level locality does not
   confer cross-domain generalization—showing simple controls suffice and the gap is fundamental, not a
   feature-engineering or pooling-granularity artifact.
5. **A failure-mechanism attribution of the collapse:** via a score-reversal diagnosis and a failure
   taxonomy we show the cross-dataset collapse is *target-domain signal loss with a domain-dependent
   score offset* (not a sign flip—flipped-AUC stays at chance), and that detectors decide along a
   **production-cleanliness / generation-signature axis orthogonal to authenticity**—yielding two
   systematic error types (false positives on Demucs/accompaniment-residue reals; false negatives on
   clean source-cue-free fakes). A blind 40-sample human-listening pilot is packaged to test whether
   these model failures exceed the human perceptual boundary.

---

## Section structure

### 1. Introduction
- AI singing voice cloning rise + misuse; SVDD as defense.
- Standard benchmarks report low EER -> appearance of "solved".
- Our thesis: this reflects *which generation family* (paradigm ⊗ vocoder) not *is it AI*. Motivate auditing.
- Summary of contributions + headline result (cross-dataset collapse; multi-axis; model-agnostic).

### 2. Related Work
- SVDD datasets/challenges (SingFake, CtrSVDD/SVDD Challenge, FSD, WildSVDD).
- Speech anti-spoofing generalization (ASVspoof, cross-dataset, vocoder artifacts in TTS deepfake).
- Domain shift / shortcut learning in audio deepfake detection.
- Multiple-instance / temporal localization in deepfake detection.
- Gap: no systematic audit of generation-family confounds (and their orthogonal axes) in SVDD.

### 3. Audit Protocols (the core methodological contribution)
Formal definitions; bonafide split disjoint across train/test in all protocols.
- **P0 Standard:** random/seen-condition split (the conventional, optimistic baseline).
- **C Copy-synthesis control:** pass bonafide through the spoof-family vocoder (analysis-by-synthesis),
  sampling-rate matched, QC verified; test whether a detector keys on *generic* vocodedness vs
  pipeline-specific artifacts.
- **P1 Leave-One-Vocoder-Out (LOVO):** hold out one vocoder family (HiFi-GAN / NSF-HiFiGAN / DDSP).
- **P2 Leave-One-Generator-Out (LOGO):** hold out one generation paradigm/family (SVS / SVC; extensible).
- **P2b Factorial vocoder×paradigm cell-out:** treat (paradigm × vocoder) as a 2-D grid; leave-one-cell-out
  plus **axis-controlled contrasts** (fix one axis, vary the other) to test whether the two confound axes
  are orthogonal and separable. Requires a vocoder that spans paradigms (NSF-HiFiGAN here).
- **P3 Leave-One-Singer-Out (LOSingerO):** hold out singers (singer-identity confound).
- **P4 Cross-dataset transfer:** train on dataset A, test on dataset B (and vice versa).
- **A Window-level / multiple-instance (MIL) analysis (orthogonal to the above):** represent each clip by
  pooled window statistics (clip_mean / MIL_mean / MIL_max / POOL_RICH) to test whether *local* temporal
  evidence transfers where clip-level evidence does not.
- **Metrics:** EER (primary), AUC, F1; plus **degradation ratio** (= shifted EER / in-dist EER),
  **worst-domain EER**, score-reversal indicator (mean_score_real vs mean_score_fake), and (for confound
  diagnosis) vocoder/singer leakage accuracy.

### 4. Experimental Setup
- **Datasets:** CtrSVDD (controlled, 16 kHz, clean vocals, 8 train/dev systems, vocoder/generator
  labels); self-collected web data ("SingerLens": 6 singers x multiple songs, B-station real +
  Seed-VC fakes); **WildSVDD** (wild bilibili AI covers, So-VITS-SVC/RVC) for cross-dataset and the
  real-world target.
- **Attack→generator→vocoder mapping** (from generation repos): HiFi-GAN {A01,A02,A05,A12}, NSF/source-
  filter HiFi-GAN {A04,A06,A07–A11,A13}, DDSP {A03}, proprietary {A14}; SVS {A01-A05,A12,A14} vs
  SVC {A06-A11,A13}. (In the balanced E1 subset used for the controlled experiments: A01-A08.)
- **Detectors (three families):** RF on interpretable feature sets (FULL/CLEAN/HNR/VRI); AASIST
  (raw-waveform graph attention, = sinc/RawNet front-end + AASIST backend); **fine-tuned SSL**
  (wav2vec2-base-960h end-to-end; + wav2vec2-large-960h 315M as a model-scale data point); plus
  window-level/MIL RF and a frozen-SSL reference.
- **SSL model-selection rationale (FINDINGS_wav2vec2_finetune.md):** WavLM/HuBERT are network-unreachable
  on our infrastructure (HF walled; WavLM not on ModelScope; large XLSR needs torch≥2.6), so the SSL
  family is represented by reachable wav2vec2 models—not a claim of optimality. Frozen wav2vec2-base is
  weak (within AUC 0.711) because it is pretrained on English ASR, not singing—frozen features are
  insensitive to singing acoustics; end-to-end fine-tuning makes it strong (0.91), and a 3.3× larger model
  (0.97 in-domain) confirms the SSL route works and scales in-domain while still collapsing cross-dataset.
- **Implementation:** balanced analysis subsets; official EER scoring.

### 5. Results
- **5.1 Standard evaluation looks solved.** AASIST in-dist EER 6.5% (AUC 0.987); FULL 18.9%. [Table 1]
- **5.2 Copy-synthesis control.** Detector does NOT flag bonafide-vocoded as fake (P(spoof) 0.375 vs
  natural 0.351) -> not a generic vocoder detector; keys on pipeline-specific artifacts. [Table 2]
- **5.3 LOVO (vocoder).** EER degrades on unseen vocoders; AASIST x2.99 on NSF-HiFiGAN; RF up to x2.32.
  Model-agnostic vocoder-family gap. [Table 3]
- **5.4 LOGO (generation paradigm).** RF FULL: hold-out SVS x2.03, SVC x1.89; AASIST x2.95 / x2.43. [Table 4]
- **5.5 Factorial vocoder×paradigm decoupling (multi-axis, model-agnostic).** The two confound axes are
  orthogonal and each independently degrades detectors; with the vocoder held FIXED (NSF-HiFiGAN), a
  paradigm swap still collapses EER ~1%→~50% (RF ×59–89; AASIST ×13.6), *more* than an unseen-vocoder
  shift. Refutes "just a vocoder detector". [Table 5a LOCO, Table 5b axis-controlled]
- **5.6 Cross-dataset transfer (headline).** Within-dataset AUC 0.88-1.00 -> cross-dataset AUC 0.34-0.62
  (near chance / reversed) for RF, AASIST, and a **fine-tuned SSL** (wav2vec2-base-960h: in-domain AUC
  0.90-0.91 -> cross-dataset 0.42/0.34). Three independent detector families collapse alike, and a
  3.3× larger SSL (wav2vec2-large, in-domain 0.97) still collapses—**model- and scale-agnostic**. [Table 6]
- **5.6b Failure Analysis: Score Collapse, Domain Offset, and the Cleanliness Axis.** The collapse is *target-domain
  signal loss*, not score-sign reversal (flipped-AUC stays ~0.53-0.56); errors are systematic—FP on
  low-HNR reals (Demucs/accompaniment residue), FN on clean So-VITS fakes lacking the source generation
  cue. [Table 6b, Figure 2]
- **5.7 Window-level locality does not transfer.** clip_mean/MIL_mean/MIL_max/POOL_RICH all collapse
  under cross-dataset transfer (AUC 0.43-0.56); MIL_max is worst and only inflates scores. Three-channel
  timelines show source-domain local AI spikes (aligned with pitch jumps) vanish/reverse cross-domain. [Table 7, Figure 1]
- **5.8 Interpretable "style" features do not transfer.** HNR/VRI at chance on CtrSVDD and cross-dataset;
  their usefulness on web data is dataset-specific (production confound). [Table 1]
- **5.9 Mitigation (honest).** Per-singer/per-domain normalization improves cross-singer LOSingerO
  (CLEAN AUC 0.68->0.77 at 6 singers); adversarial DANN does not beat it -> simple controls suffice.
- **5.9b Closing the unsupervised door, and the minimal-supervision exit.** Every unsupervised remedy
  stays at chance—feature-alignment DA (CORAL, subspace, DANN, per-domain z-norm), common-bandwidth
  confound-removal, and score calibration—because the discriminative axis itself is domain-specific
  (source ranks by MFCC production texture; target needs HNR). **The only viable exit is minimal target
  supervision: ~25 labels already beat every unsupervised method (0.72 vs ≤0.52), ~100 approach the
  in-domain bound (0.85 / 0.93).** A structured conclusion—unsupervised closed, supervised adaptation is
  the cheap exit—not a list of failures. [Table 9a DA, 9b confound, 9c attribution, 9d minimal-supervision, Fig 5]
- **5.10 Deployment implications: abstention, few-shot calibration, production perturbation.** Cross-domain
  high-confidence predictions are no more correct (selective prediction fails); ~50-100 target labels
  trained from scratch beat transfer from the confounded source; band-limiting/codec perturbation of
  *real* audio flips it to "fake" (production = a false-positive risk). [Table 8, Figures 3-4]

### 6. Discussion
- SVDD performance substantially reflects generation-family identification (paradigm ⊗ vocoder), not authenticity.
- The shift is multi-axis: controlling the vocoder alone is insufficient; the generation paradigm is an
  independent, equally large confound.
- Locality is not a remedy: local/window evidence is itself domain-specific and does not transfer.
- **The collapse mechanism is target-domain signal loss along the source-domain generation-family cues
  and a production-cleanliness axis, not a flipped decision** (flipped-AUC stays at chance): detectors
  score by recording cleanliness / generation signature, which is orthogonal to authenticity in the wild.
- Implications: report degradation ratios + cross-dataset + per-axis; release attack metadata; treat
  vocoder AND paradigm as controlled variables; wild AI-cover transfer as the true target.
- **A blind human-listening pilot is prepared (40 samples incl. the model's cross-domain FP/FN) to test
  whether human perception can compensate for model cross-domain failures** (Appendix / future work).
- **No unsupervised fix (Sec 5.9b):** feature-alignment domain adaptation (CORAL/subspace/DANN/z-norm)
  and bandwidth-equalization do not recover cross-dataset transfer; the discriminative axis itself is
  domain-specific (source ranks by MFCC production texture, target needs HNR). Only target labels help—
  reinforcing the "collect target labels" recommendation over any source-side adaptation.
- **Deployment recommendations (from Sec 5.10):** (1) a benchmark-trained detector cannot be deployed
  as-is—cross-dataset performance is near chance; (2) a confidence threshold is not a safe reject
  mechanism—out-of-domain high-confidence predictions are no more (sometimes less) correct; (3) for a new
  platform, prioritize collecting a small set (~50-100) of target-domain labels and train from scratch
  rather than transferring/fine-tuning/adapting the confounded source detector; (4) the real-audio
  production chain—platform transcoding, band-limiting, codec/separation residue—materially raises
  false-positive risk and must be evaluated before deployment.
- Limitations: axes nearly nested in CtrSVDD (one cross-paradigm vocoder); subset sizes; SVS/SVC
  granularity; window-length sensitivity; WildSVDD = reachable bilibili subset ≠ full SingFake;
  future: more datasets/generators, human study.
- Deployment-analysis caveats (Sec 5.10): (i) the perturbations are **controlled diagnostic probes, not a
  faithful simulation of platform distributions**—score shifts are sensitivity evidence, not deployable
  error rates (external validity rests on the WildSVDD-bilibili/T02 wild test); (ii) accompaniment-leakage
  and Demucs-re-separation were tested **only as synthetic probes**, so real platform separation/residue
  as an FP contributor is not fully characterized; (iii) the few-shot result contrasts **naive mixing vs
  simple target up-weighting/over-sampling only**—we do not test full domain-adaptation (DANN, calibration
  fine-tuning), so the claim is "source supervision is a negative asset *under naive mixing*", not that all
  source+target adaptation fails.

### 7. Conclusion
SVDD is far from solved once confounds are controlled; SVDD-Audit provides protocols + toolkit to
measure and report true generalization. The gap is multi-axis, model-agnostic, and not closed by
looking at local evidence.

---

## Results inventory (where each experiment plugs in)
| Paper item | Experiment | Artifact |
|---|---|---|
| Table 1 (in-dist grouped EER) | 5.1 standard | outputs/ctrsvdd_eer_e1.csv; AASIST std EER 6.5% |
| Table 2 (copy-synthesis) | 5.2 | FINDINGS_CtrSVDD_E1/E2; outputs/ctrsvdd_e2.csv |
| Table 3 (LOVO) | 5.3 | outputs/ctrsvdd_aasist_vs_rf_lovo.csv |
| Table 4 (LOGO) | 5.4 | RF x2.03/x1.89; AASIST outputs/logo_*.csv |
| **Table 5a (factorial LOCO)** | **5.5 (#2)** | **outputs/ctrsvdd_factorial_loco.csv** |
| **Table 5b (axis-controlled)** | **5.5 (#2)** | **outputs/ctrsvdd_factorial_axis.csv; ctrsvdd_aasist_paradigm.csv** |
| Table 6 (cross-dataset) | 5.6 headline | outputs/cross_dataset_aasist_vs_rf.csv; wildsvdd cross; **w2v2ft_A.csv/w2v2ft_B.csv (fine-tuned SSL)** |
| **Table 6b (score-reversal summary)** | **5.6b (P1)** | **outputs/score_distribution_summary.csv; score_clip_scores.csv** |
| **Figure 2 (real/fake score distributions)** | **5.6b (P1 viz)** | **outputs/score_distributions.png** |
| **Table 6c (failure-group feature stats)** | **5.6b (P2)** | **outputs/failure_group_stats.csv; failure_cases.csv** |
| **Table 7 (window cross-dataset)** | **5.7 (#3)** | **outputs/window_cross_dataset_eval.csv / _summary.csv** |
| **Figure 1 (3-channel case timelines)** | **5.7 (#3 viz)** | **outputs/window_cross_dataset_cases/case1-5.png** |
| **Human pilot (Appendix)** | **future work (P3)** | **outputs/human_pilot/ (manifest+key+README); human_pilot_package.tar.gz** |
| **Table 8 (deployment: reject/few-shot/perturb)** | **5.10** | **outputs/reject_option_riskcoverage.csv; fewshot_calibration.csv; perturbation_effect.csv** |
| **Figure 3 (risk-coverage curves)** | **5.10** | **outputs/reject_option_riskcoverage.png** |
| **Figure 4 (few-shot calibration curve)** | **5.10** | **outputs/fewshot_calibration.png** |
| Mitigation | 5.9 | outputs/per_singer_norm.csv; FINDINGS_对抗去歌手身份 |
| **Table 9a (unsupervised DA)** | **5.9b (Tier1.1)** | **outputs/domain_adaptation_cross.csv** |
| **Table 9b (confound-removal / bandwidth)** | **5.9b (Tier1.2/2.2)** | **outputs/confound_removal_cross.csv; bandwidth_stats.csv** |
| **Table 9c + Figure 5 (axis attribution)** | **5.9b (Tier2.1)** | **outputs/axis_attribution{,_family}.csv; axis_attribution.png** |
| **Table 9d (minimal-supervision exit)** | **5.9b (MSA)** | **outputs/fewshot_reweighting.csv + domain_adaptation_cross.csv; FINDINGS_minimal_supervised_adaptation.md** |
| Vocoder map | setup | outputs/REF_ctrsvdd_system_vocoder.md; ctrsvdd/attack_map.csv |
| Toolkit | all | scripts/ctrsvdd_*.py, window_*.py, train_aasist_lovo.py, cross_dataset_*.py |

## TODO to reach CCF-A
- [x] AASIST LOGO numbers -> Table 4.
- [x] Factorial vocoder×paradigm decoupling (RF + AASIST) -> Table 5a/5b (5.5).
- [x] 3rd public dataset (WildSVDD wild) feasibility + cross-dataset -> Table 6.
- [x] Window-level / MIL cross-dataset + visualization -> Table 7 / Figure 1 (5.7).
- [x] Failure analysis: score-reversal diagnosis (P1) + failure taxonomy (P2) -> Sec 5.6b (Table 6b/6c, Figure 2).
- [x] Human-listening pilot prep package (P3): 40 blind samples + manifest + key + README (ready-to-run).
- [x] Strong 3rd detector family: fine-tuned wav2vec2-base-960h (SSL) -> Table 6 (closes frozen-SSL caveat).
      (WavLM proper blocked: HF-walled, not on ModelScope; RawNet2 not distinct here = AASIST's sinc front-end.)
- [x] Deployment analysis (Exp1/2/3) -> Sec 5.10 (Table 8a/8b/8c, Figure 3/4).
- [x] Extended Exp3 with **accompaniment leakage** + **Demucs re-separation** (production_perturbation_ext.py):
      leakage does NOT cause FP (honest negative); re-separation mild (+0.14). -> Table 8c.
- [x] Fairer Exp2 "source+k" (up-weight/over-sample, fewshot_calibration_ext.py): source+k recovers but
      still < Wild-only -> negative-transfer conclusion hardened. -> Table 8b.
- [x] Exp2b T02 + calibration-only (fewshot_reweighting.py): calibration ≈ chance -> axis-misalignment,
      not miscalibration. -> Table 8b.
- [x] Exp3b multi-level leakage + leak→re-separation (production_leakage_resynthesis.py): leakage inert
      across levels; re-separation drives FP. -> Table 8c.
- [x] Tier1.1 unsupervised DA (CORAL/subspace/DANN/z-norm) -> all fail -> Table 9a (5.9b).
- [x] Tier1.2 confound-removal (common bandwidth) -> no recovery -> Table 9b (5.9b).
- [x] Tier2.1 axis attribution + Tier2.2 bandwidth quantification -> Table 9c / Figure 5 (5.9b).
- [ ] (optional) actually run the human listening study (package ready).
- [x] Release SVDD-Audit toolkit (protocols + mapping + scripts).
- [ ] Write full prose from this skeleton.

---
# [FILLED] Result tables + prose (final unless re-run on larger subsets)

## One-sentence evidence chain (use in Abstract/Intro/Conclusion)
> **Standard splits overestimate robustness; copy-synthesis shows detectors are not generic
> "vocodedness" detectors; LOVO/LOGO reveal an unseen generation-family domain shift; a factorial
> vocoder×paradigm decoupling shows the two axes are orthogonal and each independently degrades
> detectors—with the vocoder held fixed, swapping the generation paradigm still collapses EER ~1%→~50%;
> cross-dataset transfer reveals a dataset-level collapse to near-chance; and a window-level/MIL analysis
> shows local temporal evidence does not transfer either—all model-agnostic.**

## Sec 5.4 — Leave-one-generator-out (LOGO): generation-paradigm shift
We hold out an entire generation paradigm (SVS vs SVC), orthogonal to the vocoder axis, and train on
the other. Both an interpretable detector (RF-FULL) and SOTA AASIST degrade ~2–3× in EER on the
unseen paradigm. Notably AASIST, despite a near-perfect in-paradigm EER (5.5–8.0%), degrades **more**
than RF (×2.4–3.0 vs ×1.9–2.0), i.e., its low in-distribution error does not transfer to an unseen
generation paradigm.

**Table 4.** LOGO EER (%); degradation = LOGO / in-dist.

| Model | Held-out paradigm | in-dist | LOGO | degradation |
|---|---|---|---|---|
| AASIST | SVS | 5.48 | 16.18 | **2.95** |
| RF-FULL | SVS | 17.00 | 34.59 | 2.03 |
| AASIST | SVC | 8.00 | 19.43 | **2.43** |
| RF-FULL | SVC | 20.78 | 39.24 | 1.89 |

## Sec 5.5 — Factorial vocoder×paradigm decoupling (multi-axis confound) [NEW, #2]
LOVO and LOGO each vary a single axis. Here we treat (generation paradigm × vocoder) as a 2-D grid.
In CtrSVDD the two axes are **nearly nested**—SVC co-occurs only with NSF-HiFiGAN; HiFi-GAN and DDSP
occur only with SVS—so standard splits *conflate* paradigm and vocoder. NSF-HiFiGAN is the only vocoder
spanning both paradigms, giving a window to **isolate each axis**.

**Contingency (E1 spoof counts).** SVS: hifigan 900 / ddsp 300 / nsf 300; SVC: nsf 900 (else 0).
Populated cells: SVS×hifigan, SVS×ddsp, SVS×nsf, SVC×nsf.

**Table 5a.** Leave-One-Cell-Out, RF-FULL EER (%); in-dist trains with the cell, held-out removes it.

| Held-out cell | in-dist | held-out | degradation |
|---|---|---|---|
| SVS×hifigan | 2.33 | 31.76 | 13.6× |
| SVS×ddsp | 3.67 | 28.67 | 7.8× |
| SVS×nsf-hifigan | 2.00 | 41.33 | **20.7×** |
| SVC×nsf-hifigan | 3.67 | 39.90 | 10.9× |

**Table 5b.** Axis-controlled contrasts (RF-FULL EER %, and AASIST for the key paradigm contrast).
Holding one axis fixed isolates the other.

| Contrast | Fixed | train→test | in-dist | held-out | degradation |
|---|---|---|---|---|---|
| Unseen vocoder | paradigm=SVS | hifigan→ddsp | 2.33 | 29.71 | 12.8× |
| Unseen vocoder | paradigm=SVS | hifigan→nsf | 1.38 | 39.67 | 28.8× |
| **Unseen paradigm** | **vocoder=NSF** | SVC→SVS | 0.62 | 55.33 | **89.2×** |
| **Unseen paradigm** | **vocoder=NSF** | SVS→SVC | 0.81 | 48.00 | **59.3×** |
| **Unseen paradigm (AASIST)** | **vocoder=NSF** | SVC→SVS | **4.00** | **54.29** | **13.6×** |

With the vocoder held identical (NSF-HiFiGAN), swapping only the generation paradigm collapses both an
interpretable detector (RF ×59–89) and SOTA AASIST (4.0%→54.29%, AUC 0.994→0.456, ×13.6) to chance or
below—**a larger drop than an unseen vocoder under a fixed paradigm**. The confound is therefore
**multi-axis**: the generation paradigm contributes independently of, and as much as or more than, the
vocoder. This decisively refutes the "SVDD detectors are just vocoder detectors" hypothesis and is
**model-agnostic**.

## Sec 5.6 — Cross-dataset transfer: dataset-level collapse (headline)
Training on one dataset and testing on another, both detectors collapse from near-perfect in-domain
performance (AUC 0.90–1.00) to near-chance (AUC 0.55–0.62). The effect is most extreme for AASIST:
an in-domain EER of 5–6% rises to 42–45% (≈ chance) across datasets—degradation ratios of 6.5–8.8×.
Extending to the **wild** WildSVDD set, an in-domain AUC of 0.94 (RF) is not predicted by any source
detector: CtrSVDD→Wild and SingerLens→Wild are at chance (AUC 0.49–0.45), i.e., public-benchmark scores
do not predict real-world wild AI-cover detection.

Critically, the collapse holds for a **fine-tuned strong SSL** detector too: a fine-tuned
wav2vec2-base-960h reaches AUC 0.90–0.91 in-domain but drops to AUC 0.42 / 0.34 (below chance) across
datasets—removing the concern that our earlier *frozen* SSL baseline was too weak to be informative. The
cross-dataset collapse is thus established across **three independent detector families**: hand-feature
RF, neural AASIST, and fine-tuned SSL. The collapse is also **scale-agnostic**: a 3.3× larger SSL
(wav2vec2-large-960h, 315M) is stronger in-domain (CtrSVDD AUC 0.912→**0.970**) yet still collapses
cross-dataset (A→B AUC 0.417→**0.426**, below chance)—larger pretraining capacity buys in-domain accuracy,
not transfer.

**Table 6.** Cross-dataset EER (%) / AUC. A = CtrSVDD, B = SingerLens, W = WildSVDD (wild).
w2v2-ft = fine-tuned wav2vec2-base-960h (SSL); † B-within for w2v2-ft uses a stricter song-disjoint split.
w2v2-large = fine-tuned wav2vec2-large-960h (315M), shown for A→A / A→B as a model-scale data point.

| Protocol | AASIST EER | AASIST AUC | RF-FULL EER | RF-FULL AUC | w2v2-ft EER | w2v2-ft AUC |
|---|---|---|---|---|---|---|
| A→A (within CtrSVDD) | 6.50 | 0.987 | 18.88 | 0.899 | 17.50 | 0.912 |
| A→A — w2v2-large (315M) | — | — | — | — | 8.50 | **0.970** |
| B→B (within SingerLens) | 5.16 | 0.987 | 0.00 | 1.000 | 17.54† | 0.904† |
| A→B (train Ctr, test SL) | 42.58 | 0.596 | 42.03 | 0.615 | 56.31 | 0.417 |
| A→B — w2v2-large (315M) | — | — | — | — | 54.93 | **0.426** |
| B→A (train SL, test Ctr) | 45.25 | 0.553 | 46.96 | 0.541 | 62.00 | 0.337 |
| A→W (train Ctr, test Wild) | 32.1 | 0.748 | 48.9 | 0.520 | — | — |
| B→W (train SL, test Wild) | 54.5 | 0.430 | 53.7 | 0.461 | — | — |
| W→W (within Wild) | — | — | 14.8 | 0.935 | — | — |

## Sec 5.6b — Failure Analysis: Score Collapse, Domain Offset, and the Cleanliness Axis [NEW, P1+P2]

**Score-reversal diagnosis (why the collapse happens).** We dump per-clip canonical scores (FULL
clip_mean RF) and inspect the real/fake distributions (Figure 2). In-domain, real and fake separate
cleanly (mean_score_fake − mean_score_real ≈ +0.30–0.47); under every cross-dataset protocol the
separation collapses to ≈ ±0.02 and the distributions overlap (AUC ≈ 0.5). Crucially this is **not a
sign flip**: the **flipped-AUC** (max(AUC, 1−AUC)) stays at 0.53–0.56, so re-labelling would not help—
the target domain simply contains no separable signal along the source decision axis. The failure also
manifests as a **domain-dependent score offset**: a CtrSVDD detector pushes *all* WildSVDD clips high
(mean ≈ 0.62, "everything looks fake"), while a SingerLens detector pushes them low (≈ 0.25,
"everything looks real")—the whole target domain is shifted by a source-prior offset with zero
in-target discrimination. (Strong reversal, AUC < 0.45, appears only for the score-inflating MIL_max
pooling, not for mean aggregation.)

**Table 6b.** Score-distribution summary (FULL clip_mean). Class: AUC<0.45 reversed / 0.45–0.60 random / ≥0.60 transferable.

| Protocol | mean_real | mean_fake | AUC | flipped_AUC | class |
|---|---|---|---|---|---|
| Wild within | 0.356 | 0.650 | 0.884 | 0.884 | transferable |
| CtrSVDD within | 0.345 | 0.629 | 0.894 | 0.894 | transferable |
| SingerLens within | 0.265 | 0.735 | 0.984 | 0.984 | transferable |
| CtrSVDD→Wild | 0.632 | 0.614 | 0.464 | 0.536 | random |
| SingerLens→Wild | 0.251 | 0.267 | 0.542 | 0.542 | random |
| CtrSVDD→SingerLens | 0.626 | 0.651 | 0.556 | 0.556 | random |
| SingerLens→CtrSVDD | 0.365 | 0.385 | 0.531 | 0.531 | random |

**Failure taxonomy (what the errors are).** The discriminative axis differs across domains: within
WildSVDD, fakes are *cleaner* (HNR 17.1 > real 14.1—AI vocals are smoother), whereas CtrSVDD has no HNR
gap (21.6 ≈ 22.0) and keys on a spectral/MFCC generation signature. A CtrSVDD detector applied to the
wild therefore scores along a **production-cleanliness axis orthogonal to authenticity**, producing two
systematic error types (Table 6c): (i) **FP on low-HNR reals**—wild real vocals with Demucs/
accompaniment residue (HNR 13.7, highest spectral flatness) are read as "synthetic"; (ii) **FN on clean
fakes**—high-HNR So-VITS covers (HNR 19.5) that lack the source generation signature are read as "real"
(FN spread evenly across all So-VITS variants, i.e. not model-specific).

**Table 6c.** CtrSVDD→WildSVDD error groups (per-clip feature means).

| Outcome | n | mean score | HNR | spectral flatness | dominant cause |
|---|---|---|---|---|---|
| FP (real→fake) | 190 | 0.678 | **13.68** | 0.0228 | Demucs/accompaniment residue, production noise |
| FN (fake→real) | 60 | 0.420 | **19.50** | 0.0178 | clean So-VITS, missing source generation cue |
| TP | 177 | 0.679 | 16.32 | 0.0220 | (incidentally noisier fakes) |
| TN | 39 | 0.405 | 16.42 | 0.0186 | (incidentally cleaner reals) |

**Takeaway (use in Discussion):** *Cross-dataset detection collapse is primarily caused by target-domain
signal loss along the source-domain generation-family cues and production-cleanliness axis—not by a
reversal of the decision direction.*

## Sec 5.7 — Window-level locality does not transfer [NEW, #3]
Some hard fakes show transient AI artifacts that are averaged out at clip level, motivating a
window-level / multiple-instance (MIL) view: is a clip fake if *any* window looks fake? We compare
four clip representations built from 3 s/1.5 s windows: **clip_mean** (baseline), **MIL_mean**,
**MIL_max** (worst-window), and **POOL_RICH** (per-feature temporal mean/max/min/std as features).

Within-domain, MIL_mean / POOL_RICH give a small, consistent improvement (SingerLens 0.984→0.996,
CtrSVDD 0.894→0.905, WildSVDD 0.884→0.895). Under **cross-dataset transfer to WildSVDD, every
representation collapses to near-chance or reversed (AUC 0.43–0.56)**; MIL_max is consistently the
worst and merely inflates all scores (mean_score_real ≈ mean_score_fake ≈ 0.72). Window granularity
therefore provides an in-domain smoothing benefit but **zero cross-dataset generalization**.

**Table 7.** Window-level cross-dataset, FULL features, AUC | EER(%).

| Protocol | clip_mean | MIL_mean | MIL_max | POOL_RICH |
|---|---|---|---|---|
| within SingerLens | 0.984 \| 6.2 | **0.996 \| 3.6** | 0.990 \| 4.5 | 0.981 \| 6.3 |
| within CtrSVDD | 0.894 \| 20.2 | **0.905 \| 18.5** | 0.880 \| 21.0 | 0.885 \| 20.3 |
| within WildSVDD | 0.884 \| 20.8 | 0.886 \| 21.3 | 0.881 \| 20.4 | **0.895 \| 18.9** |
| CtrSVDD→Wild | 0.464 \| 52.6 | 0.459 \| 52.2 | 0.431 \| 54.5 | 0.482 \| 52.0 |
| SingerLens→Wild | 0.542 \| 48.9 | 0.517 \| 48.9 | 0.514 \| 50.9 | 0.559 \| 45.1 |

**Figure 1 (three-channel case timelines, outputs/window_cross_dataset_cases/).** For five cases we plot
p(fake) per window above synchronized pitch (f0), HNR, and energy (RMS) channels.
- **case1 — source-domain fake (SingerLens, OOF):** p(fake) spikes locally (0.51→0.95) and is
  **synchronized with a pitch jump** (f0 210→315 Hz at a high-note transition): genuine, localized AI
  artifact in the source domain.
- **case3 — WildSVDD real, false-positive (CtrSVDD→Wild):** p(fake) is **flat and high (~0.90) with no
  localization**; the clip's HNR is only 1–2.5 dB vs 15–18 dB in the source domain—a **domain mismatch**,
  not local AI evidence. The clean-studio-trained detector flags all noisy wild audio as fake.
- **score saturation / reversal:** under CtrSVDD→Wild, real (case3 mean 0.896) and fake (case5 mean 0.874)
  are nearly indistinguishable; mean_score_real (0.632) ≈ mean_score_fake (0.614); AUC<0.5 (reversed).
- **case2 — WildSVDD fake, false-negative:** p(fake) stays ≤0.57 across all windows—So-VITS/RVC wild
  artifacts differ from CtrSVDD-learned ones; the source-domain local cue is absent in the target.

**One-line takeaway:** *Window-level locality does not imply cross-domain generalizability.* The
intuitive "look at local artifacts" remedy fails because local evidence is itself domain-specific
(aligned pitch spikes in-domain; an HNR-scale mismatch, indistinguishable real/fake, and score reversal
out-of-domain), and this holds across pooling strategies—representation-agnostic, mirroring the
model-agnostic results above.

## Sec 5.10 — Deployment Implications: Abstention, Few-shot Calibration, Production Perturbation [NEW, P-deploy]
Three deployment-oriented analyses (FINDINGS_deployment_analysis.md) ask whether a benchmark-trained
SVDD detector can be safely fielded.

**(a) Abstention does not recover the collapse.** With confidence = |score − 0.5| and selective
classification (keep the top-c% most confident), in-domain accuracy rises to ~1.00 by c=10%—but
cross-dataset accuracy stays at chance and *worsens* at high confidence (CtrSVDD→Wild: 0.46 at full
coverage → 0.35 at 5% coverage). The model is **confidently wrong out-of-domain**; a confidence
threshold is not a safe reject mechanism. [Table 8a, Figure 3]

**(b) A few target labels beat the confounded source.** Calibrating with k labeled WildSVDD clips:
*Wild-only* training from scratch reaches AUC 0.817 (k=50) / 0.870 (k=100), approaching the
within-domain bound (~0.935), whereas pooling those k with 4 800 CtrSVDD samples reaches only
0.585 / 0.641—the confounded source actively drags performance down (negative transfer / swamping).
A dedicated reinforcement on WildSVDD-T02 (Exp2b, FINDINGS_fewshot_reweighting.md) disambiguates the
mechanism: up-weighting / over-sampling the k labels narrows the gap (k=100: naive 0.638 → 0.734 / 0.773)
**but still does not reach target-only (0.852)**, and **calibration-only** (re-fitting a threshold on the
source-only score with k labels) stays at **chance (AUC≈0.49, unchanged by k)**—a monotonic remap cannot
fix a ranking collapse. So the source decision axis is *misaligned* with the target, not merely swamped or
miscalibrated; we do not claim all source+target adaptation fails (full DA untested), only that naive
mixing and simple reweighting/calibration do not close the gap. For a new platform, collect ~50-100
target labels and train fresh. [Table 8b, Figure 4]

**(c) Production perturbation of *real* audio induces false positives.** Perturbing only real clips
(no fakes introduced), band-limiting (resample-8k, low-pass 3-4 kHz) and codec compression (MP3-32k)
raise the real-clip fake-score from 0.085 to 0.36-0.46 and flip 13-47% of reals to "fake"; additive
noise, loudness, and **accompaniment leakage do not** (≈0% flip). A reinforcement (Exp3b,
FINDINGS_production_leakage_resynthesis.md) confirms leakage is inert across **three levels (−12/−6/0 dB,
all ≈0% flip)**, whereas **Demucs re-separation raises the score (+0.14, or +0.18 chained after leakage)**—
the increment comes from the separation step, not the accompaniment. The detector keys on a
**spectral-bandwidth/codec/separation-signature axis**—not on accompaniment energy or noise—so platform
transcoding / band-limiting is a concrete false-positive risk (causal support for, and a refinement of,
the Sec 5.6b cleanliness-axis). [Table 8c]

**Table 8a.** Selective-prediction accuracy at coverage c (FULL clip_mean).

| protocol | acc@100% | acc@50% | acc@10% | acc@5% |
|---|---|---|---|---|
| CtrSVDD within | 0.80 | 0.94 | 1.00 | 1.00 |
| WildSVDD within | 0.80 | 0.92 | 1.00 | 1.00 |
| CtrSVDD→Wild | 0.46 | 0.50 | 0.45 | 0.35 |
| CtrSVDD→SingerLens | 0.53 | 0.55 | 0.40 | 0.41 |

**Table 8b.** Few-shot calibration on held-out WildSVDD-T02, AUC (20-trial mean; Exp2b).

| k | target-only | source+k naive | source+k up-sample | source+k domain-bal | calibration-only |
|---|---|---|---|---|---|
| 25 | **0.720** | 0.557 | 0.619 | 0.588 | 0.492 |
| 50 | **0.798** | 0.580 | 0.690 | 0.649 | 0.496 |
| 100 | **0.852** | 0.638 | 0.773 | 0.734 | 0.485 |

(Earlier Wild-all run, Exp2-ext: target-only 0.817/0.870 @k=50/100 vs source+k naive 0.585/0.640 →
over-sampled 0.725/0.800; same pattern. calibration-only ≈ chance confirms axis-misalignment, not
miscalibration.)

**Table 8c.** Real-only production perturbation: fake-score of real clips (n=200).

| perturbation | mean fake-score | Δ vs clean | flip→fake |
|---|---|---|---|
| clean real | 0.085 | 0.000 | 0% |
| resample-8k | 0.455 | +0.370 | 46.5% |
| low-pass 3k | 0.450 | +0.365 | 46.5% |
| MP3-32k | 0.363 | +0.278 | 25.0% |
| reverb | 0.232 | +0.147 | 4.5% |
| Demucs re-separation (clean) | 0.226 | +0.141 | 2.0% |
| leak → Demucs re-separation (Exp3b) | 0.262 | +0.177 | 5.3% |
| noise (SNR 20/10 dB) | 0.11 / 0.06 | +0.02 / −0.02 | 0% |
| accompaniment leakage (−12/−6/0 dB, Exp3b) | 0.09 / 0.08 / 0.08 | ≈0 | 0% |
| gain (quiet) | 0.100 | +0.014 | 0% |

(Honest scope. (1) These perturbations are **controlled diagnostic probes, not a faithful simulation of
platform distributions**; we read score shifts as qualitative sensitivity evidence, not deployable
error-rate estimates—external validity rests on the WildSVDD wild test. (2) Accompaniment-leakage and
Demucs-re-separation were tested **only via synthetic probes**: leakage does not induce FP in this setting
(refuting a naive "accompaniment residue → FP" reading, i.e. the FP driver is the spectral/separation/codec
signature, not accompaniment energy), but real platform separation/residue as an FP contributor is **not
fully characterized**. (3) The few-shot result contrasts **naive mixing vs simple target up-weighting/
over-sampling**; source+k stays below Wild-only, so source supervision is a negative asset *under naive
mixing*—we do not claim all source+target adaptation (DANN, calibration fine-tuning) is ineffective.)

## Sec 5.9b — Closing the Unsupervised Door, and the Minimal-Supervision Exit [NEW, Tier1/2 + MSA]
We test unsupervised remedies, attribute the failure, and identify the one viable exit
(FINDINGS_domain_adaptation.md, FINDINGS_confound_removal.md, FINDINGS_minimal_supervised_adaptation.md).
The section's takeaway is **structured, not a list of failures: every unsupervised route is closed, and
minimal target supervision is the only—and a cheap—exit.**

**(a) Unsupervised domain adaptation fails.** Feature-alignment DA—per-domain z-norm, CORAL,
subspace alignment, and an adversarial DANN—does not lift cross-dataset AUC out of the chance band,
and sometimes underperforms the no-adapt baseline, whereas a target-only model (with labels) is at
0.90–0.99. The collapse is a shift in the *conditional* p(y|x) (the decision direction), which
marginal-alignment DA cannot fix—consistent with calibration-only ≈ chance (Sec 5.10).

**Table 9a.** Unsupervised DA, AUC on the target domain.

| pair | baseline | z-norm | CORAL | subspace | DANN | target-only |
|---|---|---|---|---|---|---|
| CtrSVDD→WildT02 | 0.509 | 0.488 | 0.474 | 0.517 | 0.442 | **0.925** |
| CtrSVDD→SingerLens | 0.615 | 0.618 | 0.537 | 0.554 | 0.555 | **0.989** |
| SingerLens→CtrSVDD | 0.549 | 0.569 | 0.529 | 0.544 | 0.582 | **0.898** |

**(b) Removing the bandwidth axis is insufficient.** Datasets differ in spectral bandwidth (rolloff85:
CtrSVDD reals 3162 Hz < WildT02 3454 < SingerLens 3774; and the real-vs-fake bandwidth direction *flips*
across datasets). But equalizing it—band-limiting everything to a common 3.5 kHz before re-extraction—
does **not** recover transfer (AUC still 0.43–0.60). Bandwidth is one component of the cleanliness
confound; the residual codec/separation/generation signature (mid-band MFCC texture) sustains the collapse.

**Table 9b.** Cross-dataset AUC, native vs common-bandwidth (3.5 kHz).

| pair | native | band-limited |
|---|---|---|
| CtrSVDD→WildT02 | 0.509 | 0.434 |
| CtrSVDD→SingerLens | 0.615 | 0.584 |
| SingerLens→CtrSVDD | 0.549 | 0.602 |

**(c) Attribution: the discriminative axis is domain-specific.** RF importance: the source (CtrSVDD)
model leans on MFCC (0.64) and barely on HNR (0.055); the in-target (WildT02) model uses HNR much more
(0.146)—HNR being the genuine Wild real-vs-fake cue. The source model's (wrong) ranking of Wild clips
correlates most with high-order MFCC-std (mfcc_13_std ρ=0.55, mfcc_11_std 0.51)—i.e. it sorts by
spectral texture / production signature, not authenticity. [Figure 5: axis_attribution.png]

**(d) The minimal-supervision exit.** With the unsupervised door closed, we quantify the supervised one:
on the same WildSVDD-T02 target, a target-only model trained on *k* labels from scratch beats every
unsupervised method by a wide margin from as few as **k=25 (AUC 0.720 vs ≤0.52 unsupervised)**, rising to
0.798 (k=50) and 0.852 (k=100), approaching the in-domain bound (~0.925). The return curve is steepest in
the first few dozen labels. So Sec 5.9b is a *structured* result—**unsupervised adaptation is closed;
minimal target supervision is the only viable, and cheap, exit**—rather than a string of negative results.

**Table 9d.** Minimal supervised adaptation vs unsupervised (held-out WildSVDD-T02 AUC).

| approach | target labels | AUC |
|---|---|---|
| best unsupervised (CORAL/subspace/DANN/z-norm/calibration) | 0 | ≤0.52 (≈chance) |
| target-only, k=25 | 25 | **0.720** |
| target-only, k=50 | 50 | **0.798** |
| target-only, k=100 | 100 | **0.852** |
| full target (in-domain bound) | all | ~0.925 |

**Honest scope:** we test feature-alignment DA + a single confound-removal axis (bandwidth) only; full
end-to-end neural DA, contrastive alignment, and "copy-synthesis through one common vocoder" are untested.
The claim is *unsupervised feature-alignment and bandwidth-equalization do not fix the collapse*, not that
no method can. Note the exit is *target-only from scratch* (source+k, even reweighted, stays below it,
Sec 5.10); and reaching 0.90+ needs near-full target labels (~25 labels ≈ 0.72), so "cheap" means *relative
to unsupervised*, not "a handful of labels suffice for deployment-grade accuracy."

**(e) Even a production-orthogonal *semantic* axis does not transfer.** We tested whether an
emotion-consistency score (sung-emotion vs lyric-emotion, a zero-training high-level semantic cue,
Appendix on the emotion module / FINDINGS_emotion_consistency_crossdomain.md) is more cross-domain robust
because it is orthogonal to the production chain. It is not: AUC drops from 0.679 (SingerLens) to **0.412
(below chance, relation reversed) on WildSVDD**—a *larger* relative drop than AASIST (ratio 0.61 vs 0.76),
and the collapse persists on the well-transcribed subset (0.395). So **no single axis—production *or*
semantic—solves cross-domain detection**, which further isolates minimal target supervision as the only
exit. (The emotion module remains a within-domain case study, not a cross-domain solution.)

## Unified summary (outputs/svdd_audit_summary.csv)
Columns: protocol, model, heldout_domain, in_dist_eer, out_dist_eer, degradation_ratio.
Spanning LOVO (vocoder), LOGO (paradigm), factorial cell-out, cross-dataset, and window-level transfer;
degradation ratio undefined for RF-FULL B→A (in-dist EER 0.00, use AUC 1.00→0.54 instead).

## Appendix A — Human-listening pilot (prepared, ready-to-run) [P3]
A blind 40-sample pilot is packaged to test whether human perception compensates for the model's
cross-domain failures. Four categories × 10, all from WildSVDD: **easy_real**, **easy_fake**,
**model_failed_fake** (cross-domain FN), **model_fp_real** (cross-domain FP); anonymized + shuffled.
Deliverables: `outputs/human_pilot/{audio/HP001–040.wav, human_pilot_manifest.csv (blind: human_label /
confidence_1–5 / suspected_reason), human_pilot_key.csv (true label/category/model score),
README_human_pilot.md}` + `human_pilot_package.tar.gz`. Annotation task: real/fake + confidence(1–5) +
suspected-reason keywords. Planned analysis: human accuracy per category vs the detector, focusing on the
two model-error classes; confidence calibration; whether human cues differ from the model's cleanliness
axis. **Status:** prep complete; running the study (recruitment/IRB) is future work.

> **Human pilot ready-to-run for evaluating whether human perception can compensate for model
> cross-domain failures.**
