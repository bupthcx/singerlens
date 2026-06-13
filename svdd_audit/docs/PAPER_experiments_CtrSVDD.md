# 4. Experiments on a Public Benchmark (CtrSVDD)

## 4.1 Setup
We use **CtrSVDD** [Zang et al., 2024], the controlled track of the SVDD Challenge 2024. CtrSVDD deliberately uses clean, unaccompanied vocals for bonafide recordings to remove source-separation artifacts, leaving the *generation/vocoder pipeline* as the dominant systematic difference between bonafide and deepfake clips—making it an ideal testbed for auditing vocoder confounds. All audio is 16 kHz. The train/dev partitions contain 8 attack systems (A01–A08) spanning singing voice synthesis (SVS) and conversion (SVC). We map each system to its **vocoder family** from the public generation pipelines: HiFi-GAN {A01, A02, A05}, source-filter / NSF-HiFiGAN {A04, A06, A07, A08}, and DDSP {A03}.

To isolate feature behaviour from data scale, we build a balanced analysis subset of 2,400 bonafide (Mandarin corpora Opencpop / M4Singer) and 2,400 spoof clips stratified by system. We report **Equal Error Rate (EER)** using the official scoring, with AUC/F1 where informative.

We compare four interpretable feature sets—FULL (low-level + MFCC + pitch + vibrato), CLEAN (octave/loudness-invariant style features), HNR (harmonics-to-noise voice quality), VRI (vibrato regularity)—classified by a random forest, against a strong neural baseline, **AASIST** [Jung et al., 2022] (raw-waveform SincConv + graph attention), trained with focal loss.

## 4.2 Interpretable "style" features do not transfer (Table 1)
Table 1 reports bonafide-vs-spoof EER, overall and grouped by vocoder family. On CtrSVDD the explainable voice-quality and vibrato features (HNR, VRI) are **at chance** (EER 45.2% and 44.5%), in stark contrast to their discriminativeness on self-collected web data. Only FULL is effective (18.9% EER), and its performance is strongly vocoder-dependent (HiFi-GAN 13.7%, NSF-HiFiGAN 20.9%, DDSP 26.0%). This already suggests that what is "detected" is tied to the generation/vocoder family rather than a generalizable notion of synthetic singing.

**Table 1.** Bonafide-vs-spoof EER (%) on CtrSVDD.

| Group | FULL | CLEAN | HNR | VRI |
|---|---|---|---|---|
| overall | 18.88 | 35.96 | 45.17 | 44.50 |
| HiFi-GAN | 13.67 | 30.33 | 44.76 | 41.12 |
| DDSP | 26.04 | 29.33 | 46.00 | 38.94 |
| NSF-HiFiGAN | 20.92 | 41.67 | 45.25 | 48.92 |

## 4.3 Copy-synthesis control (Table 2)
To probe whether detection rides a *generic* vocoder footprint, we pass each bonafide clip through an off-the-shelf NSF-HiFiGAN (analysis-by-synthesis: mel + F0 → vocoder), downsampling back to 16 kHz to match the spoof pipeline and avoid a sampling-rate confound. A QC check confirms matched sampling rate, duration, loudness, and F0 across bonafide, bonafide-vocoded, and spoof. We then test a FULL detector trained on (natural bonafide vs spoof): it assigns bonafide-vocoded a mean spoof-probability of 0.375 and flags only 18.7% as spoof—essentially identical to held-out natural bonafide (0.351, 18.5%), and far from spoof (0.869, 100%). Thus the detector is *not* a generic "vocoded-or-not" classifier; passing bonafide through a *different* vocoder does not make it look fake. Detection instead keys on artifacts specific to the spoofing pipeline—motivating a family-wise generalization analysis.

**Table 2.** Cross-test of a FULL detector trained on natural-bonafide vs spoof (NSF-HiFiGAN). QC matched across groups (16 kHz; dur ≈ 5 s; rms ≈ 0.082; F0 ≈ 255 Hz).

| Test set | mean P(spoof) | % flagged spoof |
|---|---|---|
| natural bonafide (held-out) | 0.351 | 18.5 |
| spoof (NSF-HiFiGAN) | 0.869 | 100.0 |
| bonafide-vocoded (NSF-HiFiGAN) | 0.375 | 18.7 |

## 4.4 Leave-one-vocoder-out generalization (Table 3)
We hold out one vocoder family at a time: a detector is trained on spoofs from the other families (plus a fixed bonafide split) and tested on the held-out family's spoofs (plus disjoint bonafide). For the interpretable FULL detector, EER degrades sharply on unseen vocoders (e.g., HiFi-GAN 13.7% → 31.8%, ×2.3), confirming reliance on vocoder-specific cues.

Crucially, we ask whether a SOTA neural model escapes this. AASIST reaches an in-distribution EER of 6.5% (AUC 0.987), on par with the official baseline. Yet under leave-one-vocoder-out, AASIST **also degrades severely** on the most common vocoder family, NSF-HiFiGAN (7.3% → 21.9%, ×3.0), while remaining robust to held-out HiFi-GAN (×0.93). Since NSF / source-filter HiFi-GAN underlies 8 of 13 CtrSVDD systems (all So-VITS-SVC variants), this degradation is practically significant.

**Table 3.** In-distribution vs leave-one-vocoder-out EER (%); degradation = LOVO / in-dist.

| Model | Held-out vocoder | in-dist | LOVO | degradation |
|---|---|---|---|---|
| AASIST | HiFi-GAN | 5.10 | 4.76 | 0.93 |
| RF-FULL | HiFi-GAN | 13.67 | 31.76 | 2.32 |
| AASIST | NSF-HiFiGAN | 7.33 | 21.92 | **2.99** |
| RF-FULL | NSF-HiFiGAN | 20.92 | 36.33 | 1.74 |
| AASIST | DDSP | 4.00 | 5.67 | 1.42 |
| RF-FULL | DDSP | 26.04 | 28.67 | 1.10 |

## 4.5 Takeaways
(i) On a controlled benchmark, interpretable singing-style features (HNR/VRI) carry little transferable authenticity signal—their apparent usefulness on web data is dataset-specific. (ii) Detection is not a generic vocoder detector, but keys on generation/vocoder-family-specific artifacts. (iii) This unseen-vocoder generalization gap is **model-agnostic**: even SOTA AASIST loses 3× EER on the dominant NSF-HiFiGAN family. Together, these results indicate that current SVDD performance substantially reflects *"which generation family is this"* rather than *"is this an AI cover"*, and that copy-synthesis controls and leave-one-vocoder-out are necessary diagnostics for the field.
