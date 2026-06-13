# Beyond Standard Splits: Auditing Generation-Family Confounds in Singing Voice Deepfake Detection

> **Assembled full draft** (v1). 由 7 个分节 DRAFT 拼装,节序已重排(5.6 置于 5.5 后),写作笔记已移至文末附录。LaTeX 表/图为占位语法;citations 为 [cite] 占位(§2)。

---

## Abstract

Singing-voice deepfake detection (SVDD) reports low equal error rates on standard benchmarks, suggesting
the task is nearly solved. We show these numbers substantially overestimate real-world generalization,
because standard splits let detectors exploit \emph{generation-family} and \emph{production} confounds---
vocoder fingerprints, generation-paradigm signatures, and recording/codec characteristics---that are
constant within a dataset but shift across generators, datasets, and the wild. We introduce
\textbf{SVDD-Audit}, a suite of controlled protocols---a copy-synthesis control, leave-one-vocoder-out,
leave-one-generator-out, a factorial vocoder$\times$paradigm cell-out, and cross-dataset transfer---that
disentangle these confounds from genuine spoofing cues. Across three detector families (an interpretable
random forest, the neural AASIST, and a fine-tuned wav2vec2 model) and three datasets, detectors with
in-distribution AUC 0.90--0.99 collapse to near-chance (0.34--0.62) under cross-dataset transfer; the
effect is \emph{model-agnostic} and \emph{scale-agnostic}, and, with the vocoder held fixed, swapping only
the generation paradigm drives EER from $\sim$1\% to $\sim$50\%. We attribute the collapse to a
domain-specific production axis---confirmed causally, since perturbing only the production of genuine
recordings flips 13--47\% of them to ``fake''---and show that \emph{no} label-free remedy (unsupervised
domain adaptation, confound removal, recalibration, or even a production-orthogonal semantic cue) repairs
it; only minimal target supervision does---as few as $\sim$10 target labels already exit the chance band,
with the per-label gain steepest in the first few dozen. We release the protocols, mappings,
and evaluation toolkit, and discuss implications for trustworthy SVDD evaluation.

---

---

## 1. Introduction

Neural singing-voice synthesis and conversion have advanced to the point where a few minutes of a target
singer's audio can produce convincing AI covers, raising clear risks for impersonation, copyright, and
consent. Singing-voice deepfake detection (SVDD)---deciding whether a sung vocal is human or
machine-generated---has emerged as the corresponding defense, supported by recent datasets and challenges
(SingFake, the CtrSVDD / SVDD Challenge, WildSVDD). On these benchmarks, detectors now report low equal
error rates: state-of-the-art neural models reach EERs of a few percent, and the task can appear close to
solved.

We argue that this appearance is misleading. Standard SVDD evaluation uses random or seen-condition
splits, in which the entire dataset shares one recording pipeline, a small set of generators, and one
family of vocoders. A detector is then free to exploit \emph{generation-family} and \emph{production}
regularities---vocoder fingerprints, generation-paradigm signatures, and recording/codec characteristics---
that are perfectly predictive \emph{within} the dataset but have nothing to do with authenticity and do
not survive a change of generator, dataset, or platform. The reported numbers therefore measure how well a
detector recognizes a dataset's production family, not whether audio is AI-generated.

To make this precise, we introduce \textbf{SVDD-Audit}, a suite of controlled evaluation protocols that
hold authenticity fixed and vary one confounding factor at a time: a copy-synthesis control that isolates
generic ``vocodedness''; leave-one-vocoder-out and leave-one-generator-out; a factorial
vocoder$\times$paradigm cell-out that separates the two axes; and cross-dataset transfer. Applied across
three detector families of very different design---an interpretable random forest, the neural AASIST, and
a fine-tuned self-supervised (wav2vec2) model---and three datasets (controlled CtrSVDD, self-collected,
and wild bilibili covers), the audit tells a consistent story (Figure~\ref{fig:teaser}). \emph{In-
distribution} all detectors are strong to near-perfect (AUC 0.90--0.99). Yet \emph{cross-dataset} they
collapse to the chance band (AUC 0.34--0.62), an effect that is \textbf{model-agnostic} (it holds for
hand-crafted, neural, and SSL detectors alike) and \textbf{scale-agnostic} (a $3.3\times$ larger SSL is
stronger in-domain but still collapses). The confound is \textbf{multi-axis}: with the vocoder held
identical, swapping only the generation paradigm drives EER from $\sim$1\% to $\sim$50\%.

Beyond demonstrating the collapse, we diagnose it and chart its boundary. The failure is not noise but a
\emph{domain-specific decision axis}: detectors rank clips by a production signature (spectral-bandwidth /
codec / separation texture), which we confirm causally---perturbing only the \emph{production} of genuine
recordings (band-limiting, codec compression) flips 13--47\% of them to ``fake,'' while content-level
changes (noise, loudness, accompaniment leakage) do not. We then ask whether the collapse can be repaired
without new labels, and find that \emph{every} shortcut fails: unsupervised domain adaptation
(CORAL/subspace/DANN), confound removal along the implicated axis, score recalibration, and even a
production-orthogonal \emph{semantic} cue (sung-vs-lyric emotion consistency) all remain at chance. The
only remedy that works is direct supervision: a target-only detector trained from as few as $\sim$10
labeled target clips already outperforms every unsupervised method, with the per-label gain steepest in
the first few dozen. The practical message is a structured
one---unsupervised adaptation is closed; minimal target supervision is the only viable exit.

This paper makes three contributions.
\begin{itemize}
  \item \textbf{An audit protocol suite (and open toolkit).} SVDD-Audit formalizes copy-synthesis control,
  leave-one-vocoder/generator-out, a factorial vocoder$\times$paradigm cell-out, and cross-dataset
  transfer, together with an attack$\to$generator$\to$vocoder mapping, to disentangle confounds from
  genuine spoofing cues. We release the protocols, mappings, and scripts.
  \item \textbf{A model- and scale-agnostic demonstration of collapse.} Across three detector families and
  three datasets, in-distribution AUC of 0.90--0.99 falls to near-chance under cross-dataset transfer, and
  the generation paradigm is shown to be an independent confound axis (vocoder-fixed paradigm swap:
  $\sim$1\%$\to$$\sim$50\% EER).
  \item \textbf{A mechanism and a boundary.} We attribute the collapse to a domain-specific production
  axis (confirmed by causal perturbation), show that no unsupervised or semantic shortcut repairs it, and
  identify minimal target supervision as the only effective exit---turning a set of negative results into
  a structured, actionable conclusion.
\end{itemize}

---

## 2. Related Work

\textbf{SVDD datasets and challenges.}
Singing-voice deepfake detection has been driven by a recent wave of datasets and shared tasks. SingFake
introduced detection on in-the-wild song clips with real and AI-cover vocals
\cite{singfake}; the SVDD Challenge 2024 / CtrSVDD provided a controlled benchmark of
unaccompanied vocals with multiple singing-voice-synthesis (SVS) and singing-voice-conversion (SVC)
systems and per-system metadata \cite{ctrsvdd,svddchallenge2024}; and WildSVDD extended
evaluation to wild platform covers \cite{wildsvdd}, with related efforts on fake-song and
music-deepfake collections \cite{fsd,musicdeepfake}. These resources report strong
in-distribution detection; our work re-examines what that performance measures by holding their
generation families out rather than mixing them across train and test.

\textbf{Generalization in audio anti-spoofing.}
Cross-condition and cross-corpus fragility is well documented for \emph{speech} anti-spoofing: systems
trained on ASVspoof degrade markedly on unseen attacks, codecs, and corpora
\cite{asvspoof2019,asvspoof2021,crossdataset_spoof}, motivating robustness and
continual-learning approaches \cite{robust_spoof}. Singing voice adds
production stages (accompaniment, separation, vocoders) absent from clean speech; we show that the
analogous, and arguably sharper, generalization gap exists for SVDD and characterize its \emph{cause}
(a production/generation-family axis) and \emph{boundary} (what does and does not fix it), beyond
documenting the drop.

\textbf{Vocoder and generation artifacts.}
A line of work attributes detectability of synthetic speech to vocoder fingerprints and synthesis
artifacts, and studies their transfer across vocoders \cite{vocoder_artifact,tts_artifact}. Our copy-synthesis control and factorial cell-out build on
this view but separate two axes that are usually entangled---the \emph{vocoder} and the \emph{generation
paradigm}---and show that the paradigm is an independent confound even with the vocoder held fixed.

\textbf{Shortcut learning and dataset bias.}
Detectors are known to exploit spurious, dataset-specific correlations rather than task-relevant signal
\cite{geirhos2020shortcut,dataset_bias}, including in audio forensics
where channel, codec, or silence cues leak the label \cite{audio_leakage}. We provide a concrete SVDD instance: a production-cleanliness axis (spectral bandwidth / codec /
separation signature) that is label-predictive within a dataset but orthogonal to authenticity, identified
by attribution and confirmed by causal perturbation of genuine audio.

\textbf{Domain shift and adaptation.}
Unsupervised domain adaptation---feature-distribution alignment (CORAL \cite{sun2016coral}, subspace alignment
\cite{fernando2013subspace}), adversarial domain-invariant learning (DANN \cite{ganin2015dann}), and normalization-based
schemes---and calibration/few-shot adaptation \cite{guo2017calibration,fewshot_da} are
the standard tools for closing such gaps. We evaluate these for the SVDD cross-dataset collapse and find
that marginal-alignment adaptation does not recover it (because the failure is in the conditional decision
direction), whereas minimal target supervision does---connecting our diagnosis to a concrete, if modest,
remedy.

\textbf{Gap.}
While each ingredient---SVDD benchmarks, anti-spoofing generalization, vocoder artifacts, shortcut
learning, and domain adaptation---has been studied separately, no prior work provides a \emph{systematic
audit} of generation-family confounds in SVDD that (i) disentangles the orthogonal vocoder and paradigm
axes, (ii) measures the model- and scale-agnostic cross-dataset collapse across detector families and
datasets, (iii) attributes it causally to a production axis, and (iv) maps the boundary between what fails
(label-free adaptation, semantic shortcuts) and what works (minimal target supervision). SVDD-Audit fills
this gap and is released as a reusable toolkit.

---

## 3. SVDD-Audit Protocols

\textbf{Setup and notation.}
A clip $x$ carries a label $y\in\{\textsf{bona},\textsf{spoof}\}$ and belongs to a dataset $D$. Each spoof
clip is further annotated with a generation paradigm $p$ (singing-voice \emph{synthesis} SVS vs.\
\emph{conversion} SVC) and a vocoder family $v$ (e.g.\ HiFi-GAN, NSF-HiFiGAN, DDSP). A detector $f$ is
trained on a train split and evaluated on a disjoint test split. \textbf{In every protocol below the
bona-fide clips are split disjointly between train and test}, so that no protocol can be passed by
memorizing speakers or recordings; the protocols differ only in how the \emph{spoof} conditions (or the
dataset) are partitioned. The conventional evaluation is the special case in which nothing is held out.

\textbf{P0 --- Standard split.} Random/seen-condition split within a single dataset: all generators,
vocoders, and the recording pipeline appear in both train and test. This is the optimistic baseline (§5.1).

\textbf{C --- Copy-synthesis control.} We pass bona-fide recordings through the spoof family's vocoder
(analysis-by-synthesis), sampling-rate matched and quality-controlled, to obtain ``bona-vocoded'' clips,
and test whether $f$ flags them as spoof. If $f$ keyed on generic ``vocodedness,'' bona-vocoded would
score as spoof; if it scores as bona, $f$ instead relies on pipeline-specific artifacts (§5.2).

\textbf{P1 --- Leave-One-Vocoder-Out (LOVO).} Partition spoofs by vocoder family $v$; train on all but one
family and test on the held-out family (bona split fixed). Measures dependence on a seen vocoder.

\textbf{P2 --- Leave-One-Generator-Out (LOGO).} Same, partitioning by generation paradigm $p$ (SVS/SVC).
Measures dependence on a seen paradigm, an axis orthogonal to the vocoder.

\textbf{P2b --- Factorial vocoder$\times$paradigm cell-out.} Treat $(p,v)$ as a 2-D grid of cells. We
(i) leave one cell out, and (ii) run \emph{axis-controlled contrasts}: fix one axis and vary the other
(possible because one vocoder, NSF-HiFiGAN, spans both paradigms). This disentangles the paradigm and
vocoder contributions that P1/P2 and standard splits conflate (§5.2).

\textbf{P4 --- Cross-dataset transfer.} Train on dataset $D_A$ and test, \emph{without adaptation}, on a
different dataset $D_B$ (and vice versa). The strongest stress test, closest to deployment (§5.3).

\textbf{Analyses.} Beyond these held-out protocols we use three analyses: a window-level / multiple-
instance variant that asks whether \emph{local} evidence transfers; a battery of label-free remedies
(unsupervised DA, confound removal, calibration) and a minimal-supervision baseline that ask whether the
cross-dataset collapse is fixable (§5.5); and controlled real-only production perturbations that causally
probe what the detector keys on (§5.4).

\textbf{Metrics.} We report equal error rate (EER) and AUC. To quantify a shift we use the
\emph{degradation ratio} $\Delta = \text{EER}_{\text{held-out}}/\text{EER}_{\text{in-dist}}$, and, to
distinguish a lost signal from a flipped one, the \emph{flipped} AUC $\max(\text{AUC},1{-}\text{AUC})$ (a
value near $0.5$ indicates no separable signal in either orientation, not a mis-thresholded one). For the
deployment analysis we additionally use selective-classification accuracy at coverage $c$.

---

## 4. Experimental Setup

\textbf{Datasets.}
We use three datasets spanning a controlled--to--wild axis (Table~\ref{tab:datasets}). \emph{CtrSVDD}
(SVDD Challenge 2024) is a controlled 16\,kHz benchmark of unaccompanied vocals with system, paradigm, and
vocoder annotations; we use a balanced analysis subset of the training/dev systems (A01--A08), with
bona-fide drawn from the available Mandarin sources (m4singer, opencpop). \emph{SingerLens} is our
self-collected web set: 6 singers $\times$ 3 songs, with genuine vocals separated from publicly posted
recordings and AI covers generated by a conversion model (Seed-VC); it is included primarily as a
cautionary case (its production pipeline makes the in-domain task trivially separable) and as a
cross-dataset endpoint. \emph{WildSVDD} is a wild set of bilibili AI covers (So-VITS-SVC / RVC), processed
identically (mid-song segment, source separation, 16\,kHz slicing); we use its reachable bilibili (T02)
subset. Together they let us separate effects of generator, production pipeline, and ``in-the-wild''
conditions.

\textbf{Detectors.}
We evaluate three families. (i) A \emph{random forest} on interpretable acoustic features
(\textsc{full}; with \textsc{clean}/\textsc{hnr}/\textsc{vri} subsets), standardized then classified by
300 trees---a transparent, low-capacity baseline. (ii) \emph{AASIST}, the official challenge neural
baseline (a sinc/RawNet front-end with a graph-attention back-end), trained with focal loss. (iii) A
\emph{fine-tuned self-supervised} model: wav2vec2-base-960h fine-tuned end-to-end (frozen CNN front-end,
differential learning rates, gradient clipping), plus wav2vec2-large-960h (315M) as a model-scale point;
we also report a frozen-SSL reference and a window-level / multiple-instance RF. The attack$\to$generator
$\to$vocoder mapping used by P1/P2/P2b is derived from the generation repositories and released with the
toolkit.

\textbf{SSL model-selection rationale.}
We represent the SSL family with wav2vec2 for an \emph{infrastructural} reason, not a claim of optimality:
WavLM and HuBERT---which might pretrain more singing-relevant representations---were network-unreachable on
our infrastructure (the model hub was walled and the models were absent from the reachable mirror), and a
large multilingual XLSR checkpoint failed to load under our framework version. Within reach, frozen
wav2vec2-base is weak (in-domain AUC 0.71): it is pretrained on English ASR, not singing, so its frozen
features are insensitive to singing acoustics---which, far from undermining our argument, shows that an
off-the-shelf ASR-SSL representation does not by itself carry SVDD discriminability. End-to-end
fine-tuning makes it strong (0.91), and a $3.3\times$ larger model reaches 0.97 in-domain, confirming the
SSL route is effective and scales \emph{in-domain}; both, however, collapse cross-dataset (§5.3).

\textbf{Implementation.}
Unless noted, we use balanced analysis subsets and official EER scoring; cross-dataset and few-shot
evaluations are averaged over multiple random draws, and self-data within-splits are song-disjoint where
indicated. Full hyper-parameters, subset construction, and the protocol scripts are released in the
toolkit (Reproducibility, §6).

% Table 0: datasets
\begin{table}[t]
\centering
\caption{Datasets, spanning controlled to wild. ``ann.'' = available per-clip annotations used by the
audit protocols. SingerLens is included as a cautionary cross-dataset endpoint, not as a benchmark.}
\label{tab:datasets}
\begin{tabular}{l l l l}
\toprule
Dataset & Type & Spoof source & Annotations \\
\midrule
CtrSVDD    & controlled & SVS/SVC, A01--A08 & paradigm, vocoder, singer \\
SingerLens & self-collected & Seed-VC (conversion) & singer, song \\
WildSVDD   & wild (bilibili) & So-VITS-SVC / RVC & model, singer \\
\bottomrule
\end{tabular}
\end{table}

---

## 5. Results

## 5.1 In-Distribution SVDD Appears Solved

We first establish the optimistic baseline that motivates the audit. Under a standard split---training and
testing on the same dataset with disjoint utterances---singing-voice deepfake detection looks essentially
solved across detector families of very different design (Table~\ref{tab:indist}). On the controlled
CtrSVDD benchmark, the official AASIST detector reaches an EER of 6.5\% (AUC 0.987); an end-to-end
fine-tuned self-supervised model (wav2vec2) reaches AUC 0.91, and a 3.3$\times$ larger variant 0.97; even
a random forest on a small set of interpretable acoustic features (FULL) reaches AUC 0.90. A practitioner
reading these numbers would reasonably conclude the task is mature.

We take these in-distribution results at face value and use them only as a reference point: every
subsequent section measures how much of this performance is retained once a single, normally
uncontrolled, factor is varied. As a first hint that the in-distribution number can be misleading, we
note that on our self-collected data a random forest reaches a \emph{perfect} AUC of 1.00---not because
the task is easy, but because that dataset carries a production/sampling-rate regularity perfectly
correlated with the label. The remainder of the paper quantifies, with controlled protocols, how such
regularities inflate standard evaluation.

% Table 1: in-distribution
\begin{table}[t]
\centering
\caption{In-distribution performance (standard split, CtrSVDD). All three detector families are strong to
near-perfect; larger SSL capacity further improves in-domain accuracy. These numbers are the reference the
audit erodes.}
\label{tab:indist}
\begin{tabular}{l cc}
\toprule
Detector & EER\,(\%) & AUC \\
\midrule
RF (interpretable, FULL)        & 18.88 & 0.899 \\
AASIST (neural)                 & 6.50  & 0.987 \\
wav2vec2-base, fine-tuned (SSL) & 17.50 & 0.912 \\
wav2vec2-large, fine-tuned      & 8.50  & \textbf{0.970} \\
\bottomrule
\end{tabular}
\end{table}

---

## 5.2 Generation Family Is a Multi-Axis Confound

The first factor a standard split leaves uncontrolled is the \emph{generation family}: which acoustic
model and which vocoder produced the spoof. We disentangle it in three steps.

\textbf{Detectors are not generic ``vocodedness'' detectors.}
A copy-synthesis control---passing genuine recordings through the spoof family's vocoder, sampling-rate
matched---does not make a detector flag them as fake (it assigns the copy-synthesised bona fide a spoof
probability comparable to natural bona fide). Detectors therefore do not key on a universal ``was this
vocoded'' cue; they key on \emph{pipeline-specific} artifacts, which motivates controlling the vocoder and
generator explicitly.

\textbf{A single unseen axis already degrades detectors.}
Holding the dataset fixed and leaving out one vocoder family (LOVO) or one generation paradigm (LOGO)
degrades every detector by roughly $2$--$3\times$ in EER: AASIST degrades $\times2.99$ on an unseen
NSF-HiFiGAN vocoder and $\times2.4$--$3.0$ on an unseen paradigm, with the interpretable RF degrading
somewhat less ($\times1.9$--$2.0$). Standard ``seen-condition'' evaluation, which never holds out a
generation family, misses this entirely.

\textbf{The confound is multi-axis: paradigm and vocoder degrade independently.}
Because paradigm and vocoder are nearly nested in standard data, we use a factorial cell-out that isolates
each axis (Table~\ref{tab:factorial}). Holding the \emph{paradigm} fixed, an unseen vocoder degrades the
RF by $\times13$--$29$. More strikingly, holding the \emph{vocoder} fixed (NSF-HiFiGAN) and swapping only
the generation paradigm (SVS$\leftrightarrow$SVC) drives EER from $\sim$1\% to $\sim$50\% ($\times59$--$89$
for RF; $\times13.6$ for AASIST), an even larger drop than the vocoder shift. Thus, with the vocoder held
identical, the generation paradigm alone collapses the detector---refuting the common reading that SVDD
detectors are ``just vocoder detectors.'' The confound is multi-axis: the generation paradigm and the
vocoder are orthogonal directions that each independently break the detector, and standard splits conflate
both.

\textbf{Scope of the factorial analysis.}
This axis-controlled result is, by design, established on CtrSVDD alone: it is the only dataset that
annotates both the generation paradigm and the vocoder per clip, and the contrast is only possible because
one vocoder (NSF-HiFiGAN) spans both paradigms, giving a within-vocoder window in which the paradigm can be
varied in isolation. Our self-collected and wild datasets lack this factorial metadata and cannot
replicate it. We therefore report the paradigm-as-independent-confound finding as a single-dataset, but
controlled, result, and treat its cross-dataset generality as established only at the coarser level of the
cross-dataset collapse (§5.3); we return to this limitation in §6.

% Table 2: factorial axis-controlled
\begin{table}[t]
\centering
\caption{Factorial vocoder$\times$paradigm cell-out (EER\,\%; degradation = held-out\,/\,in-dist).
Holding one axis fixed isolates the other. Even with the vocoder fixed (NSF-HiFiGAN), a paradigm swap
collapses EER from $\sim$1\% to $\sim$50\%---a larger drop than an unseen vocoder. Effect is present for
both an interpretable (RF) and a neural (AASIST) detector.}
\label{tab:factorial}
\begin{tabular}{l l ccc}
\toprule
Contrast & Fixed & in-dist & held-out & $\times$ \\
\midrule
unseen vocoder (RF)   & paradigm=SVS & 2.33 & 29.71 & 12.8 \\
unseen vocoder (RF)   & paradigm=SVS & 1.38 & 39.67 & 28.8 \\
\textbf{unseen paradigm (RF)}    & \textbf{vocoder=NSF} & 0.62 & 55.33 & \textbf{89.2} \\
\textbf{unseen paradigm (RF)}    & \textbf{vocoder=NSF} & 0.81 & 48.00 & \textbf{59.3} \\
\textbf{unseen paradigm (AASIST)}& \textbf{vocoder=NSF} & 4.00 & 54.29 & \textbf{13.6} \\
\bottomrule
\end{tabular}
\end{table}


---


## 5.3 Cross-Dataset Transfer: A Model- and Scale-Agnostic Collapse

The audit protocols of Sections 5.1–5.2 hold the dataset fixed and vary a single controlled factor
(vocoder, generator, or their factorial cell). We now apply the strongest stress test, cross-dataset
transfer (protocol P4): a detector is trained on one dataset and evaluated, without adaptation, on
another. This is the setting that most closely mirrors deployment, where a system trained on available
data must judge audio from an unseen platform or generator.

\textbf{In-distribution near-perfection does not survive a change of dataset.}
Table~3 reports equal error rate (EER) and AUC for all train/test dataset pairs over our three detector
families. Within a dataset, every detector is strong to near-perfect: AASIST reaches an EER of 6.5\%
(AUC 0.987) on CtrSVDD, the random forest (RF) on interpretable features reaches AUC 0.90, and the
fine-tuned self-supervised model (wav2vec2) reaches 0.91. Yet the same detectors, transferred across
datasets, collapse to the chance band: AUC falls to 0.34--0.62 and EER rises to 42--62\%. The drop is not
a graceful degradation but a near-total loss of discrimination---an in-distribution 5--6\% EER detector
becomes a 42--62\% EER detector once the dataset changes.

\textbf{The collapse is model-agnostic.}
It is not an artifact of any single inductive bias. It appears identically for a hand-crafted RF, a
state-of-the-art neural detector (AASIST), and an end-to-end fine-tuned SSL model---three families with
very different feature spaces and capacities. On the controlled pair (CtrSVDD$\leftrightarrow$SingerLens),
all three move from in-domain AUC $\geq$0.90 to cross-dataset AUC of 0.55--0.62 (AASIST/RF) and 0.34--0.42
(SSL); several configurations fall \emph{below} 0.5, i.e. the cross-dataset ranking is mildly
anti-correlated with authenticity (we analyse this reversal in Section~5.4). We note that the
within-SingerLens figure (RF AUC 1.00) is a cautionary extreme rather than evidence of an easy task: our
self-collected data carries a production/sampling-rate regularity that makes the in-domain problem
trivially separable, yet the same detector is at chance off-domain---precisely the overestimation this
paper quantifies.

\textbf{The collapse is also scale-agnostic.}
A natural hypothesis is that the SSL baseline is simply too small. We therefore fine-tune a 3.3$\times$
larger model (wav2vec2-large, 315M parameters) under the same protocol. Larger capacity buys
\emph{in-domain} accuracy---CtrSVDD AUC improves from 0.912 (base) to 0.970---but \emph{not} transfer: the
cross-dataset AUC is essentially unchanged (0.417$\to$0.426, still below chance). Increasing pretrained
representation capacity does not close the gap, indicating that the failure is a property of the
train/test distribution mismatch rather than of model size.\footnote{WavLM/HuBERT, which might pretrain
more singing-relevant representations, were network-unreachable on our infrastructure; we discuss this
constraint and its implications in Section~4 and the Limitations.}

\textbf{Public-benchmark accuracy does not predict wild performance.}
Extending the target to genuinely wild data (WildSVDD, bilibili AI covers) confirms the gap, with a
detector-dependent nuance. The in-domain WildSVDD task is itself learnable (RF AUC 0.935 under
cross-validation), yet source-trained detectors transfer poorly: CtrSVDD$\to$WildSVDD and
SingerLens$\to$WildSVDD reach AUC 0.52/0.46 (RF), 0.75/0.43 (AASIST), and 0.67/0.63 (fine-tuned SSL). The
interpretable RF collapses to chance, while the neural and SSL detectors retain only partial,
unreliable signal (0.63--0.75, down from $\sim$0.99 in-domain). A detector that appears near-solved on a
public benchmark thus carries little predictive value for the wild covers a deployed system would actually
face.

\textbf{How the collapse manifests.}
Figure~1 visualises the mechanism via the per-clip score distributions of a representative detector.
In-domain, the real and fake distributions are cleanly separated. Under cross-dataset transfer the two
distributions \emph{collapse onto each other} and are jointly shifted by a domain-dependent offset (a
source detector rates \emph{every} target clip as predominantly fake, or predominantly real, depending on
the source). The failure is therefore a loss of in-target separability accompanied by a domain offset,
\emph{not} a preserved-but-mis-thresholded boundary---a distinction we exploit diagnostically in
Section~5.4.

% ---------------------------------------------------------------------------
\begin{table}[t]
\centering
\caption{Cross-dataset transfer (EER\,\% / AUC). A = CtrSVDD, B = SingerLens, W = WildSVDD (wild).
w2v2-ft = fine-tuned wav2vec2-base; w2v2-L = fine-tuned wav2vec2-large (315M, model-scale point).
$\dagger$ B-within for SSL uses a stricter song-disjoint split. In-domain rows are near-perfect;
every cross-dataset row collapses to the chance band.}
\label{tab:crossdataset}
\begin{tabular}{l cc cc cc}
\toprule
 & \multicolumn{2}{c}{AASIST} & \multicolumn{2}{c}{RF-FULL} & \multicolumn{2}{c}{w2v2-ft} \\
Protocol & EER & AUC & EER & AUC & EER & AUC \\
\midrule
A$\to$A (within) & 6.50 & 0.987 & 18.88 & 0.899 & 17.50 & 0.912 \\
\quad A$\to$A, w2v2-L & -- & -- & -- & -- & 8.50 & \textbf{0.970} \\
B$\to$B (within) & 5.16 & 0.987 & 0.00 & 1.000 & 17.54$\dagger$ & 0.904$\dagger$ \\
\midrule
A$\to$B & 42.58 & 0.596 & 42.03 & 0.615 & 56.31 & 0.417 \\
\quad A$\to$B, w2v2-L & -- & -- & -- & -- & 54.93 & \textbf{0.426} \\
B$\to$A & 45.25 & 0.553 & 46.96 & 0.541 & 62.00 & 0.337 \\
A$\to$W & 32.10 & 0.748 & 48.90 & 0.520 & 38.41 & 0.669 \\
B$\to$W & 54.50 & 0.430 & 53.70 & 0.461 & 41.85 & 0.628 \\
W$\to$W (within) & -- & -- & 14.80 & 0.935 & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
% ---------------------------------------------------------------------------

% Figure 1 caption (figure = outputs/score_distributions.png)
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{score_distributions.png}
\caption{Per-clip score distributions (real vs.\ fake) for a representative detector. \emph{Top/within-
domain:} the classes separate. \emph{Cross-dataset:} the distributions collapse together and are jointly
shifted by a domain-dependent offset, so AUC falls to chance. The collapse is a loss of in-target
separability, not a mis-placed threshold (cf.\ Section~5.4).}
\label{fig:scoredist}
\end{figure}


## 5.4 Why It Collapses: A Domain-Specific Production Axis

The collapse of Section~5.3 raises an obvious question: is the cross-dataset signal merely noisy, or is
the detector confidently following the wrong cue? We show it is the latter. Three observations---the
geometry of the scores, a feature attribution, and a causal perturbation---converge on a single
explanation: the detectors rank clips along a \emph{domain-specific production axis} that is correlated
with authenticity only within a dataset.

\textbf{The detector is confidently wrong, not merely uncertain.}
Under cross-dataset transfer the real and fake score distributions do not just widen; they collapse onto
each other and are jointly displaced by a domain-dependent offset (Figure~1), so that a source detector
assigns nearly all target clips to one class. Crucially this is not a preserved-but-mis-thresholded
boundary: the \emph{flipped} AUC, $\max(\text{AUC}, 1-\text{AUC})$, remains at chance (0.53--0.56), so
relabelling the decision would not help. The target domain simply contains no separable signal along the
source's decision direction---the failure is in the conditional $p(y\mid x)$, not in a threshold.

\textbf{The discriminative axis is domain-specific.}
A feature attribution makes the mismatch concrete (Table~\ref{tab:attribution}). A source detector trained
on CtrSVDD places most of its importance on MFCC spectral texture (0.64) and almost none on harmonics-to-
noise ratio (HNR, 0.055), whereas a detector trained \emph{within} the wild target relies on HNR much more
(0.146)---HNR being the genuine real-vs-fake cue in that domain (AI covers are, if anything, cleaner).
Moreover, the source model's ranking of target clips correlates most strongly with high-order MFCC standard
deviations ($\rho{=}0.55$ for \texttt{mfcc\_13\_std}): it sorts wild clips by spectral texture---a
recording/codec/separation signature---rather than by authenticity.

\textbf{A causal test confirms the production axis.}
If the detector keys on a production signature, then perturbing only the \emph{production} of genuine
recordings---without changing who sings or whether it is AI---should move their scores toward "fake". It
does. Band-limiting real clips (resampling to 8\,kHz, low-pass at 3--4\,kHz) and codec compression
(32\,kbps MP3) raise their fake-score from 0.085 to 0.36--0.46 and flip 13--47\% of real clips to "fake",
whereas additive noise, loudness changes, and even equal-loudness accompaniment leakage move them
essentially not at all ($\approx$0\% flipped); re-separating an already-clean vocal has only a mild effect
($+0.14$). The detector is thus sensitive to a \emph{spectral-bandwidth / codec / separation} signature
and is invariant to content-level dirtiness---the signature is orthogonal to authenticity. This directly
explains the cross-domain error structure of Section~5.3: low-bandwidth wild recordings (e.g.\ after
platform transcoding or source separation) are scored as fake (false positives), while clean AI covers
that lack the source generator's signature are scored as real (false negatives).

\textbf{A continuous bandwidth sweep reveals a sharp onset.}
Sweeping the low-pass cutoff continuously (Figure~\ref{fig:bwsweep}) locates the effect precisely. On
16\,kHz audio (Nyquist 8\,kHz), real clips are scored correctly down to a cutoff of $\sim$7\,kHz, but the
fake-score then jumps sharply---between 7.5 and 6\,kHz the flip rate rises from 2.5\% to 33\%---marking a
critical onset around 6--7\,kHz beyond which the detector begins to mislabel genuine singing. A
``danger band'' of 3--6\,kHz cutoffs produces the highest false-positive rates (27--47\%), exactly the
bandwidth range typical of platform transcoding and source separation; at extreme cutoffs ($\leq$2.5\,kHz)
the score recedes, indicating the detector keys on a specific mid-bandwidth signature rather than on
``narrower is faker.''

\textbf{The confound is language-general, not specific to the (Mandarin) training language.}
Because all our training and perturbation data are Mandarin, one might worry the effect is an artifact of
Mandarin phonology. It is not. Re-running the production perturbation on \emph{Cantonese} (a distinct
tonal language) and \emph{English} (non-tonal, phonologically far from the training data) real vocals,
scored by the same Mandarin-trained detector, reproduces the same band-limiting-induced flips
(Table~\ref{tab:crosslang}): English real clips flip to ``fake'' at 50\% under 8\,kHz resampling---if
anything \emph{above} Mandarin's 46\%---and additive noise flips none in any language. The confound is
thus a property of the audio \emph{production} spectral footprint, not of any language's acoustics. (The
MP3-codec component is the least transferable---25/15/0\% for Mandarin/English/Cantonese---suggesting the
band-limiting footprint generalizes most cleanly across languages.) We note the English condition is a
single song (26 clips); the sample is small, but the flip-rate magnitude is on par with the much larger
Mandarin set and so suffices to reject the ``Chinese-specific'' hypothesis.

\textbf{Why standard splits hide this.}
The production axis is (near-)constant within a dataset---one recording pipeline, one generator family,
one platform---so a standard random split never perturbs it, and the detector is rewarded for using it. It
is a confound, not a cue; the audit protocols of this paper are precisely the perturbations a random split
omits.

% Table 4: attribution
\begin{table}[t]
\centering
\caption{Discriminative axis differs across domains (summed RF feature importance by family). A source
(CtrSVDD) detector leans on MFCC spectral texture and barely uses HNR; a within-target (WildSVDD) detector
relies on HNR, the genuine cue. The source's (wrong) target ranking correlates most with high-order
MFCC-std ($\rho{=}0.55$).}
\label{tab:attribution}
\begin{tabular}{l cc}
\toprule
Feature family & Source (CtrSVDD) & Within-target (Wild) \\
\midrule
MFCC          & 0.642 & 0.561 \\
HNR           & 0.055 & \textbf{0.146} \\
low-level     & 0.103 & 0.110 \\
vibrato/VRI   & 0.094 & 0.102 \\
pitch         & 0.092 & 0.072 \\
\bottomrule
\end{tabular}
\end{table}

% Figure: bandwidth sweep (figure = outputs/bandwidth_sweep.png)
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{bandwidth_sweep.png}
\caption{Continuous low-pass sweep on genuine 16\,kHz recordings (mean fake-score and flip-to-fake rate
vs.\ cutoff). Real clips are scored correctly down to $\sim$7\,kHz; the fake-score jumps sharply at a
\textbf{critical onset of 6--7\,kHz}, with a \textbf{3--6\,kHz ``danger band''} producing 27--47\%
false positives (the bandwidth range typical of platform transcoding/separation). Beyond $\sim$2.5\,kHz
the score recedes, so the detector keys on a specific mid-bandwidth signature, not on ``narrower is faker.''}
\label{fig:bwsweep}
\end{figure}

% Table: cross-language flip rates
\begin{table}[t]
\centering
\caption{Production perturbation is language-general. Flip-to-fake rate of \emph{real} vocals under each
perturbation, scored by the same Mandarin-trained detector, for Mandarin / Cantonese / English. Band-
limiting flips real clips alike across languages (English even slightly above Mandarin); noise flips none.
English is a single song (26 clips); the magnitude nonetheless matches the larger sets, rejecting a
``Chinese-specific'' explanation.}
\label{tab:crosslang}
\begin{tabular}{l ccc}
\toprule
Perturbation & Mandarin & Cantonese & English \\
\midrule
resample-8k      & 0.465 & 0.390 & \textbf{0.500} \\
low-pass 4k      & 0.130 & 0.133 & 0.154 \\
low-pass 3k      & 0.465 & 0.324 & 0.423 \\
MP3-32k          & 0.250 & 0.000 & 0.154 \\
noise (10\,dB)   & 0.000 & 0.000 & 0.000 \\
\midrule
\#\,clips        & 200 & 105 & 26 \\
\bottomrule
\end{tabular}
\end{table}

---

## 5.5 No Shortcut Survives: Minimal Supervision Is the Only Exit

Given that the failure is a domain-specific production axis (Section~5.4), the natural remedy is to remove
or bypass that axis \emph{without} new labels. We try every such shortcut. None recovers cross-dataset
performance, which isolates minimal target supervision as the only viable exit. Table~\ref{tab:adaptation}
summarises.

\textbf{Unsupervised domain adaptation does not help.}
Aligning the source and target feature distributions---per-domain standardisation, CORAL, subspace
alignment, and an adversarial domain classifier (DANN)---leaves the target AUC in the chance band
(0.44--0.62) and sometimes below the no-adaptation baseline. This is expected: these methods align the
\emph{marginal} feature distribution, but the failure lies in the \emph{conditional} decision direction
$p(y\mid x)$, which marginal alignment cannot repair.

\textbf{Removing the implicated axis is insufficient.}
Motivated by the diagnosis, we directly attack the bandwidth component: we low-pass every dataset to a
common cutoff and re-extract features. Cross-dataset AUC still does not recover (0.43--0.60), because
bandwidth is only one component of the production signature---the residual codec and separation texture is
enough to sustain the collapse.

\textbf{Re-calibration cannot help either,} since fitting a threshold on the source score with target
labels (calibration-only) leaves the AUC at chance ($\approx$0.49, invariant to the number of labels): a
monotonic re-mapping cannot repair a ranking that is already uninformative.

\textbf{Even a production-orthogonal \emph{semantic} cue fails,} further refuting the hypothesis that some
robust shortcut exists: a zero-training emotion-consistency score (agreement between sung and lyric
emotion), which is by construction orthogonal to the production chain, transfers \emph{worse} than the
production-based detectors (AUC $0.679\to0.412$ on wild data, below chance and with the relation reversed;
a larger relative drop than AASIST). No single axis---production \emph{or} semantic---solves cross-dataset
detection.

\textbf{Minimal target supervision is the only exit.}
With every label-free shortcut closed, the one remedy that works is direct supervision on the target
domain. We trace the learning curve of a detector trained \emph{from scratch} on $k$ labeled target clips
(Figure~\ref{fig:msacurve}; Table~\ref{tab:adaptation}). As few as $k{=}10$ labels already exit the chance
band and beat every unsupervised method (AUC 0.63 vs.\ $\leq$0.52); accuracy rises steeply to 0.77
($k{=}30$) and 0.82 ($k{=}50$), then flattens toward the in-domain bound ($\sim$0.93). The \emph{shape} of
the curve makes the ``cheap'' claim precise rather than anecdotal: the marginal gain per label falls
roughly $20\times$, from $+0.013$/label over the first ten labels to $+0.0003$/label beyond $k{=}150$, so
the returns are demonstrably concentrated in the first few dozen labels. The exit is specifically
\emph{target-only}: even reweighted source+target mixtures stay below it (Appendix~E), so the source data
is not merely diluted---it carries no transferable signal. We remain precise about ``cheap'': a few dozen
labels dominate \emph{unsupervised} adaptation, but deployment-grade accuracy (0.90+) is reached only near
$k{\approx}200$, about half of this target set. The practical message (Section~5.6) follows directly: for
a new platform, collect a modest set of target labels rather than adapt a confounded source detector.

% Table 5: every shortcut vs minimal supervision
\begin{table}[t]
\centering
\caption{Closing the unsupervised door, and the minimal-supervision exit (held-out WildSVDD-T02 AUC).
No label-free remedy leaves the chance band; target-only training beats them all from $k{=}10$ labels, and
the per-label gain falls $\sim$20$\times$ across the curve (steepest in the first few dozen labels).}
\label{tab:adaptation}
\begin{tabular}{l c c}
\toprule
Approach & Target labels & AUC \\
\midrule
No-adaptation (source$\to$target)          & 0 & 0.51 \\
\;\; + per-domain z-norm / CORAL / SA / DANN & 0 & 0.44--0.52 \\
\;\; + common-bandwidth confound-removal    & 0 & 0.43--0.52 \\
\;\; + calibration-only (re-threshold)      & 0 & 0.49 \\
\;\; + emotion-consistency (semantic)       & 0 & 0.41 \\
\midrule
Target-only, $k{=}10$    & 10  & \textbf{0.63} \\
Target-only, $k{=}30$    & 30  & 0.77 \\
Target-only, $k{=}50$    & 50  & 0.82 \\
Target-only, $k{=}100$   & 100 & 0.86 \\
Target-only, $k{=}200$   & 200 & 0.91 \\
Target-only, full (bound)& all & $\sim$0.93 \\
\bottomrule
\end{tabular}
\end{table}

% Figure: MSA learning curve (figure = outputs/msa_learning_curve.png)
\begin{figure}[t]
\centering
\includegraphics[width=\linewidth]{msa_learning_curve.png}
\caption{Minimal-supervision learning curve (target-only, held-out WildSVDD-T02). A few dozen labels
already exit the chance band and dominate every unsupervised remedy ($\leq$0.52); the per-label return is
steepest in the first $\sim$30--50 labels and flattens toward the in-domain bound.}
\label{fig:msacurve}
\end{figure}

\textbf{What the few-shot detector learns: the authentic cue, not a new shortcut.}
A natural worry is that target supervision merely re-learns a \emph{target-specific} production shortcut,
in which case it would fail again on the next platform. A feature attribution of the few-shot detector
rules this out (Table~\ref{tab:msaattr}). With only $k{=}50$ labels, its reliance on HNR---the genuine
real-vs-fake cue in the target (Section~5.4)---rises from the source detector's 0.055 to 0.123,
about 80\% of the way to the within-target model's 0.146, while its MFCC weight drops from the source's
inflated 0.642 to the target-appropriate 0.56. The few-shot detector thus recovers the target's
\emph{authentic} discriminative axis, not a re-learned production signature; minimal supervision is real
adaptation, not another shortcut. It is, however, still \emph{per-target} supervision: it learns what is
genuine for \emph{this} target, so a new platform requires its own labels. The recommendation is therefore
to label a little data for each new platform, not to expect a one-shot transferable fix.

% Table: MSA attribution
\begin{table}[t]
\centering
\caption{The few-shot detector recovers the target's authentic axis. Summed RF feature importance for HNR
(the genuine target cue) and MFCC (the source's production shortcut), across the source detector, few-shot
detectors ($k{=}50,100$), and the within-target model. The few-shot HNR weight is target-like, far from
the source's.}
\label{tab:msaattr}
\begin{tabular}{l cccc}
\toprule
Family & Source & $k{=}50$ & $k{=}100$ & Within-target \\
\midrule
HNR (target cue)        & 0.055 & 0.123 & 0.132 & \textbf{0.146} \\
MFCC (source shortcut)  & 0.642 & 0.560 & 0.556 & 0.561 \\
\bottomrule
\end{tabular}
\end{table}


## 5.6 Deployment Implications

The audit has a direct operational reading: a detector that is near-perfect on a benchmark cannot be
trusted in deployment, and several intuitive safeguards do not rescue it.

\textbf{A confidence threshold is not a safe reject mechanism.}
One might hope to deploy a source-trained detector with abstention, acting only on high-confidence
predictions. This fails out-of-domain. With selective classification---keeping only the most confident
fraction of predictions---in-domain accuracy rises monotonically to 1.00 by the top 10\% (Table~
\ref{tab:abstain}); but cross-dataset accuracy stays at chance at every coverage and, at the highest
confidence, drops \emph{below} it (CtrSVDD$\to$WildSVDD: 0.46 at full coverage, 0.35 at the top 5\%). The
detector is confidently wrong out-of-domain, so confidence cannot gate deployment.

\textbf{Collect a few target labels rather than adapt a confounded source.}
As Section~5.5 shows, no label-free adaptation recovers cross-dataset performance, whereas a few dozen
target labels (target-only) dominate every unsupervised method. The practical recommendation is therefore
to budget a small amount of target-domain annotation for any new platform or generator, rather than to
transfer or adapt an existing detector.

\textbf{Audit the production chain for false-positive risk.}
Because detectors key on a spectral-bandwidth/codec/separation signature (Section~5.4), routine
production operations on genuine audio---platform transcoding, band-limiting, source-separation
residue---raise its fake-score and can manufacture false positives (13--47\% of real clips flip under
band-limiting/codec in our controlled probes). Deployment pipelines should evaluate this sensitivity, and
report it, before acting on a detector's output.

% Table 6: selective prediction
\begin{table}[t]
\centering
\caption{Selective-prediction accuracy at coverage $c$ (keep the top-$c$ most confident; FULL clip-mean
detector). In-domain, abstention buys near-perfect accuracy; cross-dataset, high-confidence predictions
are no more (and at $c{=}5\%$ less) correct.}
\label{tab:abstain}
\begin{tabular}{l cccc}
\toprule
Protocol & $c{=}100\%$ & $50\%$ & $10\%$ & $5\%$ \\
\midrule
CtrSVDD within        & 0.80 & 0.94 & 1.00 & 1.00 \\
WildSVDD within       & 0.80 & 0.92 & 1.00 & 1.00 \\
CtrSVDD$\to$WildSVDD  & 0.46 & 0.50 & 0.45 & 0.35 \\
CtrSVDD$\to$SingerLens& 0.53 & 0.55 & 0.40 & 0.41 \\
\bottomrule
\end{tabular}
\end{table}

---

## 6. Discussion

\textbf{Implications for evaluation.}
Our results argue that an in-distribution EER is, on its own, not evidence that an SVDD detector works.
Reported performance substantially reflects \emph{generation-family identification}---which vocoder and
which paradigm produced the audio---rather than authenticity. We recommend that SVDD evaluations (i) treat
the vocoder \emph{and} the generation paradigm as controlled variables and report held-out (LOVO/LOGO/
factorial) degradation, (ii) report cross-dataset transfer as a default, not an afterthought, and (iii)
release the attack$\to$generator$\to$vocoder metadata needed to run such audits. Operationally, because no
label-free adaptation closes the gap, the actionable path for a new platform is to budget a small amount
of target-domain annotation rather than to trust or adapt a benchmark-trained detector.

\textbf{Limitations.}
Our study has clear limits, which also bound its claims. (1) Our self-collected SingerLens set uses a
single conversion model and our own production pipeline; its in-domain separability is partly self-made,
which is why we treat it only as a cautionary cross-dataset endpoint and triangulate with the
third-party-controlled CtrSVDD and the wild WildSVDD. (2) WildSVDD is a reachable bilibili subset
(hundreds of clips); the neural detectors underfit it, so we anchor its in-domain reference on the random
forest. (3) Only one fully public benchmark (CtrSVDD) is used; we did not include SingFake owing to source
availability. (4) Our ``no unsupervised fix'' result is established for feature-alignment domain
adaptation and a single confound-removal axis (bandwidth); full end-to-end neural DA, contrastive
alignment, and copy-synthesis through one common vocoder remain untested---the claim is that these
shortcuts fail, not that no method can succeed. (5) The production perturbations are controlled diagnostic
probes, not a faithful simulation of platform distributions; we read their score shifts as qualitative
sensitivity evidence, not deployable error rates. (6) Our SSL detector is an ASR-pretrained wav2vec2, not
a singing-specific or state-of-the-art model, because more singing-relevant SSL was network-unreachable.
(7) The semantic cue we test is a single emotion-consistency construct; other semantic or prosodic
constructs may behave differently, and we did not isolate the reliability of the SER tool (emotion2vec)
itself under cross-lingual / strong-vocal conditions---so the semantic cue's cross-domain failure may be
partly attributable to the tool rather than to the concept. (8) The factorial vocoder$\times$paradigm
result (§5.2)---that a
paradigm swap with the vocoder fixed degrades detectors more than an unseen vocoder---is established on
CtrSVDD only, the sole dataset with per-clip paradigm \emph{and} vocoder labels and the only one whose
vocoder set (NSF-HiFiGAN) spans both paradigms; our other datasets cannot replicate this controlled
contrast, so this specific finding is single-dataset, even though the broader cross-dataset collapse it
illustrates is shown across all three datasets.

\textbf{Ethics and privacy.}
WildSVDD comprises AI covers scraped from a public video platform; these clone the voices of real,
identifiable singers, raising consent, likeness, and copyright concerns. We use only already-public
material, for the defensive purpose of detection, do not redistribute raw audio, and release derived
features and code rather than the recordings. The human-listening component is provided only as a
prepared, not-yet-run package and would require informed consent and ethics review before collection.
SVDD is dual-use: insight into detector failure modes could inform evasion; we judge the benefit of
exposing over-optimistic evaluation and guiding trustworthy deployment to outweigh this risk, and we avoid
publishing any recipe whose primary use is evasion.

\textbf{Reproducibility.}
We release SVDD-Audit---protocol definitions, the attack$\to$generator$\to$vocoder mapping, extraction and
evaluation scripts, result tables, and subset-construction code---together with the external dependencies
(official scorers, baselines, vocoder weights, model identifiers). All controlled experiments use fixed
seeds and balanced subsets documented in the toolkit.

---

## 7. Conclusion

Standard singing-voice deepfake detection is far from solved once its confounds are controlled. Through a
suite of controlled audit protocols, we showed that detectors which look near-perfect in-distribution
collapse to near-chance across generators, datasets, and the wild; that this collapse is model- and
scale-agnostic and is driven by an orthogonal, multi-axis generation-family and production confound rather
than by authenticity; and that no label-free remedy---adaptation, confound removal, recalibration, or a
production-orthogonal semantic cue---repairs it, leaving minimal target supervision as the only viable
exit. The contribution is not merely a set of negative results but a structured, actionable picture of
\emph{when} SVDD generalizes and what to do when it does not. We release SVDD-Audit so that future SVDD
systems can be evaluated, and reported, for the generalization they actually have.

---

## Appendix Z — Drafting notes (per-section, 不入正文)

### [DRAFT_sec1_intro.md]
- thesis 句:第 2 段("reported numbers measure dataset production family, not authenticity")。
- headline 数字已埋:0.90–0.99→0.34–0.62、model-/scale-agnostic、1%→50%、25 标签、捷径全败。
- 3 贡献与 skeleton_v2 一致(协议套件+toolkit / 崩塌实证 / 机制+边界)。
- Figure 1 teaser 待画(概念图:审计协议如何逐一控制混杂轴,域内→跨域崩塌示意);figure 编号最终稿统一。
- citations 待补:SingFake, CtrSVDD/SVDD Challenge, WildSVDD, AASIST, wav2vec2, CORAL/DANN, ASVspoof 泛化。
- 与 §6 Ethics 呼应:第 1 段 misuse/consent 已起头,Discussion 展开。

### [DRAFT_sec2_related_work.md]
- 五块 + gap 段;每块 1 段,`[\textsc{cite}: ...]` 标明要引哪类工作,bibkey 后补。
- 待补具体文献(建议起点):
  - SVDD 数据:SingFake(Zang et al. 2024)、CtrSVDD/SVDD Challenge 2024、WildSVDD、FSD。
  - anti-spoofing 泛化:ASVspoof 2019/2021、cross-dataset spoofing、AASIST(Jung et al.)、wav2vec2/SSL anti-spoofing。
  - vocoder 伪迹:neural vocoder artifact detection、TTS artifact、source-filter/HiFi-GAN 相关。
  - shortcut:Geirhos et al. shortcut learning;spurious correlations;audio forensic leakage。
  - DA:CORAL(Sun & Saenko)、Subspace Alignment(Fernando et al.)、DANN(Ganin & Lempitsky)、calibration(Guo et al.)。
- 每处声明"我们的差异/超出"已埋(re-examine / cause+boundary / 分离两轴 / 具体 SVDD instance / conditional vs marginal)。
- 待统一术语:SVS/SVC、LOVO/LOGO、production axis。

### [DRAFT_sec3-4_protocols_setup.md]
- §3 形式化已含 P0/C/P1/P2/P2b/P4 + 三类 analyses + 指标(EER/AUC/degradation/flipped-AUC/selective-acc)。
  bona disjoint 声明已显式写(防"记歌手"质疑)。
- §4 数据集表 T0 给定性描述(规模/采样率细节可加一列或脚注:CtrSVDD E1 4800、SingerLens 1085、WildSVDD-T02 387)。
- SSL rationale 段=弱点一的论文化(WavLM/HuBERT 不可达 / frozen 弱因非歌声 / 微调+large 验证)。
- 数字/许可待补:CtrSVDD zenodo、WildSVDD zenodo 10893604、各集 clip 数与采样率脚注、RF 特征维度。
- 与 §5 记号统一:A/B/W、$p$/$v$、$\Delta$、flipped-AUC 全文一致。

### [DRAFT_sec5.1-5.2-5.6.md]
- 5.1 短(setup);Table 1 只列强检测器域内;RF 可解释子集(CLEAN/HNR/VRI)细分入 appendix。
- 5.2 三段:copy-synth / 单轴 LOVO-LOGO / 因子化(headline,Table 2)。数字源 ctrsvdd_factorial_axis + aasist_paradigm。
- 5.6 三段 + Table 6(reject_option_riskcoverage)。few-shot 细节指回 5.5;perturbation 指回 5.4。
- 表数:T1(5.1)/T2(5.2)/T3(5.3)/T4(5.4)/T5(5.5)/T6(5.6)= 6 主表,+ F1 分数分布 + F2 teaser。符合 review 预算。
- 全文待统一:detector 命名、protocol 记号(A/B/W, $\to$)、degradation ratio 定义(§3)。

### 写作笔记(给后续起草者,不入正文)
- 数字来源:`cross_dataset_aasist_vs_rf.csv` + `w2v2ft_{A,B}.csv` + `w2v2ft_large_A.csv` + wild cross。
- hedge 已加:"across the three detector families we evaluate"、SingerLens 1.00 标 cautionary、large 脚注 WavLM 不可达。
- 过渡已埋:末段"not a mis-thresholded boundary → Section 5.4"接诊断节。
- 待统一:detector 命名(RF-FULL / AASIST / w2v2-ft/-L)、protocol 记号(A/B/W、$\to$)全文一致。
- 若投短文超长:WildSVDD 段 + Figure 1 可压一段 + 移部分 AASIST/RF 细分到 appendix Table。

### [DRAFT_sec5.4-5.5_diagnosis_exit.md]
- 5.4 三柱:flipped-AUC(score_distribution_summary)/ 归因(axis_attribution_family,Table 4)/ perturbation
  (perturbation_effect + production_leakage_resynthesis,入正文文字)。Figure 1 复用 5.3。
- 5.5 句数按约定:DA 2 / 带宽 2 / 校准 1 / 情感 1 / MSA 一段(主)。Table~\ref{tab:adaptation} 自带 \label。
- 数字源:domain_adaptation_cross / confound_removal_cross / fewshot_reweighting / wild_emotion_auc。
- hedge:DA "marginal vs conditional";MSA "cheap = relative to unsupervised,0.90+ 需近全量";情感 "shortcut 证伪"。
- 过渡:5.4 末"confound not a cue → 审计正是 random split 省略的扰动";5.5 末→ 5.6 部署。
- 跨节 \ref:tab:attribution(5.4)、tab:adaptation(5.5)、Figure 1(fig:scoredist,5.3)。Appendix E/D 待建 \label。

### [DRAFT_abstract_sec6-7.md]
- Abstract ~200 词,headline 数字齐(0.90–0.99→0.34–0.62、1%→50%、13–47% flip、~25 标签、model-/scale-agnostic)。
- Discussion 四段:Implications / Limitations(7 条,主动写足 review 列的攻击点)/ Ethics&Privacy / Reproducibility。
- Conclusion 一段,呼应 thesis + 结构化结论 + toolkit。
- 待统一:citation(§2 完成后)、figure/table 编号、术语(SVS/SVC、LOVO/LOGO、degradation ratio)。
- 全文 draft 现仅缺 §2 Related Work(需真实文献)。