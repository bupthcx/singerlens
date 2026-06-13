# DRAFT — §5.4 Diagnosis + §5.5 Exit (paper prose, v1)

> 接 §5.3。诊断→修复→出口的递进主干。同 5.3 文风(paragraph-lead 加粗 + LaTeX 表)。
> 5.4 复用 Figure 1(不新增图),perturbation 入正文文字;5.5 自带 Table~\ref{tab:adaptation}。

---

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
domain. A detector trained \emph{from scratch} on $k$ labeled target clips beats every unsupervised method
from as few as $k{=}25$ (AUC 0.72 vs.\ $\leq$0.52), and improves to 0.80 ($k{=}50$) and 0.85 ($k{=}100$)
toward the in-domain bound ($\sim$0.93), with the steepest returns in the first few dozen labels
(Table~\ref{tab:adaptation}). The exit is specifically \emph{target-only}: even reweighted source+target
mixtures stay below it (Appendix~E), so the source data is not merely diluted---it carries no transferable
signal. We are deliberate about the scope of "cheap": a few dozen labels dominate \emph{unsupervised}
adaptation, but reaching deployment-grade accuracy (0.90+) still requires near-complete target labelling.
The practical message (Section~5.6) follows directly: for a new platform, collect a small set of target
labels rather than adapt a confounded source detector.

% Table 5: every shortcut vs minimal supervision
\begin{table}[t]
\centering
\caption{Closing the unsupervised door, and the minimal-supervision exit (held-out WildSVDD AUC).
No label-free remedy leaves the chance band; target-only training beats them all from $k{=}25$ labels.}
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
Target-only, $k{=}25$                       & 25  & \textbf{0.72} \\
Target-only, $k{=}50$                       & 50  & 0.80 \\
Target-only, $k{=}100$                      & 100 & 0.85 \\
Target-only, full (in-domain bound)         & all & $\sim$0.93 \\
\bottomrule
\end{tabular}
\end{table}

---

### 写作笔记(不入正文)
- 5.4 三柱:flipped-AUC(score_distribution_summary)/ 归因(axis_attribution_family,Table 4)/ perturbation
  (perturbation_effect + production_leakage_resynthesis,入正文文字)。Figure 1 复用 5.3。
- 5.5 句数按约定:DA 2 / 带宽 2 / 校准 1 / 情感 1 / MSA 一段(主)。Table~\ref{tab:adaptation} 自带 \label。
- 数字源:domain_adaptation_cross / confound_removal_cross / fewshot_reweighting / wild_emotion_auc。
- hedge:DA "marginal vs conditional";MSA "cheap = relative to unsupervised,0.90+ 需近全量";情感 "shortcut 证伪"。
- 过渡:5.4 末"confound not a cue → 审计正是 random split 省略的扰动";5.5 末→ 5.6 部署。
- 跨节 \ref:tab:attribution(5.4)、tab:adaptation(5.5)、Figure 1(fig:scoredist,5.3)。Appendix E/D 待建 \label。
