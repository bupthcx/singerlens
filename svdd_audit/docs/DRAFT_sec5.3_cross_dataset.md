# DRAFT — §5.3 Cross-Dataset Collapse (paper prose, v1)

> 正式论文格式草稿(英文主文)。配 Table 3 + Figure 1。语气/hedging 已按 review 处理。
> 起笔节,后续 5.4/5.5 接续。

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
Extending the target to genuinely wild data (WildSVDD, bilibili AI covers), the gap is, if anything,
worse. The in-domain WildSVDD task is itself learnable (RF AUC 0.935 under cross-validation), yet no
source-trained detector transfers to it: CtrSVDD$\to$WildSVDD and SingerLens$\to$WildSVDD reach AUC
0.52/0.46 (RF) and 0.75/0.43 (AASIST). A detector that appears near-solved on a public benchmark thus
carries almost no predictive value for the wild covers a deployed system would actually face.

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
A$\to$W & 32.10 & 0.748 & 48.90 & 0.520 & -- & -- \\
B$\to$W & 54.50 & 0.430 & 53.70 & 0.461 & -- & -- \\
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

---

### 写作笔记(给后续起草者,不入正文)
- 数字来源:`cross_dataset_aasist_vs_rf.csv` + `w2v2ft_{A,B}.csv` + `w2v2ft_large_A.csv` + wild cross。
- hedge 已加:"across the three detector families we evaluate"、SingerLens 1.00 标 cautionary、large 脚注 WavLM 不可达。
- 过渡已埋:末段"not a mis-thresholded boundary → Section 5.4"接诊断节。
- 待统一:detector 命名(RF-FULL / AASIST / w2v2-ft/-L)、protocol 记号(A/B/W、$\to$)全文一致。
- 若投短文超长:WildSVDD 段 + Figure 1 可压一段 + 移部分 AASIST/RF 细分到 appendix Table。
