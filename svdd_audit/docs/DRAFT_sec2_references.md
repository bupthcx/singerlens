# §2 References — bibkey → 文献映射(分置信度)

> §2 正文已用 `\cite{key}` 引用。下表给出每个 key 对应的文献。**分三档置信度**:
> ✅ 高(经典方法学,可直接用)/ 🟡 中(知名但请核对确切 venue/年份)/ ⚠ 待核实/待选(近期 SVDD 专属或类别占位,
> 我**不杜撰**确切作者/标题,请你据真实文献补全)。

## ✅ 高置信度(基础方法学,可直接采用 bibtex)

- **sun2016coral** — B. Sun, J. Feng, K. Saenko. *Return of Frustratingly Easy Domain Adaptation.* AAAI 2016.
- **fernando2013subspace** — B. Fernando, A. Habrard, M. Sebban, T. Tuytelaars. *Unsupervised Visual Domain
  Adaptation Using Subspace Alignment.* ICCV 2013.
- **ganin2015dann** — Y. Ganin, V. Lempitsky. *Unsupervised Domain Adaptation by Backpropagation.* ICML 2015.
  (或引 Ganin et al., *Domain-Adversarial Training of Neural Networks*, JMLR 2016。)
- **geirhos2020shortcut** — R. Geirhos, J. Jacobsen, C. Michaelis, R. Zemel, W. Brendel, M. Bethge,
  F. Wichmann. *Shortcut Learning in Deep Neural Networks.* Nature Machine Intelligence 2020.
- **guo2017calibration** — C. Guo, G. Pleiss, Y. Sun, K. Q. Weinberger. *On Calibration of Modern Neural
  Networks.* ICML 2017.

(§4 另用到、同属高置信:**baevski2020wav2vec2** — Baevski et al., *wav2vec 2.0*, NeurIPS 2020;
**jung2022aasist** — Jung et al., *AASIST*, ICASSP 2022。)

## 🟡 中置信度(知名,请核对确切引用)

- **asvspoof2019** — ASVspoof 2019 数据库/挑战。候选:Todisco et al., *ASVspoof 2019...*, Interspeech 2019;
  或 X. Wang et al., *ASVspoof 2019: a large-scale public database...*, Computer Speech & Language 2020。
- **asvspoof2021** — ASVspoof 2021。候选:Yamagishi et al.(ASVspoof 2021 workshop),
  或 Liu et al., *ASVspoof 2021...*, IEEE/ACM TASLP 2023。

## ⚠ 待核实 / 待选(近期 SVDD 专属 + 类别占位,请据真实文献补全)

**SVDD 数据集与挑战(2024,务必核对作者/会议/年份):**
- **singfake** — "SingFake: Singing Voice Deepfake Detection"(疑 ICASSP 2024,作者待核)。
- **ctrsvdd** / **svddchallenge2024** — "CtrSVDD" / "SVDD Challenge 2024"(疑 Interspeech 2024,作者待核)。
- **wildsvdd** — "WildSVDD"(疑 SingFake 作者扩展;数据 zenodo 10893604,文献待核)。
- **fsd** / **musicdeepfake** — FSD(FakeSong)/ 音乐深伪数据集(请选一篇代表作)。

**类别占位(请各选 1–2 篇代表作填入):**
- **crossdataset_spoof** — 语音反欺骗跨库泛化研究(代表作待选)。
- **robust_spoof** — 鲁棒/持续学习反欺骗(代表作待选)。
- **vocoder_artifact** — 神经声码器指纹/伪迹检测(代表作待选)。
- **tts_artifact** — TTS 合成伪迹检测(代表作待选)。
- **dataset_bias** — 虚假相关/数据集偏置(可选 Torralba & Efros, *Unbiased Look at Dataset Bias*, CVPR 2011)。
- **audio_leakage** — 音频取证中信道/编解码/静音线索泄露(代表作待选)。
- **fewshot_da** — 少样本域适应/标定式适应(代表作待选)。

## 使用说明
- ✅ 档可直接写进 .bib;🟡 档核对后写入;⚠ 档需你据真实文献补全(我未杜撰具体条目)。
- 若某 ⚠ key 暂无合适文献,可在 §2 删去该 `\cite{key}` 或合并到相邻引用,避免空引用。
- 全文除本 §2 外无其它 [cite] 占位;Figure teaser(fig:teaser)仍待画。
