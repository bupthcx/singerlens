# Real-only Production Perturbation (FINDINGS_production_perturbation.md)

## Purpose
因果验证 P2 的"制作干净度轴":制作噪声/平台压缩/带宽变化是否会诱发**真实样本**的假阳性。
**只扰动 real，不引入任何 fake**——若真唱经制作类扰动后 fake_score 上升，则检测器是在沿
"制作/频谱足迹"轴打分而非真伪,直接解释 cross-domain 的 FP。

## Setup
- 检测器:SingerLens FULL RF(原始 real vs fake,n=1085)。
- 取 200 个真唱,逐个施加 8 种制作类扰动,re-extract FULL 特征(同 extract_clip_features 口径)后打分。
  扰动均**不改变"谁在唱/是否 AI"**:重采样(16k→8k→16k)、低通(3k/4k)、加噪(SNR 20/10dB)、
  减增益(×0.3)、MP3 编解码(32kbps)、合成混响。脚本 `scripts/production_perturbation.py`。
- 指标:mean_fake_score、delta_vs_original、high_score_ratio(=flip→fake, P(fake)>0.5)。

## Results
原始真唱基线 mean P(fake)=0.085,flip 0%(检测器正确放行)。扰动后:

| perturbation | type | mean_fake_score | delta_score | high_score_ratio(flip) |
|---|---|---:|---:|---:|
| original (clean real) | — | 0.085 | 0.000 | 0.0% |
| resample_8k | bandwidth | **0.455** | **+0.370** | **46.5%** |
| lowpass_3k | bandwidth | 0.450 | +0.365 | 46.5% |
| lowpass_4k | bandwidth | 0.397 | +0.312 | 13.0% |
| mp3_32k | compression/codec | 0.363 | +0.278 | 25.0% |
| reverb | room/reverb | 0.232 | +0.147 | 4.5% |
| noise_20db | additive noise | 0.108 | +0.023 | 0.0% |
| gain_quiet | loudness | 0.100 | +0.014 | 0.0% |
| noise_10db | additive noise | 0.062 | −0.023 | 0.0% |
| accomp_leak_-6db | accompaniment leakage | 0.085 | 0.000 | 0.0% |
| accomp_leak_0db | accompaniment leakage | 0.079 | −0.006 | 0.0% |
| demucs_resep | Demucs re-separation | 0.226 | +0.141 | 2.0% |

(后三行为 Exp3-ext 2026-06-11 夜补跑，`production_perturbation_ext.py`，accomp leakage 用同曲 Demucs no_vocals
按 SNR -6/0dB 混回人声 [188/200 真唱有伴奏 stem]；demucs_resep 把已分离人声再过一次 htdemucs 取 vocals。)

## Interpretation
**部分支持 cleanliness-axis,但需精确限定为"频谱带宽/编解码足迹轴",非任意脏化。**
- **带宽缩减(重采样 8k、低通 3-4k)与有损编解码(MP3 32k)显著推高真唱 fake_score**
  (Δ+0.28~0.37,flip 13-47%):真唱在不改变演唱/真伪的前提下,仅因频谱高频被裁/编码足迹改变,
  即被判为 fake → **制作链路/平台压缩确实诱发真实样本假阳性**,因果支撑 P2 的 FP 机制
  (野生真唱经 bilibili 压缩/分离后低带宽 → FP)。
- **加性噪声与响度变化几乎不触发**(flip 0%,Δ≈0,noise_10db 甚至略降):说明检测器读的**不是
  "脏/吵",而是频谱包络/带宽形状**(MFCC/spectral_flatness 偏移)。这是诚实的限定——
  把结论精确化为 spectral-bandwidth/codec axis,而非泛化的"任何制作噪声"。
- 混响居中(+0.147,flip 4.5%)。
- **伴奏残留(accompaniment leakage)不触发假阳性(诚实负结果,重要)**:即便把同曲伴奏按等响度(0dB)
  混回真唱,fake_score 几乎不动(Δ 0.000 / −0.006,flip 0%)。这**反驳**了"伴奏残留 → FP"的朴素假设——
  检测器的 FP 驱动**不是**伴奏能量,而是频谱带宽/编解码足迹。精确化了 P2:5.6b 里 FP 真唱的低 HNR
  应理解为**分离/编码产生的频谱签名**,而非"有伴奏内容"本身。
- **Demucs 重分离轻微推高**(+0.141,mean 0.226,flip 2%,≈混响量级):把已干净人声再过一次 Demucs
  引入的分离伪迹有真实但温和的 fake 化效果,符合"分离产线足迹"解读,但远弱于带宽/编解码。

## Limitation
- **合成扰动是 controlled diagnostic probe,不等于真实平台分布**:本实验用 librosa/ffmpeg 合成的
  重采样/低通/MP3/混响/加噪/伴奏混入/二次分离,**非真实 bilibili/YouTube 转码与分离链路**。
  fake-score shift / flip rate 的**绝对值不可外推**到真实互联网平台;我们只把分数变化作**定性的
  敏感性证据**(模型分数是否对 production cleanliness/compression/separation 敏感),
  **不作可部署的错误率估计**。最终外部效度仍以 WildSVDD-bilibili/T02 的真实野生测试为准。
  *("These perturbations are controlled probes rather than a faithful simulation of platform
  distributions. We therefore interpret score shifts qualitatively as sensitivity evidence, not as
  deployable error-rate estimates.")*
- **accompaniment leakage / Demucs re-separation 仅用合成探针测过,未忠实复现真实分离/平台残留**:
  本实验的 accompaniment leakage 是把**同曲伴奏随机段(非时间对齐)**按 -6/0dB 混入,Demucs
  re-separation 是对已分离人声**再过一次 htdemucs**;二者均为受控合成探针。其结果(伴奏残留 flip 0%、
  再分离 +0.141)表明**在该合成设置下伴奏能量本身不致 FP**,但**不能据此排除真实平台的伴奏残留/分离
  链路对 FP 的贡献**。*("Accompaniment leakage and re-separation artifacts are plausible contributors
  to false positives, but were tested only with synthetic probes—not faithful platform residue—so their
  real-world contribution is not fully characterized.")* 因此 Exp3 支持的是**诊断性结论**
  ("模型分数沿 bandwidth/codec/separation 的 cleanliness-axis 敏感"),**不声称已完整覆盖伴奏残留或分离链路**。
- 仅在 SingerLens 真唱上施扰 + SingerLens 域内检测器;跨检测器/跨数据集的扰动敏感性未测。

## 产物
- `scripts/production_perturbation.py`、`scripts/production_perturbation_ext.py`(accomp leakage + demucs re-sep)
- `outputs/perturbation_{effect,scores}.csv` + `outputs/perturbation_ext_{effect,scores}.csv`
