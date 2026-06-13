# Exp3b — Production Leakage & Re-synthesis (FINDINGS_production_leakage_resynthesis.md)

## Purpose
补强 production-cleanliness axis:检验真实样本中的**伴奏残留(多级)**与**分离链路(Demucs 重分离,
含 leak→resep 链式)**是否会诱发 fake_score 上升。**只用 real vocal,不引入 fake。**
脚本 `scripts/production_leakage_resynthesis.py`。

## 核心结论(精炼版)
1. **伴奏残留不致 FP(三级稳健)**:-12/-6/0 dB 三个混入级别下,真唱 fake_score 几乎不动
   (Δ +0.006/-0.010/-0.002,high_score_ratio 0%)。即便等响度(0dB)伴奏混入也不翻 → **伴奏能量
   根本不是误报驱动**。
2. **真正推高的是 re-separation 本身**:对已干净人声再过一次 Demucs(demucs_resep)Δ+0.142、flip 3%;
   leak→resep 链式 Δ+0.177、flip 5.3%。链式略高于纯 resep,但**增量来自分离步骤,非伴奏**
   (伴奏单独 = 0)。
3. **量级对照**:re-separation(+0.14~0.18)≈ reverb(+0.147)< MP3 编解码(+0.278,flip 25%)<<
   带宽缩减(Exp3 resample/lowpass +0.31~0.37,flip 13-47%)。
4. **诊断结论**:模型分数对**分离/编码的频谱签名**敏感(production-chain sensitivity),
   **不对伴奏能量或噪声敏感**。精确化 5.6b cleanliness-axis = spectral-bandwidth / codec /
   separation-signature axis。

## Results(真唱 fake_score,clean=0.085;伴奏条件 n=188[94% 有同曲 stem],其余 n=200)

| perturbation | mean_fake_score | Δ vs clean | high_score_ratio |
|---|---:|---:|---:|
| clean real | 0.085 | 0.000 | 0% |
| accomp_leak low (-12 dB) | 0.091 | +0.006 | 0% |
| accomp_leak mid (-6 dB) | 0.075 | -0.010 | 0% |
| accomp_leak high (0 dB) | 0.083 | -0.002 | 0% |
| leak → Demucs re-separation | 0.262 | +0.177 | 5.3% |
| Demucs re-separation (clean) | 0.227 | +0.142 | 3.0% |
| MP3-32k (context) | 0.363 | +0.278 | 25.0% |
| reverb (context) | 0.232 | +0.147 | 4.5% |

## Interpretation
production-chain 的 FP 风险来自**重编码/重分离引入的频谱签名**,而非伴奏内容。这把"野生真唱低 HNR → FP"
(5.6b)的机制定位到**分离/编码产线足迹**:bilibili 真唱经分离+转码后带上这种签名 → 被干净训练的
检测器判 fake;但**单纯混入伴奏(分离不彻底)本身不会**触发。

## Limitation
- **controlled probe**:伴奏混合比例(-12/-6/0 dB)与合成二次分离是受控探针,**不代表真实平台分布**;
  结论只写 **production-chain sensitivity**,**不写部署误报率**。绝对 Δ/flip 不可外推到真实互联网平台。
- accomp leakage 用同曲伴奏随机段(非时间对齐)混入;leak→resep 是合成链式,非真实平台分离/转码链路。
- 仅 SingerLens 真唱 + SingerLens 域内检测器;外部效度仍以 WildSVDD-bilibili/T02 野生测试为准。

## 产物
- `scripts/production_leakage_resynthesis.py`
- `outputs/production_leakage_resynthesis.csv`、`outputs/production_leakage_resynthesis_scores.csv`
