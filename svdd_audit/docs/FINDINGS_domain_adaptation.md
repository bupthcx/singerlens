# Domain Adaptation & Discriminative-Axis Attribution (FINDINGS_domain_adaptation.md)

## Purpose
关闭论文里"只测了 naive mixing/简单加权"的口子:测**完整无监督 DA**(特征对齐式)能否修复
cross-dataset collapse(Tier1.1);并**归因为什么**——源域与目标域的判别轴是否不同(Tier2.1)。
脚本 `scripts/domain_adaptation_cross.py`、`scripts/axis_attribution.py`。

## 核心结论(精炼版)
1. **无监督 DA 全线失败**:per-domain z-norm / CORAL / Subspace Alignment / DANN 均无法把 cross-dataset
   AUC 拉离随机(0.44–0.62),没有任何方法稳定优于 no-adapt baseline,部分甚至更差。
2. **target-only(有标签)才行**:同设置下 target-only CV 上界 0.90–0.99。
3. **归因**:源(CtrSVDD)模型过押**高阶 MFCC 频谱纹理**,域内(Wild)模型额外依赖 **HNR**(Wild 真伪线索);
   源在 Wild 上的(错)分数主要由 mfcc_*_std 驱动 → 判别轴不同。
4. **结论**:cross-dataset 失败是**判别方向 p(y|x) 跨域错位**(非边缘分布偏移),所以对齐边缘的 DA 救不了;
   与 calibration-only≈随机、cleanliness-axis 一致。**修复需目标域标签,不是特征对齐。**

## Tier1.1 — 无监督 DA cross-dataset AUC

| pair | baseline | perdomain z-norm | CORAL | subspace-align | DANN | target-only(参考) |
|---|---:|---:|---:|---:|---:|---:|
| CtrSVDD→WildT02 | 0.509 | 0.488 | 0.474 | 0.517 | 0.442 | **0.925** |
| CtrSVDD→SingerLens | 0.615 | 0.618 | 0.537 | 0.554 | 0.555 | **0.989** |
| SingerLens→CtrSVDD | 0.549 | 0.569 | 0.529 | 0.544 | 0.582 | **0.898** |

无一方法把任一 pair 拉出随机带;CORAL/subspace/DANN 在 Ctr→Wild 反而 ≤baseline。
→ **特征对齐式 DA(对齐协方差/子空间/对抗域不变)不能修复 cross-dataset**。

## Tier2.1 — 判别轴归因(RF importance / 源分数相关)

**家族级重要性**(`outputs/axis_attribution_family.csv`):

| family | 源 CtrSVDD | 域内 WildT02 |
|---|---:|---:|
| MFCC | 0.642 | 0.561 |
| **HNR** | **0.055** | **0.146** |
| lowlevel | 0.103 | 0.110 |
| VRI | 0.094 | 0.102 |
| pitch | 0.092 | 0.072 |

- 域内 Wild 模型对 **HNR 权重 0.146**(源仅 0.055)——HNR 正是 Wild real-vs-fake 的真实线索
  (失败归因 P2:Wild fake 更干净 HNR 更高);**源模型欠用 HNR、过押 MFCC**。
- 源模型在 Wild 上的(错)分数 spearman 相关 top:**mfcc_13_std 0.55 / mfcc_11_std 0.51 / mfcc_8_std 0.41**
  → 源把 Wild 按**高阶 MFCC 频谱纹理(=编码/分离/带宽签名,cleanliness 轴)**排序,而非真伪。

图 `outputs/axis_attribution.png`(家族级重要性对比)。

## Limitation
- DA 仅测特征对齐家族(z-norm/CORAL/SA/DANN)+ RF/浅层;未测端到端神经 DA、对比学习式对齐、
  或大模型表示上的 DA。结论限定为"**特征对齐式无监督 DA 不能修复**",不外推为"任何方法都不可能"。
- 归因基于 RF gini importance + spearman,关联非严格因果;但与 Tier1.2 confound-removal 互证。

## 产物
- `scripts/{domain_adaptation_cross,axis_attribution}.py`
- `outputs/domain_adaptation_cross.csv`、`outputs/axis_attribution{,_family}.csv`、`outputs/axis_attribution.png`
