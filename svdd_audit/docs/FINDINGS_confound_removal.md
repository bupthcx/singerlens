# Confound-Removal & Spectral-Bandwidth Quantification (FINDINGS_confound_removal.md)

## Purpose
- **Tier2.2**:量化各数据集**原生频谱带宽**(real vs fake),看 cross-dataset 的域偏移是否伴随带宽差异。
- **Tier1.2**:**主动消除带宽混杂**——把所有数据统一低通到共同带宽(3.5kHz)再 cross-dataset,
  检验"抹掉带宽轴能否让迁移回升"(建设性因果控制)。脚本 `scripts/confound_removal.py`。

## 核心结论(精炼版)
1. **数据集带宽不同 + 真假方向跨域翻转**:rolloff85 域级差 ~600Hz;且 real-vs-fake 带宽方向
   CtrSVDD/WildT02(fake>real)与 SingerLens(real>fake)**相反** → 带宽对标签的关系本身域特异。
2. **统一带宽不修复 cross-dataset**:band-limited(3.5kHz)后 AUC 仍在随机带(0.434/0.584/0.602),
   相比原始(0.509/0.615/0.549)**未回升**。
3. **结论**:带宽只是 cleanliness 混杂的**一个分量**,单轴抹除不足;残留的**编码/分离/生成签名
   (中频 MFCC 纹理,见 Tier2.1)**仍主导 → 混杂是**多分量**的,无单一廉价 confound-removal 能修复。

## Tier2.2 — 原生频谱带宽(rolloff85 / centroid,Hz)

| dataset | label | rolloff85 | centroid | n |
|---|---|---:|---:|---:|
| CtrSVDD | real | 3162 | 1709 | 250 |
| CtrSVDD | fake | 3379 | 1822 | 244 |
| SingerLens | real | 3774 | 1993 | 250 |
| SingerLens | fake | 3548 | 2014 | 250 |
| WildT02 | real | 3454 | 1761 | 226 |
| WildT02 | fake | 3623 | 1899 | 161 |

域级带宽偏移(SingerLens 最宽、CtrSVDD 最窄)对应 cross-dataset 的整域分数 offset(P1);
**真假方向翻转**(CtrSVDD/Wild fake 更宽、SingerLens real 更宽)再次坐实判别轴跨域错位。

## Tier1.2 — 统一带宽后的 cross-dataset AUC

| pair | 原始 AUC | band-limited 3.5kHz AUC |
|---|---:|---:|
| CtrSVDD→WildT02 | 0.509 | 0.434 |
| CtrSVDD→SingerLens | 0.615 | 0.584 |
| SingerLens→CtrSVDD | 0.549 | 0.602 |

统一带宽**未带来回升**(仍随机)→ 带宽混杂被消除后,**其余混杂分量(codec/separation/生成签名)
足以维持崩塌**。与 Tier1.1(特征对齐 DA 失败)+ Tier2.1(源押 MFCC 纹理)互证:cross-dataset 失败
是**多分量、判别轴级**的,不是单一带宽差异。

## Limitation
- band-limiting 是受控探针(统一低通),非真实平台分布;3.5kHz 截断会损失信息,绝对 AUC 不可外推。
- 仅测了"统一带宽"一种 confound-removal;"统一过一个声码器(copy-synth all)"未做(需全量重合成,
  成本高,留待后续)。因此结论是"**带宽轴单独消除不足**",非"任何 confound-removal 都无效"。
- 子采样(CtrSVDD/SingerLens 各 ~500、WildT02 387)。

## 产物
- `scripts/confound_removal.py`
- `outputs/bandwidth_stats.csv`、`outputs/bandlimited_features.csv`、`outputs/confound_removal_cross.csv`
