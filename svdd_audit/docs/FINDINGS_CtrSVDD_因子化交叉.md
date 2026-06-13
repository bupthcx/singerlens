# CtrSVDD 因子化交叉：范式 × 声码器 (FINDINGS_CtrSVDD_因子化交叉.md)

**动机**：既有 LOVO(留一声码器)/LOGO(留一范式)是单轴。本实验把二者升级为
**二维 (范式 × 声码器) cell**，回答"domain shift 是否多轴、能否归约为单一声码器"，
直接锐化主线"Are we detecting AI covers or vocoders?"。E1 平衡子集(A01–A08, 16k 无采样率混杂,
每攻击 300 spoof + 2400 bonafide)。脚本 `scripts/ctrsvdd_factorial.py`。

## (1) 列联表：两轴在 CtrSVDD 内嵌套混杂

| spoof 数 | ddsp | hifigan | nsf-hifigan |
| --- | ---: | ---: | ---: |
| **SVC** | 0 | 0 | 900 |
| **SVS** | 300 | 900 | 300 |

SVC 仅与 nsf-hifigan 共现，hifigan/ddsp 仅与 SVS → **范式与声码器近乎嵌套**(方法学警示：
标准评测里二者无法解耦)。**nsf-hifigan 是唯一跨范式声码器**，提供了固定声码器隔离范式的窗口。

## (2) Leave-One-Cell-Out (LOCO)：留一(范式×声码器)格

留出一个 cell 的 spoof 不训练，测同一 held-out test(in-dist 训练含该格作对照)。FULL EER%：

| held-out cell | in-dist | held-out | ratio |
| --- | ---: | ---: | ---: |
| SVS×hifigan | 2.33 | 31.76 | 13.6× |
| SVS×ddsp | 3.67 | 28.67 | 7.8× |
| SVS×nsf-hifigan | 2.00 | 41.33 | **20.7×** |
| SVC×nsf-hifigan | 3.67 | 39.90 | 10.9× |

每个 cell 留出后全部塌到近随机(28–41% EER)。CLEAN 同形(6–9×)。HNR/VRI 在 CtrSVDD
本就近随机(HNR in-dist 已 15–17%)，对迁移无用——印证 HNR 是 SingerLens 特异信号、非通用 AI 痕迹。

## (3) 轴控解耦：单独隔离声码器/范式效应（核心新贡献）

FULL EER%(in-dist | held-out | ratio)：

| 对照 | 固定轴 | train→test | in-dist | held-out | ratio |
| --- | --- | --- | ---: | ---: | ---: |
| 换声码器 | 范式=SVS | hifigan→ddsp | 2.33 | 29.71 | 12.8× |
| 换声码器 | 范式=SVS | hifigan→nsf | 1.38 | 39.67 | 28.8× |
| **换范式** | **声码器=nsf** | SVC→SVS | 0.62 | 55.33 | **89.2×** |
| **换范式** | **声码器=nsf** | SVS→SVC | 0.81 | 48.00 | **59.3×** |

**铁证**：**声码器完全相同(都是 NSF-HiFiGAN)**时，仅把生成范式 SVS↔SVC 一换，
检测器从近完美(EER 0.6–0.8%)崩到掷硬币(48–55%)，退化 **59–89×**，
比"固定范式换声码器"(13–29×)还严重。CLEAN 同向(换范式 18–25× > 换声码器 8–10×)。

## 结论

1. **混杂是多轴的，不能归约为声码器**。"SVDD 只是声码器探测器"的简化说法被证伪：
   即便声码器一致，生成范式(声学模型/mel 生成方式)造成的 domain shift 与声码器同量级或更大。
   主线由"AI covers or vocoders" 锐化为 **"generation-family（范式 ⊗ 声码器）"**。
2. **标准评测内两轴嵌套混杂**(SVC⊂nsf)，域内高分同时吃了范式与声码器两重捷径，
   任一轴 held-out 都崩 → 标准 split 系统性高估鲁棒性(model-agnostic 主命题再添一维证据)。
3. HNR/VRI 手工嗓音特征对 CtrSVDD 迁移无用，与 SingerLens 上最强形成对照——
   "可解释特征的判别力是数据集特异、非通用"。

## (4) AASIST 确认：多轴混杂 model-agnostic（定锤）

同一关键对照（**声码器固定 NSF-HiFiGAN，只换范式**），官方 AASIST(rawnet 前端，
train_set 12000 平衡训练，qwen3-asr 环境，15 epoch)：

| AASIST (vocoder=nsf 固定) | EER | AUC | F1 |
| --- | ---: | ---: | ---: |
| in-dist SVC×nsf | 4.00 | 0.994 | 0.951 |
| **held-out 范式 SVS×nsf** | **54.29** | **0.456** | 0.052 |

SVC 训练的 AASIST 测同声码器的 SVS，从近完美(AUC 0.994)崩到**比随机还差**(AUC 0.456，
反相关)，退化 13.6×。与 RF 同向(RF 0.62→55.33，89×)。

**→ "SVDD 检测器不只是声码器探测器、混杂是多轴(范式⊗声码器)" 对手工特征与 SOTA 神经均成立，
是 model-agnostic 的根本问题。** 协议 CSV: `/home/admin2/xf/ctrsvdd/aasist_csv/para_nsf_*.csv`；
结果 `outputs/ctrsvdd_aasist_paradigm.csv`。

## 产物

- `scripts/ctrsvdd_factorial.py`
- `outputs/ctrsvdd_factorial_loco.csv`、`outputs/ctrsvdd_factorial_axis.csv`
