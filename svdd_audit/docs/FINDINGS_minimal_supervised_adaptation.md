# Minimal Supervised Adaptation — The Only Viable Exit (FINDINGS_minimal_supervised_adaptation.md)

## 立意
阶段 E(能否修复 cross-dataset)不应读成"一串失败",而是一个**有结构的正面结论**:
**无监督路线全部关闭 → 监督适应是唯一可行出口,且代价极小**(几十个目标标签即碾压所有无监督方法、
逼近域内上界)。本 finding 把 Tier1.1(无监督 DA 失败)+ Exp2/Exp2b(target-only 有效)合并为一条
"关门 + 开门"的论断。

## 两列对比(target = WildSVDD-T02,held-out AUC)

| 方法 | 是否用目标标签 | AUC |
| --- | --- | ---: |
| no-adapt baseline(源直迁) | 否 | 0.509 |
| per-domain z-norm | 否 | 0.488 |
| CORAL | 否 | 0.474 |
| Subspace Alignment | 否 | 0.517 |
| DANN(对抗域不变) | 否 | 0.442 |
| calibration-only(源分数重标定) | 否(仅校准) | 0.485 |
| **— 无监督最佳 —** | **否** | **≤0.52(≈随机)** |
| **few-shot target-only, k=25** | **是(25)** | **0.720** |
| **few-shot target-only, k=50** | **是(50)** | **0.798** |
| **few-shot target-only, k=100** | **是(100)** | **0.852** |
| full target(域内上界) | 是(全量) | ~0.925 |

(数据:`domain_adaptation_cross.csv`、`fewshot_reweighting.csv`。)

## 结论(结构化,非"失败串")
1. **无监督出口全关**:特征对齐(CORAL/subspace/DANN/z-norm)、单轴 confound-removal(统一带宽)、
   纯校准(calibration-only)**无一离开随机带**——因为失败是判别方向 p(y|x) 跨域错位,
   对齐边缘分布或重标定都改不了排序。
2. **监督出口唯一可行且廉价**:**仅 25 个目标标签(0.720)就已超过每一个无监督方法**;
   50→0.798、100→0.852,逼近域内上界 ~0.925。**收益曲线在前几十个标签最陡**。
3. **一句话**:*"Unsupervised adaptation is closed; minimal target supervision is the only viable exit—
   a few dozen target labels already dominate every unsupervised method and approach the in-domain bound."*

## 诚实限定
- 监督出口的"廉价"指**相对无监督**;0.90+ 需接近全量目标标签,**非 10–20 个**(k=25≈0.72,k=100≈0.85)。
- few-shot 用 RF + 简单从头训练;source+k(即便公平加权)仍 < target-only(见 Exp2b),故"出口"是
  **target-only / 收集目标标签**,不是"源+少量目标自适应"。
- 未测端到端神经 DA / 对比对齐 / 歌声专用预训练;结论限定为"**特征对齐式无监督关闭**",非"任何方法不可能"。

## 部署含义(与 5.10 一致)
新平台**不要**指望无监督迁移现成检测器;应**优先标注几十个目标域样本从头训练**——这是 ROI 最高、
唯一被验证有效的路径。

## 产物
- 复用 `scripts/{domain_adaptation_cross,fewshot_reweighting}.py`
- `outputs/domain_adaptation_cross.csv`、`outputs/fewshot_reweighting.csv`
