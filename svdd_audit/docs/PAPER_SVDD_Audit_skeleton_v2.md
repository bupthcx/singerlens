# SVDD-Audit — Main-Text Skeleton v2 (draft-ready, compressed)

> 由 v1(490+ 行、13 子节、实验日志式)压缩为**会议论文主文 5 节 + 6 表 2 图 + 附录**。
> 重点:(A) 阶段 D/E 串成"**诊断 → 修复尝试 → 出口**"的递进线;(B) 情感轴失败定性为"**进一步证伪捷径假说**"。
> 全文落点一句话:**跨域 SVDD 没有捷径,只有标注少量目标域数据。**

---

## Thesis(one sentence)
标准 SVDD 评测严重高估泛化:检测器学的是"生成/制作家族"而非真伪;审计解耦后域内近完美→跨域随机,
且 model-/scale-agnostic、多轴、**任何无监督或语义捷径都修不了——唯一出口是最小目标监督**。

## Contributions(3,独立)
1. **SVDD-Audit 协议套件**(copy-synthesis / LOVO / LOGO / **因子化范式×声码器** / cross-dataset)
   + attack→generator→vocoder 映射 + 开源 toolkit。
2. **Model-/scale-agnostic 崩塌实证**:3 检测器家族(RF / AASIST / 微调 SSL,含 95M→315M 规模轴)× 3 数据集,
   域内 0.88–0.99 → 跨数据集 ≈ 随机;多轴(固定声码器换范式仍塌)。
3. **机制 + 边界(递进结论)**:诊断失败=域特异"制作干净度"判别轴;证明**每条捷径(无监督 DA、带宽消除、
   校准、语义情感轴)都失败**,**最小目标监督是唯一可行出口**。

---

## §1 Introduction
- AI 翻唱滥用 → SVDD 防御;benchmark 低 EER 的 "solved" 假象。
- Thesis + 3 贡献 + headline(域内 0.99 → 跨域 0.5);**Fig.1 teaser**(审计如何解耦混杂的概念图,需新画)。

## §2 Related Work
- SVDD 数据/挑战赛(SingFake, CtrSVDD/SVDD Challenge 2024, WildSVDD);anti-spoofing 跨库泛化(ASVspoof);
  vocoder/TTS 伪迹检测;shortcut learning;domain shift & unsupervised DA;(MIL/情感作次要)。

## §3 SVDD-Audit Protocols(方法核心)
- 形式化定义 + 记号;train/test bonafide disjoint 声明;**P0 标准 / C copy-synth / P1 LOVO / P2 LOGO /
  P2b 因子化 cell-out / P4 cross-dataset**;指标 EER / AUC / **degradation ratio** / flipped-AUC。

## §4 Experimental Setup
- **数据集表**(T0,可并入正文):CtrSVDD(受控,16k,A01-A08,声码器/范式标注)/ SingerLens(自建 6 歌手,
  Seed-VC)/ WildSVDD-bilibili(野生,So-VITS/RVC);来源/规模/采样率/许可。
- **3 检测器家族** + **SSL 选型 rationale**(WavLM/HuBERT 网络不可达=客观约束;wav2vec2-base frozen 弱因
  非歌声预训练;微调 0.91、large-960h 0.97 验证 SSL 路线有效且规模可扩,但跨域仍塌)。
- 实现/平衡子集/官方 EER 评分。

## §5 Results(5 节,递进)

### 5.1 In-distribution SVDD looks "solved" — [T1]
3 检测器域内近完美(AASIST EER 6.5%/AUC 0.987;RF 0.90;微调 SSL 0.91,large 0.97)→ 立"看似 solved"。

### 5.2 Generation-family is a multi-axis confound — [T2]
- copy-synthesis(一段):检测器非通用 vocodedness 探测器,押 pipeline 特异伪迹。
- 单轴 LOVO/LOGO(T2 行):未见声码器/范式退化 ×2–3。
- **因子化(headline of this section)**:**声码器固定(NSF)仅换范式 SVS↔SVC,EER ~1%→~50%**
  (RF ×59–89,AASIST ×13.6)→ 混杂**多轴正交**,"不只是声码器"。

### 5.3 Cross-dataset collapse — model- & scale-agnostic — [T3, F1]
- 域内 AUC 0.88–1.00 → 跨数据集 0.34–0.62(随机/反相关),RF / AASIST / **微调 SSL** 同塌;
  **3.3× 大 SSL(0.97 域内)跨域仍塌(0.426)→ scale-agnostic**。野生 WildSVDD:公开 benchmark 高分
  完全不预测真实野生检测。
- **F1**:real/fake 分数分布(域内分离 → 跨域塌缩+域 offset)。

### 5.4 WHY it collapses: a domain-specific *production* axis(诊断)— [T4]
> **递进转折句**:崩塌不是随机噪声,而是检测器沿一条**域特异的判别轴**打分。
- score collapse + 域 offset(flipped-AUC 仍随机 → 目标域内真无可分信号,非符号反)。
- **归因(T4)**:源押高阶 MFCC 频谱纹理(=制作/编码/分离签名),域内 Wild 还用 HNR(真伪线索)→ 判别轴不同。
- **因果确认(perturbation)**:只扰真唱,带宽/编解码/再分离把真唱翻 fake(flip 13–47%),
  **加噪/响度/伴奏残留不动** → 该轴 = spectral-bandwidth/codec/separation 签名,与真伪正交。

### 5.5 CAN it be fixed? Every shortcut fails; minimal supervision is the only exit(修复 → 出口)— [T5]
> **递进转折句**:既然失败源于这条域特异制作轴,我们逐一尝试"绕过它"的捷径——全部失败,最终只剩一条路。
- (a) **无监督 DA**(CORAL/subspace/DANN/z-norm):≈随机 → 对齐边缘分布修不了 p(y|x) 错位。
- (b) **消除带宽轴**(统一低通):仍崩 → 混杂多分量,单轴抹除不够。
- (c) **纯校准**(calibration-only):AUC 不变(单调)→ 排序本身错,非阈值问题。
- (d) **语义捷径(情感一致性,production-orthogonal)**:**进一步证伪"找个鲁棒捷径"的假说**——
  本以为与制作链路正交应更稳健,实测 SingerLens 0.679 → WildSVDD **0.412(低于随机、关系翻转)**,
  降幅比 AASIST 还大 → **连语义轴也是域特异的;没有任何单一轴(制作 OR 语义)能单独解决跨域**。
- (e) **唯一出口 = 最小目标监督**(T5 两列):无监督最佳 ≤0.52 vs target-only k=25 **0.720** / k=50 0.798 /
  k=100 0.852 / 全量 ~0.925。**几十个目标标签碾压所有无监督方法**(廉价 = 相对无监督)。

### 5.6 Deployment implications(简)
benchmark 检测器不能直接上线;**置信阈值不是安全网**(跨域高置信也错);**优先收集少量目标标签从头训练**;
真实音频制作链路/转码/分离残留是 FP 风险源(需上线前评估)。

## §6 Discussion
- 含义:把声码器**与范式**当受控变量;报告 degradation + cross-dataset;野生迁移是真目标。
- **Limitations**(主动写):自建单生成器(用 CtrSVDD/Wild 三角验证缓解)/ WildSVDD 子集小 / 仅 1 公开集 /
  DA 只测特征对齐+单轴带宽 / 合成扰动非真实平台 / SSL 非歌声专用且非 SOTA 上限 / 情感构造单一。
- **Ethics & Privacy**(必写):WildSVDD 抓取 bilibili AI 翻唱涉真实歌手声纹克隆(版权/肖像/同意);
  human pilot IRB/知情同意;deepfake 检测双重用途声明。
- **Reproducibility**:toolkit 开源 + 种子 + 子集脚本 + 外部依赖清单。

## §7 Conclusion
受控审计后 SVDD 远未 solved;崩塌 model-/scale-agnostic、多轴;**没有捷径,唯有最小目标监督**;
SVDD-Audit 提供协议 + toolkit 量化真实泛化。

---

## Table / Figure plan(主文 6 表 + 2 图)
| # | 内容 | 节 | 来源 |
|---|---|---|---|
| T0 | 数据集详情(3 集) | §4 | (新整理) |
| T1 | 标准域内 EER/AUC(3 检测器,含 large) | 5.1 | ctrsvdd_eer_e1 + aasist/w2v2 within |
| T2 | 生成家族混杂:LOVO/LOGO 行 + 因子化退化(RF+AASIST) | 5.2 | factorial_{loco,axis} + aasist_paradigm |
| **T3** | **Cross-dataset collapse(3 集 × 3 检测器 + large 行)** | 5.3 | cross_dataset_aasist_vs_rf + w2v2ft{,_large} + wild |
| T4 | 失败归因(family importance 源 vs 域内) | 5.4 | axis_attribution_family |
| **T5** | **每条捷径失败 + 最小监督出口(两列)** | 5.5 | domain_adaptation_cross + fewshot_reweighting + wild_emotion_auc |
| **F1** | real/fake 分数分布(域内分离→跨域塌缩) | 5.3 | score_distributions.png |
| F2 | teaser/概念图(审计解耦混杂) | §1 | 新画 |

## Appendices(supplementary,v1 全部下沉)
- A 协议形式化定义 + attack→vocoder 映射全表
- B 单轴 LOVO/LOGO 完整退化表
- C 失败分析细节(score-reversal 全表 / failure taxonomy FP/FN 特征 / Fig 三通道案例)
- D Production perturbation 全表(带宽/codec/reverb/noise/leakage/re-sep,含 Exp3b 多级)
- E Few-shot / DA / confound-removal / bandwidth 全表 + axis_attribution 图
- F **Emotion-consistency 模块**(域内案例 + 跨域证伪;明确为 case study 非跨域方案)
- G Window-level / MIL cross-dataset(局部性≠泛化性,负面 side-result)
- H 跨歌手 mitigation(per-singer-z vs DANN)/ Human-listening pilot 包 / Reproducibility & toolkit

---

## 与 v1 的映射(裁剪记录)
- v1 5.3 LOVO + 5.4 LOGO → **并入 5.2**(降为 T2 行) | v1 5.5 因子化 → 5.2 headline
- v1 5.6 cross-dataset → **5.3** | v1 5.6b 失败归因 → **5.4** | v1 5.10 perturbation → 5.4 因果确认
- v1 5.9b(DA/confound/MSA/情感 e)→ **5.5 递进**(诊断已在 5.4) | v1 5.10 abstention/few-shot → 5.6
- v1 5.7 窗级 / 5.8 风格特征 / 5.9 跨歌手 mitigation / 情感模块 / human pilot → **Appendix**
- 5 贡献 → 3 | 9+ 表 → 6 主表 + 2 图

## 判断
**结构已 draft-ready。** 主线递进清晰(solved → 多轴混杂 → 跨域塌 → 诊断制作轴 → 捷径全败 → 最小监督出口),
情感失败已纳入递进(捷径证伪)。下一步可逐节写正文(建议从 §5.3→5.4→5.5 主干起笔)。
