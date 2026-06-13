# Paper-Level Review — Before Full Draft (PAPER_REVIEW_BEFORE_FULL_DRAFT.md)

> 审查对象:`PAPER_SVDD_Audit_skeleton.md`(当前 ~490 行,5.1–5.10 + 5.5/5.6b/5.9b + Appendix A)。
> 目标:**压缩主线、确定主文/附录边界**,不补实验、不改结论。

---

## 1. One-sentence thesis
**标准 SVDD 评测严重高估泛化,因为检测器利用了"生成家族/制作产线"混杂(在数据集内恒定、跨数据集漂移);
一旦用受控审计协议(copy-synthesis、LOVO/LOGO/因子化、cross-dataset)解耦,域内近完美的检测器跨域塌到随机
——且这是 model-agnostic、多轴、并且无法靠局部性或无监督自适应修复的。**

> 备选更短版(投稿标题/摘要首句用):*"In-distribution SVDD is solved; cross-distribution SVDD is at chance,
> because detectors learn the generation/production family, not authenticity."*

---

## 2. Main contributions sanity check

**当前 5 条存在两组重复:**
- **C1(系统性发现)≈ C3(大规模实证)**:一个是"结论",一个是"证据规模",应合并。
- **C2(协议套件)≈ C4(toolkit)**:toolkit 就是协议的开源实现,应合并(toolkit 作为 C2 的交付物)。
- **C5(失败机制归因)** 独立且有价值,但目前把"human pilot"塞进 C5 显得杂。

**建议压成 3 条独立贡献(主文):**
1. **审计协议套件 SVDD-Audit**:copy-synthesis 控制 + LOVO + LOGO + **因子化(范式×声码器)** + cross-dataset,
   附 attack→generator→vocoder 映射;形式化定义 + 开源 toolkit(把旧 C2+C4 合并,toolkit 是交付物不是单列贡献)。
2. **Model-agnostic 崩塌实证**:3 检测器家族(手工 RF / 神经 AASIST / 微调 SSL)× 3 数据集,
   域内 AUC 0.88–0.99 → 跨数据集 ≈ 随机;**多轴**(固定声码器换范式仍塌,RF×59–89/AASIST×13.6)(合并旧 C1+C3)。
3. **机制 + 边界**:失败 = score collapse + 域 offset + cleanliness/判别轴错位;
   **什么修不了**(局部性、无监督 DA、单轴 confound-removal)**什么能修**(少量目标标签)+ 部署含义
   (合并旧 C4 mitigation + C5)。

> 降级:human-listening pilot → **不是贡献**,放 Appendix(它是 ready-to-run 包,未实际跑)。
> toolkit → 作为贡献 1 的"开源交付",一句话即可,不单列。

---

## 3. Section 5 overload check

**结论:5.1–5.10 + 5.5/5.6b/5.9b = 13 个子节,严重超载,读起来像 experiment log,必须压缩到 ~5–6 个。**

| 现子节 | 处置 | 理由 |
|---|---|---|
| 5.1 标准看似 solved | **主文(保留,精简为半节)** | motivation 必需 |
| 5.2 copy-synthesis | **主文一段 + 细节进 appendix** | 概念重要(非通用 vocodedness),但可一段带过 |
| 5.3 LOVO + 5.4 LOGO | **合并进 5.5,降为一张表的两行** | 单轴是因子化的特例,因子化更强 |
| 5.5 因子化(多轴) | **主文(核心,作"生成家族混杂"节的主体)** | 最强的"不只是声码器"novel 结果 |
| 5.6 cross-dataset | **主文(headline)** | 论文卖点 |
| 5.6b 失败归因(cleanliness) | **主文(机制节,压缩)** | 机制必需,但 P1+P2 两套表可压成一图一表 |
| 5.7 窗级局部性 | **降 appendix(主文一句)** | 诚实负面 side-quest,非主线 |
| 5.8 风格特征不迁移 | **并入 5.6b 一句话** | 不值一个 section |
| 5.9 跨歌手 mitigation(per-singer-z/DANN) | **降 appendix** | 源于课程项目,跨歌手≠跨数据集主线 |
| 5.9b 能否修复(DA/confound) | **主文(边界节,压缩)** | "无监督修不了、需目标标签"是强结论 |
| 5.10 部署(abstention/few-shot/perturb) | **主文保留 abstention+few-shot;perturbation 并入 5.6b 作 cleanliness 因果证据** | 部署落点有价值,但 3 张子表太多 |
| Appendix A human pilot | **保留 appendix** | OK |

**压缩后主文 Section 5(建议 5 节):**
5.1 Standard looks solved → 5.2 Generation-family confound(copy-synth 一段 + 单轴/因子化)→
5.3 Cross-dataset collapse(headline)→ 5.4 Why & where it fails(机制 + 归因 + perturbation 因果)→
5.5 Can it be fixed?(局部性/DA/confound 都不行,目标标签行)+ 部署含义。

---

## 4. Table / Figure plan(主文最多 6–8)

**主文(建议 6 表 + 2 图):**
| # | 内容 | 作用 | 来源 |
|---|---|---|---|
| **T1** | 标准域内 EER/AUC(3 检测器 × 特征/数据集) | 立"看似 solved" | ctrsvdd_eer_e1 + AASIST/w2v2 within |
| **T2** | 因子化 vocoder×paradigm 退化(RF+AASIST,含单轴 LOVO/LOGO 行) | 多轴混杂、不只声码器 | factorial_loco/axis + aasist_paradigm |
| **T3** | **Cross-dataset collapse(3 数据集 × 3 检测器,AUC/EER)** | **headline** | cross_dataset_aasist_vs_rf + w2v2ft + wild |
| **T4** | 失败归因(family importance 源vs域内 + 错轴特征) | 机制:判别轴错位 | axis_attribution_family |
| **T5** | 能否修复(few-shot target-only vs DA/confound/calibration) | 边界:只有目标标签行 | domain_adaptation_cross + fewshot_reweighting |
| **T6** | 部署:selective-prediction acc@coverage(域内 vs 跨域) | 置信阈值不安全 | reject_option_riskcoverage |
| **F1** | real/fake 分数分布(域内分离 → 跨域塌缩/offset) | 一图说清崩塌机制 | score_distributions.png |
| **F2** | 三通道案例时间轴 或 概念审计流程图(二选一) | 可解释/teaser | window_cross_dataset_cases 或新画 concept fig |

**降 appendix 的表图**:LOVO/LOGO 单轴完整表、窗级 MIL 全表 + 案例图、production perturbation 全表(8c)、
few-shot 全 5 方法 × 多 k、bandwidth_stats、confound-removal 全表、Table 8a/8b/8c 细分、human pilot、per-singer-z。

---

## 5. Narrative risk

**(a) "实验多但主线散"——最大风险。** 当前 13 子节 + 9 表读如日志。**必须收成一条脊柱**:
*overestimate → 生成家族多轴混杂 → cross-dataset 塌 → 机制(判别轴错位/cleanliness)→ 修不了(除非目标标签)*。
凡不在这条脊柱上的(窗级、跨歌手 mitigation、perturbation 细分)一律降 appendix 或一句话。

**(b) 过强表述需软化:**
- "**model-agnostic**":只测 3 家族,w2v2 是 base-960h 非 SOTA 大 SSL → 改"**across the three detector
  families we evaluate**",并明说 w2v2-ft 是 fine-tuned base 而非 large。
- "**SVDD is far from solved**":保留但加"once generation-family/production confounds are controlled"。
- **SingerLens within AUC=1.0 "trivially solved"**:别让它代表 SVDD 一般情况,明确标为
  "self-collected data 的 production-confound 极端案例,用作 cautionary,不作 benchmark"。
- "cleanliness axis":已诚实限定为 spectral-bandwidth/codec/separation;保持,别扩成"任何制作噪声"。

**(c) 易被攻击的 limitation(需主动在 Discussion 写明):**
1. **自建 SingerLens 单生成器(Seed-VC 为主)+ 自造产线** → "你的混杂是你自己造的"。回应:正因如此才用
   CtrSVDD(受控、第三方标注)与 WildSVDD(野生)三角验证;自建集仅作 cautionary 极端。
2. **WildSVDD = 可达 bilibili 子集(387–466 clips),小**;AASIST 在其上欠拟合(以 RF 0.935 为域内基准)。
3. **公开数据集仅 CtrSVDD**(+ 自建 + 野生子集);SingFake 因 YouTube 可获取性未用 → 写明。
4. **DA/confound-removal 只测特征对齐 + 单轴带宽**;未测端到端神经 DA / copy-synth 统一声码器 → 已限定,保持。
5. **合成 perturbation ≠ 真实平台**;只作定性敏感性,不作误报率 → 已限定,保持。

---

## 6. Missing pieces before full draft(写作层面,非实验)

- **Related Work**(需从零写 + 引文):SingFake、CtrSVDD/SVDD Challenge 2024、ASVspoof 跨库泛化、
  TTS/vocoder 伪迹检测、shortcut learning/捷径、domain shift & DA、multiple-instance learning。
- **Dataset details 表**:三数据集来源/规模/采样率/生成器/声码器/许可(CtrSVDD zenodo、WildSVDD zenodo、自建)。
- **形式化协议定义**(Sec 3):P0/copy-synth/LOVO/LOGO/factorial/cross-dataset/(window/DA 作 analysis)的
  统一记号、train/test bonafide disjoint 声明、degradation ratio 定义。
- **Metric 定义**:EER、AUC、degradation ratio、flipped-AUC、balanced-acc。
- **Figure captions / Table notes**:每张需自解释(尤其 †song-split、n、subset 说明)。
- **Ethics / Privacy statement**(重要,易漏):WildSVDD 抓取 bilibili AI 翻唱涉及**真实歌手声纹克隆**——
  版权/肖像/同意;human pilot 的 IRB/知情同意;deepfake 检测的双重用途声明。
- **Reproducibility statement**:toolkit 开源、随机种子、子集构造脚本、外部依赖(CtrSVDD eer.py/AASIST/
  openvpi NSF-HiFiGAN/wav2vec2-base-960h)。
- **Teaser/concept figure**(Fig 1):审计协议如何解耦混杂的示意图(目前没有,建议画)。

---

## 7. Proposed full paper outline(正式会议论文,非日志)

> 目标体量:ICASSP/INTERSPEECH 短文(~4–6 页正文)或 ACM MM/TASLP(更长可容纳更多主文表)。
> 下面按"短文"裁剪;若投 TASLP 可把部分 appendix 上提。

```
Title: Beyond Standard Splits: Auditing Generation-Family Confounds in Singing Voice Deepfake Detection

1. Introduction
   - AI 翻唱滥用 → SVDD;benchmark 低 EER 的"solved"假象;thesis;贡献(3 条);headline 数字;Fig.1 teaser
2. Related Work
   - SVDD 数据/挑战赛;anti-spoofing 跨库泛化;vocoder/产线伪迹;shortcut learning;domain shift/DA;MIL
3. SVDD-Audit Protocols(方法核心)
   - 形式化:P0 / copy-synth / LOVO / LOGO / factorial cell-out / cross-dataset;指标(EER/AUC/degradation)
4. Experimental Setup
   - 3 数据集(Table: details);3 检测器家族;实现/子集/评分
5. Results
   5.1 Standard evaluation looks solved (T1)
   5.2 Generation-family is a multi-axis confound: copy-synth(一段)+ 单轴/因子化 (T2)
   5.3 Cross-dataset collapse, model-agnostic (T3, headline)  + F1 分数分布
   5.4 Why & where it fails: score collapse/offset + cleanliness/判别轴错位 + 因果 perturbation (T4, F2)
   5.5 Can it be fixed? locality✗ / unsupervised DA✗ / bandwidth-removal✗ / target labels✓ + 部署含义 (T5, T6)
6. Discussion
   - 含义(把声码器+范式当受控变量;报告 degradation+cross-dataset);Limitations;Ethics
7. Conclusion
Appendices (supplementary):
   A. 完整协议定义与映射表  B. 单轴 LOVO/LOGO 全表  C. 窗级 MIL + 案例图
   D. Production perturbation 全表(含 leakage/re-sep)  E. Few-shot/DA/confound 全表 + bandwidth
   F. 跨歌手 mitigation(per-singer-z vs DANN)  G. Human-listening pilot 包  H. Reproducibility/toolkit
```

---

## 最终判断

**结论:还需一次"整理"(结构压缩)再开 full draft —— 不需要任何新实验。**

证据/实验侧已**充分且闭环**(3 数据集 × 3 家族 × 多协议 + 机制 + 边界 + 部署);
缺的全是**写作/结构**:把 13 子节 → 5 节、9 表 → 6 表 + 2 图、5 贡献 → 3 贡献,软化 3 处过强表述,
补 Related Work / Ethics / Dataset 表 / captions。

**建议的下一步(一步到位):** 我按本 review 的 §3/§4/§7 出一版**压缩后的 main-text skeleton v2**
(主文 5 节锁定 + 6 表 2 图占位 + appendix 清单),你确认后再逐节写正文。
要我现在就出 skeleton v2 吗?
