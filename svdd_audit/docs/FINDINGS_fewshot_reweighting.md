# Exp2b — T02 Target Few-Shot Calibration / Source+k Reweighting (FINDINGS_fewshot_reweighting.md)

## Purpose
回答 Exp2 的"source+k 不如 target-only"究竟是 **naive mixing 把 k 个目标标签淹没**,
还是**源域监督真负迁移(判别轴错位)**。脚本 `scripts/fewshot_reweighting.py`。

## 核心结论(精炼版)
1. **Target-only 最强**:仅用 25–100 个 Wild T02 标签从头训练,target-only AUC=0.720–0.852,
   明显高于任意 source+k 配置。
2. **Naive 淹没存在**:k 个 Wild 标签与 4 800 CtrSVDD 样本 naive 合并性能下降(0.638 @ k=100),
   证实源样本淹没了目标信号。
3. **判别轴错位**:即便最优的上采样/加权 reweighting 也未超过 target-only(0.773 < 0.852),
   说明源监督的判别轴与目标域排序错位,源信息在 WildSVDD 不能直接迁移。
4. **校准-only 无效**:calibration-only 全程 AUC≈0.49,单调重标定无法修复源域排序错位。
5. **整体结论**:source+k 既有淹没效应也有负迁移,但**核心问题是源域判别轴对目标域不匹配**;
   廉价修复仅靠简单加权/上采样无法消除。**不外推为"任何 source+target 方案都无效"**——
   仅描述 naive/简单加权/上采样;完整 DA(domain-balanced minibatch、DANN 特征对齐、calibration
   fine-tune)留待后续(本轮不新开)。

## Setup
- source = CtrSVDD(4800,e1);target = **WildSVDD/T02**(387 clips,226 real/161 fake)。
- k = 25/50/100 个 T02 标签(均衡 real/fake),其余 T02 作 held-out 测试;20 次平均。
- 同 Exp2 特征(FULL)/模型(RF),保证可比。五方法:
  1 target-only / 2 source+k naive / 3 source+k target-upsampling /
  4 source+k domain-balanced weighting / 5 calibration-only(source-only RF + Platt 在 k 标签上重标定)。
- 指标:AUC(主)+ balanced-accuracy。**calibration 为单调映射 → AUC≡source-only**(诊断点)。

## Results (held-out WildSVDD-T02 AUC)

| k | target_only | src naive | src upsample | src domain-bal | calibration_only |
|---:|---:|---:|---:|---:|---:|
| 25 | **0.720** | 0.557 | 0.619 | 0.588 | 0.492 |
| 50 | **0.798** | 0.580 | 0.690 | 0.649 | 0.496 |
| 100 | **0.852** | 0.638 | 0.773 | 0.734 | 0.485 |

balanced-accuracy 同向(k=100:target_only 0.769 > upsample 0.707 > domain-bal 0.670 > naive 0.598 > calib 0.489)。

## 重点回答

**reweighting 能否救 source+k?——部分能,但不足以追平 target-only。**
- **naive mixing 确有淹没**:对 k 上采样/domain-balanced 加权后,source+k 从 naive 0.638 回升到
  upsample 0.773 / domain-bal 0.734(k=100)→ 原 Exp2 的部分差距确是 naive 合并把 k 淹没所致。
- **但任何 reweighting 变体仍 < target-only**(0.773 < 0.852)→ 加入源数据始终拖累,源监督是**非正资产**。

**calibration-only ≈ 随机(0.49)、AUC 不随 k 变 → 源判别轴对目标域排序本身错位,非 miscalibration。**
单调重标定不改变 AUC,而源 score 对 Wild 的 AUC 已≈0.5;说明问题**不是阈值/校准**,而是源域学到的
判别方向在目标域无效。这与 cross-dataset collapse + cleanliness-axis 一致。

## 结论(精确化 Exp2)
**naive mixing 失败 + balanced adaptation 部分有效但仍不及 target-only + 源 score 排序在目标域≈随机
(calibration 救不了)** → 综合判定:**源域监督在朴素合并下是负资产,简单 reweighting 缩小但不能关闭差距,
且源判别轴与 Wild 目标域错位**。准确表述:
*"Simple reweighting/upsampling narrows but does not close the source+k gap, and calibration-only stays
at chance, indicating the source decision axis is misaligned with the target domain—not merely a mixing
or threshold artifact. Target-only training remains best."*
**不外推为"所有 source+target 自适应都无效"**:未测完整 DA(DANN 特征对齐、神经校准层微调、
domain-balanced minibatch 等),留待后续(本轮不新开)。

## Limitation
- 仅 RF + 手工特征 + 简单加权/上采样/Platt 校准;未测特征对齐式 DA。
- target=T02(可达 bilibili 子集),非完整 SingFake;k≤100。

## 产物
- `scripts/fewshot_reweighting.py`、`outputs/fewshot_reweighting.{csv,png}`
