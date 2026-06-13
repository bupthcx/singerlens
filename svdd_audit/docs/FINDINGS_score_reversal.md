# Score Reversal 诊断 (FINDINGS_score_reversal.md)

## 目的
汇总所有 cross-dataset / WildSVDD / T02 协议的分数分布，量化迁移失败的**机制**：
是真假分数仍可分(transferable)、塌成一团(random)、还是语义翻转(reversed)。
canonical 检测器 = FULL clip_mean RF。脚本 `scripts/score_reversal_diag.py`。
分类阈值: AUC<0.45 reversed / 0.45–0.60 random / ≥0.60 transferable。

## 主表 (FULL clip_mean)

| 协议 | mean_real | mean_fake | med_real | med_fake | AUC | flipped_AUC | EER | 类别 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| Wild within | 0.356 | 0.650 | 0.357 | 0.640 | 0.884 | 0.884 | 20.8 | **transferable** |
| Wild-T02 within | 0.306 | 0.538 | 0.287 | 0.517 | 0.852 | 0.852 | 22.2 | **transferable** |
| CtrSVDD within | 0.345 | 0.629 | 0.343 | 0.633 | 0.894 | 0.894 | 20.2 | **transferable** |
| SingerLens within | 0.265 | 0.735 | 0.247 | 0.757 | 0.984 | 0.984 | 6.2 | **transferable** |
| CtrSVDD→Wild_all | 0.632 | 0.614 | 0.647 | 0.630 | 0.464 | 0.536 | 52.6 | random |
| CtrSVDD→Wild_T02 | 0.629 | 0.629 | 0.643 | 0.647 | 0.503 | 0.503 | 49.3 | random |
| SingerLens→Wild_all | 0.251 | 0.267 | 0.250 | 0.260 | 0.542 | 0.542 | 48.9 | random |
| SingerLens→Wild_T02 | 0.246 | 0.265 | 0.248 | 0.257 | 0.535 | 0.535 | 49.5 | random |
| CtrSVDD→SingerLens | 0.626 | 0.651 | 0.628 | 0.650 | 0.556 | 0.556 | 46.3 | random |
| SingerLens→CtrSVDD | 0.365 | 0.385 | 0.337 | 0.363 | 0.531 | 0.531 | 46.8 | random |

(完整 FULL/CLEAN/HNR × 全协议见 `outputs/score_distribution_summary.csv`；逐 clip 分数
`outputs/score_clip_scores.csv`；分布图 `outputs/score_distributions.png`。)

## 关键发现

**1. 失败机制 = 分数塌缩/饱和，不是强语义反转。**
域内 real/fake 分数清晰分离(separation = mean_fake − mean_real ≈ +0.30~0.47)；
所有 cross-dataset 协议 separation 坍缩到 ≈ ±0.02，真假分布几乎重叠 → AUC≈0.5。
canonical clip_mean 无一落入 reversed(AUC 最低 0.464，仅轻微低于 0.5)。

**2. 强反转(AUC<0.45)只在某些表示下出现，非主因。**
窗级 MIL_max 的 CtrSVDD→Wild AUC=0.431(`window_cross_dataset_eval.csv`)是唯一明确 reversed，
说明 max-pooling 的抬分假象会把弱信号推过 0.5 翻成反相关；clip_mean/mean 聚合下则是塌缩。

**3. 域相关分数偏移 —— 迁移失败表现为整个目标域的分数平移。**
CtrSVDD 检测器把 WildSVDD **一切判高**(mean_real 0.632 / mean_fake 0.614，全部 ~0.6+ 像 fake)；
SingerLens 检测器把 WildSVDD **一切判低**(0.251 / 0.267，全部 ~0.25 像 real)。
即跨域不是"区分能力下降"，而是**整域被一个固定 offset 平移、域内零分辨**——
offset 的方向由源域产线先验决定(CtrSVDD 干净录音室 vs SingerLens 自建产线)。

**4. flipped_AUC 也不救。**
若按"知道要翻转"取 max(AUC,1−AUC)，cross-dataset flipped_AUC 仍只有 0.53–0.56 ≈ 随机
→ 失败不是"信号反了符号"(那样 flipped 会高)，而是**目标域内真无可分信号**。

## 结论
跨数据集迁移失败的主导机制是 **score collapse + 域相关 offset**(真假分数塌在一起并被整体平移)，
而非语义反转。reversed 仅在 max-pooling 这类抬分表示下偶发。这与"局部性/窗级救不了泛化"一致：
目标域里源域学到的判别轴根本不存在，换聚合、翻符号都无济于事。

## 产物
- `scripts/score_reversal_diag.py`
- `outputs/score_distribution_summary.csv`(协议×特征集×方法)
- `outputs/score_clip_scores.csv`(逐 clip canonical 分数，供 failure taxonomy / human pilot 复用)
- `outputs/score_distributions.png`(6 协议 real/fake 直方图)
