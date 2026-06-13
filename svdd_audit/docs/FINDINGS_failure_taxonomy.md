# Failure Taxonomy (FINDINGS_failure_taxonomy.md)

## 目的
基于 canonical FULL clip_mean 分数(`score_clip_scores.csv`)+ 窗级特征聚合 + metadata，
对 cross-domain 错误做定量归因，回答错误是否集中在 production artifacts / Demucs 伴奏残留 /
source-domain generation cue 缺失。脚本 `scripts/failure_taxonomy.py`。
代表样本 `outputs/failure_cases.csv`(每协议×TP/FN/FP/TN 各 8 条)，组级统计 `failure_group_stats.csv`。

## 根因：判别轴在域间错位

| 数据集(域内轴) | fake HNR | real HNR | 说明 |
|---|---:|---:|---|
| **WildSVDD** | **17.1** | **14.1** | 野生 AI 人声更干净/谐波强 → "高 HNR=fake" |
| **CtrSVDD(源)** | 21.6 | 22.0 | 录音室 HNR 无差 → 源检测器靠频谱/MFCC 生成签名，非 HNR |

WildSVDD 域内真伪轴(HNR: fake 更干净)既**反直觉**又与 CtrSVDD 源域轴**正交**：源检测器从未学过这个轴。

## CtrSVDD→WildSVDD 错误的定量结构(全样本组均值)

| outcome | n | 平均分 | HNR | spectral_flatness | 归因 |
|---|---:|---:|---:|---:|---|
| **FP**(真判假) | 190 | 0.678 | **13.68(最低)** | 0.0228(高) | 低 HNR/高 flatness 嘈杂真唱(Demucs 伴奏残留) |
| **FN**(假判真) | 60 | 0.420 | **19.50(最高)** | 0.0178(低) | 最干净 So-VITS 假唱，缺源域生成签名 |
| TP | 177 | 0.679 | 16.32 | 0.0220 | 恰好较脏的假唱 |
| TN | 39 | 0.405 | 16.42 | 0.0186 | 恰好较干净的真唱 |

**检测器实际决策轴 = 频谱 flatness / 干净度**(高分组 FP+TP flatness≈0.022，低分组 FN+TN≈0.018)，
**与目标域真伪完全正交**。两类系统性错误：

1. **Production-artifact / Demucs 残留致 FP**：野生真唱经分离后伴奏/混响残留 → 低 HNR、高 flatness，
   被干净录音室训练的检测器读成"合成感" → 误判 fake(FP HNR 13.68 vs TN 16.42，整组最低)。
2. **Source-domain generation-cue 缺失致 FN**：So-VITS/RVC 野生翻唱干净、谐波强，但**不携带 CtrSVDD
   学到的频谱/声码器生成签名** → 检测器找不到熟悉的"假"证据，判 real(FN HNR 19.50 整组最高)。
   FN 跨所有 So-VITS 变体均匀(Sovits4.0 33/98、4.1 7/33、unknown 16/76)→ 非单一模型问题，是通用的"干净假唱"盲区。

## SingerLens→WildSVDD：相反方向的整域偏移
SingerLens 检测器把 Wild **几乎全判 real**(分数 ~0.23-0.25)：FN=227(假几乎全漏)、TN=217(真凑巧对)、
FP 仅 12、TP 仅 10。即源域产线先验把整个目标域平移到"真"一侧，零真伪分辨——与 P1 的域相关 offset 一致。

## 与窗级结果呼应
窗级可视化(FINDINGS_window_cross_dataset_v2 Figure1)已示：Wild 真唱 FP 段 HNR 仅 1-2.5dB(段内)、
p(fake) 平直高；此处组级统计坐实——**FP=低 HNR 制作噪声、FN=高 HNR 干净假唱**，错误不是随机而是
沿"制作干净度"这条与真伪正交的轴系统排列。

## 结论
cross-domain 失败可完全归因为**检测器沿"制作干净度/生成签名"轴打分，而该轴与目标域真伪正交甚至翻转**：
(1) 脏真唱(Demucs 残留)→ FP；(2) 干净假唱(缺源域生成 cue)→ FN。这是产线/生成家族混杂在
野生域的直接后果，也是"标准评测高估泛化"在错误层面的微观证据。

## 产物
- `scripts/failure_taxonomy.py`
- `outputs/failure_cases.csv`(代表样本 TP/FN/FP/TN×协议)
- `outputs/failure_group_stats.csv`(组级特征均值)
