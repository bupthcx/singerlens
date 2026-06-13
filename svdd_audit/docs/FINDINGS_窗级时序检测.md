# 窗级时序检测 (FINDINGS_窗级时序检测.md)

**动机**：难度分层发现部分难 fake 有"局部窗 AI 痕迹冒头(win_max>0.5)被整段平均掉"，
推测窗级 max-pooling 能抓住被平均稀释的局部伪迹。本实验直接证伪/证实该假说。

## 设置

- 每 clip 滑窗 win=3s / hop=1.5s（10s clip→6 窗），逐窗抽 49 维 base 特征。
  1637 clip → **9822 窗**(real552/fake533/real_vocoded552)。脚本 `scripts/window_extract.py`，
  长表 `outputs/window_features.csv`。
- 三种段级表示对比（脚本 `scripts/window_temporal_eval.py`）：
  - **POOL_MEAN**：每特征窗间均值（≈现有整段聚合，基线）
  - **POOL_RICH**：每特征 {mean,max,min,std} 拼接（把时序极值/离散度当**特征**喂给分类器）
  - **MIL_{mean,max,top2,top3}**：窗级 RF 逐窗打分→段分数按对应规则池化（多示例学习）
- 协议：within(StratifiedGroupKFold by song) + **LOSO-singer**(泛化)。特征集 FULL/CLEAN/HNR。
- 审计版 `--neg vocoded`：负类换 real_vocoded(共享声码器足迹)，检验窗级是否只是抓声码器。

## 核心结果（LOSO-singer AUC，负类=real）

| 特征集 | POOL_MEAN(基线) | POOL_RICH | MIL_mean | MIL_max | MIL_top3 |
| --- | ---: | ---: | ---: | ---: | ---: |
| FULL  | 0.885 | 0.881 | **0.906** | 0.856 | 0.883 |
| CLEAN | 0.680 | 0.713 | **0.715** | 0.670 | 0.703 |
| HNR   | 0.652 | **0.700** | 0.669 | 0.616 | 0.651 |

## 三条结论

**(1) max-pooling 反而更差——"局部痕迹"假说被证伪。**
MIL_max 在所有特征集/协议下 AUC 一致**低于** mean(FULL LOSO 0.885→0.856，HNR 0.652→0.616)。
难歌 fake recall 看似飙升(句号 0.10→0.66、walkonwater 0.08→0.77，见 hardsong_recall.csv)
纯属 **max 全局抬分假象**——所有段(含真唱)分数被一起抬高，固定 0.5 阈值下 fake recall 涨但
可分性(AUC)降、真唱被误判增多。被平均掉的"局部窗"在真唱里同样存在，不是 AI 专属信号。

**(2) 真正小赚的是平庸方案——窗级均值 / 时序方差特征。**
- **MIL_mean**(窗级训练+均值聚合)给一致小幅提升：FULL LOSO 0.885→**0.906**、CLEAN 0.68→**0.715**。
  收益来自窗级样本更多(9822>1637)+ 段分数更平滑，非局部峰值。
- **POOL_RICH**(时序 max/min/std 当特征)救弱嗓音通道：HNR LOSO 0.652→**0.700**、within 0.693→0.741。
  即"嗓音质量随时间的离散程度"携带歌手无关的真伪信号，但**作为特征**用、不作为池化规则。

**(3) 窗级不修复泛化鸿沟，且 FULL 仍骑产线/生成混杂。**
可解释特征 LOSO 仍 ≤0.72；FULL 0.906 在审计版(负类 real_vocoded)仍达 LOSO 0.88
→ 不是纯声码器探测器(否则 fake 与 real_vocoded 不可分)，而是抓 DiT 生成 mel 签名；
CLEAN/HNR 在审计版部分塌(HNR LOSO 0.669→0.624)证其判别力部分来自 real(Demucs) vs vocoded 产线差。
窗级三结论对声码器对照完全鲁棒(两版同形)。

## 叙事价值（第 5 个反转）

延续"层层识破自己"主线：**上窗级 max 想抓局部痕迹 → 反被抬分假象坑、AUC 更差；
真正有效的是不起眼的窗级均值与时序方差特征**。又一个"简单赢花哨的显然修复"诚实负面结果，
与 DANN<per-singer-z、max-pool<mean-pool 同构——花哨的直觉修复打不过平庸的统计量。

## 产物

- `scripts/window_extract.py`、`scripts/window_temporal_eval.py`
- `outputs/window_features.csv`(9822 窗)、`outputs/window_temporal_real.csv`、
  `outputs/window_temporal_vocoded.csv`、`outputs/window_temporal_hardsong_recall.csv`
