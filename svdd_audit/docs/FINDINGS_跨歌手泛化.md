# 实验发现 II：跨歌手泛化（报告第四/六章核心）

## 数据集（v2，三歌手）
| 歌手 | 歌曲 | real | fake | 来源 |
|---|---|---|---|---|
| Singer A 单依纯 | DearFriend | 31 | 31 | B站→Demucs→Seed-VC(低音target,低八度) |
| Singer B 孙燕姿 | 开始懂了 | 32 | 27 | B站→Demucs→Seed-VC(自身target,同音区) |
| Singer C 邓紫棋 | 天空没有极限 | 35 | 34 | B站→Demucs→Seed-VC(自身target,同音区) |
| 合计 | | 98 | 92 | 共 190 片段 |

> fake 生成：每位歌手用自己的高质清唱片段做 Seed-VC target，--f0-condition True --auto-f0-adjust False
> （保留源音区，避免 Singer A 早期的低八度混杂）。6 个弱源产生的垃圾 fake 经质量门槛
> (rms>=0.01, voiced>=0.25) 过滤，移至 fake_rejected/。

## 实验一：合并交叉验证（5 折，按特征家族）
| 组 | 单歌手(仅A) AUC | 三歌手 AUC | 变化 |
|---|---|---|---|
| pitch_dynamics | 0.989 | 0.534 | 暴跌 → 单歌手是过拟合 |
| vri | 0.823 | 0.580 | 大幅下降 |
| voice_quality(HNR) | 0.937 | 0.859 | 最稳健 |
| CLEAN(14维干净特征) | 0.989 | 0.847 | 下降 |
| lowlevel / mfcc | 1.0 / 1.0 | 1.0 / 0.973 | 仍为产线混杂 |

## 实验二：留一歌手交叉验证 LOSO（训练2位测第3位）⭐
| 特征组 | testA | testB | testC | 平均AUC | 平均F1 |
|---|---|---|---|---|---|
| vri | 0.491 | 0.521 | 0.505 | **0.505** | ~0.50 |
| voice_quality | 0.596 | 0.752 | 0.474 | 0.608 | 低 |
| CLEAN | 0.643 | 0.669 | 0.539 | **0.617** | 低 |
| FULL | 1.000 | 0.958 | 1.000 | **0.986** | **≈0.0** |

## 两条核心结论
1. **FULL 的高 AUC(0.986) 是陷阱**：其 LOSO 准确率仅 ~0.50、F1≈0。MFCC/energy_dynamic
   等产线特征能在同一歌手内部完美排序真假，但判决阈值无法跨歌手迁移 → 表面性能是产线
   混杂的假象，并非真正的 AI 翻唱泛化检测能力。
2. **VRI 跨歌手 AUC=0.505 ≈ 随机**：单歌手时 VRI 看似有效(0.82)，LOSO 揭示它主要捕捉
   「某歌手的颤音个人习惯」而非通用 AI 痕迹。可解释风格特征整体仅泛化到 ~0.62，
   其中 HNR(气声/嗓音质量) 相对最稳健但在 Singer C 上失效。

## 对报告的意义
- 第四章：用 LOSO 表格 + 「单歌手 vs 三歌手」对比，展示泛化是核心挑战。
- 第六章局限性：① 高 AUC 可能来自产线混杂而非真信号(用 F1/准确率交叉验证)；
  ② 风格特征易过拟合单一歌手；③ 三歌手仍不足以得出强泛化结论。
- 这是一个诚实且有学术价值的「负面/受限结果」，比虚高的 1.0 更可信。

## 复现
```bash
python scripts/rebuild_metadata.py
python scripts/extract_features.py --metadata data/demo_data/metadata.csv --output outputs/features_fixed.csv
python scripts/honest_eval.py --features outputs/features_fixed.csv
python scripts/cross_singer_eval.py --features outputs/features_fixed.csv
```

## 待确认事项
- Singer A 的 fake 仍是低八度(早期用低音 target)，与 B/C 的 recipe 不一致。
  已备好 data/reference/target_voice_singer_a.wav，可一键重生成以完全统一三歌手。
  （CLEAN/LOSO 用八度不变特征，已部分规避；但统一后产线混杂分析更干净。）

---

## 补充：Singer A 统一 recipe 后重跑（v3，三歌手伪造方式完全一致）
将 Singer A 的 fake 由「低音target/低八度」改为与 B/C 相同的「自身target/保留音区」后重跑：

### LOSO 平均 AUC 对比
| 特征组 | 统一前 | 统一后 |
|---|---|---|
| voice_quality(HNR) | 0.608 | 0.690 |
| vri | 0.505 | 0.552 |
| CLEAN | 0.617 | 0.699 |
| FULL | 0.986(F1崩) | 1.000(F1仍崩,singer_c=0.11) |

### 结论
1. 伪造方式跨歌手一致后，可解释风格特征的真实泛化提升（CLEAN 0.62→0.70，HNR 0.61→0.69），
   说明统一的 AI 生成流程让检测器能学到可迁移信号 → 验证了统一 recipe 的必要性。
2. HNR 现对三位歌手 LOSO 全为正(A0.58/B0.70/C0.79)，是最可迁移的真实风格信号。
3. CV 中 pitch 由 0.581 崩到 0.327，确认旧的 pitch 区分力来自 Singer A 低八度混杂，现已消除。
4. FULL 仍 AUC=1.0 但 singer_c F1=0.11，残留 MFCC 产线混杂(real走Demucs/fake走Seed-VC+bigvgan,
   频谱系统性不同)，与音区无关，为后续需控制的混杂(如对 real/fake 统一重采样链或加噪对齐)。

> 备份：旧低八度 fake 存于 data/demo_data/fake_singer_a_octavelow.bak/（可回退对照）。
