# Emotion-Consistency Cross-Domain Test (FINDINGS_emotion_consistency_crossdomain.md)

## Purpose
检验假说:情感一致性(歌词情感 vs 演唱情感)是**高层语义信号、与制作链路正交**,
是否因此**跨域更稳健**(降幅小于 AASIST/手工特征)?若成立 → 情感模块从支线变正面出口;
若同样崩 → 诚实补充"情感轴也无法单独解决跨域"。脚本 `scripts/wild_emotion_consistency.py`。

## Setup
- 零训练语义分数:per-song consistency = 1−|lyric_valence − singing_valence|/2;
  lyric_valence = structbert 中文情感(Whisper large-v3 转录歌词);singing_valence = emotion2vec SER。
  score = −consistency(低一致性→fake)。WildSVDD-T02 平衡子集 **80 首(40 real / 40 fake)**。
- 基线:SingerLens 既有 9 真+9 假 consistency 的 AUC。对照:AASIST/RF/SSL 跨域到 Wild。

## 结果 —— 假说被证伪(诚实负面)

| 设置 | AUC |
| --- | ---: |
| 情感一致性 @ SingerLens(基线) | 0.679 |
| **情感一致性 @ WildSVDD(80 首)** | **0.412(低于随机)** |
| └ 仅转录较好子集(lines≥5,n=43) | 0.395 |
| 情感降幅 ratio(SL→Wild) | **0.61** |
| 对照 AASIST CtrSVDD→Wild | 0.748(降幅 ratio 0.76) |
| 对照 RF-FULL CtrSVDD→Wild / SL→Wild | 0.520 / 0.447 |

- WildSVDD 上 **real 一致性 0.626 < fake 0.686 → 关系翻转**(AI 翻唱反而"更一致")。
- **降幅比 AASIST 更大**(0.61 < 0.76):情感一致性跨域**比制作链路特征还脆**,非更稳健。

## Interpretation(诚实)
1. **假说不成立**:情感一致性**不是**跨域稳健信号;在野生数据上塌到 0.41 且反相关,降幅大于 AASIST。
   "与制作链路正交 → 跨域更稳健"无数据支撑。
2. **非转录质量 artifact**:28% 野生歌 Whisper 仅转出 ≤2 行(分离人声+多语种退化),但**剔除后(lines≥5)
   AUC=0.395 仍崩** → 崩塌来自情感信号本身的脆弱/域特异,不只是转录退化。
3. **机制推测**:emotion2vec SER 对野生多歌手/多语种(粤语)/强唱腔噪声大(自建数据已知把强唱腔误判 happy);
   wild 歌词语义价分布与自建集不同;故"歌词-演唱一致性"这一语义轴在野生域不可靠。
4. **补充证据(对主线有用)**:**没有任何单一轴(制作 OR 语义情感)能单独解决跨域问题** →
   进一步坐实 **minimal supervised adaptation(目标标签)是唯一可行出口**(5.9b);情感模块保持
   **支线/案例解释**定位(SingerLens 域内 0.68 + 案例),**不提升为跨域正面出口**。

## Limitation
- 80 首子集、per-song;Whisper 在野生分离人声上转录退化(28% ≤2 行)是已知噪声源(已用 lines≥5 子集旁证)。
- emotion2vec SER 的 valence 映射粗糙(happy/sad/...),对强唱腔/多语种鲁棒性差(自建集已记)。
- 仅测一致性这一种情感构造;不排除更精细的情感/韵律构造在跨域有信号,但**本构造无**。

## 产物
- `scripts/wild_emotion_consistency.py`、`outputs/wild_emotion_consistency.csv`、`outputs/wild_emotion_auc.csv`
- 转录 `outputs/wild_lyrics/*.json`(80 首),子集 `outputs/wild_emotion_subsample.csv`
