# 实验发现 IV：最终数据集(3歌手×3歌,520片段)泛化总表

## 数据集
| 歌手 | 歌曲(3首) | real | fake |
|---|---|---|---|
| A 单依纯 | DearFriend, 开始懂了, 天空 | 92 | 92 |
| B 孙燕姿 | 开始懂了, 天黑黑, 我要的幸福 | 83 | 74 |
| C 邓紫棋 | 天空没有极限, 句号, Walk on Water | 90 | 89 |
| 合计 | 9首 | 265 | 255 |
fake=Seed-VC(每歌手自身target,保留音区); real_vocoded=real过同一bigvgan(声码器对照); 弱源垃圾fake已过滤。

## 三种泛化协议对比 (AUC)
| 特征组 | Leave-One-Song-Out(同歌手留一首) | Leave-One-Singer-Out(留一歌手) | LOSinger(vocoded对照) |
|---|---|---|---|
| vri 颤音 | 0.548 | 0.429 | 0.563 |
| voice_quality HNR | 0.764 | 0.673 | 0.620 |
| CLEAN 干净集 | 0.776 | 0.679 | 0.627 |
| FULL 全特征 | 1.000 | 1.000 | 0.583 |

## 核心结论
1. **泛化难度排序**：见过该歌手(Song-Out, CLEAN 0.78) > 没见过该歌手(Singer-Out, 0.68)。
   检测器对'同一歌手的新歌'泛化明显好于'全新歌手'，说明仍有歌手身份成分。
2. **FULL 的完美(1.0)是产线假象**：声码器对照(real_vocoded vs fake)下 Singer-Out 从 1.000 崩到 0.583。
   证明 MFCC/能量的'完美'来自 real=Demucs vs fake=声码器 的制作链差异,而非真实 AI 检测能力。
3. **HNR(气声/嗓音质量)是唯一稳健可迁移的真信号**：三协议均 0.62-0.76,过声码器仍保持,
   是可解释检测应依赖的核心维度。
4. **VRI 颤音规律性跨歌手≈随机(0.43)**:单歌手时的高分是过拟合个人习惯,不具通用性。

## 给报告的建议
- 第四章主表用本三协议对比 + 声码器对照,展示'看似完美实为混杂'的发现。
- 主检测器性能引用 CLEAN 在 Song-Out(0.78,有歌手先验)与 Singer-Out(0.68,无先验)两个数字。
- 第六章局限性:特征仍部分编码歌手身份;3歌手不足以强泛化;HNR单点最稳但仍有限。

## 产物
outputs/{cross_song_loso.csv, cross_singer_loso.csv, vocoder_control.csv,
features_fixed.csv(520), features_vocoded.csv(520)}
