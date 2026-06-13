# 实验发现 V：歌词-演唱情感一致性（方案模块四,案例解释用）

## 方法
- 歌词识别：Whisper large-v3 转录分离人声(中文,带时间戳) -> outputs/lyrics/*.json。
  清洗去除 Whisper 在前奏/安静段的幻觉(作词/作曲等credit刷屏、单字重复、整行去重)。
- 歌词情感：ModelScope iic/nlp_structbert_emotion-classification_chinese-base(7类),
  逐行求 valence=P(高兴+喜好)-P(悲伤+愤怒+恐惧+厌恶),全歌平均。
- 演唱情感：emotion2vec_plus_large(funasr,9类),逐片段求 valence=P(happy)-P(sad+angry+fearful+disgusted),
  对每首真唱/AI翻唱各取≤12片段平均。
- 一致性 = 1 - |歌词valence - 演唱valence|/2,范围[0,1]。

## 结果(每首真唱 vs AI翻唱)
| 歌曲 | 歌词val | 真唱val | AI val | 一致性(真) | 一致性(AI) |
|---|---|---|---|---|---|
| 单依纯 DearFriend | -0.42 | -0.41 | -0.80 | 1.00 | 0.81 |
| 单依纯 天空 | -0.36 | -0.15 | -0.44 | 0.89 | 0.96 |
| 孙燕姿 开始懂了 | -0.07 | -0.07 | -0.10 | 1.00 | 0.98 |
| 孙燕姿 天黑黑 | -0.36 | -0.61 | -0.08 | 0.88 | 0.86 |
| 孙燕姿 我要的幸福 | +0.10 | -0.33 | -0.30 | 0.78 | 0.80 |
| 邓紫棋 句号 | -0.43 | +0.67 | +1.00 | 0.45 | 0.28 |
| 邓紫棋 天空没有极限 | -0.26 | +0.25 | +1.00 | 0.74 | 0.37 |
| 邓紫棋 Walk on Water | -0.10 | -0.08 | +0.73 | 0.99 | 0.58 |
| **平均** | | | | **0.831** | **0.696** |

## 核心发现
真唱的歌词-演唱情感一致性(0.831)高于 AI 翻唱(0.696)。AI 翻唱更难保持与歌词情感的契合,
支撑'AI 难复制情感微表达'的论点。邓紫棋的歌尤为明显:AI 把悲伤歌词的演唱情感推到极端 happy(+1.0)。

## 局限性(报告需写明)
1. emotion2vec 对高能量唱腔(如邓紫棋强爆发)倾向把高唤醒(arousal)误读为 happy(valence),
   SER 的 arousal/valence 混淆是已知问题 -> 情感一致性仅作案例解释,不作主检测特征(与方案一致)。
2. Whisper 在歌曲上会幻觉(前奏credit刷屏/重复),清洗后单依纯《开始懂了》仅余1行歌词,该首歌词情感不可靠。
3. 歌词与演唱未做精确时间对齐(按整首平均),细粒度情感波动未捕捉。

## 产物
outputs/lyrics/*.json(9首歌词), outputs/emotion_consistency.csv,
scripts/{transcribe_lyrics.sh, emotion_consistency.py}; 环境 qwen3-asr(funasr+modelscope+emotion2vec)。
