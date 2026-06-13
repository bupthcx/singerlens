# 跨语种 cleanliness-axis:普通话 / 粤语 / 英文 (FINDINGS_crosslingual_cleanliness.md)

## 目的
全部训练/扰动数据均为普通话歌曲。本实验检验**制作干净度混杂(cleanliness axis)是否中文特有**:
用**粤语**(独立汉语方言、声调语言)与**英文**(非声调、音系与中文迥异)真唱走同一套制作扰动 +
同一个普通话训练检测器,看带限是否同样把它们翻成 fake。脚本 `scripts/cantonese_cleanliness.py`、
`scripts/english_cleanliness.py`。

## 设置
- 真唱:粤语 = WildSVDD 粤语真唱 35 首 → 105 片段;英文 = Taylor Swift《Love Story》(Fearless,bilibili
  官方 MV,原生 48kHz,真人独唱)经 Demucs 分离 → 26 个有人声 10s 片段。
- 检测器:**SingerLens 普通话域内 FULL RF**(与 Exp3 同一个,即跨语言应用)。
- 扰动:与 Exp3 同口径(resample-8k / lowpass-4k/3k / MP3-32k / noise / reverb)。

## 结果:三语种 flip→fake 率对比(真唱被翻成 fake 的比率)

| 扰动 | 普通话 | 粤语 | **英文** |
|---|---:|---:|---:|
| clean | 0% | 0% | **0%** |
| **resample_8k** | 46.5% | 39.0% | **50.0%** |
| lowpass_4k | 13.0% | 13.3% | **15.4%** |
| **lowpass_3k** | 46.5% | 32.4% | **42.3%** |
| mp3_32k | 25.0% | 0.0% | **15.4%** |
| noise_10db | 0% | 0% | **0%** |
| (clean 基线 mean fake-score) | 0.085 | 0.223 | 0.200 |

(英文 mean/Δ:resample_8k 0.486/+0.287、lowpass_3k 0.494/+0.294、lowpass_4k 0.405/+0.205;noise −0.148;
完整见 `english_cleanliness.csv`。)
| reverb | 0.232 / 0% / +0.008 | 0.232 / 4.5% / +0.147 |

## 关键发现
1. **cleanliness 轴语言一般、非中文特有**:带限把**粤语与英文**真唱同样翻成 fake。英文(非声调、与训练
   数据音系迥异)的 resample-8k flip **50%**(甚至略高于普通话 46.5%)、lowpass-3k 42%;粤语 39%/32%。
   三语种方向与量级一致 → **混杂关于音频制作/频谱足迹,与语言无关**。lowpass-4k 三语种几乎相同(13–15%)。
2. **最普适的分量是频谱带限本身**:resample/lowpass 三语种强成立;**MP3 编解码分量跨语种偏弱且不稳**
   (普通话 25% / 英文 15% / 粤语 0%)→ codec 足迹更语料/语言相关,带宽足迹最普适。
3. **加噪跨语种同样不触发**(三语种 noise flip 0%)→ 再次排除"任意脏化",锁定带宽/频谱轴。
4. **诚实 nuance**:粤语/英文 clean 基线偏高(0.22/0.20 vs 普通话 0.085),即普通话检测器对非普通话真唱
   本就略偏"假"(轻微语言-域偏移),但仍 0% flip(<0.5),需经带限才翻 → 偏移与 cleanliness 翻转是两回事。

## 结论
制作干净度/带宽混杂在**粤语(声调汉语)与英文(非声调、音系迥异)**上同样成立,**该问题非中文特有,
是语言一般的**;最普适的成分是频谱带限。
*"The production-cleanliness confound is not language-specific: band-limiting flips real Cantonese and
English vocals to fake just as it does Mandarin (English resample-8k flip 50%, even above Mandarin's 46%)."*

## 限制
- 英文为**单首歌(Love Story,26 片段)**;粤语 35 首/105 片段;单检测器(SingerLens 普通话);合成扰动
  非真实平台分布。英文样本量小,但 flip 量级与中文一致已足以驳"中文特有"。
- 未做英文的连续带宽扫描/跨数据集检测(只测了 cleanliness 扰动这一因果点)。

## 产物
- `scripts/{cantonese,english}_cleanliness.py`
- `outputs/cantonese_cleanliness.{csv,_scores.csv}`、`outputs/english_cleanliness.{csv,_scores.csv}`
- 英文音频:`/home/admin2/xf/wildsvdd/english_test/`(lovestory.wav + demucs vocals)
