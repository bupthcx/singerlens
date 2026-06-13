# 实验发现 III：声码器对齐对照(copy-synthesis)——处理 MFCC/产线混杂

## 动机
之前 FULL 在 LOSO 下 AUC=1.0 但 F1≈0，疑为产线混杂：fake 过了 bigvgan 声码器，real 没过。
诊断显示泄露主要在静态倒谱均值 mfcc_mean（信道着色）。为根因控制，做声码器对齐对照：
将 real 也过同一个 bigvgan(仅 mel->vocoder,不做转换)得到 real_vocoded，两类共享声码器足迹。

## 方法
copy_synth_batch.py：real(44.1k) -> mel_fn -> bigvgan vocoder_fn -> real_vocoded。
验证：f0/有声率/能量与原 real 一致，仅过声码器。98 个 real 全部处理。
对照评估 vocoder_control_eval.py：比较 (real原始 vs fake) 与 (real_vocoded vs fake)。

## 结果（CV=5折合并, LOSO=留一歌手）
| 特征组 | orig CV | orig LOSO | voc CV | voc LOSO |
|---|---|---|---|---|
| lowlevel | 1.000 | 1.000 | 0.757 | 0.719 |
| mfcc_mean | 0.953 | 0.604 | 0.948 | 0.547 |
| mfcc | 0.955 | 0.604 | 0.964 | 0.563 |
| pitch_dynamics | 0.372 | 0.522 | 0.478 | 0.555 |
| vri | 0.517 | 0.552 | 0.499 | 0.567 |
| voice_quality(HNR) | 0.814 | 0.690 | 0.765 | 0.675 |
| CLEAN | 0.784 | 0.699 | 0.721 | 0.649 |
| FULL | 1.000 | 1.000 | 0.983 | 0.657 |

## 结论
1. FULL 的跨歌手 AUC 从虚高的 1.000 降到 0.657，lowlevel 从 1.000 降到 0.719：
   证实之前的'完美'泛化主要是产线(能量/静音结构 = Demucs vs 声码器)混杂，对照后回归诚实。
2. MFCC 差异并非声码器伪迹：real_vocoded 过了同一 bigvgan，MFCC 仍分到 CV 0.96。
   它来自 DiT 扩散模型生成的 mel 频谱签名(真实的'被AI转换'线索)，但跨歌手不迁移(LOSO 0.56)，
   属模型特异、与歌手纠缠，不能作为通用风格证据。
3. HNR(voice_quality)过同声码器后 LOSO 仍 0.675≈不变——气声/嗓音质量是既非声码器、
   也非音区伪迹的真正可迁移信号，应作为可解释检测的核心维度。

## 对报告的意义
- 提供方法学严谨性：用声码器对齐对照区分'真信号 vs 声码器伪迹'(反欺骗标准做法)。
- 诚实修正：报告主检测器的跨歌手性能应引用对照后的数值(FULL/CLEAN LOSO≈0.65)，
  而非未对照的虚高 1.0。
- 推荐协议：训练负类用 real_vocoded 而非 real原始，可避免检测器退化为'声码器检测器'。

## 产物
data/demo_data/real_vocoded/(98), outputs/features_vocoded.csv, outputs/vocoder_control.csv,
scripts/{copy_synth_batch.py(seed-vc目录), vocoder_control_eval.py}
