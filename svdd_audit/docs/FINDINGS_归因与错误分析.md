# 实验发现 VI：归因与错误分析

## 1-3. 三组二分类 × 四特征集 (5折CV, AUC)
| 设置 | FULL | CLEAN | HNR | VRI |
|---|---|---|---|---|
| real vs fake | 1.000 | 0.752 | 0.787 | 0.419 |
| real vs real_vocoded | 1.000 | 0.271 | 0.456 | 0.365 |
| real_vocoded vs fake | 0.988 | 0.714 | 0.743 | 0.507 |
(完整 F1/Acc 见 attribution_classification.csv)

## 三个归因结论
1. **FULL = 声码器/产线探测器**：FULL 连 real vs real_vocoded(仅差一个 bigvgan,演唱内容相同)
   都 AUC=1.0,证明其检测力主要来自声码器指纹而非演唱真伪。real_vocoded vs fake 仍 0.988
   说明 DiT 生成的 mel 也带可分签名。
2. **HNR 是真信号**：real_vocoded vs fake(两者都过声码器)HNR 仍 0.743;且 real vs real_vocoded
   HNR 仅 0.456≈随机(声码器不改 HNR)。故 HNR 的真伪区分非声码器伪迹,是真实嗓音质量差异。
3. **VRI 无效**:三设置 0.42-0.51,连 pooled real vs fake 都≈随机。
4. **CLEAN 声码器不变**:real vs real_vocoded 仅 0.271(声码器保留风格特征),证明 CLEAN 非产线伪迹。

## 4. FP/FN 案例 (CLEAN 模型, 见 attribution_fp_fn.csv)
每协议各 2 FP + 2 FN,含路径/真实标签/预测分/HNR/VRI/f0_jitter/异常维度(z-score)。典型:
- real_vs_fake FP: singer_c_tiankong_017(真唱误判AI,prob0.88) 因 vibrato_depth/f0_jitter/micro_variation z>+2.4(颤音异常强);
- real_vs_fake FN: singer_c_juhao_029_fake(AI误判真,prob0.12) 因 hnr_mean z=-1.2(气声重,像真人);
- realvoc_vs_fake FN: singer_c_juhao 的 fake 反复被误判真(气声/vri 低),juhao(句号)是最难的歌。

## 5. 特征归因: Cohen's d + 方差分解 η² (real vs fake 数据, 见 attribution_feature_stats.csv)
| 特征 | Cohen d | η²_label | η²_singer | L/S |
|---|---|---|---|---|
| hnr_low_ratio | 0.56 | 0.073 | 0.236 | 0.31 |
| mfcc_2_mean | 0.33 | 0.027 | 0.186 | 0.15 |
| energy_dynamic | -0.26 | 0.016 | 0.071 | 0.23 |
| hnr_mean | -0.21 | 0.011 | 0.471 | 0.02 |
| f0_jitter | 0.05 | 0.001 | 0.134 | 0.00 |
| vibrato_depth_mean | -0.04 | 0.000 | 0.121 | 0.00 |

**核心机制发现**:所有可解释特征的方差都被歌手身份主导(η²_singer ≫ η²_label)。
即便判别力最强的 hnr_mean,47%方差来自歌手、仅1%来自真伪。这解释了为何跨歌手泛化崩溃:
特征本质是'歌手指纹'多于'真伪指纹'。相对最偏真伪的是 hnr_low_ratio(气声比例,L/S=0.31)。

## 对报告的意义
- 用 real-vs-real_vocoded 的 FULL=1.0 作为'高分来自产线'的决定性证据。
- 用 η²_label vs η²_singer 表解释泛化失败的根因(特征编码歌手身份)。
- 指出可改进方向:做歌手内标准化(per-singer z-score)或对抗歌手身份的特征,
  以剥离歌手成分、放大真伪成分(尤其 hnr_low_ratio)。

## 产物
scripts/attribution_analysis.py; outputs/attribution_{classification,fp_fn,feature_stats}.csv
