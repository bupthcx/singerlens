# 实验发现：真假检测中的'制作链路泄露'诊断与修复

## 1. 现象
对 31 真 + 31 假（singer_a / DearFriend）提取特征后训练 RandomForest，
**所有特征组（含仅 4 维的 baseline）交叉验证 AUC 均为 1.0**。满分本身是危险信号。

## 2. 诊断：真假之间存在平凡的系统性差异（捷径）
| 混杂因素 | 证据 | 性质 |
|---|---|---|
| 采样率不一致 | real=16kHz（原生），fake=44.1kHz | 产线 |
| 响度不一致 | rms_mean real=0.052 vs fake=0.043 | 产线 |
| F0 八度跳变 | real f0_mean=381Hz（向上跳谐波），fake f0_min 踩 C2 地板（向下跳次谐波） | 提取 |
| fake 整体低八度 | f0_mean real≈390 vs fake≈140 | **数据生成** |
| 静音结构不同 | energy_dynamic real=404 vs fake=74 | 产线（Demucs 有空隙） |
| 谱形差异 | MFCC 多维完全不重叠（cohen_d>5） | 产线 |

分类器靠这些'非演唱风格'的差异走捷径，VRI 等核心特征未被真正检验。

## 3. 修复（features.py）
- **响度归一化**：load_audio 统一到 -20 dBFS（消除 rms 泄露）。
- **F0 八度校正**：extract_f0 收紧音域 A2–A5，用中值滤波估稳健参考音高，
  按整数八度折叠每帧到参考 ±0.5 八度内（只去突跳，保留 VRI 所需微扰）。
- 效果：f0_range_semitones 由 real18.4/fake20.4 → real11.0/fake11.1（合理且不再泄露）；
  rms_mean → 0.080/0.085（基本相等）。

## 4. 残留混杂（需在报告局限性中说明，并在主检测器中剔除）
- 绝对音高 f0_mean/min/max：fake 低八度（数据生成缺陷）→ 剔除，改用八度不变的 f0 动态。
- energy_dynamic：尺度无关比值，反映静音结构差异 → 剔除。
- MFCC、原始 RMS：产线相关 → 剔除。

## 5. 诚实结果（5 折交叉验证）
| 组 | AUC |
|---|---|
| pitch_dynamics（f0_std/range/jitter，八度不变） | 0.989 |
| voice_quality（HNR） | 0.937 |
| vri（颤音规律性） | 0.823 |
| stability（长音稳定，单特征） | 0.530（≈随机，诚实局限） |
| **CLEAN（14 个干净特征，报告主检测器）** | **0.989 / Acc 0.95** |

> 结论：剔除产线/数据混杂后，仅凭八度不变、响度不变的可解释风格特征即可达到
> AUC≈0.99，支撑核心论点；同时长音稳定性单独接近随机，作为诚实局限写入报告。

## 6. 复现
```bash
python scripts/extract_features.py --metadata data/demo_data/metadata.csv --output outputs/features_fixed.csv
python scripts/honest_eval.py --features outputs/features_fixed.csv
```
