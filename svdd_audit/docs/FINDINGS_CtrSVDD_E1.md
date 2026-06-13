# CtrSVDD 公开基准 E1：分组 EER（bonafide vs spoof）

## 设置
CtrSVDD train+dev,全16kHz(无采样率混杂)。平衡子集:每attack(A01-A08)300 spoof + 2400 bonafide(m4singer/opencpop),共4800。
我们的特征 FULL/CLEAN/HNR/VRI,5折CV得分,EER%(越低越好)。指标用官方 eer.py。

## 结果(EER%)
| 分组 | CLEAN | FULL | HNR | VRI |
|---|---|---|---|---|
| overall | 35.96 | 18.88 | 45.17 | 44.50 |
| voc:hifigan(A01,A02,A05) | 30.33 | 13.67 | 44.76 | 41.12 |
| voc:ddsp(A03) | 29.33 | 26.04 | 46.00 | 38.94 |
| voc:nsf-hifigan(A04,A06,A07,A08) | 41.67 | 20.92 | 45.25 | 48.92 |
| atk:A05(NaiveRNN,hifigan) | 27.33 | 8.06 | - | - |
| atk:A06/A07/A08(NSF-SVC) | ~43-47 | ~18-25 | - | - |
(官方AASIST基线 EER 10.4%, 仅供参考;我们是手工特征+RF的分析,非追SOTA)

## 三个核心发现
1. **HNR/VRI 在 CtrSVDD ≈随机(45%/44.5%)**:在自建数据(B站real vs Seed-VC)上HNR是最强真信号(AUC0.74),
   但CtrSVDD(干净录音室bonafide vs录音室级SVS/SVC)上完全失效。=> HNR判别力是数据集特异的(依赖录音条件),
   非通用AI痕迹。强烈支撑论文'别乱归因'主线:跨数据集,所谓'风格真信号'不成立。
2. **只有FULL有效(18.88%)**,靠MFCC/谱;且随声码器强变:hifigan13.7%<nsf20.9%<ddsp26%。
   声码器家族决定可检测性 => 为E2(声码器是否为混杂)铺垫。
3. CtrSVDD全16k,声码器混杂是纯频谱/伪迹(无采样率混杂),比自建数据更干净的检验场。

## 产物
scripts/{ctrsvdd_subset_extract,ctrsvdd_eer_analysis}.py;
outputs/{ctrsvdd_features_e1.csv(4800), ctrsvdd_eer_e1.csv}
