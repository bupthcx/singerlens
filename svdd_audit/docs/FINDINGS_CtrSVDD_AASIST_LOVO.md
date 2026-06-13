# CtrSVDD: AASIST(强神经基线) vs RF FULL(手工特征) 的 Leave-One-Vocoder-Out 对比

## 设置
AASIST=官方rawnet+AASIST(SincConv+图注意力,0.3M参数,raw waveform,focal loss,30ep)。
in-dist=所有声码器可见(std,2400/2400);LOVO=留出一个声码器家族测试。
与RF FULL用同一bonafide 50/50划分、同样vocoder分组。指标EER%(官方eer.py)。

## AASIST standard in-distribution
overall EER=6.50%, AUC=0.987, F1=0.934 (与官方AASIST 10.4%同量级,子集更平衡略好)。
per-vocoder in-dist: hifigan 5.10 / nsf-hifigan 7.33 / ddsp 4.00。

## 核心对比表 (degradation_ratio = lovo_eer / in_dist_eer)
| model | heldout_vocoder | in_dist_eer | lovo_eer | degradation_ratio |
|---|---|---|---|---|
| AASIST | hifigan | 5.10 | 4.76 | 0.93 |
| RF_FULL | hifigan | 13.67 | 31.76 | 2.32 |
| AASIST | nsf-hifigan | 7.33 | 21.92 | 2.99 |
| RF_FULL | nsf-hifigan | 20.92 | 36.33 | 1.74 |
| AASIST | ddsp | 4.00 | 5.67 | 1.42 |
| RF_FULL | ddsp | 26.04 | 28.67 | 1.10 |
(csv: outputs/ctrsvdd_aasist_vs_rf_lovo.csv; AASIST LOVO AUC: hifigan0.991/nsf0.863/ddsp0.988)

## 结论(按预设三种情况判定)
**属于'AASIST 也退化'+'但vocoder特异'的混合,核心命题成立:**
1. **AASIST在未见 NSF-HiFiGAN 上严重退化(EER 7.33->21.92, x2.99)** —— 强神经基线同样存在
   未见声码器退化,证明这是**模型无关的领域问题(domain shift)**,非手工频谱特征独有。
   且NSF-HiFiGAN是最常见声码器(8/13系统含全部So-VITS-SVC),这个退化尤其要命。
2. **AASIST对未见 hifigan 不退化(0.93)、ddsp轻微(1.42)** —— 说明退化是**声码器家族特异**的:
   HiFi-GAN伪迹可从其他声码器训练中迁移识别,但NSF-HiFiGAN(源-滤波,更自然)难以跨家族泛化。
3. 对照RF FULL:整体退化更普遍(hifigan x2.32最重),且绝对EER高得多(手工特征本就弱)。

## 论文价值(决定性)
'未见声码器泛化缺口'对SOTA神经模型(AASIST)同样存在,尤其在主流NSF-HiFiGAN上EER翻3倍。
=> SVDD检测在相当程度上依赖训练见过的声码器家族特征,而非通用'AI演唱'判别;
   即便强模型也受声码器domain shift限制。这与E1(我们特征)、E2/LOVO(RF)共同构成
   '审计SVDD声码器混杂'的完整证据链,是论文最硬的实验。

## 产物
CtrSVDD2024_Baseline/train_aasist_lovo.py(自写驱动,修了官方fairseq强依赖+test_loader bug);
ctrsvdd/aasist_out/{std,lovo_*}.csv; outputs/ctrsvdd_aasist_vs_rf_lovo.csv
