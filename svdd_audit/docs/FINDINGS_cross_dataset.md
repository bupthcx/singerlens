# Cross-dataset Transfer: 标准域内评测高估泛化 (CCF-A 主命题证据)

## 协议
两数据集(A=CtrSVDD子集2400+2400; B=SingerLens 6歌手552+533),同一特征管线。
域内: A->A / B->B (5折CV); 跨数据集: A->B / B->A (训练一个测另一个,scaler在训练集拟合)。
特征 FULL/CLEAN/HNR/VRI; 指标 EER/AUC/F1 (官方eer.py)。

## 结果 EER% (越低越好)
| 协议 | FULL | CLEAN | HNR | VRI |
|---|---|---|---|---|
| A->A CtrSVDD域内 | 18.88 | 35.96 | 45.17 | 44.50 |
| B->B SingerLens域内 | 0.00 | 34.75 | 36.59 | 53.09 |
| A->B 训Ctr测SL | 42.03 | 50.69 | 46.17 | 48.57 |
| B->A 训SL测Ctr | 46.96 | 46.67 | 47.33 | 49.83 |

## AUC (FULL): A->A 0.899 / B->B 1.000 / A->B 0.615 / B->A 0.541

## 核心结论(论文主命题)
1. 域内评测严重高估泛化:FULL 域内 AUC 0.90(A)/1.00(B)、EER 0-19%,
   跨数据集全部塌到 AUC 0.54-0.62、EER 42-47% ≈ 随机。
2. B->B FULL EER=0.00 是极端教学案例:SingerLens 的采样率/产线混杂被 FULL 轻松利用,
   标准评测会报'100%解决';但同一检测器迁到 CtrSVDD(B->A)=46.96% ≈ 随机 => 完全无用。
3. 不同数据集混杂来源不同(SingerLens=采样率/产线; CtrSVDD=声码器家族),
   但'标准域内评测高估真实泛化'是普遍现象。
4. HNR/VRI 跨数据集亦≈随机,再次印证可解释风格特征无通用真伪信号。

## 论文价值
这是 SVDD-Audit 的主命题直接证据:current SVDD eval (within-dataset/standard split)
substantially overestimates cross-distribution generalization。配合 LOVO(声码器家族)、
copy-synthesis(混杂诊断),构成'评测范式不可靠'的完整论证。

## 待加强(下一步)
- 用 AASIST/WavLM 重做 cross-dataset(证明 model-agnostic,与LOVO一致);
- 加第三个公开数据集(SingFake/WildSVDD,需先验证可获取性);
- Leave-one-generator-family-out(LOGO)。

## 产物
scripts/cross_dataset_transfer.py; outputs/cross_dataset_transfer.csv

---
## 补充: AASIST(强神经基线) cross-dataset —— 证明 model-agnostic
AASIST(rawnet)在各向训练30ep。与RF FULL并列:
| protocol | AASIST EER | AASIST AUC | RF_FULL EER | RF_FULL AUC |
|---|---|---|---|---|
| A->A within CtrSVDD | 6.50 | 0.987 | 18.88 | 0.899 |
| B->B within SingerLens | 5.16 | 0.987 | 0.00 | 1.000 |
| A->B train Ctr test SL | 42.58 | 0.596 | 42.03 | 0.615 |
| B->A train SL test Ctr | 45.25 | 0.553 | 46.96 | 0.541 |

**决定性结论**: 域内 AUC 0.90-1.00(两模型两数据集均近乎完美) -> 跨数据集 AUC 0.55-0.62(≈随机),
对手工特征(RF FULL)与SOTA神经(AASIST)**同样成立**。即便AASIST域内仅5-6% EER(近完美),
跨数据集EER飙到42-45%≈随机。'标准域内评测高估泛化'是**模型无关**的根本问题。
csv: outputs/cross_dataset_aasist_vs_rf.csv

---
## 补充: 第三数据集 WildSVDD(真实野生AI翻唱, bilibili子集) cross-dataset
WildSVDD test_A bilibili子集: 下载165/192(86%存活),处理后466 clips(229real+237fake,164首),
每首中段30s->Demucs人声->切10s片->SingerLens特征。带label/singer/model(So-VITS-SVC/RVC等)。

### AUC
| 协议 | FULL | CLEAN | HNR | VRI |
|---|---|---|---|---|
| WildSVDD 域内CV | 0.935 | 0.793 | 0.746 | 0.624 |
| CtrSVDD -> WildSVDD | 0.494 | 0.338 | 0.378 | 0.535 |
| SingerLens -> WildSVDD | 0.447 | 0.470 | 0.451 | 0.537 |
### EER%: WildSVDD域内FULL 14.81; CtrSVDD->Wild 49.36; SingerLens->Wild 56.44

### 决定性现实价值结论
WildSVDD自身可解(域内AUC0.935),但CtrSVDD/SingerLens训练的检测器迁到野生AI翻唱AUC≈0.45-0.49
(随机/反相关)。=> 公开benchmark高分完全无法预测真实世界检测能力;AUC<0.5示严重域错配(特征语义翻转)。
WildSVDD带伴奏(Demucs分离),含真实in-the-wild的分离伪迹+混音+多模型,是最贴近现实的检验。
此结果直击审稿人Q5(现实价值):当前benchmark训练的SVDD检测器在真实互联网AI翻唱上失效。
脚本scripts/cross_dataset_wild.py; outputs/cross_dataset_wild.csv; outputs/wildsvdd_features.csv。
