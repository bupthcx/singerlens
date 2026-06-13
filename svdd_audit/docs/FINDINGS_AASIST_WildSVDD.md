# AASIST -> WildSVDD: 强神经模型在真实野生AI翻唱上的跨域泛化(vs RF FULL)

## 设置
AASIST(rawnet,30ep)训练于 CtrSVDD(std_train2400)/SingerLens(sl_train542)/WildSVDD(wild_train82,per-song vocals),
测试于 WildSVDD(165首 Demucs vocals,81real/84fake)。统一16kHz,pad/crop 4s,label 1=bonafide,score=P(spoof)。

## 结果(vs RF FULL)
| 协议 | AASIST EER | AASIST AUC | RF_FULL EER | RF_FULL AUC |
|---|---|---|---|---|
| Wild->Wild 域内 | 59.03* | 0.400* | 14.81 | 0.935 |
| CtrSVDD->Wild | 32.12 | 0.748 | 49.36 | 0.494 |
| SingerLens->Wild | 54.54 | 0.430 | 56.44 | 0.447 |
*AASIST域内不可靠:wild_train仅82 per-song样本,欠拟合(AUC0.40,F1 0)。RF域内466clips/5折AUC0.935为有效参照。

## 诚实结论(不过度宣称)
1. 两模型在真实野生数据上均严重退化:CtrSVDD->Wild AASIST 0.987->0.748、RF 0.90->0.49;
   SingerLens->Wild 两者≈随机(0.43/0.45)。
2. AASIST从受控benchmark(CtrSVDD)迁移保留部分信号(AUC0.748),优于手工特征(0.49)——
   神经模型学到的更可迁移,**非完全随机**。诚实写明,不宣称'AASIST也归零'。
3. 从自建SingerLens(Seed-VC)迁移两模型都崩到随机:自数据与野生AI翻唱差异过大。

## 核心命题(成立,带nuance)
real-world generalization gap 对强模型AASIST同样存在(EER6.5->32,AUC0.99->0.75),
但退化幅度model-dependent:AASIST-from-CtrSVDD非随机(0.748),AASIST/RF-from-SingerLens及RF-from-CtrSVDD≈随机。
=> 真实野生AI翻唱的跨域崩塌不是手工特征独有问题,而是强模型也面临的real-world gap;
   神经模型从受控benchmark迁移略好,但远未达可用,且自建数据完全不迁移。

## 局限
WildSVDD域内AASIST样本太少(82)欠拟合,域内参照以RF(466clips)为准;AASIST per-song vs RF per-clip粒度不同;
WildSVDD带伴奏经Demucs(含分离伪迹),为真实in-the-wild条件。

## 产物
CtrSVDD2024_Baseline/train_aasist_lovo.py(加--extra-tests); ctrsvdd/aasist_out/wild_*.csv;
ctrsvdd/aasist_csv/wild_{all,train,test}.csv
