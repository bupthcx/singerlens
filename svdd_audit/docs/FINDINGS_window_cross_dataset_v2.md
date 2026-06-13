# 窗级跨数据集迁移 v2 (FINDINGS_window_cross_dataset_v2.md)

## 1. 目的

窗级时序检测在 SingerLens 域内只能小幅提升、修不了 LOSO 泛化。本实验把窗级/MIL 方法搬到
**最难的 cross-dataset 设定**(CtrSVDD/SingerLens → WildSVDD 野生),回答:**局部窗级证据能否弥合
跨数据集泛化鸿沟?** 并用 pitch/HNR/energy 三通道时间轴解释失败机制。

口径:每 clip 滑窗 3s/1.5s 同 `extract_clip_features`。窗表 `window_features{,_ctrsvdd,_wild}.csv`
(SingerLens 1085clip / CtrSVDD E1 4800clip→12241 窗 / WildSVDD 466clip→2796 窗)。
窗级表示 clip_mean / MIL_mean / MIL_max / POOL_RICH;特征 FULL/CLEAN/HNR/VRI;
指标 AUC/EER + mean_score_real/fake(分数反转)。脚本 `window_cross_dataset_eval.py`、`..._cases.py`。

## 2. 主要结果(FULL,AUC | EER）

| 协议 | clip_mean | MIL_mean | MIL_max | POOL_RICH |
| --- | --- | --- | --- | --- |
| **域内** SingerLens | 0.984 \| 6.2 | **0.996 \| 3.6** | 0.990 \| 4.5 | 0.981 \| 6.3 |
| **域内** CtrSVDD | 0.894 \| 20.2 | **0.905 \| 18.5** | 0.880 \| 21.0 | 0.885 \| 20.3 |
| **域内** WildSVDD | 0.884 \| 20.8 | 0.886 \| 21.3 | 0.881 \| 20.4 | **0.895 \| 18.9** |
| CtrSVDD→Wild(all) | 0.464 \| 52.6 | 0.459 \| 52.2 | 0.431 \| 54.5 | 0.482 \| 52.0 |
| CtrSVDD→Wild(T02) | 0.503 \| 49.3 | 0.471 \| 50.1 | 0.447 \| 53.5 | 0.500 \| 49.3 |
| SingerLens→Wild(all) | 0.542 \| 48.9 | 0.517 \| 48.9 | 0.514 \| 50.9 | 0.559 \| 45.1 |
| SingerLens→Wild(T02) | 0.535 \| 49.5 | 0.523 \| 48.3 | 0.505 \| 52.8 | 0.549 \| 46.1 |

- **域内全部可学**(AUC 0.88–1.00),WildSVDD 自身 0.88–0.90 → Wild 非不可学,纯属迁移崩塌。
- **跨数据集全部塌到随机/反相关**(AUC 0.43–0.56,EER 45–55,退化 ~2.3–2.8×)。
  **没有任何窗级表示能弥合鸿沟**。T02 子集同形(那 30 NaN 行不影响)。

## 3. 是否复现 SingerLens 内部发现 —— 全部复现且更尖锐

1. **MIL_max 抬分不提升可分性**:域内 MIL_max 的 mean_score 系统性最高(Wild 域内 real0.50/fake0.77,
   SingerLens real0.43/fake0.85),但 AUC 不优于 mean;**跨数据集 MIL_max 一致最差**
   (CtrSVDD→Wild 0.431,SingerLens→Wild 0.514),且把 real/fake 分数一起抬到 ~0.72 → 抬分假象跨域照旧。
2. **MIL_mean / POOL_RICH 仅小幅平滑**:域内给 +0.01~0.02 AUC(SingerLens 0.984→0.996,CtrSVDD 0.894→0.905,
   Wild 0.884→0.895),**跨数据集仍 ~随机**(POOL_RICH 略好 0.482/0.559 但仍崩)。
3. **窗级无法关闭 domain gap**:窗级粒度的收益是"域内小赚",对跨数据集泛化零贡献。

## 4. 跨域失败机制(三通道可视化,outputs/window_cross_dataset_cases/)

- **Case1 源域 in-domain fake**(SingerLens,OOF):p(fake) 在 t≈7.5–9s **局部尖峰** 0.51→0.95,
  与 pitch 跳变(f0 210→315 Hz,高音区转换)**同步** → 源域确有局部 AI 痕迹、且对齐音高异常。
- **Case3 Wild 真唱 False-Positive**:p(fake) **整段平直 ~0.90 无任何局部性**,判 fake(WRONG);
  其 HNR 仅 1–2.5 dB,而源域 SingerLens 为 15–18 dB → **域错配**(野生录音嘈杂低 HNR,被干净录音室
  训练的检测器一律当 fake),不是局部 AI 证据。
- **分数饱和/不可分**:CtrSVDD→Wild 下真唱(case3 mean0.896)与假唱(case5 mean0.874)分数几乎相同,
  mean_score_real0.632≈mean_score_fake0.614 → 真假都被判高 fake,局部证据真假共有、无区分力。
- **分数反转**:CtrSVDD→Wild 四方法 reversed=True(AUC<0.5),源域学到的"局部伪迹"在野生域语义翻转。
- **跨域 fake 缺源域伪迹**:Case2 Wild fake False-Negative,p(fake) 全窗 ≤0.57 无尖峰
  → So-VITS/RVC 野生翻唱的局部特征 ≠ CtrSVDD SVS/SVC 学到的,源域局部痕迹在目标域缺失。

## 5. 结论

**Window-level locality does not imply cross-domain generalizability.**
窗级/MIL 在域内可小幅改善(MIL_mean/POOL_RICH),但对最难的 cross-dataset collapse 零修复;
MIL_max 跨域仍是抬分假象(最差+反转)。局部证据是**域特异**的:源域有对齐音高的局部 AI 尖峰,
野生域则表现为整段域错配(HNR 量级差)、真假共有高分、分数反转。坐实主线"标准评测高估泛化"——
连"看局部"这一直觉补救也救不了,且 model/representation-agnostic。

## 6. 限制

- 定性三通道可视化仅作解释,**非新 detector / 新主指标**。
- RF + 窗级特征对窗口长度(3s/1.5s)敏感,未做窗长敏感性扫描。
- WildSVDD/T02 是下载可达的 bilibili 子集,**不等于完整 SingFake**;Wild 部分 clip f0 检测退化
  (嘈杂分离人声)亦反映野生数据本身难度。

## 产物

- `scripts/window_extract_xdataset.py`、`window_cross_dataset_eval.py`、`window_cross_dataset_cases.py`
- `outputs/window_features_{ctrsvdd,wild}.csv`
- `outputs/window_cross_dataset_eval.csv`、`outputs/window_cross_dataset_summary.csv`
- `outputs/window_cross_dataset_cases/`(case1–5 png + cases_index.csv)
