# Deployment Analysis: Abstention, Few-shot Calibration, Production Perturbation (FINDINGS_deployment_analysis.md)

把三个部署导向的小实验合为一组,回答:**一个 benchmark 上训练好的 SVDD 检测器,能否安全上线?**
结论:不能直接上线;置信阈值不可靠;但少量目标域标签可廉价修复;制作链路是误报风险源。

---

## Exp1 — Selective prediction / reject option（弃权能否逃离崩塌）
confidence=|score−0.5|,保留 top-c% 最自信预测时的 accuracy(canonical FULL clip_mean,脚本 `reject_option.py`)。

| protocol | acc@100% | acc@50% | acc@10% | acc@5% |
|---|---:|---:|---:|---:|
| CtrSVDD within | 0.80 | 0.94 | 1.00 | 1.00 |
| WildSVDD within | 0.80 | 0.92 | 1.00 | 1.00 |
| SingerLens within | 0.94 | 1.00 | 1.00 | 1.00 |
| CtrSVDD→Wild | 0.46 | 0.50 | 0.45 | **0.35** |
| CtrSVDD→SingerLens | 0.53 | 0.55 | 0.40 | 0.41 |
| SingerLens→CtrSVDD | 0.55 | 0.54 | 0.46 | 0.43 |

**结论**:域内 selective prediction 完美(留 10% 最自信 → accuracy 100%);**跨数据集弃权救不了——
高置信预测不仅不更对,反而更错**(CtrSVDD→Wild 留 5% 最自信 acc 跌到 0.35,confidently reversed)。
模型是"自信地错"。**confidence threshold 不能作为跨域安全拒识手段。**
产物 `outputs/reject_option_riskcoverage.{csv,png}`。

## Exp2 — Few-shot Wild calibration（少量目标标签能否修复）
源(CtrSVDD-RF-FULL)+k 个标注 Wild vs 仅 k 个 Wild 从头训练,测 held-out Wild AUC(20 次平均,脚本 `fewshot_calibration.py`)。

| k | source + k Wild | **Wild-only k** |
|---:|---:|---:|
| 0 | 0.494 | — |
| 10 | 0.525 | 0.681 |
| 25 | 0.542 | 0.752 |
| 50 | 0.585 | **0.817** |
| 100 | 0.641 | **0.870** |
| (上界 Wild-within) | | ~0.935 |

**结论**:**仅 50–100 个 Wild 标签从头训练(0.82–0.87,逼近域内上界 0.935)远胜"源+k"**;
加 CtrSVDD 源数据反而拖累(4800 源样本淹没少量目标 + 混杂特征负迁移)。
**新平台应优先收集少量目标域标签从头训练,而非迁移/微调被混杂污染的源检测器。**
产物 `outputs/fewshot_calibration.{csv,png}`。
**公平变体(Exp2-ext)**:对 k Wild 上采样/加权后 source+k 回升(k=100:naive 0.640→upweight 0.764→
oversample 0.800),证实"被淹没"诊断;**但即便公平加权仍打不过 wild-only(0.859)→ 源数据真无可迁移信号,
负迁移结论更硬**。产物 `outputs/fewshot_calibration_ext.{csv,png}`。

**结论的诚实限定(避免过度声称)**:本实验对比的是 **naive 合并** 与 **简单 target 上采样/加权**
两类组合方式,**未测**更完整的领域自适应(DANN 式特征对齐、神经模型的校准层/warm-start 微调、
domain-balanced sampling 等)。因此准确的结论是:
*"Naively mixing large source-domain data with a small number of target labels underperforms target-only
training; simple target up-weighting narrows but does not close the gap—source supervision can become a
negative asset under naive mixing."* **不应外推为"任何 source+target 自适应都无效"**。后续可选变体:
target upsampling / class-balanced source-target sampling / domain-balanced weighting / calibration-only
fine-tune,留待后续(本轮不新开)。

**Exp2b 正式版(target=WildSVDD-T02,FINDINGS_fewshot_reweighting.md)**:5 方法 × k=25/50/100。
target-only 0.720/0.798/0.852 全程最强;reweighting(k=100:upsample 0.773 / domain-bal 0.734)缩小但
不及 target-only;**calibration-only 全程 AUC≈0.49(单调不变)→ 源排序对 Wild 本身随机,非 miscalibration**。
精确结论:**源判别轴与目标域错位(非仅淹没/校准),简单 reweighting 不能关闭差距,target-only 最优**;
不外推为"所有 source+target 自适应无效"(完整 DA 未测)。

## Exp3 — Real-only production perturbation（误报风险源）
只扰动真唱,看 fake_score 是否被推高(脚本 `production_perturbation.py` + `_ext.py`,详见 FINDINGS_production_perturbation.md)。

**结论**:**带宽缩减(重采样 8k/低通 3-4k)与 MP3 编解码把真唱翻成 fake(Δ+0.28~0.37,flip 13-47%);
加噪/响度/伴奏残留几乎不动**。检测器沿"频谱带宽/编解码足迹"轴打分 → **真实音频的平台压缩/带宽
是直接的误报(FP)风险源**;但精确说是 spectral-bandwidth/codec,非任意脏化(诚实限定)。
**Exp3-ext / Exp3b 关键负结果:伴奏残留三级(−12/−6/0dB)全 flip 0%、Δ≈0 → 反驳"伴奏残留→FP"假设**;
**Demucs 重分离推高(+0.142,leak→resep 链式 +0.177),增量来自分离步骤非伴奏**。即 FP 驱动是
分离/编码的**频谱签名**(spectral-bandwidth/codec/separation),非伴奏能量。详见 FINDINGS_production_leakage_resynthesis.md。

---

## 合并部署结论
1. **benchmark 上近完美的检测器不能直接上线**:跨数据集塌到随机/反相关(Exp1 域外 acc≈0.35-0.55)。
2. **置信阈值不是安全网**:跨域高置信预测同样错(Exp1)。
3. **新平台优先收集少量(~50-100)目标域标签从头训练**,胜过迁移混杂源模型(Exp2)。
4. **上线前需评估制作链路敏感性**:平台转码/带宽/编解码会显著推高真实样本误报(Exp3);
   但伴奏残留本身不致 FP(Exp3-ext),FP 驱动是分离/编码的频谱签名。

## 产物索引
- Exp1 `reject_option.py` / `reject_option_riskcoverage.{csv,png}`
- Exp2 `fewshot_calibration.py` + `_ext.py` / `fewshot_calibration{,_ext}.{csv,png}`
- Exp3 `production_perturbation.py` + `_ext.py` / `perturbation{,_ext}_{effect,scores}.csv`(+ FINDINGS_production_perturbation.md)
