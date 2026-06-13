# Nightly Summary — after Exp1/2/3 (+ ext) (2026-06-11 晚)

> **更新(夜间补跑完成)**:两个 user-approved 扩展实验今晚也已跑完并折叠进文档:
> - **Exp2-ext(公平 source+k)**:上采样/加权后 source+k 回升(k=100:0.640→0.800)但**仍 < wild-only 0.859**
>   → 源数据真无可迁移信号,负迁移结论更硬(非加权 artifact)。
> - **Exp3-ext(伴奏残留 + Demucs 重分离)**:**伴奏残留 0dB 混入 flip 0%、Δ≈0 → 反驳"伴奏残留→FP"**;
>   Demucs 重分离仅 +0.141(≈混响)。FP 驱动是分离/编码的频谱签名,非伴奏能量。
> 已更新:FINDINGS_production_perturbation / FINDINGS_deployment_analysis / 骨架 Table 8b(4列)+8c(含两新行)。

## 一句话结论
- **Exp1 Selective prediction**:域内置信度有效(留 10% 最自信 → acc 1.00),**跨数据集高置信也会错
  (CtrSVDD→Wild acc@5%=0.35,confidently reversed)→ confidence threshold 不能作跨域安全拒识**。
- **Exp2 Few-shot Wild calibration**:**仅 50–100 个目标域标签从头训练(AUC 0.82–0.87)远胜"源+k"
  (0.59–0.64)→ 混杂源监督是负迁移,新平台应优先收少量目标标签从头训练**。
- **Exp3 Real-only production perturbation**:**带宽/编解码扰动(重采样8k/低通3k/MP3)把真唱翻成 fake
  (Δ+0.28~0.37,flip 13–47%),加噪/响度不动 → 检测器沿"频谱带宽/编解码足迹"轴打分,平台压缩是
  真实样本误报(FP)风险源**(诚实限定:非任意脏化)。

## 关键表格数字
**Exp1 selective accuracy**

| protocol | @100% | @50% | @10% | @5% |
|---|---|---|---|---|
| CtrSVDD within | 0.80 | 0.94 | 1.00 | 1.00 |
| CtrSVDD→Wild | 0.46 | 0.50 | 0.45 | 0.35 |
| CtrSVDD→SingerLens | 0.53 | 0.55 | 0.40 | 0.41 |

**Exp2 few-shot AUC(held-out Wild)**

| k | source+k | Wild-only k |
|---|---|---|
| 25 | 0.542 | 0.752 |
| 50 | 0.585 | 0.817 |
| 100 | 0.641 | 0.870 |
| 上界 | — | ~0.935 |

**Exp3 real-only 扰动(真唱 fake-score,clean=0.085)**

| 扰动 | mean | Δ | flip |
|---|---|---|---|
| resample-8k | 0.455 | +0.370 | 46.5% |
| low-pass 3k | 0.450 | +0.365 | 46.5% |
| MP3-32k | 0.363 | +0.278 | 25.0% |
| reverb | 0.232 | +0.147 | 4.5% |
| noise 20/10dB | 0.11/0.06 | +0.02/−0.02 | 0% |
| gain quiet | 0.100 | +0.014 | 0% |

## 已更新/新增文件
- `outputs/FINDINGS_production_perturbation.md`(Exp3)
- `outputs/FINDINGS_deployment_analysis.md`(Exp1+2+3 合并部署小节)
- `outputs/PAPER_SVDD_Audit_skeleton.md`(451 行;新增 **5.10 Deployment Implications** + Table 8a/8b/8c
  + Figure 3/4 inventory + Discussion 四条部署建议;两处 outputs/ 与 svdd_audit/docs/ 已同步)
- 数据/图:`reject_option_riskcoverage.{csv,png}`、`fewshot_calibration.{csv,png}`、
  `perturbation_{effect,scores}.csv`
- 脚本:`scripts/{reject_option,fewshot_calibration,production_perturbation}.py`

## 异常 / 失败 / 需人工拍板
- **无失败**:五个实验(Exp1/2/3 + Exp2-ext + Exp3-ext)均正常完成,产物完整。
- **昨夜两条 caveat 已闭环(user 今晚直接开,已跑完)**:
  1. ~~Exp3 未覆盖 accompaniment leakage / Demucs re-separation~~ → **已跑(Exp3-ext)**:伴奏残留 flip 0%
     (反驳"伴奏残留→FP"),Demucs 重分离 +0.141 ≈ 混响;FP 驱动是分离/编码频谱签名,非伴奏能量。
  2. ~~Exp2 source+k 未上采样~~ → **已做公平变体(Exp2-ext)**:upweight/oversample 后回升到 0.80
     但仍 < wild-only 0.859,负迁移结论更硬。
- **剩余范围限制(非失败,写进 Limitation 即可)**:合成扰动 ≠ 真实平台转码分布,flip 绝对值不可外推;
  扰动仅在 SingerLens 真唱 + SingerLens 域内检测器上测。

## Decisions / Caveats（已写进 FINDINGS + 骨架 5.10 Limitations,诚实限定版）
> 注:下列三条不是"未做",而是对**已做实验**的诚实范围限定(避免过度声称)。已在
> FINDINGS_production_perturbation / FINDINGS_deployment_analysis / PAPER skeleton 5.10 中落实。
1. **Accompaniment leakage / Demucs re-separation 只用合成探针测过,未独立控制真实分离/平台残留**。
   合成设置下伴奏残留不致 FP(反驳"伴奏残留→FP"),但不能据此排除真实平台分离链路对 FP 的贡献。
2. **Few-shot source+k 只比了 naive 合并 vs 简单 target 上采样/加权**,未测完整领域自适应(DANN/校准微调)。
   结论限定为"source supervision is a negative asset **under naive mixing**",不外推为"所有自适应都无效"。
3. **合成扰动是 controlled diagnostic probe,非真实平台分布**。分数 shift / flip rate 作**定性敏感性证据**,
   非可部署错误率;外部效度以 WildSVDD-bilibili/T02 真实野生测试为准。

## 明天建议第一步
**写论文全文**(骨架结构完整:5.1–5.10 + Appendix A,含 5.10 ext 更新,每个实验表/图/csv 三对齐)。
建议从 **5.6→5.6b→5.10 这条"崩塌→归因→部署"主线**先成文,这是审稿人最关心的实用价值落点。
可选(非必需,需新工作,先不自动开):实际跑 40 条 human pilot。
