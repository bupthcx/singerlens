# Daily Summary — Exp2b / Exp3b 补强 + Tier1/2 (2026-06-12)

今日:两个 limitation 补强(Exp2b/Exp3b)+ 用户批准的 Tier1/2 四个强化实验(完整 DA / confound-removal /
轴归因 / 带宽量化)。未开其它新线。

## Tier1/2 — 能否修复 cross-dataset?(四实验,FINDINGS_domain_adaptation.md + FINDINGS_confound_removal.md)
**一句话:无监督修复全线失败,判别轴本身跨域错位,只有目标标签能修。**

- **Tier1.1 完整 DA**:per-domain z-norm / CORAL / subspace-align / DANN 全部 ≈ 随机(0.44–0.62),
  无一稳定优于 baseline,部分更差;target-only 0.90–0.99。→ **特征对齐式 DA 不能修复**(p(y|x) 跨域错位)。

  | pair | baseline | CORAL | subspace | DANN | target-only |
  |---|---|---|---|---|---|
  | Ctr→WildT02 | 0.509 | 0.474 | 0.517 | 0.442 | **0.925** |
  | Ctr→SL | 0.615 | 0.537 | 0.554 | 0.555 | **0.989** |

- **Tier1.2 confound-removal(统一带宽 3.5kHz)**:cross-dataset 仍崩(0.434/0.584/0.602,无回升)
  → 带宽只是混杂一个分量,残留 codec/separation/生成签名维持崩塌。**多分量混杂,无单轴廉价修复**。
- **Tier2.1 轴归因**:源押 MFCC(0.64)/几乎不用 HNR(0.055),域内 Wild 用 HNR(0.146);源对 Wild 的
  错分数由高阶 mfcc_*_std(ρ≈0.5)驱动 = 频谱纹理/产线签名,非真伪 → 判别轴不同。图 axis_attribution.png。
- **Tier2.2 带宽量化**:rolloff85 域级差 ~600Hz(SL 最宽 3774 / Ctr 最窄 3162);真假带宽方向**跨域翻转**
  (Ctr/Wild fake>real,SL real>fake)→ 坐实轴错位。

**骨架**:新增 **5.9b "Can the cross-dataset collapse be fixed?"**(Table 9a DA / 9b confound / 9c 归因 +
Figure 5)+ Discussion 加"无监督修复无效、需目标标签"+ inventory/TODO 更新,两处同步。

---
（以下为今日早些时候的 Exp2b/Exp3b）

今日只做两个昨晚标为 limitation 的补强实验(未开其它新线)。

## Exp2b — Source+k Reweighting / Calibration(target=WildSVDD-T02)
**一句话**:reweighting/upsampling 缩小但追不平 target-only,calibration-only 全程≈随机
→ **源判别轴对目标域错位(非仅 naive 淹没/非 miscalibration),target-only 最优**。

| k | target-only | src naive | src upsample | src domain-bal | calibration-only |
|---|---|---|---|---|---|
| 25 | **0.720** | 0.557 | 0.619 | 0.588 | 0.492 |
| 50 | **0.798** | 0.580 | 0.690 | 0.649 | 0.496 |
| 100 | **0.852** | 0.638 | 0.773 | 0.734 | 0.485 |

诚实限定:仅 RF + 简单加权/上采样/Platt 校准;完整 DA(DANN/校准层微调/domain-balanced minibatch)未测,
不外推为"所有 source+target 自适应无效"。产物 `fewshot_reweighting.{csv,png}` / FINDINGS_fewshot_reweighting.md。

## Exp3b — Production Leakage & Re-synthesis(real-only)
**一句话**:伴奏残留三级(−12/−6/0dB)全 flip 0% → 伴奏能量不是 FP 驱动;**re-separation 才推高**
(+0.142,leak→resep +0.177),FP 驱动是分离/编码的频谱签名。

| 扰动 | mean | Δ | flip |
|---|---|---|---|
| clean | 0.085 | 0 | 0% |
| accomp_leak −12/−6/0db | 0.09/0.08/0.08 | ≈0 | 0% |
| leak → re-separate | 0.262 | +0.177 | 5.3% |
| demucs_resep(clean) | 0.227 | +0.142 | 3% |
| mp3_32k(对照) | 0.363 | +0.278 | 25% |

诚实限定:controlled probe,不代表真实平台分布;只写 production-chain sensitivity,不写部署误报率;
外部效度以 WildSVDD-bilibili/T02 为准。产物 `production_leakage_resynthesis.{,_scores}.csv` / FINDINGS_production_leakage_resynthesis.md。

## 综合(两条 limitation 已补强闭环)
- **Exp2b**:Exp2 的"source+k 不如 target-only"= naive 淹没 **+** 判别轴错位,核心是后者
  (calibration-only≈随机坐实);简单廉价修复不够,需 target labels 从头训练。
- **Exp3b**:production-cleanliness axis 精确化为 **spectral-bandwidth/codec/separation-signature**,
  伴奏内容本身无关;支持 5.6b 的 FP 归因。

## 已更新文件
- 新增:`FINDINGS_fewshot_reweighting.md`、`FINDINGS_production_leakage_resynthesis.md`
- 更新:`FINDINGS_deployment_analysis.md`(Exp2b/Exp3b 段)、`PAPER_SVDD_Audit_skeleton.md` 5.10
  (Table 8b 改为 T02 5 方法 + Table 8c 加 leak 三级/leak→resep + (b)(c) prose,两处同步)
- 脚本:`scripts/{fewshot_reweighting,production_leakage_resynthesis}.py`

## 异常 / 失败
- 无失败,五结果(含对照)完整。

## 明天建议第一步
**写论文全文**(骨架 5.1–5.10 + Appendix A 结构完整,5.10 已含 Exp2b/Exp3b 补强)。
从 5.6→5.6b→5.10 主线先成文。可选(需新工作,先不自动开):完整 DA(DANN/校准层)对照、实际跑 human pilot。
