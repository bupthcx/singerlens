# CtrSVDD E2': Leave-One-Vocoder-Out (LOVO) — 核心混杂证据

## 协议
vocoder_group(hifigan/nsf-hifigan/ddsp)为domain。固定bonafide 50/50(跨折一致)。
每折:训练=train_bona + 其余vocoder的spoof;测试=test_bona + held-out vocoder的spoof。
报告 FULL/CLEAN/HNR/VRI 的 EER/AUC/F1。检验:检测器是否依赖训练见过的声码器家族特征。

## FULL EER%: in-distribution(见过) vs LOVO(未见)
| held-out vocoder | in-dist(E1 CV) | LOVO(held-out) | 退化 |
|---|---|---|---|
| hifigan | 13.67 | 31.76 | x2.3 |
| nsf-hifigan | 20.92 | 36.33 | x1.7 |
| ddsp | 26.04 | 28.67 | 小(ddsp本就难) |

## LOVO 全表
| held_out | metric | FULL | CLEAN | HNR | VRI |
|---|---|---|---|---|---|
| ddsp | EER% | 28.67 | 32.96 | 49.33 | 41.33 |
| hifigan | EER% | 31.76 | 40.57 | 48.33 | 47.67 |
| nsf-hifigan | EER% | 36.33 | 46.25 | 47.75 | 51.25 |
| ddsp | AUC | 0.799 | 0.714 | 0.527 | 0.607 |
| hifigan | AUC | 0.773 | 0.621 | 0.529 | 0.532 |
| nsf-hifigan | AUC | 0.696 | 0.546 | 0.545 | 0.486 |

## 核心结论
1. FULL 检测力高度依赖训练见过的声码器家族:未见时EER近翻倍(hifigan13.7->31.8,nsf20.9->36.3)。
   => 学的是声码器家族特异伪迹,非通用'AI演唱'线索。
2. HNR/VRI 风格特征无可迁移信号(in/out 均≈随机 EER~48%/AUC~0.5)。
3. 与E2跨测试自洽:非通用声码器探测器(不同声码器bonafide_vocoded不被误判),
   但声码器家族特异(LOVO换家族即崩)。

## 论文主线(精炼)
SVDD检测靠'这是哪个生成/声码器家族'而非'是不是AI演唱';留一声码器EER翻倍,可解释风格特征无迁移。
copy-synthesis(E2)与LOVO共同构成审计SVDD混杂的诊断协议。

## 待加强(step7)
用官方AASIST/WavLM强基线重做LOVO:若SOTA也在未见声码器上暴跌,则混杂是领域性的(非我们弱特征所致),论文力度大增。

## 产物
scripts/ctrsvdd_lovo.py; outputs/ctrsvdd_lovo.csv
