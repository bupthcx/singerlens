# SVDD-Audit — MANIFEST (experiment → script → result → finding → paper)

> 一张索引:每个实验对应的脚本、结果文件、FINDINGS、以及在论文 skeleton_v2 中的表/图位置。
> 论文骨架见 `docs/PAPER_SVDD_Audit_skeleton_v2.md`(主文 v2)/ `..._skeleton.md`(v1 全表);
> 正文草稿 `docs/DRAFT_sec5.3_cross_dataset.md`、`docs/DRAFT_sec5.4-5.5_diagnosis_exit.md`。

## 主文实验(skeleton_v2 §5)

| 论文位置 | 实验 | 脚本(protocols/) | 结果(results/) | FINDINGS(docs/) |
|---|---|---|---|---|
| §5.1 T1 | 标准域内 EER/AUC | honest_eval.py, ctrsvdd_eer_analysis.py | ctrsvdd_eer_e1.csv | CtrSVDD_E1 |
| §5.2 T2 | LOVO 留一声码器 | ctrsvdd_lovo.py | ctrsvdd_lovo.csv | CtrSVDD_LOVO |
| §5.2 T2 | LOGO + AASIST LOVO | train_aasist_lovo.py | ctrsvdd_aasist_vs_rf_lovo.csv | CtrSVDD_AASIST_LOVO |
| §5.2 T2(headline) | **因子化 范式×声码器** | ctrsvdd_factorial.py | ctrsvdd_factorial_{loco,axis}.csv, ctrsvdd_aasist_paradigm.csv | CtrSVDD_因子化交叉 |
| §5.3 T3,F1(headline) | **cross-dataset 崩塌** | cross_dataset_transfer.py, cross_dataset_wild.py, train_w2v2_ft.py | cross_dataset_aasist_vs_rf.csv, w2v2ft_{A,B}.csv, w2v2ft_large_A.csv, score_distributions.png | cross_dataset, AASIST_WildSVDD, WildSVDD_T02, wav2vec2_finetune |
| §5.4 T4 | 失败归因(score collapse + 轴) | score_reversal_diag.py, failure_taxonomy.py, axis_attribution.py | score_distribution_summary.csv, failure_{cases,group_stats}.csv, axis_attribution{,_family}.csv, axis_attribution.png | score_reversal, failure_taxonomy, domain_adaptation |
| §5.4(因果) | production perturbation | production_perturbation{,_ext}.py, production_leakage_resynthesis.py | perturbation_effect.csv, production_leakage_resynthesis.csv | production_perturbation, production_leakage_resynthesis |
| §5.5 T5(出口) | DA / confound / MSA | domain_adaptation_cross.py, confound_removal.py, fewshot_{calibration,reweighting}.py | domain_adaptation_cross.csv, confound_removal_cross.csv, bandwidth_stats.csv, fewshot_{calibration,reweighting}.csv | domain_adaptation, confound_removal, fewshot_reweighting, minimal_supervised_adaptation |
| §5.5 T5(出口) | U-MSA 第二目标域复验 | umsa_fixedtest_second_target.py | d3_secondtarget_ctrsvdd_fixedtest.{csv,png} | active_msa_second_target |
| §5.5(d 情感) | 情感语义捷径证伪 | wild_emotion_consistency.py | wild_emotion_{consistency,auc}.csv | emotion_consistency_crossdomain |
| §5.6 | 部署(弃权/few-shot/扰动) | reject_option.py, fewshot_calibration.py | reject_option_riskcoverage.{csv,png}, fewshot_calibration.{csv,png} | deployment_analysis |

## 附录实验(Appendix,skeleton_v2)

| Appendix | 实验 | 脚本 | FINDINGS |
|---|---|---|---|
| C | 错误分析/难度分层 | difficulty_stratification.py, fp_fn_timelines.py | 错误分析与难度分层, 归因与错误分析 |
| D | 扰动全表 | (同上 production_*) | production_perturbation, production_leakage_resynthesis |
| F | 情感模块(域内案例) | emotion_consistency.py, transcribe_lyrics.sh | 情感一致性(域内), emotion_consistency_crossdomain(跨域证伪) |
| G | 窗级 MIL 局部性 | window_extract.py, window_temporal_eval.py, window_{extract_xdataset,cross_dataset_eval,cross_dataset_cases}.py | 窗级时序检测, window_cross_dataset_v2 |
| H | 跨歌手 mitigation / pilot | per_singer_norm_eval.py, dann_singer_invariant.py, human_pilot_prep.py | per_singer归一化改进, 对抗去歌手身份 |
| — | 声码器对照/泛化(自建) | vocoder_control_eval.py, cross_singer_eval.py, cross_song_eval.py | 声码器对齐对照, 跨歌手泛化, 最终_泛化总表 |

## 关键映射 / 数据
- attack→generator→vocoder 映射:`mapping/`(REF_ctrsvdd_system_vocoder.md, attack_map.csv)
- 数据集:CtrSVDD(zenodo)/ SingerLens(自建)/ WildSVDD-bilibili(zenodo 10893604, T02 子集)
- 模型(不在本包):wav2vec2-base/large-960h(ModelScope)、AASIST/openvpi NSF-HiFiGAN(NOTES.md)

## 大文件(不入本包,服务器 /home/admin2/xf/SingerLens/outputs/)
window_features{,_ctrsvdd,_wild}.csv、ctrsvdd_features_e1.csv、features_{fixed,vocoded}.csv、
bandlimited_features.csv、score_clip_scores.csv 等(原始特征矩阵,>100KB,复跑脚本可再生)。

## 论文文档
- `docs/PAPER_SVDD_Audit_skeleton_v2.md` — 主文 v2(5 节,6 表 2 图,draft-ready)
- `docs/PAPER_SVDD_Audit_skeleton.md` — v1(全表全数据,附录素材)
- `docs/PAPER_REVIEW_BEFORE_FULL_DRAFT.md` — 结构审查
- `docs/DRAFT_sec5.3_cross_dataset.md`、`docs/DRAFT_sec5.4-5.5_diagnosis_exit.md` — 正文草稿(已写)
- `docs/DAILY_SUMMARY_2026-06-12.md`、`docs/NIGHTLY_SUMMARY_after_exp3.md` — 进度战报
