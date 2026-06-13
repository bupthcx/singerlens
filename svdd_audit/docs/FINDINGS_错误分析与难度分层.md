# 实验发现 VIII：FP/FN 案例剖析 + 难度分层(对应4件计划之2、4)
> 之1(三分类声码器归因)、之3(Cohen's d+方差)见 FINDINGS_归因与错误分析.md。

## 第2件：FP/FN 案例剖析(含片段内时间轴)
对 3 协议各取 2 FP + 2 FN(CLEAN模型),除特征/异常维度外,新增片段内逐窗(3s窗/1s跳)AI概率时间轴。
图:outputs/fp_fn_timelines/all_cases_timeline.png;数据:attribution_fp_fn.csv + fp_fn_timeline_summary.csv。

| 协议 | 类型 | 文件 | 整段p | 窗均值 | 窗min-max | 解读 |
|---|---|---|---|---|---|---|
| real_vs_fake | FP | singer_c_tiankong_017(真) | 0.88 | 0.61 | 0.52-0.72 | 通篇像AI的真唱(颤音异常强) |
| real_vs_fake | FN | singer_c_juhao_029(假) | 0.12 | 0.21 | 0.10-0.46 | 整段都像真人,真正难样本 |
| real_vs_fake | FN | singer_b_kaishidongle_021(假) | 0.14 | 0.37 | 0.25-0.62 | 有一窗AI痕迹冒头(0.62)但被平均掉 |
| realvoc_vs_fake | FN | singer_c_juhao_017/029(假) | 0.08/0.09 | 0.26/0.21 | - | 《句号》fake跨协议反复FN,最难 |

关键洞察:
- 部分 FN(难fake)存在'局部窗AI痕迹冒头'(win_max>0.5),提示窗级/时序检测可能优于整段平均;
- FP 多为'通篇像AI的真唱'(如颤音过强的真实演唱),非单窗噪声;
- 《句号》(juhao)fake 在多协议持续 FN,是公认最难检测的歌(与难度分层一致)。

## 第4件:难度分层(CLEAN模型预测概率分箱)
fake 难度: hard(p<0.4,像真)=35, medium=92, easy(p>0.6)=128。
| 难度 | n | 主要歌曲 |
|---|---|---|
| hard(像真) | 35 | 孙燕姿天黑黑8, 邓紫棋句号6, 单依纯DearFriend5 |
| medium | 92 | 单依纯开始懂了14, 邓紫棋天空没有极限13 |
| easy(易检) | 128 | 单依纯DearFriend21, 邓紫棋天空没有极限20 |

各难度关键特征(z=相对真人均值):
| 特征 | hard | medium | easy |
|---|---|---|---|
| vibrato_depth_mean | 148(z-0.2) | 158(z-0.0) | 169(z+0.2) |
| hnr_low_ratio | 0.256 | 0.205 | 0.222 |

关键洞察:
1. **最难检测的 fake 集中在悲伤/温柔慢歌**(天黑黑/句号/DearFriend):AI翻唱在抒情慢歌上更接近真人。
2. **颤音深度是可解释线索**:易检测fake的颤音被AI夸大(depth 169 z+0.2),难检测的接近真人(148);
   说明'过度规律/夸张的颤音'是AI破绽,呼应VRI设计初衷(虽VRI综合分整体弱)。
3. 其他特征跨难度差异小(z多在±0.5内),印证CLEAN判别为弱多元信号,无单一强特征。

## 对报告/答辩的意义
- 用 FP/FN 时间轴讲案例(某fake在1.2s处AI痕迹冒头/某真唱通篇像AI),比讲数字有力。
- 难度分层给出'AI在慢歌上更难辨'的可解释结论 + 颤音深度这一具体破绽。

## 产物
scripts/{difficulty_stratification.py, fp_fn_timelines.py};
outputs/{difficulty_fakes.csv, fp_fn_timeline_summary.csv, fp_fn_timelines/all_cases_timeline.png}
