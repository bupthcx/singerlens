# wav2vec2 frozen SSL embedding baseline (第三模型家族)

## 设置
WavLM不在ModelScope(404),改用 AI-ModelScope/wav2vec2-base-960h(safetensors,768维)冻结embedding
(均值池化last_hidden_state)+RF。三数据集 ctr4800/sl1085/wild165。指标EER/AUC。
注:XLSR中文版因.bin需torch>=2.6无法在qwen3-asr(2.5.1)加载,故用base-960h。

## 结果(AUC)
| protocol | EER | AUC |
|---|---|---|
| CtrSVDD within | 35.17 | 0.711 |
| SingerLens within | 41.11 | 0.632 |
| WildSVDD within | 50.31 | 0.581 |
| CtrSVDD->SingerLens | 57.42 | 0.411 |
| SingerLens->CtrSVDD | 53.17 | 0.448 |
| CtrSVDD->WildSVDD | 50.90 | 0.499 |
| SingerLens->WildSVDD | 49.10 | 0.569 |
| LOVO hifigan/nsf/ddsp | 36.0/40.8/33.3 | 0.701/0.622/0.741 |

## 诚实结论
1. frozen wav2vec2 embedding即使域内也弱(CtrSVDD AUC0.711,远低于AASIST0.987/RF FULL0.90)——
   冻结SSL embedding(未微调)本不适合反欺骗,SVDD文献均微调SSL前端。须写明,非强基线。
2. 但跨数据集同样崩到随机(0.41-0.57),LOVO也退化——第三特征家族佐证主命题。
3. 三模型家族cross-dataset汇总:RF手工0.90-1.00->随机; AASIST0.987->随机~0.75; wav2vec2-frozen0.71(弱)->随机。
   => cross-dataset崩塌model-agnostic(3家族),但wav2vec2-frozen弱基线限制了'连强SSL也崩'的力度。

## 改进建议(论文)
若要强SSL基线,应微调wav2vec2/WavLM前端(需torch>=2.6或fairseq xlsr2_300m.pt+AASIST后端);
当前frozen版作为'通用SSL表示也不跨域迁移'的轻量佐证。

## 产物
scripts/{wavlm_extract.py(改用AutoModel+wav2vec2-base-960h),wavlm_analysis.py};
wavlm/{ctr,sl,wild}_emb.csv; outputs/wavlm_results.csv
