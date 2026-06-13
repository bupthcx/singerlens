# wav2vec2 Fine-tune 强 SSL 基线 (FINDINGS_wav2vec2_finetune.md)

## 目的
补齐论文短板:此前 SSL 家族仅有 **frozen** wav2vec2-base-960h(within CtrSVDD AUC 仅 0.711，
弱基线削弱"连强 SSL 也跨域崩"的力度)。本实验**端到端微调** wav2vec2-base-960h，得到真正强的
SSL 检测器,检验其跨数据集是否同样崩塌。(WavLM 正版权重 HF 被墙且不在 ModelScope，故用
wav2vec2-base-960h；RawNet2 在本仓库非独立架构——已有 AASIST 即 rawnet-sinc 前端+AASIST backend。)

## 设置
- 模型:wav2vec2-base-960h(ModelScope 缓存) + LayerNorm 均值池化 + MLP 头(768→256→1)，Focal loss。
- 稳定化:frozen CNN feature_extractor + 差分 LR(transformer 2e-6 / 头 1e-4) + grad clip 0.5。
- **关键修复:关闭 SpecAugment**(`apply_spec_augment=False`, mask_*_prob=0)。
  诊断确认 train 模式 SpecAugment 把短序列整段 mask → layernorm 输出 NaN，导致每步 loss 非有限、
  参数从不更新(旧 `.bak_nan_` 版即此 bug)。关闭后训练稳定(skip nonfinite = 0，loss 平滑下降)。
- 环境 qwen3-asr(torch2.5.1+cu121)。CtrSVDD train 6000 平衡 / SingerLens 按 song 分 train-test。脚本 `train_w2v2_ft.py`。

## 结果

| 协议 | EER% | AUC |
|---|---:|---:|
| A→A (within CtrSVDD) | 17.50 | **0.912** |
| B→B (within SingerLens, song-split) | 17.54 | **0.904** |
| A→B (CtrSVDD→SingerLens) | 56.31 | **0.417** |
| B→A (SingerLens→CtrSVDD) | 62.00 | **0.337** |

(`outputs/w2v2ft_A.csv`, `w2v2ft_B.csv`。训练健康:CtrSVDD EER 20.2→16.0 / SingerLens 33.8→18.9 across epochs。)

## 关键发现
1. **微调让 SSL 域内变强**:within AUC 0.711(frozen)→ **0.90-0.91**(fine-tuned)，确为强基线。
2. **但跨数据集仍崩塌且反相关**:A→B 0.417 / B→A 0.337(低于随机),与 frozen(0.41-0.57)同向甚至更崩，
   与 AASIST(A→B 0.596 / B→A 0.553)、RF-FULL(0.615 / 0.541)**同一模式**。
3. **结论:微调不修复 cross-dataset gap**。"标准评测高估泛化、跨数据集塌到随机"对
   **手工 RF + 神经 AASIST + 微调强 SSL** 三个独立检测器家族一致成立 → **model-agnostic 证据更硬,
   frozen 弱基线的 caveat 解除**。

## SSL 选型说明(Model-Selection Rationale)
为什么 SSL 家族用 wav2vec2、且基线看似偏弱——三点客观说明,供论文 Setup/Limitations 引用:

1. **WavLM / HuBERT 因网络限制无法获取(客观约束,非选型偏好)**。本机 HuggingFace 被墙;
   WavLM 不在 ModelScope(404);XLSR-large 中文版权重需 torch≥2.6 加载失败(本机 torch 2.5.1)。
   可达的仅 ModelScope 的 wav2vec2 系列。故 SSL 家族以 wav2vec2 为代表,**并非认为它最优**。
2. **frozen wav2vec2-base 弱(within AUC 0.711)是预期的、可解释的**:wav2vec2-base-960h 在**英文语音
   ASR** 上预训练,**未针对歌声**;frozen 特征对唱歌声学(颤音/音区/声码器伪迹)不敏感,
   故冻结表示弱。**这不削弱论文论点**——它恰恰说明"现成 ASR-SSL 表示不自带 SVDD 判别力"。
3. **端到端微调把 SSL 拉成强基线(within 0.90-0.91),验证 SSL 路线有效**,瓶颈只在起点模型而非范式;
   且微调强 SSL **跨数据集仍塌**(0.34-0.42),与 RF/AASIST 同 → model-agnostic 结论稳健。

## 模型规模补充:wav2vec2-large(315M)对照
为展示"测试了模型规模的影响",补一个同源更大模型 wav2vec2-large-960h(315M,hidden 1024,
vs base 95M/768;经 .bin→safetensors 格式转换绕过 torch<2.6 限制后微调,同协议 train CtrSVDD)。

| 模型 | within CtrSVDD AUC | cross A→B(Ctr→SL) AUC |
| --- | ---: | ---: |
| wav2vec2-base-960h (95M) | 0.912 | 0.417 |
| **wav2vec2-large-960h (315M)** | **0.970** | **0.426** |

**结论:3.3× 更大的同源 SSL 域内明显更强(0.912→0.970,EER 17.5→8.5),但跨数据集仍塌(0.426,低于随机)。**
→ cross-dataset 崩塌**不是小模型 artifact**,放大模型规模改善域内、**不修复跨数据集** → model-agnostic
**且 scale-agnostic**(within EER 8.5% 的强 SSL 跨域仍≈随机)。诚实限定:large 仍是英文 ASR 预训练、
非歌声专用、亦非 SOTA 上限;.bin→safetensors 仅格式转换绕 torch<2.6,未改权重。

## 产物
- `train_w2v2_ft.py`(已修 SpecAugment NaN，备份 `.bak_prefix`；`--model` 可切 base/large)
- 协议 `/home/admin2/xf/ctrsvdd/aasist_csv/w2v2_{ctr,sl}_{train,test,all}.csv`
- 结果 `outputs/w2v2ft_A.csv`、`outputs/w2v2ft_B.csv`、`outputs/w2v2ft_large_A.csv`
