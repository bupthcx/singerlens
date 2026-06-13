# CtrSVDD 系统→声码器映射（据 arXiv:2406.02438 §2.2 核对）

| 系统 | 类型 | 生成模型 | 声码器 | 划分 | SSL先验(SVC) |
|---|---|---|---|---|---|
| A01 | SVS | XiaoiceSing(非自回归,Transformer) | HiFi-GAN | train | - |
| A02 | SVS | VISinger(端到端VAE+对抗) | VITS端到端(集成GAN解码器) | train | - |
| A03 | SVS | VISinger2 | DDSP | train | - |
| A04 | SVS | NNSVS(diffusion声学) | source-filter HiFi-GAN | train | - |
| A05 | SVS | Naive RNN(BiLSTM,FastSpeech式) | HiFi-GAN | train | - |
| A06 | SVC | NU-SVC(diffusion,ContentVec) | source-filter HiFi-GAN | train | ContentVec |
| A07 | SVC | Soft-VITS-SVC | VITS端到端 | train | (WavLM/ContentVec/MR-HuBERT/WavLabLM/中文HuBERT之一) |
| A08 | SVC | Soft-VITS-SVC | VITS端到端 | train | 同上 |
| A09 | SVC | Soft-VITS-SVC变体 | VITS端到端(★存疑:challenge文档称sf-HiFiGAN) | eval | 同上 |
| A10 | SVC | Soft-VITS-SVC变体 | VITS端到端 | eval | 同上 |
| A11 | SVC | Soft-VITS-SVC变体 | VITS端到端 | eval | 同上 |
| A12 | SVS | DiffSinger(FastSpeech+diffusion解码) | HiFi-GAN | eval | - |
| A13 | SVC | Soft-VITS-SVC变体 | VITS端到端 | eval | 同上 |
| A14 | SVS | ACESinger(商用ACE-Studio,人工调音) | 专有/未知 | eval | - |

## 声码器分组(混杂分析用)
- HiFi-GAN: A01,A05,A12
- source-filter HiFi-GAN: A04,A06 (★A09可能也属此,待核)
- DDSP: A03
- VITS端到端(集成GAN解码器,无独立声码器): A02,A07,A08,A09,A10,A11,A13
- 专有: A14(ACE-Studio)

## 待最终核实(写论文前从生成配置/CtrSVDD_Utils确认)
1. A09 声码器: §2.2(VITS端到端) vs challenge overview(sf-HiFiGAN) 冲突 → 查生成config。
2. A07-A11,A13 各自的SSL先验(WavLM/ContentVec/MR-HuBERT/WavLabLM/中文HuBERT)精确分配。
3. A14(ACE-Studio)声码器(商用,可能无法获知,论文标proprietary)。

## 关键观察(支撑论文)
bonafide=自然清唱录音(刻意去分离伪迹); 所有deepfake经神经声码器/VITS解码器合成。
→ bonafide与spoof间存在系统性声码器/合成产线差异,正是本文要量化的混杂。

---
# 【最终核实版】据生成仓库 HANJionghao/so-vits-svc2 + eval plan 修正
**关键修正**: So-VITS-SVC(A07-A11,A13) 用 **NSF-HiFiGAN(=source-filter HiFi-GAN)**,非端到端VITS。
证据: so-vits-svc2 README the vocoder was replaced with NSF HiFiGAN; 默认 vocoder_name=nsf-hifigan。

## 最终声码器分组(用于混杂分析)
| 声码器 | 系统 |
|---|---|
| HiFi-GAN | A01(XiaoiceSing), A02(VISinger端到端,集成HiFiGAN解码器), A05(NaiveRNN), A12(DiffSinger) |
| NSF/source-filter HiFi-GAN | A04(NNSVS), A06(NU-SVC), A07,A08,A09,A10,A11,A13(全部So-VITS-SVC) |
| DDSP | A03(VISinger2) |
| 专有 | A14(ACE-Studio商用) |

## copy-synthesis 对照实现要点(E2)
用与spoof相同的 **NSF-HiFiGAN** 过 bonafide → bonafide_vocoded。
预训练权重: github openvpi/vocoders releases nsf_hifigan_20221211.zip (github可达)。
这样 bonafide_vocoded 与 So-VITS-SVC类spoof共享同一声码器足迹,是公开基准上最干净的对照。

## SSL先验轴(次要,So-VITS-SVC内容编码器,每A0x分配TBD)
so-vits-svc2 支持: vec768l12(ContentVec)/hubertsoft/whisper-ppg/cnhubertlarge/dphubert/wavlmbase+(WavLM)/mrhubert(MR-HuBERT)/wavlablm(WavLabLM)。
A07-A13 各用哪个: 生成配置(官方赛后释放)未在仓库,标TBD;与声码器轴正交,可作内容编码器子分析。

## 官方基线(eval plan §7)
raw waveform + AASIST: EER 10.39%; LFCC+AASIST backend: 11.37%。均离线可复现(无需SSL),repo rawnet/lfcc encoder。
