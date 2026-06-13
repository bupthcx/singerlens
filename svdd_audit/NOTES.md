# External components (not vendored)
- CtrSVDD official EER: github.com/SVDDChallenge/CtrSVDD_Utils (eer.py). Protocol scripts import compute_eer from it.
- AASIST baseline: github.com/SVDDChallenge/CtrSVDD2024_Baseline (models/, used by models/train_aasist_lovo.py; patch: guard fairseq import, custom driver avoids the test_loader bug).
- NSF-HiFiGAN vocoder (copy-synthesis): openvpi/vocoders nsf_hifigan_20221211.zip (sha256 d86ea84b7e2c9169afb5ccbb720b5542704be519c643f698332f2014a8f2d6bd).
- SSL frontend: ModelScope AI-ModelScope/wav2vec2-base-960h (safetensors). WavLM/XLSR-zh unavailable/blocked.
- Datasets: CtrSVDD (zenodo 10467648 train/dev, 12703261 eval), WildSVDD (zenodo 10893604 URL annotations; bilibili subset only from CN).
- attack_map.csv columns: attack, system, type(SVS/SVC=generator paradigm for LOGO), vocoder_group(for LOVO), split.
