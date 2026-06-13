# Continuing Experiments

This note is for teammates who want to continue the project instead of only reading the report.

## 1. Where to Work

Use the full server workspace:

```bash
ssh 3090-shahe
cd /home/admin2/xf/SingerLens
```

This GitHub repository is a lightweight snapshot. The server workspace has the raw data, large feature CSVs, generated audio, logs, trained detector, and the latest full `outputs/`.

## 2. Environments

Common setup:

```bash
source /home/admin2/anaconda3/etc/profile.d/conda.sh
```

Use these environments:

```bash
conda activate singerlens   # feature extraction, RF experiments, figures, Gradio demo
conda activate seedvc       # Seed-VC / vocoder generation
conda activate qwen3-asr    # Whisper + emotion2vec emotion service
```

Important: Seed-VC should run offline because Hugging Face is unreliable from the server.

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
```

## 3. Server Layout

```text
/home/admin2/xf/SingerLens
├── data/             # raw/separated/demo audio
├── outputs/          # full experiment outputs, feature tables, logs, figures
├── scripts/          # full experiment scripts
├── src/singerlens/   # Gradio demo and core feature code
└── svdd_audit/       # cleaned toolkit snapshot
```

In this GitHub repo, the matching lightweight toolkit is under `svdd_audit/`.

## 4. Start With Existing Indices

Before running anything new, check:

```text
svdd_audit/MANIFEST.md
svdd_audit/README.md
outputs/FINDINGS_*.md
outputs/ASSEMBLED_SVDD_Audit_paper.md
outputs/COURSE_REPORT_SingerLens_SVDD.md
```

The manifest maps experiments to scripts and result files. The FINDINGS files explain what each result means.

## 5. Useful Existing Scripts

Core SingerLens:

```bash
python scripts/extract_features.py
python scripts/train_detector.py
python scripts/cross_singer_eval.py
python scripts/vocoder_control_eval.py
python scripts/per_singer_norm_eval.py
```

SVDD-Audit:

```bash
python scripts/ctrsvdd_eer_analysis.py
python scripts/ctrsvdd_lovo.py
python scripts/ctrsvdd_factorial.py
python scripts/cross_dataset_transfer.py
python scripts/cross_dataset_wild.py
python scripts/wavlm_extract.py
python scripts/wavlm_analysis.py
```

Production/confound controls:

```bash
python scripts/bandwidth_sweep.py
python scripts/production_perturbation.py
python scripts/prod_randomize_aug.py
python scripts/d1_eval.py
python scripts/d3b_fixedtest.py
python scripts/d3c_sanity.py
```

Figures:

```bash
python scripts/make_figures_data.py
```

## 6. Long-Running Job Pattern

The SSH connection to the server can drop. Do not run long experiments as a foreground command.

Recommended pattern:

```bash
setsid bash -lc 'source /home/admin2/anaconda3/etc/profile.d/conda.sh && conda activate singerlens && python scripts/your_exp.py > outputs/your_exp.log 2>&1; echo done > outputs/YOUR_EXP_DONE.marker' >/tmp/your_exp.out 2>&1 &
```

Then poll with short SSH commands:

```bash
tail -50 outputs/your_exp.log
ls outputs/*DONE.marker
```

Avoid this pattern:

```bash
pkill -f singerlens.app
```

It can match its own command line. Use a safer regex if needed:

```bash
pkill -f 'singerlens[.]app'
```

## 7. Suggested Next Experiments

The current story is already strong enough for the course report. Good follow-up experiments would be:

1. **Stronger SSL baseline**
   Fine-tune WavLM/XLSR/wav2vec2 instead of using frozen embeddings. Current SSL baseline is intentionally marked weak.

2. **RawNet2 or another SVDD baseline**
   Adds a second neural baseline beyond AASIST.

3. **Fixed-test active learning on another target domain**
   U-MSA is positive on WildSVDD-T02. Repeating it on another target split would make the method claim stronger.

4. **Human pilot**
   The human-pilot package is prepared on the server. Running even a small blind study could support the failure-analysis section.

5. **More cross-language cleanliness tests**
   Mandarin/Cantonese/English already support the production-signature claim. More English songs or another language would reduce the "single English song" limitation.

6. **Teaser/PPT integration**
   Turn the 10 generated Chinese figures in `report_figures/` into a short presentation narrative.

## 8. Rules for Adding New Results

For each new experiment, please add:

- script name and command,
- input data path,
- output CSV/PNG path,
- short result interpretation,
- whether the result supports or weakens the main claim,
- any caveats.

Put the detailed note in `outputs/FINDINGS_<name>.md`, then copy small final artifacts into `svdd_audit/results/` if they should be part of the public snapshot.

After syncing back to this GitHub repo, update:

```text
svdd_audit/MANIFEST.md
README.md
EXPERIMENTS_NEXT.md
```

