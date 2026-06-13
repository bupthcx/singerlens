#!/usr/bin/env bash
# 单歌曲全流程：demucs 分离 -> 切片 -> Seed-VC 生成fake(用该歌手已有target) -> copy-synth(声码器对照)
# 用法: bash scripts/process_song.sh <singer_id> <song_id> <raw_wav_path>
set -euo pipefail
SINGER=$1; SONG=$2; RAW=$3
ROOT=/home/admin2/xf/SingerLens
SEEDVC=/home/admin2/xf/tools/seed-vc
TGT=$ROOT/data/reference/target_voice_${SINGER}.wav
source /home/admin2/anaconda3/etc/profile.d/conda.sh

echo "[$SINGER/$SONG] 1/4 Demucs 分离"
conda activate singerlens
cd $ROOT
demucs --two-stems=vocals -o data/separated "$RAW" >/dev/null 2>&1
VOCALS=$(find data/separated -path "*$(basename "${RAW%.*}")*" -name vocals.wav | head -1)
echo "  vocals: $VOCALS"

echo "[$SINGER/$SONG] 2/4 切片"
python scripts/slice_real.py --audio "$VOCALS" --singer $SINGER --song $SONG 2>&1 | tail -1

echo "[$SINGER/$SONG] 3/4 Seed-VC 生成 fake"
conda activate seedvc
cd $SEEDVC
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python gen_fakes_batch.py   --source-glob "$ROOT/data/demo_data/real/${SINGER}_${SONG}_*.wav"   --target "$TGT" --out-dir $ROOT/data/demo_data/fake --singer $SINGER --song $SONG   2>&1 | grep -E 'sources=|FAIL|done'

echo "[$SINGER/$SONG] 4/4 copy-synth 声码器对照"
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 python copy_synth_batch.py   --source-glob "$ROOT/data/demo_data/real/${SINGER}_${SONG}_*.wav"   --out-dir $ROOT/data/demo_data/real_vocoded 2>&1 | grep -E 'sources=|FAIL|done'
echo "[$SINGER/$SONG] 完成"
