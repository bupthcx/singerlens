"""窗级特征抽取：对每个 clip 滑窗 (win=3s, hop=1.5s)，逐窗抽 base 特征。
输出 long-format outputs/window_features.csv，供窗级 MIL 时序检测评测。

覆盖三类：real + fake (metadata.csv) + real_vocoded (metadata_vocoded.csv 中 source_type==real_vocoded)。
增量落盘 + 完成写 marker，配合后台 setsid 轮询（服务器 SSH 不稳）。
"""
from __future__ import annotations
import os, sys, tempfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from singerlens.features import extract_clip_features

SR = 16000
WIN = 3.0
HOP = 1.5
MIN_WIN = 2.0          # 末窗至少 2s 才保留
OUT = 'outputs/window_features.csv'
MARKER = 'outputs/.window_extract.done'


def windows(y, sr):
    dur = len(y) / sr
    if dur <= WIN:
        return [(0.0, y)]
    out, st = [], 0.0
    while st < dur - MIN_WIN + 1e-6:
        seg = y[int(st * sr):int((st + WIN) * sr)]
        if len(seg) >= int(MIN_WIN * sr):
            out.append((round(st, 3), seg))
        st += HOP
    return out


def build_metadata():
    base = pd.read_csv('data/demo_data/metadata.csv')          # 552 real + 533 fake
    frames = [base]
    voc_path = 'data/demo_data/metadata_vocoded.csv'
    if os.path.exists(voc_path):
        mv = pd.read_csv(voc_path)
        if 'source_type' in mv.columns:
            rv = mv[mv['source_type'] == 'real_vocoded'].copy()
            rv['label'] = 'real_vocoded'   # 单独成一类，便于审计分析
            frames.append(rv)
    meta = pd.concat(frames, ignore_index=True)
    meta = meta[meta['file_path'].astype(str).str.endswith('.wav')]
    return meta.reset_index(drop=True)


def main():
    meta = build_metadata()
    print(f'clips to process: {len(meta)}  '
          f'({meta["label"].value_counts().to_dict()})', flush=True)
    rows = []
    tmp = Path(tempfile.mkdtemp())
    wp = tmp / 'w.wav'
    n = len(meta)
    for i, r in meta.iterrows():
        path = str(r['file_path'])
        if not os.path.exists(path):
            continue
        try:
            y, sr = librosa.load(path, sr=SR, mono=True)
        except Exception as e:
            print('load fail', path, e, flush=True)
            continue
        clip_id = Path(path).stem
        for wi, (st, seg) in enumerate(windows(y, sr)):
            sf.write(wp, seg, sr)
            try:
                feat = extract_clip_features(wp)
            except Exception:
                continue
            row = {'clip_id': clip_id, 'win_idx': wi, 'win_start': st,
                   'file_path': path, 'label': r['label'],
                   'singer_id': r.get('singer_id', ''),
                   'song_id': r.get('song_id', ''),
                   'source_type': r.get('source_type', '')}
            row.update(feat)
            rows.append(row)
        if (i + 1) % 25 == 0:
            print(f'{i + 1}/{n} clips -> {len(rows)} windows', flush=True)
            pd.DataFrame(rows).to_csv(OUT, index=False)
    pd.DataFrame(rows).to_csv(OUT, index=False)
    Path(MARKER).write_text('ok\n')
    print(f'DONE {len(rows)} windows -> {OUT}', flush=True)


if __name__ == '__main__':
    main()
