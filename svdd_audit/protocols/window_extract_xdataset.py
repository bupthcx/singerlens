"""跨数据集窗级特征抽取 (CtrSVDD / WildSVDD)。
对每个 clip 滑窗 win=3s/hop=1.5s，逐窗抽 49 维 base 特征，长表输出。
与 SingerLens 的 window_features.csv 同口径(extract_clip_features)，供跨数据集窗级迁移。

  --dataset ctrsvdd : clip = E1 子集的整段 flac
  --dataset wild    : clip = WildSVDD 每首 vocals.wav 按 10s/8s-hop 重切的片段(对齐 wildsvdd_features)
增量落盘 + 完成写 marker。
"""
from __future__ import annotations
import os, sys, glob, tempfile, warnings, argparse
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from singerlens.features import extract_clip_features

SR = 16000; WIN = 3.0; HOP = 1.5; MIN_WIN = 2.0
SVS = {'A01', 'A02', 'A03', 'A04', 'A05'}; SVC = {'A06', 'A07', 'A08'}


def windows(y, sr):
    dur = len(y) / sr
    if dur <= WIN:
        return [(0.0, y)] if dur >= MIN_WIN else []
    out, st = [], 0.0
    while st < dur - MIN_WIN + 1e-6:
        seg = y[int(st * sr):int((st + WIN) * sr)]
        if len(seg) >= int(MIN_WIN * sr):
            out.append((round(st, 3), seg))
        st += HOP
    return out


def emit(rows, y, sr, base, tmp, wp):
    for wi, (st, seg) in enumerate(windows(y, sr)):
        sf.write(wp, seg, sr)
        try:
            feat = extract_clip_features(wp)
        except Exception:
            continue
        rows.append({**base, 'win_idx': wi, 'win_start': st, **feat})


def run_ctrsvdd(out):
    e = pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    m = pd.read_csv('/home/admin2/xf/ctrsvdd/ctrsvdd_meta.csv').set_index('filename')['split'].to_dict()
    rows = []; tmp = Path(tempfile.mkdtemp()); wp = tmp / 'w.wav'; n = len(e)
    for i, r in e.iterrows():
        fn = r['filename']; sp = m.get(fn)
        path = f'/home/admin2/xf/ctrsvdd/{sp}_set/{fn}.flac'
        if not os.path.exists(path):
            continue
        try:
            y, sr = librosa.load(path, sr=SR, mono=True)
        except Exception:
            continue
        atk = r['attack']
        para = 'SVS' if atk in SVS else ('SVC' if atk in SVC else 'bonafide')
        base = {'clip_id': fn, 'label': r['label'], 'attack': atk,
                'vocoder_group': r['vocoder_group'], 'paradigm': para, 'singer_id': r['singer_id']}
        emit(rows, y, sr, base, tmp, wp)
        if (i + 1) % 200 == 0:
            print(f'{i+1}/{n} clips -> {len(rows)} win', flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    return len(rows)


def run_wild(out):
    feat = pd.read_csv('outputs/wildsvdd_features.csv')
    inv = feat.groupby('idx')['clip'].apply(set).to_dict()       # idx -> 存在的 clip 序号
    meta = feat.drop_duplicates('idx').set_index('idx')[['label', 'singer', 'model']].to_dict('index')
    t02 = set()
    tp = '/home/admin2/xf/wildsvdd/wildsvdd_bili_t02.csv'
    if os.path.exists(tp):
        t = pd.read_csv(tp); t02 = set(t['idx']) if 'idx' in t.columns else set()
    root = '/home/admin2/xf/wildsvdd'
    rows = []; tmp = Path(tempfile.mkdtemp()); wp = tmp / 'w.wav'
    ids = list(inv.keys()); n = len(ids)
    for i, idx in enumerate(ids):
        voc = glob.glob(f'{root}/demucs_out/{idx}/htdemucs/*/vocals.wav')
        if not voc:
            continue
        try:
            v, _ = librosa.load(voc[0], sr=SR, mono=True)
        except Exception:
            continue
        ci = 0
        for st in np.arange(0, max(len(v) / SR - 10, 0.01), 8):   # 与 wild_process 同切法
            ch = v[int(st * SR):int((st + 10) * SR)]
            if len(ch) < SR * 5:
                continue
            if ci in inv[idx]:                                    # 仅对齐已有特征的 clip
                md = meta[idx]
                base = {'clip_id': f'{idx}_c{ci}', 'idx': idx, 'clip': ci,
                        'label': md['label'], 'singer_id': md['singer'], 'model': md['model'],
                        't02': idx in t02}
                emit(rows, ch, SR, base, tmp, wp)
            ci += 1
        if (i + 1) % 20 == 0:
            print(f'{i+1}/{n} songs -> {len(rows)} win', flush=True)
            pd.DataFrame(rows).to_csv(out, index=False)
    pd.DataFrame(rows).to_csv(out, index=False)
    return len(rows)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--dataset', choices=['ctrsvdd', 'wild'], required=True)
    a = ap.parse_args()
    out = f'outputs/window_features_{a.dataset}.csv'
    nw = run_ctrsvdd(out) if a.dataset == 'ctrsvdd' else run_wild(out)
    Path(f'outputs/.window_{a.dataset}.done').write_text('ok\n')
    print(f'DONE {a.dataset}: {nw} windows -> {out}', flush=True)
