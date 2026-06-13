"""Tier1.2 + Tier2.2 合并(一次音频 pass)。
Tier2.2: 各数据集原生频谱带宽(rolloff85/centroid)统计(real vs fake)。
Tier1.2: 把所有数据统一低通到共同带宽(3.5kHz)消除 bandwidth 混杂,re-extract FULL,再 cross-dataset。
对照原始 cross-dataset:若回升=bandwidth 是混杂主因;若仍崩=还有别的轴。诚实。
无人值守: 增量落盘 + marker。
"""
from __future__ import annotations
import os, sys, glob, tempfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf
from scipy.signal import butter, sosfilt
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from singerlens.features import extract_clip_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

SR = 16000; CUT = 3500
MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])
rng = np.random.RandomState(0)


def lowpass(y, cut=CUT):
    sos = butter(8, cut / (SR / 2), btype='low', output='sos'); return sosfilt(sos, y).astype(np.float32)


def band_stats(y):
    roll = float(np.mean(librosa.feature.spectral_rolloff(y=y, sr=SR, roll_percent=0.85)))
    cent = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=SR)))
    return roll, cent


def ctr_items(n):
    e = pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    m = pd.read_csv('/home/admin2/xf/ctrsvdd/ctrsvdd_meta.csv').set_index('filename')['split'].to_dict()
    e = e.groupby('label', group_keys=False).apply(lambda d: d.sample(min(len(d), n // 2), random_state=0))
    out = []
    for _, r in e.iterrows():
        p = f"/home/admin2/xf/ctrsvdd/{m.get(r['filename'])}_set/{r['filename']}.flac"
        out.append(('CtrSVDD', p, None, None, r['label']))
    return out


def sl_items(n):
    f = pd.read_csv('outputs/features_fixed.csv')
    f = f.groupby('label', group_keys=False).apply(lambda d: d.sample(min(len(d), n // 2), random_state=0))
    return [('SingerLens', r['file_path'], None, None, r['label']) for _, r in f.iterrows()]


def wild_items():
    w = pd.read_csv('outputs/wildsvdd_features.csv')
    t02 = set(pd.read_csv('/home/admin2/xf/wildsvdd/wildsvdd_bili_t02.csv')['idx'])
    w = w[w['idx'].isin(t02)]
    return [('WildT02', None, r['idx'], int(r['clip']), r['label']) for _, r in w.iterrows()]


def load_wild_clip(idx, ci):
    voc = glob.glob(f'/home/admin2/xf/wildsvdd/demucs_out/{idx}/htdemucs/*/vocals.wav')
    if not voc: return None
    v, _ = librosa.load(voc[0], sr=SR, mono=True); c = 0
    for st in np.arange(0, max(len(v) / SR - 10, 0.01), 8):
        ch = v[int(st * SR):int((st + 10) * SR)]
        if len(ch) < SR * 5: continue
        if c == ci: return ch
        c += 1
    return None


def main():
    items = ctr_items(500) + sl_items(500) + wild_items()
    print(f'clips: {len(items)}', flush=True)
    tmp = Path(tempfile.mkdtemp()); wp = tmp / 'w.wav'
    rows = []; bw = []
    for i, (ds, path, idx, ci, label) in enumerate(items):
        try:
            y = librosa.load(path, sr=SR, mono=True)[0] if path else load_wild_clip(idx, ci)
            if y is None or len(y) < SR: continue
            roll, cent = band_stats(y)
            bw.append(dict(dataset=ds, label=label, rolloff85=round(roll, 1), centroid=round(cent, 1)))
            yl = lowpass(y); sf.write(wp, yl, SR)
            feat = extract_clip_features(wp)
            rows.append({'dataset': ds, 'label': label, **{c: feat.get(c, 0.0) for c in FULL}})
        except Exception:
            continue
        if (i + 1) % 100 == 0:
            print(f'{i+1}/{len(items)}', flush=True)
            pd.DataFrame(bw).to_csv('outputs/bandwidth_stats.csv', index=False)
            pd.DataFrame(rows).to_csv('outputs/bandlimited_features.csv', index=False)
    bwdf = pd.DataFrame(bw); bwdf.to_csv('outputs/bandwidth_stats.csv', index=False)
    bl = pd.DataFrame(rows); bl.to_csv('outputs/bandlimited_features.csv', index=False)

    # Tier2.2 汇总
    print('\n=== Tier2.2 原生频谱带宽 (rolloff85 / centroid, Hz) ===')
    print(bwdf.groupby(['dataset', 'label'])[['rolloff85', 'centroid']].mean().round(0).to_string())

    # Tier1.2 cross-dataset on band-limited
    def rf(): return RandomForestClassifier(n_estimators=300, random_state=0, class_weight='balanced', n_jobs=-1)
    def XY(d):
        X = d[FULL].replace([np.inf, -np.inf], np.nan).fillna(0); y = (d['label'] == 'fake').astype(int).values
        return StandardScaler().fit_transform(X), y, X
    groups = {g: bl[bl.dataset == g] for g in bl.dataset.unique()}
    print('\n=== Tier1.2 band-limited cross-dataset AUC (统一 3.5kHz) ===')
    res = []
    for a, b in [('CtrSVDD', 'WildT02'), ('CtrSVDD', 'SingerLens'), ('SingerLens', 'CtrSVDD')]:
        if a not in groups or b not in groups: continue
        _, ya, Xa = XY(groups[a]); _, yb, Xb = XY(groups[b])
        sc = StandardScaler().fit(Xa)
        auc = roc_auc_score(yb, rf().fit(sc.transform(Xa), ya).predict_proba(sc.transform(Xb))[:, 1])
        res.append(dict(pair=f'{a}->{b}', bandlimited_AUC=round(auc, 3)))
        print(f'  {a}->{b}: {round(auc,3)}')
    pd.DataFrame(res).to_csv('outputs/confound_removal_cross.csv', index=False)
    open('outputs/.confound.done', 'w').write('ok\n')
    print('\nSaved -> outputs/{bandwidth_stats,bandlimited_features,confound_removal_cross}.csv', flush=True)


if __name__ == '__main__':
    main()
