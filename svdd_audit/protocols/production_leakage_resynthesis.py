"""Exp3b: accompaniment leakage(low/mid/high)+ Demucs re-separation(含 leak->resep 链式),正式版。
只扰动 real vocal，re-extract FULL 特征，用同一 SingerLens 检测器打分。
controlled probe：伴奏混合比例不代表真实平台分布；只写 production-chain sensitivity，不写误报率。
无人值守: try/except + 增量落盘 + marker。
"""
from __future__ import annotations
import os, sys, tempfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src'); sys.path.insert(0, 'scripts')
from singerlens.features import extract_clip_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
# 复用昨晚函数(import 时会加载 demucs MODEL 一次)
from production_perturbation_ext import accomp_leak, demucs_resep, song_key
from production_perturbation import reverb as p_reverb, mp3_roundtrip

SR = 16000
MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])


def leak_then_resep(y, key):
    yl = accomp_leak(y, key, 6)            # mid 级 -6dB 伴奏残留
    if yl is None:
        return None
    return demucs_resep(yl)                # 残留后再过一次分离


def main():
    feat = pd.read_csv('outputs/features_fixed.csv')
    cols = [c for c in FULL if c in feat.columns]
    y = (feat['label'] == 'fake').astype(int).values
    X = feat[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = Pipeline([('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
                                                     class_weight='balanced', n_jobs=-1))]).fit(X, y)
    reals = feat[feat.label == 'real']['file_path'].tolist()
    rng = np.random.RandomState(7); rng.shuffle(reals); reals = reals[:200]
    tmp = Path(tempfile.mkdtemp()); wp = tmp / 'w.wav'

    def score(sig):
        sf.write(wp, sig, SR); f = extract_clip_features(wp)
        x = pd.DataFrame([{c: f.get(c, 0.0) for c in cols}]).replace([np.inf, -np.inf], np.nan).fillna(0)
        return float(model.predict_proba(x)[0, 1])

    PERTS = {
        'clean': lambda y0, key: y0,
        'accomp_leak_low_-12db': lambda y0, key: accomp_leak(y0, key, 12),
        'accomp_leak_mid_-6db': lambda y0, key: accomp_leak(y0, key, 6),
        'accomp_leak_high_0db': lambda y0, key: accomp_leak(y0, key, 0),
        'leak_then_resep': lambda y0, key: leak_then_resep(y0, key),
        'demucs_resep_clean': lambda y0, key: demucs_resep(y0),
        'mp3_32k': lambda y0, key: mp3_roundtrip(y0, '32k', tmp),     # 对照(已有结论)
        'reverb': lambda y0, key: p_reverb(y0),                       # 对照
    }
    rows = []
    for i, rp in enumerate(reals):
        if not os.path.exists(rp):
            continue
        try:
            y0, _ = librosa.load(rp, sr=SR, mono=True)
        except Exception:
            continue
        key = song_key(rp)
        rec = {'file': os.path.basename(rp), 'song_key': key}
        for name, fn in PERTS.items():
            try:
                yp = fn(y0, key)
                rec[name] = round(score(yp), 4) if yp is not None and len(yp) >= SR else np.nan
            except Exception:
                rec[name] = np.nan
        rows.append(rec)
        if (i + 1) % 10 == 0:
            print(f'{i+1}/{len(reals)}', flush=True)
            pd.DataFrame(rows).to_csv('outputs/production_leakage_resynthesis_scores.csv', index=False)
    df = pd.DataFrame(rows); df.to_csv('outputs/production_leakage_resynthesis_scores.csv', index=False)
    base = df['clean']
    summ = []
    for name in PERTS:
        s = df[name].dropna()
        summ.append(dict(perturbation=name, n=len(s),
                         mean_fake_score=round(s.mean(), 3) if len(s) else np.nan,
                         median=round(s.median(), 3) if len(s) else np.nan,
                         high_score_ratio=round(float((s > 0.5).mean()), 3) if len(s) else np.nan,
                         delta_vs_clean=round(s.mean() - base.mean(), 3) if len(s) else np.nan))
    pd.DataFrame(summ).to_csv('outputs/production_leakage_resynthesis.csv', index=False)
    open('outputs/.exp3b.done', 'w').write('ok\n')
    print('\n=== Exp3b production leakage / re-synthesis ===', flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True)
    print('DONE -> outputs/production_leakage_resynthesis{,_scores}.csv', flush=True)


if __name__ == '__main__':
    main()
