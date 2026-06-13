"""Exp3-ext: 补两类 real-only 制作扰动 —— 伴奏残留(accompaniment leakage)与 Demucs 重分离。
只扰动真唱，re-extract FULL 特征，用同一 SingerLens 检测器打分，看 fake_score 是否上升。
无人值守: 逐样本 try/except + 增量落盘 + 完成写 marker。Demucs 走 CPU(model 加载一次)。
"""
from __future__ import annotations
import os, sys, tempfile, warnings
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf
warnings.filterwarnings('ignore')
sys.path.insert(0, 'src')
from singerlens.features import extract_clip_features
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SR = 16000
MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])
SEP_DIR = 'data/separated/htdemucs'
rng = np.random.RandomState(7)

# ---- demucs (load once, 4.0.1 旧 API) ----
MODEL = None
try:
    import torch
    from demucs.pretrained import get_model
    from demucs.apply import apply_model
    MODEL = get_model('htdemucs'); MODEL.cpu().eval()
    VIDX = MODEL.sources.index('vocals')
    print('demucs htdemucs loaded, sources', MODEL.sources, flush=True)
except Exception as e:
    print('demucs unavailable:', e, flush=True)


def rms(y): return float(np.sqrt(np.mean(y ** 2)) + 1e-9)


def accomp_leak(y, song_key, snr_db):
    p = f'{SEP_DIR}/{song_key}/no_vocals.wav'
    if not os.path.exists(p):
        return None
    a, _ = librosa.load(p, sr=SR, mono=True)
    if len(a) < len(y):
        a = np.pad(a, (0, len(y) - len(a)))
    st = rng.randint(0, max(1, len(a) - len(y)))
    seg = a[st:st + len(y)]
    alpha = rms(y) / rms(seg) * (10 ** (-snr_db / 20.0))   # 伴奏比人声低 snr_db
    z = y + alpha * seg
    return (z / (np.max(np.abs(z)) + 1e-9) * np.max(np.abs(y))).astype(np.float32)


def demucs_resep(y):
    if MODEL is None:
        return None
    import torch
    y44 = librosa.resample(y, orig_sr=SR, target_sr=44100)
    wav = torch.tensor(np.stack([y44, y44]))[None]         # (1, 2, L)
    with torch.no_grad():
        out = apply_model(MODEL, wav, device='cpu')[0]     # (sources, 2, L)
    voc = out[VIDX].mean(0).numpy()
    return librosa.resample(voc, orig_sr=44100, target_sr=SR).astype(np.float32)


def song_key(fname):
    return '_'.join(Path(fname).stem.split('_')[:-1])      # singer_X_song


def main():
    feat = pd.read_csv('outputs/features_fixed.csv')
    cols = [c for c in FULL if c in feat.columns]
    y = (feat['label'] == 'fake').astype(int).values
    X = feat[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = Pipeline([('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
                                                     class_weight='balanced', n_jobs=-1))]).fit(X, y)
    reals = feat[feat.label == 'real']['file_path'].tolist()
    rng.shuffle(reals); reals = reals[:200]
    tmp = Path(tempfile.mkdtemp()); wp = tmp / 'w.wav'

    def score(sig):
        sf.write(wp, sig, SR)
        f = extract_clip_features(wp)
        x = pd.DataFrame([{c: f.get(c, 0.0) for c in cols}]).replace([np.inf, -np.inf], np.nan).fillna(0)
        return float(model.predict_proba(x)[0, 1])

    PERTS = {
        'original': lambda y0, key: y0,
        'accomp_leak_-6db': lambda y0, key: accomp_leak(y0, key, 6),
        'accomp_leak_0db': lambda y0, key: accomp_leak(y0, key, 0),
        'demucs_resep': lambda y0, key: demucs_resep(y0),
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
            pd.DataFrame(rows).to_csv('outputs/perturbation_ext_scores.csv', index=False)
    df = pd.DataFrame(rows); df.to_csv('outputs/perturbation_ext_scores.csv', index=False)
    base = df['original']
    summ = []
    for name in PERTS:
        s = df[name].dropna()
        summ.append(dict(perturbation=name, n=len(s),
                         mean_p_fake=round(s.mean(), 3) if len(s) else np.nan,
                         median_p_fake=round(s.median(), 3) if len(s) else np.nan,
                         flip_rate_to_fake=round(float((s > 0.5).mean()), 3) if len(s) else np.nan,
                         delta_vs_original=round(s.mean() - base.mean(), 3) if len(s) else np.nan))
    pd.DataFrame(summ).to_csv('outputs/perturbation_ext_effect.csv', index=False)
    open('outputs/.exp3ext.done', 'w').write('ok\n')
    print('\n=== Exp3-ext (accomp leakage + demucs re-sep) ===', flush=True)
    print(pd.DataFrame(summ).to_string(index=False), flush=True)
    print('DONE -> outputs/perturbation_ext_{effect,scores}.csv', flush=True)


if __name__ == '__main__':
    main()
