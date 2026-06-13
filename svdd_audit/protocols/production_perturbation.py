"""Exp3: Real-only production perturbation。
只对真唱施加制作类扰动(重采样/低通/加噪/增益/MP3编解码/混响)——均不改变"谁在唱/是否AI"，
re-extract FULL 特征，用域内 SingerLens 检测器打分，看真唱是否被翻成 fake。
因果验证 P2:检测器是否沿"制作干净度"轴而非真伪打分。
"""
from __future__ import annotations
import os, sys, tempfile, subprocess, warnings
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf
from scipy.signal import butter, sosfilt
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


def lowpass(y, cut):
    sos = butter(8, cut / (SR / 2), btype='low', output='sos'); return sosfilt(sos, y).astype(np.float32)

def add_noise(y, snr_db):
    p = np.mean(y ** 2); n = np.random.RandomState(0).randn(len(y))
    n = n * np.sqrt(p / (10 ** (snr_db / 10)) / np.mean(n ** 2)); return (y + n).astype(np.float32)

def mp3_roundtrip(y, br, tmp):
    wavp = tmp / 'a.wav'; mp3p = tmp / 'a.mp3'; outp = tmp / 'b.wav'
    sf.write(wavp, y, SR)
    subprocess.run(['ffmpeg', '-y', '-i', str(wavp), '-b:a', br, str(mp3p)], capture_output=True)
    subprocess.run(['ffmpeg', '-y', '-i', str(mp3p), '-ar', str(SR), str(outp)], capture_output=True)
    z, _ = librosa.load(outp, sr=SR, mono=True); return z.astype(np.float32)

def reverb(y):
    ir = np.random.RandomState(1).randn(int(0.25 * SR)) * np.exp(-np.linspace(0, 6, int(0.25 * SR)))
    z = np.convolve(y, ir)[:len(y)]; return (z / (np.max(np.abs(z)) + 1e-9) * np.max(np.abs(y))).astype(np.float32)

PERTS = {
    'original':    lambda y, t: y,
    'resample_8k': lambda y, t: librosa.resample(librosa.resample(y, orig_sr=SR, target_sr=8000), orig_sr=8000, target_sr=SR).astype(np.float32),
    'lowpass_4k':  lambda y, t: lowpass(y, 4000),
    'lowpass_3k':  lambda y, t: lowpass(y, 3000),
    'noise_20db':  lambda y, t: add_noise(y, 20),
    'noise_10db':  lambda y, t: add_noise(y, 10),
    'gain_quiet':  lambda y, t: (y * 0.3).astype(np.float32),
    'mp3_32k':     lambda y, t: mp3_roundtrip(y, '32k', t),
    'reverb':      lambda y, t: reverb(y),
}


def main():
    feat = pd.read_csv('outputs/features_fixed.csv')
    cols = [c for c in FULL if c in feat.columns]
    y = (feat['label'] == 'fake').astype(int).values
    X = feat[cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    model = Pipeline([('sc', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
                                                     class_weight='balanced', n_jobs=-1))]).fit(X, y)
    print('detector trained on SingerLens FULL (real vs fake), n=', len(y))

    reals = feat[feat.label == 'real']['file_path'].tolist()
    rng = np.random.RandomState(7); rng.shuffle(reals)
    reals = reals[:200]
    tmp = Path(tempfile.mkdtemp()); wp = tmp / 'w.wav'

    def score(path_or_y):
        sf.write(wp, path_or_y, SR)
        f = extract_clip_features(wp)
        x = pd.DataFrame([{c: f.get(c, 0.0) for c in cols}]).replace([np.inf, -np.inf], np.nan).fillna(0)
        return float(model.predict_proba(x)[0, 1])

    rows = []
    for i, rp in enumerate(reals):
        p = rp if rp.startswith('/') else rp
        if not os.path.exists(p):
            continue
        try:
            y0, _ = librosa.load(p, sr=SR, mono=True)
        except Exception:
            continue
        rec = {'file': os.path.basename(p)}
        for name, fn in PERTS.items():
            try:
                yp = fn(y0, tmp)
                if len(yp) < SR:
                    continue
                rec[name] = round(score(yp), 4)
            except Exception as e:
                rec[name] = np.nan
        rows.append(rec)
        if (i + 1) % 50 == 0:
            print(f'{i+1}/{len(reals)}', flush=True)
    df = pd.DataFrame(rows)
    df.to_csv('outputs/perturbation_scores.csv', index=False)

    summ = []
    base = df['original']
    for name in PERTS:
        s = df[name].dropna()
        summ.append(dict(perturbation=name, n=len(s),
                         mean_p_fake=round(s.mean(), 3), median_p_fake=round(s.median(), 3),
                         flip_rate_to_fake=round(float((s > 0.5).mean()), 3),
                         delta_vs_original=round(s.mean() - base.mean(), 3)))
    sm = pd.DataFrame(summ)
    sm.to_csv('outputs/perturbation_effect.csv', index=False)
    print('\n=== Real-only production perturbation (P(fake) of REAL clips) ===')
    print(sm.to_string(index=False))
    print('\nSaved -> outputs/perturbation_effect.csv, perturbation_scores.csv')


if __name__ == '__main__':
    main()
