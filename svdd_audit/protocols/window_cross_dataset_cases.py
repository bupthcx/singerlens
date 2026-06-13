"""窗级跨数据集案例可视化：每案例画 fake_prob + pitch/HNR/energy 三通道时间轴，
解释"局部性≠泛化性"。源域(SingerLens)in-domain fake 有局部 AI 痕迹冒头；
跨域(CtrSVDD->Wild) fake 的局部证据消失 / 真假共有高分 / 分数反转。
图供论文解释，非新指标。
"""
from __future__ import annotations
import warnings, os
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] +
        ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
         'periodicity_score', 'micro_variation', 'vri_score', 'hnr_mean', 'hnr_std', 'hnr_low_ratio', 'long_note_stability'])
OUT = Path('outputs/window_cross_dataset_cases'); OUT.mkdir(parents=True, exist_ok=True)


def rf():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
                                                    class_weight='balanced', n_jobs=-1))])


def load(p, song):
    w = pd.read_csv(p); w = w[w['label'].isin(['real', 'fake'])].copy()
    w['y'] = (w['label'] == 'fake').astype(int); w['song'] = w[song].astype(str)
    w['t'] = w['win_start'] + 1.5
    return w


def Xof(df): return df[[c for c in FULL if c in df.columns]].replace([np.inf, -np.inf], np.nan).fillna(0)


def oof_scores(df):
    X = Xof(df); y = df['y'].values; ws = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in cv.split(X, y, df['song']):
        ws[te] = rf().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    return ws


def plot_case(df_clip, title, fname):
    d = df_clip.sort_values('win_idx')
    t = d['t'].values; p = d['ws'].values
    fig, ax = plt.subplots(4, 1, figsize=(7, 6.5), sharex=True)
    ax[0].plot(t, p, '-o', ms=4, color='#B3261E'); ax[0].axhline(0.5, ls='--', c='gray', lw=0.8)
    ax[0].set_ylim(-0.02, 1.02); ax[0].set_ylabel('p(fake)')
    ax[0].text(0.01, 0.92, 'mean=%.2f  max=%.2f  p90=%.2f' % (p.mean(), p.max(), np.percentile(p, 90)),
               transform=ax[0].transAxes, fontsize=8, va='top')
    for k, (col, lab, c) in enumerate([('f0_mean', 'pitch f0 (Hz)', '#1f77b4'),
                                       ('hnr_mean', 'HNR (dB)', '#2ca02c'),
                                       ('rms_mean', 'energy RMS', '#ff7f0e')], start=1):
        ax[k].plot(t, d[col].values, '-o', ms=3, color=c); ax[k].set_ylabel(lab, fontsize=9)
    ax[3].set_xlabel('clip time (s)')
    fig.suptitle(title, fontsize=9)
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(OUT / fname, dpi=130); plt.close(fig)
    return dict(mean=round(float(p.mean()), 3), max=round(float(p.max()), 3), p90=round(float(np.percentile(p, 90)), 3))


def clip_mean(df, ws):
    d = df.copy(); d['ws'] = ws
    return d.groupby('clip_id').agg(y=('y', 'first'), m=('ws', 'mean'))


def main():
    SL = load('outputs/window_features.csv', 'song_id')
    CT = load('outputs/window_features_ctrsvdd.csv', 'clip_id')
    WD = load('outputs/window_features_wild.csv', 'idx')

    # 1) 源域 in-domain: SingerLens OOF, 选局部尖峰明显的 fake (max 高 mean 中等)
    sl_ws = oof_scores(SL); SL = SL.assign(ws=sl_ws)
    slc = clip_mean(SL, sl_ws)
    fakes = slc[slc.y == 1]
    # 局部性: 该 clip max 高但 mean 不极端 → 局部尖峰
    perclip = SL[SL.y == 1].groupby('clip_id')['ws'].agg(['mean', 'max'])
    perclip['spike'] = perclip['max'] - perclip['mean']
    pick_sl = perclip.sort_values('spike', ascending=False).index[0]

    # 2) 跨域 CtrSVDD->Wild: 训练 CtrSVDD 窗模型, 打分 Wild 窗 (非泄漏)
    mdl = rf().fit(Xof(CT), CT['y'].values)
    wd_ws = mdl.predict_proba(Xof(WD))[:, 1]; WD = WD.assign(ws=wd_ws)
    wdc = clip_mean(WD, wd_ws)
    # 选案例
    fn = wdc[(wdc.y == 1) & (wdc.m < 0.5)].sort_values('m').index            # fake FN
    fp = wdc[(wdc.y == 0) & (wdc.m > 0.5)].sort_values('m', ascending=False).index  # real FP
    tn = wdc[(wdc.y == 0) & (wdc.m < 0.5)].sort_values('m').index            # real TN
    tp = wdc[(wdc.y == 1) & (wdc.m > 0.5)].sort_values('m', ascending=False).index  # fake TP (可能空)

    cases = []
    cases.append(('SingerLens', SL, pick_sl, 'SOURCE in-domain FAKE (OOF)'))
    if len(fn): cases.append(('Wild', WD, fn[0], 'CtrSVDD->Wild  FAKE  False-Negative'))
    if len(fp): cases.append(('Wild', WD, fp[0], 'CtrSVDD->Wild  REAL  False-Positive'))
    if len(tn): cases.append(('Wild', WD, tn[0], 'CtrSVDD->Wild  REAL  True-Negative'))
    if len(tp): cases.append(('Wild', WD, tp[0], 'CtrSVDD->Wild  FAKE  True-Positive'))

    rows = []
    for k, (ds, df, cid, desc) in enumerate(cases):
        dc = df[df.clip_id == cid]
        ytrue = int(dc['y'].iloc[0]); mscore = float(dc['ws'].mean())
        correct = (mscore > 0.5) == bool(ytrue)
        title = '[%s] %s\nclip=%s  true=%s  mean_p=%.2f  decision=%s' % (
            ds, desc, cid, 'FAKE' if ytrue else 'REAL', mscore, 'OK' if correct else 'WRONG')
        st = plot_case(dc, title, f'case{k+1}_{ds}_{"fake" if ytrue else "real"}.png')
        rows.append(dict(case=k + 1, dataset=ds, desc=desc, clip_id=cid,
                         true='fake' if ytrue else 'real', correct=correct, **st))
        print('case', k + 1, ds, desc, cid, st, flush=True)
    pd.DataFrame(rows).to_csv(OUT / 'cases_index.csv', index=False)
    print('Saved cases ->', OUT, flush=True)


if __name__ == '__main__':
    main()
