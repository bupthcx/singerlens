"""P1: score reversal 诊断。汇总所有 cross-dataset / WildSVDD / T02 结果的分数分布。
输出每 (协议×方法): mean/median score(real/fake), AUC, flipped_AUC, EER, 类别(transferable/random/reversed)。
逐 clip 分数(canonical = clip_mean FULL)落盘供 P2/P3 复用。real/fake 分布图。
"""
from __future__ import annotations
import warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])
CLEAN = ['f0_std', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability']
SETS = {'FULL': FULL, 'CLEAN': CLEAN, 'HNR': VQ}


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[i] + fnr[i]) / 2 * 100)


def rf():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
                                                    class_weight='balanced', n_jobs=-1))])


def load(p, song):
    w = pd.read_csv(p); w = w[w['label'].isin(['real', 'fake'])].copy()
    w['y'] = (w['label'] == 'fake').astype(int); w['song'] = w[song].astype(str)
    return w


def clip_mean_rep(win, cols):
    g = win.groupby('clip_id'); cols = [c for c in cols if c in win.columns]
    meta = g[['y', 'song']].first(); X = g[cols].mean()
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return meta.join(X), list(X.columns)


def transfer_scores(src, tgt, cols):
    Xs, fc = clip_mean_rep(src, cols); Xt, _ = clip_mean_rep(tgt, cols)
    fc = [c for c in fc if c in Xt.columns]
    s = rf().fit(Xs[fc], Xs['y']).predict_proba(Xt[fc])[:, 1]
    return Xt['y'].values, s, Xt.index.values


def within_scores(df, cols):
    X, fc = clip_mean_rep(df, cols); y = X['y'].values; s = np.zeros(len(y))
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    for tr, te in cv.split(X[fc], y, X['song']):
        s[te] = rf().fit(X[fc].iloc[tr], y[tr]).predict_proba(X[fc].iloc[te])[:, 1]
    return y, s, X.index.values


def classify(auc):
    if auc < 0.45: return 'reversed'
    if auc < 0.60: return 'random'
    return 'transferable'


def summarize(proto, fs, method, y, s):
    auc = roc_auc_score(y, s)
    return dict(protocol=proto, feature_set=fs, method=method, n=len(y),
                n_real=int((y == 0).sum()), n_fake=int((y == 1).sum()),
                mean_score_real=round(float(s[y == 0].mean()), 3),
                mean_score_fake=round(float(s[y == 1].mean()), 3),
                median_score_real=round(float(np.median(s[y == 0])), 3),
                median_score_fake=round(float(np.median(s[y == 1])), 3),
                AUC=round(auc, 3), flipped_AUC=round(max(auc, 1 - auc), 3),
                EER=round(eer(y, s), 2),
                F1=round(f1_score(y, (s > 0.5).astype(int), zero_division=0), 3),
                separation=round(float(s[y == 1].mean() - s[y == 0].mean()), 3),
                klass=classify(auc))


def main():
    SL = load('outputs/window_features.csv', 'song_id')
    CT = load('outputs/window_features_ctrsvdd.csv', 'clip_id')
    WD = load('outputs/window_features_wild.csv', 'idx')
    WDt = WD[WD.get('t02', False) == True].copy() if 't02' in WD.columns else WD.iloc[0:0]

    protos = [
        ('CtrSVDD->Wild_all', 'transfer', CT, WD), ('CtrSVDD->Wild_T02', 'transfer', CT, WDt),
        ('SingerLens->Wild_all', 'transfer', SL, WD), ('SingerLens->Wild_T02', 'transfer', SL, WDt),
        ('CtrSVDD->SingerLens', 'transfer', CT, SL), ('SingerLens->CtrSVDD', 'transfer', SL, CT),
        ('Wild_all_within', 'within', WD, WD), ('Wild_T02_within', 'within', WDt, WDt),
        ('CtrSVDD_within', 'within', CT, CT), ('SingerLens_within', 'within', SL, SL),
    ]
    rows = []; clip_rows = []
    for proto, kind, src, tgt in protos:
        for fs, cols in SETS.items():
            y, s, idx = (within_scores(src, cols) if kind == 'within' else transfer_scores(src, tgt, cols))
            rows.append(summarize(proto, fs, 'clip_mean', y, s))
            if fs == 'FULL':   # canonical 逐 clip 落盘
                ds = {'Wild': 'WildSVDD', 'CtrSVDD': 'CtrSVDD', 'SingerLens': 'SingerLens'}
                tgt_ds = 'WildSVDD' if 'Wild' in proto.split('->')[-1] else (
                    'CtrSVDD' if proto.split('->')[-1].startswith('CtrSVDD') else 'SingerLens')
                for ci, yy, ss in zip(idx, y, s):
                    clip_rows.append(dict(protocol=proto, target=tgt_ds, clip_id=ci,
                                          y=int(yy), label='fake' if yy else 'real', score=round(float(ss), 4)))
        print('  done', proto, flush=True)
    summ = pd.DataFrame(rows)
    summ.to_csv('outputs/score_distribution_summary.csv', index=False)
    pd.DataFrame(clip_rows).to_csv('outputs/score_clip_scores.csv', index=False)

    print('\n=== score_distribution_summary (FULL) ===')
    print(summ[summ.feature_set == 'FULL'][
        ['protocol', 'mean_score_real', 'mean_score_fake', 'median_score_real', 'median_score_fake',
         'AUC', 'flipped_AUC', 'EER', 'klass']].to_string(index=False))

    # 分布图: FULL clip_mean 各 transfer-to-Wild + Wild within
    cs = pd.read_csv('outputs/score_clip_scores.csv')
    plot_ps = ['CtrSVDD->Wild_all', 'SingerLens->Wild_all', 'Wild_all_within',
               'CtrSVDD->SingerLens', 'SingerLens->CtrSVDD', 'CtrSVDD_within']
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, p in zip(axes.ravel(), plot_ps):
        d = cs[cs.protocol == p]
        if len(d) == 0: ax.set_visible(False); continue
        bins = np.linspace(0, 1, 26)
        ax.hist(d[d.y == 0]['score'], bins=bins, alpha=0.6, label='real', color='#2ca02c', density=True)
        ax.hist(d[d.y == 1]['score'], bins=bins, alpha=0.6, label='fake', color='#B3261E', density=True)
        ax.axvline(0.5, ls='--', c='gray', lw=0.8)
        kl = summ[(summ.protocol == p) & (summ.feature_set == 'FULL')].iloc[0]
        ax.set_title('%s\nAUC=%.2f (%s)' % (p, kl['AUC'], kl['klass']), fontsize=9)
        ax.legend(fontsize=7); ax.set_xlabel('p(fake)', fontsize=8)
    fig.suptitle('FULL clip_mean: real vs fake score distributions', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('outputs/score_distributions.png', dpi=130); plt.close(fig)
    print('\nSaved -> outputs/score_distribution_summary.csv, score_clip_scores.csv, score_distributions.png', flush=True)


if __name__ == '__main__':
    main()
