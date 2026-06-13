"""窗级跨数据集迁移评测。核心问题：窗级/MIL 能否修复 cross-dataset collapse？

数据(窗级 long 表): SingerLens(window_features.csv) / CtrSVDD(window_features_ctrsvdd.csv) /
                    WildSVDD(window_features_wild.csv, 含 t02 标记)。
协议: CtrSVDD->Wild(all/T02), SingerLens->Wild(all/T02), Wild 域内CV, 以及源域内CV(参考)。
特征集: FULL/CLEAN/HNR/VRI。窗级表示: clip_mean / MIL_mean / MIL_max / POOL_RICH。
指标: AUC/EER/F1 + mean_score_real / mean_score_fake(分数反转判定) + degradation ratio。
"""
from __future__ import annotations
import warnings, itertools
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, f1_score, roc_curve

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
PITCHD = ['f0_std', 'f0_range_semitones', 'f0_jitter']
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
STAB = ['long_note_stability']
LOW = ['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std']
CLEAN = PITCHD + VRI + VQ + STAB
FULL = LOW + MFCC + ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + STAB
SETS = {'FULL': FULL, 'CLEAN': CLEAN, 'HNR': VQ, 'VRI': VRI}


def eer(y, s):
    fpr, tpr, _ = roc_curve(y, s); fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr)); return float((fpr[i] + fnr[i]) / 2 * 100)


def rf():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
                                                    class_weight='balanced', n_jobs=-1))])


def avail(cols, df): return [c for c in cols if c in df.columns]


def load(path, song_col):
    w = pd.read_csv(path)
    w = w[w['label'].isin(['real', 'fake'])].copy()
    w['y'] = (w['label'] == 'fake').astype(int)
    w['song'] = w[song_col].astype(str)
    return w


def clip_rep(win, cols, mode):
    g = win.groupby('clip_id'); cols = avail(cols, win)
    meta = g[['y', 'song']].first()
    if mode == 'mean':
        X = g[cols].mean(); X.columns = [f'{c}_m' for c in cols]
    else:
        parts = []
        for st, fn in [('m', 'mean'), ('mx', 'max'), ('mn', 'min'), ('sd', 'std')]:
            p = getattr(g[cols], fn)(); p.columns = [f'{c}_{st}' for c in cols]; parts.append(p)
        X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return meta.join(X), list(X.columns)


def metrics(y, s):
    return dict(AUC=round(roc_auc_score(y, s), 3), EER=round(eer(y, s), 2),
                F1=round(f1_score(y, (s > 0.5).astype(int), zero_division=0), 3),
                mean_score_real=round(float(s[y == 0].mean()), 3),
                mean_score_fake=round(float(s[y == 1].mean()), 3),
                reversed=bool(s[y == 1].mean() < s[y == 0].mean()))


# ---- pooled (clip_mean / POOL_RICH) transfer & within ----
def pooled_transfer(src, tgt, cols, mode):
    Xs, fc = clip_rep(src, cols, mode); Xt, _ = clip_rep(tgt, cols, mode)
    fc = [c for c in fc if c in Xt.columns]
    m = rf().fit(Xs[fc], Xs['y']); s = m.predict_proba(Xt[fc])[:, 1]
    return metrics(Xt['y'].values, s)


def pooled_within(df, cols, mode):
    X, fc = clip_rep(df, cols, mode); y = X['y'].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    s = np.zeros(len(y))
    for tr, te in cv.split(X[fc], y, X['song']):
        s[te] = rf().fit(X[fc].iloc[tr], y[tr]).predict_proba(X[fc].iloc[te])[:, 1]
    return metrics(y, s)


# ---- MIL (window classifier) transfer & within ----
def mil_transfer(src, tgt, cols, pool):
    cols = avail(cols, src)
    m = rf().fit(src[cols].replace([np.inf, -np.inf], np.nan).fillna(0), src['y'])
    tw = tgt.copy(); tw['ws'] = m.predict_proba(tgt[cols].replace([np.inf, -np.inf], np.nan).fillna(0))[:, 1]
    return mil_pool(tw, pool)


def mil_within(df, cols, pool):
    cols = avail(cols, df); X = df[cols].replace([np.inf, -np.inf], np.nan).fillna(0); y = df['y'].values
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    ws = np.zeros(len(y))
    for tr, te in cv.split(X, y, df['song']):
        ws[te] = rf().fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
    dd = df.copy(); dd['ws'] = ws
    return mil_pool(dd, pool)


def mil_pool(tw, pool):
    agg = (lambda s: s.max()) if pool == 'max' else (lambda s: s.mean())
    cs = tw.groupby('clip_id').agg(y=('y', 'first'), ws=('ws', agg))
    return metrics(cs['y'].values, cs['ws'].values)


def main():
    SL = load('outputs/window_features.csv', 'song_id')
    CT = load('outputs/window_features_ctrsvdd.csv', 'clip_id')   # CtrSVDD: 每 clip 独立(无 song 分组)→按 clip 分折
    WD = load('outputs/window_features_wild.csv', 'idx')
    WDt = WD[WD['t02'] == True].copy() if 't02' in WD.columns else WD
    print(f'SL win {len(SL)} clips {SL.clip_id.nunique()} | CT win {len(CT)} clips {CT.clip_id.nunique()} | '
          f'WD win {len(WD)} clips {WD.clip_id.nunique()} | WDt02 clips {WDt.clip_id.nunique()}', flush=True)

    protocols = [
        ('CtrSVDD->Wild_all', CT, WD, 'transfer'),
        ('CtrSVDD->Wild_T02', CT, WDt, 'transfer'),
        ('SingerLens->Wild_all', SL, WD, 'transfer'),
        ('SingerLens->Wild_T02', SL, WDt, 'transfer'),
        ('Wild_all_within', WD, WD, 'within'),
        ('Wild_T02_within', WDt, WDt, 'within'),
        ('CtrSVDD_within', CT, CT, 'within'),
        ('SingerLens_within', SL, SL, 'within'),
    ]
    rows = []
    for pname, src, tgt, kind in protocols:
        for fs, cols in SETS.items():
            for method in ['clip_mean', 'MIL_mean', 'MIL_max', 'POOL_RICH']:
                try:
                    if method == 'clip_mean':
                        r = pooled_within(src, cols, 'mean') if kind == 'within' else pooled_transfer(src, tgt, cols, 'mean')
                    elif method == 'POOL_RICH':
                        r = pooled_within(src, cols, 'rich') if kind == 'within' else pooled_transfer(src, tgt, cols, 'rich')
                    elif method == 'MIL_mean':
                        r = mil_within(src, cols, 'mean') if kind == 'within' else mil_transfer(src, tgt, cols, 'mean')
                    else:
                        r = mil_within(src, cols, 'max') if kind == 'within' else mil_transfer(src, tgt, cols, 'max')
                except Exception as e:
                    print('ERR', pname, fs, method, e, flush=True); continue
                rows.append(dict(protocol=pname, kind=kind, feature_set=fs, method=method,
                                 n_clips=tgt.clip_id.nunique(), **r))
        print(f'  done {pname}', flush=True)
    res = pd.DataFrame(rows)
    res.to_csv('outputs/window_cross_dataset_eval.csv', index=False)

    # degradation ratio: 迁移 EER / 对应 Wild-within EER (同 feature_set+method)
    wref = res[res.protocol == 'Wild_all_within'].set_index(['feature_set', 'method'])['EER'].to_dict()
    res['deg_vs_wildwithin'] = res.apply(
        lambda r: round(r['EER'] / max(wref.get((r['feature_set'], r['method']), np.nan), 1e-6), 2)
        if 'Wild' in r['protocol'] and 'within' not in r['protocol'] else np.nan, axis=1)
    res.to_csv('outputs/window_cross_dataset_eval.csv', index=False)

    # summary: 主表 FULL+CLEAN, 看 AUC/EER/reversed
    summ = res[res.feature_set.isin(['FULL', 'CLEAN', 'HNR'])][
        ['protocol', 'feature_set', 'method', 'AUC', 'EER', 'reversed',
         'mean_score_real', 'mean_score_fake', 'deg_vs_wildwithin']]
    summ.to_csv('outputs/window_cross_dataset_summary.csv', index=False)
    print('\n=== FULL transfer-to-Wild & within (AUC | EER | reversed) ===')
    print(res[(res.feature_set == 'FULL')][
        ['protocol', 'method', 'AUC', 'EER', 'reversed', 'mean_score_real', 'mean_score_fake']].to_string(index=False))
    print('\nSaved -> outputs/window_cross_dataset_eval.csv + _summary.csv', flush=True)


if __name__ == '__main__':
    main()
