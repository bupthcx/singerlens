"""窗级时序检测评测 (MIL)。

对比三种段级表示，检验"难 fake 有局部窗 AI 痕迹被整段平均掉"假说：
  1) CLIP_MEAN   : 每特征窗间均值 (≈现有整段聚合，基线)
  2) POOL_RICH   : 每特征 {mean,max,min,std} 拼接 (给分类器看时序极值)
  3) MIL_*       : 窗级 RF 逐窗打分，再按 {mean,max,top2,top3} 池化到段
协议: within (StratifiedGroupKFold by song) + LOSO-singer (留一歌手)。
特征集: FULL / CLEAN / HNR。
审计: 负类 = real(默认) 或 real_vocoded(--neg vocoded)，后者若窗级 max 仍强=真信号，塌=只是声码器。
难度: 输出难 fake 慢歌 (dearfriend/天黑黑/句号 等) 的逐歌 fake recall。
"""
from __future__ import annotations
import argparse, warnings
from pathlib import Path
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedGroupKFold, LeaveOneGroupOut
from sklearn.metrics import roc_auc_score, f1_score

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
PITCH_DYN = ['f0_std', 'f0_range_semitones', 'f0_jitter']
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean',
       'vibrato_depth_std', 'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
STAB = ['long_note_stability']
LOWLEVEL = ['rms_mean', 'rms_std', 'energy_dynamic',
            'spectral_flatness_mean', 'spectral_flatness_std']
CLEAN = PITCH_DYN + VRI + VQ + STAB
FULL = LOWLEVEL + MFCC + ['f0_mean', 'f0_std', 'f0_min', 'f0_max',
                          'f0_range_semitones', 'f0_jitter'] + VRI + VQ + STAB
FEATURE_SETS = {'FULL': FULL, 'CLEAN': CLEAN, 'HNR': VQ}

HARD_SONGS = ['dearfriend', '天黑黑', '句号', 'tianheihei', 'juhao']  # 悲伤慢歌


def eer(y, s):
    from sklearn.metrics import roc_curve
    fpr, tpr, _ = roc_curve(y, s)
    fnr = 1 - tpr
    i = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[i] + fnr[i]) / 2 * 100)


def rf():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=42,
                                                    class_weight='balanced', n_jobs=-1))])


def avail(cols, df):
    return [c for c in cols if c in df.columns]


# ---------- 段级表示 ----------
def clip_table(win, cols, mode):
    """把 long-format 窗表聚合成段级 (clip_id 一行)。mode in {mean, rich}."""
    g = win.groupby('clip_id')
    meta = g[['label', 'singer_id', 'song_id', 'source_type']].first()
    cols = avail(cols, win)
    if mode == 'mean':
        X = g[cols].mean()
        X.columns = [f'{c}_mean' for c in cols]
    else:  # rich
        parts = []
        for stat, fn in [('mean', 'mean'), ('max', 'max'), ('min', 'min'), ('std', 'std')]:
            p = getattr(g[cols], fn)()
            p.columns = [f'{c}_{stat}' for c in cols]
            parts.append(p)
        X = pd.concat(parts, axis=1)
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    out = meta.join(X)
    return out, [c for c in X.columns]


# ---------- 评测：池化段级表示 (paradigm A) ----------
def eval_pooled(df, feats, y, groups, protocol):
    X = df[feats]
    if protocol == 'within':
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X, y, groups['song'])
    else:  # loso-singer
        cv = LeaveOneGroupOut()
        splits = cv.split(X, y, groups['singer'])
    scores = np.zeros(len(y)); seen = np.zeros(len(y), bool)
    for tr, te in splits:
        m = rf().fit(X.iloc[tr], y[tr])
        scores[te] = m.predict_proba(X.iloc[te])[:, 1]; seen[te] = True
    return scores, seen


# ---------- 评测：窗级 MIL (paradigm B) ----------
def eval_mil(win, cols, protocol):
    """窗级训练→逐窗打分→段级 {mean,max,top2,top3} 池化。返回各池化的段级分数 dict。"""
    cols = avail(cols, win)
    win = win.copy()
    win['y'] = (win['label'] == 'fake').astype(int)
    X = win[cols].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    yw = win['y'].values
    win_scores = np.zeros(len(win)); seen = np.zeros(len(win), bool)
    if protocol == 'within':
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        splits = cv.split(X, yw, win['song_id'])
    else:
        cv = LeaveOneGroupOut()
        splits = cv.split(X, yw, win['singer_id'])
    for tr, te in splits:
        m = rf().fit(X.iloc[tr], yw[tr])
        win_scores[te] = m.predict_proba(X.iloc[te])[:, 1]; seen[te] = True
    win = win.assign(ws=win_scores)
    pooled = {}
    for name in ['mean', 'max', 'top2', 'top3']:
        def agg(s, n=name):
            s = np.sort(s.values)[::-1]
            if n == 'mean': return s.mean()
            if n == 'max': return s[0]
            if n == 'top2': return s[:2].mean()
            if n == 'top3': return s[:3].mean()
        cs = win.groupby('clip_id')['ws'].apply(agg)
        pooled[name] = cs
    meta = win.groupby('clip_id')[['label', 'singer_id', 'song_id']].first()
    return pooled, meta


def run(args):
    win = pd.read_csv(args.features)
    # 选负类
    if args.neg == 'vocoded':
        win = win[win['label'].isin(['real_vocoded', 'fake'])].copy()
        win['label'] = win['label'].replace({'real_vocoded': 'real'})
    else:
        win = win[win['label'].isin(['real', 'fake'])].copy()
    print(f'[neg={args.neg}] windows={len(win)} clips={win["clip_id"].nunique()} '
          f'singers={win["singer_id"].nunique()}', flush=True)
    rows = []
    for fsname, cols in FEATURE_SETS.items():
        for protocol in ['within', 'loso']:
            # paradigm A: pooled clip table
            for mode in ['mean', 'rich']:
                tbl, fcols = clip_table(win, cols, mode)
                y = (tbl['label'] == 'fake').astype(int).values
                groups = {'song': tbl['song_id'].values, 'singer': tbl['singer_id'].values}
                sc, seen = eval_pooled(tbl, fcols, y, groups, protocol)
                rows.append(dict(feature_set=fsname, protocol=protocol,
                                 method=f'POOL_{mode.upper()}',
                                 AUC=round(roc_auc_score(y[seen], sc[seen]), 3),
                                 EER=round(eer(y[seen], sc[seen]), 2),
                                 F1=round(f1_score(y[seen], (sc[seen] > 0.5).astype(int), zero_division=0), 3),
                                 n=int(seen.sum())))
            # paradigm B: MIL window-level
            pooled, meta = eval_mil(win, cols, protocol)
            ym = (meta['label'] == 'fake').astype(int).values
            for pname, cs in pooled.items():
                cs = cs.reindex(meta.index)
                rows.append(dict(feature_set=fsname, protocol=protocol,
                                 method=f'MIL_{pname}',
                                 AUC=round(roc_auc_score(ym, cs.values), 3),
                                 EER=round(eer(ym, cs.values), 2),
                                 F1=round(f1_score(ym, (cs.values > 0.5).astype(int), zero_division=0), 3),
                                 n=len(ym)))
        print(f'  done {fsname}', flush=True)
    res = pd.DataFrame(rows)
    out = f'outputs/window_temporal_{args.neg}.csv'
    res.to_csv(out, index=False)
    print('\n=== RESULTS (neg=%s) ===' % args.neg, flush=True)
    print(res.to_string(index=False), flush=True)
    print('saved ->', out, flush=True)

    # 难度: LOSO MIL_max vs MIL_mean 的逐难歌 fake recall (FULL+CLEAN)
    if args.neg == 'real':
        drows = []
        for fsname in ['FULL', 'CLEAN', 'HNR']:
            pooled, meta = eval_mil(win, FEATURE_SETS[fsname], 'loso')
            fake = meta[meta['label'] == 'fake']
            for pname in ['mean', 'max']:
                cs = pooled[pname].reindex(meta.index)
                thr = 0.5
                for song in sorted(fake['song_id'].unique()):
                    idx = fake[fake['song_id'] == song].index
                    rec = float((cs.reindex(idx).values > thr).mean())
                    drows.append(dict(feature_set=fsname, pool=pname, song=song,
                                      n_fake=len(idx), fake_recall=round(rec, 3),
                                      hard=song in HARD_SONGS))
        dd = pd.DataFrame(drows)
        dd.to_csv('outputs/window_temporal_hardsong_recall.csv', index=False)
        print('\n=== HARD-SONG fake recall (LOSO, MIL mean vs max) ===', flush=True)
        piv = dd.pivot_table(index=['feature_set', 'song', 'hard'], columns='pool',
                             values='fake_recall').reset_index()
        print(piv.to_string(index=False), flush=True)
        print('saved -> outputs/window_temporal_hardsong_recall.csv', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='outputs/window_features.csv')
    ap.add_argument('--neg', choices=['real', 'vocoded'], default='real')
    run(ap.parse_args())
