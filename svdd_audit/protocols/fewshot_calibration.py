"""Exp2: Few-shot Wild calibration。
给源检测器(CtrSVDD-RF-FULL)加 k 个标注 Wild 片段，看多少目标标签能廉价修复跨域鸿沟。
对照: source+k vs Wild-only-k vs k=0(纯跨域崩塌) vs Wild-within 上界。
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])
KS = [0, 5, 10, 25, 50, 100]
NTRIAL = 20


def rf():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=0,
                                                    class_weight='balanced', n_jobs=-1))])


def prep(df, label_col='label'):
    df = df.copy()
    df['y'] = (df[label_col] == 'fake').astype(int) if df[label_col].dtype == object else df[label_col]
    return df


def main():
    cols = FULL
    ctr = pd.read_csv('outputs/ctrsvdd_features_e1.csv'); ctr = prep(ctr)
    wild = pd.read_csv('outputs/wildsvdd_features.csv'); wild = prep(wild)
    cols = [c for c in cols if c in ctr.columns and c in wild.columns]
    Xc = ctr[cols].replace([np.inf, -np.inf], np.nan).fillna(0); yc = ctr['y'].values
    wild = wild.reset_index(drop=True)
    Xw = wild[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values; yw = wild['y'].values
    print(f'CtrSVDD {len(yc)}  Wild {len(yw)} (real {int((yw==0).sum())}/fake {int((yw==1).sum())})')

    rng = np.random.RandomState(42)
    idx_real = np.where(yw == 0)[0]; idx_fake = np.where(yw == 1)[0]
    rows = []
    for k in KS:
        aucs_src, aucs_only = [], []
        for t in range(NTRIAL):
            if k == 0:
                cal = np.array([], int)
            else:
                kr = k // 2; kf = k - kr
                cal = np.concatenate([rng.choice(idx_real, kr, replace=False),
                                      rng.choice(idx_fake, kf, replace=False)])
            test = np.setdiff1d(np.arange(len(yw)), cal)
            Xte, yte = Xw[test], yw[test]
            if len(np.unique(yte)) < 2:
                continue
            # source + k wild
            if k == 0:
                m = rf().fit(Xc, yc)
            else:
                Xtr = np.vstack([Xc.values, Xw[cal]]); ytr = np.concatenate([yc, yw[cal]])
                m = rf().fit(Xtr, ytr)
            aucs_src.append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
            # wild-only k
            if k >= 4 and len(np.unique(yw[cal])) == 2:
                mo = rf().fit(Xw[cal], yw[cal])
                aucs_only.append(roc_auc_score(yte, mo.predict_proba(Xte)[:, 1]))
        rows.append(dict(k=k,
                         src_plus_k_AUC=round(np.mean(aucs_src), 3), src_plus_k_std=round(np.std(aucs_src), 3),
                         wild_only_k_AUC=round(np.mean(aucs_only), 3) if aucs_only else np.nan))
    res = pd.DataFrame(rows)
    # Wild-within 上界 (5-fold 近似: 用全 wild 自身 CV 的已知值)
    res.to_csv('outputs/fewshot_calibration.csv', index=False)
    print(res.to_string(index=False))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.errorbar(res.k, res.src_plus_k_AUC, yerr=res.src_plus_k_std, marker='o', label='CtrSVDD + k Wild labels')
    ax.plot(res.k, res.wild_only_k_AUC, '--s', label='Wild-only (k labels)')
    ax.axhline(0.935, ls=':', c='green', label='Wild-within upper bound (~0.935)')
    ax.axhline(0.5, ls='--', c='gray', lw=0.8, label='chance')
    ax.set_xlabel('# labeled Wild clips (k)'); ax.set_ylabel('AUC on held-out Wild')
    ax.set_title('Few-shot Wild calibration: a few target labels recover the gap')
    ax.legend(fontsize=8); ax.set_ylim(0.4, 1.0)
    fig.tight_layout(); fig.savefig('outputs/fewshot_calibration.png', dpi=130); plt.close(fig)
    print('\nSaved -> outputs/fewshot_calibration.csv + .png')


if __name__ == '__main__':
    main()
