"""Exp2-ext: 更公平的 few-shot 'source+k' 臂(上采样/加权 k Wild，避免被 4800 源淹没)。
对照: wild-only k / source+k naive / source+k upweighted / source+k oversampled。
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
KS = [0, 10, 25, 50, 100]
NTRIAL = 20


def rf():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=0,
                                                    class_weight='balanced', n_jobs=-1))])


def main():
    ctr = pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    wild = pd.read_csv('outputs/wildsvdd_features.csv')
    cols = [c for c in FULL if c in ctr.columns and c in wild.columns]
    Xc = ctr[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    yc = (ctr['label'] == 'fake').astype(int).values
    wild = wild.reset_index(drop=True)
    Xw = wild[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    yw = (wild['label'] == 'fake').astype(int).values
    rng = np.random.RandomState(42)
    ir = np.where(yw == 0)[0]; ifk = np.where(yw == 1)[0]
    rows = []
    for k in KS:
        acc = {a: [] for a in ['wild_only', 'src_naive', 'src_upweight', 'src_oversample']}
        for t in range(NTRIAL):
            if k == 0:
                cal = np.array([], int)
            else:
                kr = k // 2; cal = np.concatenate([rng.choice(ir, kr, replace=False),
                                                   rng.choice(ifk, k - kr, replace=False)])
            test = np.setdiff1d(np.arange(len(yw)), cal)
            Xte, yte = Xw[test], yw[test]
            if len(np.unique(yte)) < 2:
                continue
            if k == 0:
                m = rf().fit(Xc, yc)
                for a in acc:
                    acc[a].append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]) if a == 'src_naive' else np.nan)
                continue
            Xk, yk = Xw[cal], yw[cal]
            if len(np.unique(yk)) < 2:
                continue
            # wild-only
            acc['wild_only'].append(roc_auc_score(yte, rf().fit(Xk, yk).predict_proba(Xte)[:, 1]))
            # naive pool
            Xp = np.vstack([Xc, Xk]); yp = np.concatenate([yc, yk])
            acc['src_naive'].append(roc_auc_score(yte, rf().fit(Xp, yp).predict_proba(Xte)[:, 1]))
            # upweight: k Wild 总权重 = 源总权重
            w = np.concatenate([np.ones(len(yc)), np.full(k, len(yc) / k)])
            m = rf().fit(Xp, yp, clf__sample_weight=w)
            acc['src_upweight'].append(roc_auc_score(yte, m.predict_proba(Xte)[:, 1]))
            # oversample: 复制 k Wild 到 ~源规模
            reps = max(1, len(yc) // k)
            Xo = np.vstack([Xc, np.repeat(Xk, reps, axis=0)]); yo = np.concatenate([yc, np.repeat(yk, reps)])
            acc['src_oversample'].append(roc_auc_score(yte, rf().fit(Xo, yo).predict_proba(Xte)[:, 1]))
        rows.append(dict(k=k, **{a: round(np.nanmean(v), 3) if len(v) else np.nan for a, v in acc.items()}))
    res = pd.DataFrame(rows)
    res.to_csv('outputs/fewshot_calibration_ext.csv', index=False)
    print(res.to_string(index=False))
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for a, lab in [('wild_only', 'Wild-only k'), ('src_naive', 'source+k naive'),
                   ('src_upweight', 'source+k upweighted'), ('src_oversample', 'source+k oversampled')]:
        ax.plot(res.k, res[a], '-o', label=lab)
    ax.axhline(0.935, ls=':', c='green', label='within bound'); ax.axhline(0.5, ls='--', c='gray', lw=0.8)
    ax.set_xlabel('# labeled Wild (k)'); ax.set_ylabel('held-out Wild AUC'); ax.legend(fontsize=8); ax.set_ylim(0.4, 1.0)
    ax.set_title('Few-shot Wild calibration: fairer source+k variants')
    fig.tight_layout(); fig.savefig('outputs/fewshot_calibration_ext.png', dpi=130); plt.close(fig)
    open('outputs/.exp2ext.done', 'w').write('ok\n')
    print('Saved -> outputs/fewshot_calibration_ext.{csv,png}')


if __name__ == '__main__':
    main()
