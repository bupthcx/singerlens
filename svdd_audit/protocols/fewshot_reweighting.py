"""Exp2b: source+k reweighting/upsampling (正式版, target=WildSVDD/T02)。
验证 'source+k 不如 target-only' 是 naive mixing 淹没 还是 真负迁移。
方法: 1 target-only / 2 source+k naive / 3 source+k target-upsampling /
      4 source+k domain-balanced weighting / 5 calibration-only(source-only + Platt on k labels)。
指标: AUC(主) + balanced-accuracy。calibration 为单调映射 → AUC≡source-only(关键诊断点)。
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])
KS = [25, 50, 100]
NTRIAL = 20


def rf():
    return Pipeline([('sc', StandardScaler()),
                     ('clf', RandomForestClassifier(n_estimators=300, random_state=0,
                                                    class_weight='balanced', n_jobs=-1))])


def auc_bacc(m, Xte, yte):
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), balanced_accuracy_score(yte, (p > 0.5).astype(int)), p


def main():
    ctr = pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    wild = pd.read_csv('outputs/wildsvdd_features.csv')
    t02 = pd.read_csv('/home/admin2/xf/wildsvdd/wildsvdd_bili_t02.csv')
    tids = set(t02['idx'])
    wild = wild[wild['idx'].isin(tids)].reset_index(drop=True)
    cols = [c for c in FULL if c in ctr.columns and c in wild.columns]
    Xc = ctr[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    yc = (ctr['label'] == 'fake').astype(int).values
    Xw = wild[cols].replace([np.inf, -np.inf], np.nan).fillna(0).values
    yw = (wild['label'] == 'fake').astype(int).values
    print(f'source CtrSVDD {len(yc)} | target WildSVDD-T02 {len(yw)} (real {int((yw==0).sum())}/fake {int((yw==1).sum())})', flush=True)

    rng = np.random.RandomState(42)
    ir = np.where(yw == 0)[0]; ifk = np.where(yw == 1)[0]
    rows = []
    for k in KS:
        acc = {m: {'auc': [], 'bacc': []} for m in
               ['target_only', 'src_naive', 'src_upsample', 'src_domain_balanced', 'calibration_only']}
        for t in range(NTRIAL):
            kr = k // 2
            cal = np.concatenate([rng.choice(ir, kr, replace=False), rng.choice(ifk, k - kr, replace=False)])
            test = np.setdiff1d(np.arange(len(yw)), cal)
            Xte, yte = Xw[test], yw[test]
            Xk, yk = Xw[cal], yw[cal]
            if len(np.unique(yte)) < 2 or len(np.unique(yk)) < 2:
                continue
            # 1 target-only
            a, b, _ = auc_bacc(rf().fit(Xk, yk), Xte, yte); acc['target_only']['auc'].append(a); acc['target_only']['bacc'].append(b)
            # 2 naive
            Xp = np.vstack([Xc, Xk]); yp = np.concatenate([yc, yk])
            a, b, _ = auc_bacc(rf().fit(Xp, yp), Xte, yte); acc['src_naive']['auc'].append(a); acc['src_naive']['bacc'].append(b)
            # 3 target-upsampling (复制 k 到 ~源规模)
            reps = max(1, len(yc) // k)
            Xo = np.vstack([Xc, np.repeat(Xk, reps, axis=0)]); yo = np.concatenate([yc, np.repeat(yk, reps)])
            a, b, _ = auc_bacc(rf().fit(Xo, yo), Xte, yte); acc['src_upsample']['auc'].append(a); acc['src_upsample']['bacc'].append(b)
            # 4 domain-balanced weighting (源/目标 总权重相等)
            w = np.concatenate([np.ones(len(yc)), np.full(k, len(yc) / k)])
            a, b, _ = auc_bacc(rf().fit(Xp, yp, clf__sample_weight=w), Xte, yte)
            acc['src_domain_balanced']['auc'].append(a); acc['src_domain_balanced']['bacc'].append(b)
            # 5 calibration-only: source-only RF -> Platt(logistic) on k cal scores
            ms = rf().fit(Xc, yc)
            sc_cal = ms.predict_proba(Xk)[:, 1]; sc_te = ms.predict_proba(Xte)[:, 1]
            lr = LogisticRegression().fit(sc_cal.reshape(-1, 1), yk)
            p_te = lr.predict_proba(sc_te.reshape(-1, 1))[:, 1]
            acc['calibration_only']['auc'].append(roc_auc_score(yte, p_te))       # ≡ source-only (单调)
            acc['calibration_only']['bacc'].append(balanced_accuracy_score(yte, (p_te > 0.5).astype(int)))
        for m, d in acc.items():
            rows.append(dict(k=k, method=m, AUC=round(np.mean(d['auc']), 3), AUC_std=round(np.std(d['auc']), 3),
                             bal_acc=round(np.mean(d['bacc']), 3)))
    res = pd.DataFrame(rows)
    res.to_csv('outputs/fewshot_reweighting.csv', index=False)
    piv = res.pivot(index='k', columns='method', values='AUC')
    print('\n=== AUC by method (held-out WildSVDD-T02) ==='); print(piv.to_string())
    print('\n=== balanced-accuracy ==='); print(res.pivot(index='k', columns='method', values='bal_acc').to_string())

    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    for m in ['target_only', 'src_naive', 'src_upsample', 'src_domain_balanced', 'calibration_only']:
        d = res[res.method == m]; ax.plot(d.k, d.AUC, '-o', label=m)
    ax.axhline(0.935, ls=':', c='green', label='Wild-within bound'); ax.axhline(0.5, ls='--', c='gray', lw=0.8)
    ax.set_xlabel('# labeled Wild-T02 (k)'); ax.set_ylabel('held-out T02 AUC'); ax.set_ylim(0.4, 1.0)
    ax.legend(fontsize=8); ax.set_title('Exp2b: reweighting/upsampling cannot rescue source+k (T02)')
    fig.tight_layout(); fig.savefig('outputs/fewshot_reweighting.png', dpi=130); plt.close(fig)
    open('outputs/.exp2b.done', 'w').write('ok\n')
    print('\nSaved -> outputs/fewshot_reweighting.{csv,png}', flush=True)


if __name__ == '__main__':
    main()
