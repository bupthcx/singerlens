"""Exp1: Uncertainty / reject option (selective classification)。
用 canonical FULL clip_mean 分数(score_clip_scores.csv),confidence=|score-0.5|，
画 risk-coverage：保留 top-c% 最自信预测时的 accuracy。
核心问题：跨数据集崩塌下，弃权(只留高置信)能否恢复精度？
"""
from __future__ import annotations
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

COVS = [1.0, 0.75, 0.5, 0.3, 0.2, 0.1, 0.05]


def acc_at_cov(y, score, cov):
    conf = np.abs(score - 0.5)
    k = max(1, int(round(len(score) * cov)))
    idx = np.argsort(-conf)[:k]
    pred = (score[idx] > 0.5).astype(int)
    return float((pred == y[idx]).mean())


def main():
    cs = pd.read_csv('outputs/score_clip_scores.csv')
    protos = ['CtrSVDD_within', 'Wild_all_within', 'SingerLens_within',
              'CtrSVDD->Wild_all', 'SingerLens->Wild_all', 'CtrSVDD->SingerLens', 'SingerLens->CtrSVDD']
    rows = []
    for p in protos:
        d = cs[cs.protocol == p]
        if len(d) == 0:
            continue
        y = d['y'].values; s = d['score'].values
        kind = 'within' if 'within' in p else 'cross'
        base = max((y == 0).mean(), (y == 1).mean())   # 多数类基率
        rec = dict(protocol=p, kind=kind, n=len(y), majority_base=round(base, 3))
        for c in COVS:
            rec[f'acc@{int(c*100)}'] = round(acc_at_cov(y, s, c), 3)
        rows.append(rec)
    res = pd.DataFrame(rows)
    res.to_csv('outputs/reject_option_riskcoverage.csv', index=False)
    print(res.to_string(index=False))

    # 图: risk(=1-acc)-coverage
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
    cov_grid = np.linspace(0.05, 1.0, 20)
    for p in protos:
        d = cs[cs.protocol == p]
        if len(d) == 0:
            continue
        y = d['y'].values; s = d['score'].values
        accs = [acc_at_cov(y, s, c) for c in cov_grid]
        a = ax[0] if 'within' in p else ax[1]
        a.plot(cov_grid, accs, '-o', ms=3, label=p)
    for a, t in zip(ax, ['Within-domain (selective classification works)',
                         'Cross-dataset (abstention cannot recover)']):
        a.axhline(0.5, ls='--', c='gray', lw=0.8, label='chance')
        a.set_xlabel('coverage (fraction kept, most-confident first)')
        a.set_ylabel('accuracy on retained'); a.set_ylim(0.3, 1.02)
        a.set_title(t, fontsize=10); a.legend(fontsize=7); a.invert_xaxis()
    fig.tight_layout(); fig.savefig('outputs/reject_option_riskcoverage.png', dpi=130); plt.close(fig)
    print('\nSaved -> outputs/reject_option_riskcoverage.csv + .png')


if __name__ == '__main__':
    main()
