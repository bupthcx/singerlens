"""P2: failure taxonomy。基于 canonical FULL clip_mean 分数(score_clip_scores.csv)+
窗级特征聚合 + metadata，选 TP/FN/FP/TN 代表样本并定量归因 cross-domain 错误来源。
重点: production artifacts / Demucs 伴奏残留(低 HNR) / source-domain generation cue 缺失。
输出 failure_cases.csv + 组级归因统计供 FINDINGS。
"""
from __future__ import annotations
import numpy as np, pandas as pd

KEYCH = ['hnr_mean', 'hnr_low_ratio', 'f0_mean', 'rms_mean', 'spectral_flatness_mean', 'vri_score']


def clip_feat(path, idcols):
    w = pd.read_csv(path)
    w = w[w['label'].isin(['real', 'fake'])].copy()
    g = w.groupby('clip_id')
    agg = g[KEYCH].mean()
    agg['voiced_frac'] = g['f0_mean'].apply(lambda s: float((s > 0).mean()))  # 有声窗占比
    agg['n_win'] = g.size()
    meta = g[idcols].first()
    return meta.join(agg)


def main():
    cs = pd.read_csv('outputs/score_clip_scores.csv')
    wfeat = clip_feat('outputs/window_features_wild.csv', ['label', 'singer_id', 'model'])
    cfeat = clip_feat('outputs/window_features_ctrsvdd.csv', ['label', 'attack', 'vocoder_group', 'paradigm'])
    sfeat = clip_feat('outputs/window_features.csv', ['label', 'singer_id', 'song_id'])
    feat_by_ds = {'WildSVDD': wfeat, 'CtrSVDD': cfeat, 'SingerLens': sfeat}

    protos = ['CtrSVDD->Wild_all', 'SingerLens->Wild_all', 'CtrSVDD->Wild_T02',
              'SingerLens->Wild_T02', 'Wild_all_within']
    NSEL = 8
    rows = []
    for proto in protos:
        d = cs[cs.protocol == proto].copy()
        if len(d) == 0:
            continue
        ds = d['target'].iloc[0]; feat = feat_by_ds[ds]
        d = d.merge(feat, left_on='clip_id', right_index=True, how='left', suffixes=('', '_f'))
        d['outcome'] = np.where(d.y == 1,
                                np.where(d.score > 0.5, 'TP', 'FN'),
                                np.where(d.score > 0.5, 'FP', 'TN'))
        for oc in ['TP', 'FN', 'FP', 'TN']:
            sub = d[d.outcome == oc].copy()
            if len(sub) == 0:
                continue
            # 代表性: TP/FP 取最高分, FN/TN 取最低分
            sub = sub.sort_values('score', ascending=(oc in ['FN', 'TN'])).head(NSEL)
            for _, r in sub.iterrows():
                rows.append(dict(protocol=proto, target=ds, outcome=oc, clip_id=r['clip_id'],
                                 true=r['label'], score=round(r['score'], 3),
                                 singer=r.get('singer_id', ''), model=r.get('model', ''),
                                 attack=r.get('attack', ''),
                                 hnr=round(r.get('hnr_mean', np.nan), 2),
                                 hnr_low=round(r.get('hnr_low_ratio', np.nan), 2),
                                 voiced_frac=round(r.get('voiced_frac', np.nan), 2),
                                 flatness=round(r.get('spectral_flatness_mean', np.nan), 4),
                                 vri=round(r.get('vri_score', np.nan), 3)))
    fc = pd.DataFrame(rows)
    fc.to_csv('outputs/failure_cases.csv', index=False)

    # 组级归因: 比较 cross-domain 各 outcome 的特征均值
    print('=== 组级特征均值 (按协议×outcome, 全样本非仅代表) ===')
    grp_rows = []
    for proto in protos:
        d = cs[cs.protocol == proto].copy()
        ds = d['target'].iloc[0]; feat = feat_by_ds[ds]
        d = d.merge(feat, left_on='clip_id', right_index=True, how='left')
        d['outcome'] = np.where(d.y == 1, np.where(d.score > 0.5, 'TP', 'FN'),
                                np.where(d.score > 0.5, 'FP', 'TN'))
        for oc, sub in d.groupby('outcome'):
            grp_rows.append(dict(protocol=proto, outcome=oc, n=len(sub),
                                 hnr=round(sub['hnr_mean'].mean(), 2),
                                 hnr_low=round(sub['hnr_low_ratio'].mean(), 2),
                                 voiced=round(sub['f0_mean'].gt(0).mean(), 2),
                                 flatness=round(sub['spectral_flatness_mean'].mean(), 4),
                                 score=round(sub['score'].mean(), 3)))
    gp = pd.DataFrame(grp_rows)
    gp.to_csv('outputs/failure_group_stats.csv', index=False)
    print(gp.to_string(index=False))

    # Wild within 真假 HNR 对照 (域内可分性来源)
    wf = feat_by_ds['WildSVDD']
    print('\n=== WildSVDD 真假特征均值 (域内, 看判别轴) ===')
    print(wf.groupby('label')[['hnr_mean', 'hnr_low_ratio', 'spectral_flatness_mean', 'vri_score']].mean().round(3).to_string())
    print('\n=== CtrSVDD 真假特征均值 (源域判别轴) ===')
    print(cfeat.groupby('label')[['hnr_mean', 'hnr_low_ratio', 'spectral_flatness_mean', 'vri_score']].mean().round(3).to_string())
    print('\nFN by model (CtrSVDD->Wild_all):')
    dd = cs[cs.protocol == 'CtrSVDD->Wild_all'].merge(wf, left_on='clip_id', right_index=True)
    dd['outcome'] = np.where(dd.y == 1, np.where(dd.score > 0.5, 'TP', 'FN'), np.where(dd.score > 0.5, 'FP', 'TN'))
    print(pd.crosstab(dd[dd.y == 1]['model'], dd[dd.y == 1]['outcome']).to_string())
    print('\nSaved -> outputs/failure_cases.csv, failure_group_stats.csv', flush=True)


if __name__ == '__main__':
    main()
