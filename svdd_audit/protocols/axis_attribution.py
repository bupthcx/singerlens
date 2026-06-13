"""Tier2.1: 判别轴归因。源模型(CtrSVDD)与域内模型(Wild)各押在哪些特征?
源模型在 Wild 上的(错)分数被哪些特征驱动?量化"错的轴"。
"""
from __future__ import annotations
import warnings
import numpy as np, pandas as pd
warnings.filterwarnings('ignore')
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from scipy.stats import spearmanr

MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
VQ = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
FULL = (['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std'] + MFCC +
        ['f0_mean', 'f0_std', 'f0_min', 'f0_max', 'f0_range_semitones', 'f0_jitter'] + VRI + VQ + ['long_note_stability'])


def rf(): return RandomForestClassifier(n_estimators=400, random_state=0, class_weight='balanced', n_jobs=-1)


def fam(f):
    if f.startswith('mfcc'): return 'MFCC'
    if f.startswith('hnr'): return 'HNR'
    if f.startswith(('vibrato', 'periodicity', 'micro', 'vri')): return 'VRI'
    if f.startswith('f0'): return 'pitch'
    if f.startswith(('rms', 'energy', 'spectral_flatness')): return 'lowlevel'
    return 'stability'


def main():
    ctr = pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    wild = pd.read_csv('outputs/wildsvdd_features.csv')
    t02 = set(pd.read_csv('/home/admin2/xf/wildsvdd/wildsvdd_bili_t02.csv')['idx'])
    wt = wild[wild['idx'].isin(t02)].reset_index(drop=True)
    cols = [c for c in FULL if c in ctr.columns and c in wt.columns]

    Xc = ctr[cols].replace([np.inf, -np.inf], np.nan).fillna(0); yc = (ctr['label'] == 'fake').astype(int).values
    Xw = wt[cols].replace([np.inf, -np.inf], np.nan).fillna(0); yw = (wt['label'] == 'fake').astype(int).values

    src = rf().fit(Xc, yc); win = rf().fit(Xw, yw)
    imp_src = pd.Series(src.feature_importances_, index=cols)
    imp_win = pd.Series(win.feature_importances_, index=cols)
    # 源模型在 Wild 上的分数 → 各特征与该(错)分数的 spearman 相关
    s_on_w = src.predict_proba(Xw)[:, 1]
    corr = pd.Series({c: spearmanr(Xw[c], s_on_w).correlation for c in cols})

    df = pd.DataFrame({'imp_source(CtrSVDD)': imp_src, 'imp_within(WildT02)': imp_win,
                       'corr_feat_vs_sourceScoreOnWild': corr})
    df['family'] = [fam(c) for c in df.index]
    df = df.sort_values('imp_source(CtrSVDD)', ascending=False)
    df.round(3).to_csv('outputs/axis_attribution.csv')

    # 家族级聚合
    famagg = df.groupby('family')[['imp_source(CtrSVDD)', 'imp_within(WildT02)']].sum().round(3)
    famagg.to_csv('outputs/axis_attribution_family.csv')
    print('=== 家族级重要性 (源 CtrSVDD vs 域内 WildT02) ===')
    print(famagg.to_string())
    print('\n=== 源模型 top-8 特征 + 域内重要性对照 ===')
    print(df.head(8)[['imp_source(CtrSVDD)', 'imp_within(WildT02)', 'family']].to_string())
    print('\n=== 与源在Wild上(错)分数最相关的 top-6 特征 (=错的轴) ===')
    print(corr.abs().sort_values(ascending=False).head(6).to_string())

    # 图: 家族级重要性对比
    fig, ax = plt.subplots(figsize=(7, 4))
    famagg.plot(kind='bar', ax=ax); ax.set_ylabel('summed RF importance')
    ax.set_title('Discriminative axis differs: source(CtrSVDD) vs within-target(WildT02)')
    plt.xticks(rotation=0); fig.tight_layout(); fig.savefig('outputs/axis_attribution.png', dpi=130); plt.close(fig)
    print('\nSaved -> outputs/axis_attribution{,_family}.csv + .png')


if __name__ == '__main__':
    main()
