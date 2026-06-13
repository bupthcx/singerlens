"""CtrSVDD 因子化交叉：范式(SVS/SVC) × 声码器(hifigan/nsf-hifigan/ddsp) 联合 domain。

延伸单轴 LOVO/LOGO → 二维 cell。三块：
  (1) 列联表：量化 CtrSVDD 内范式×声码器的嵌套混杂。
  (2) Leave-One-Cell-Out (LOCO)：留一(范式×声码器)格 spoof，in-dist vs held-out 退化倍数。
  (3) 轴控解耦：固定范式换声码器 / 固定声码器(NSF)换范式，证两轴各自独立致退化。
RF 四特征集 FULL/CLEAN/HNR/VRI。E1 平衡子集(A01-A08, 16k 无采样率混杂)。
"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0, '/home/admin2/xf/CtrSVDD_Utils')
from eer import compute_eer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

HNR = ['hnr_mean', 'hnr_std', 'hnr_low_ratio']
VRI = ['vibrato_rate_mean', 'vibrato_rate_std', 'vibrato_depth_mean', 'vibrato_depth_std',
       'periodicity_score', 'micro_variation', 'vri_score']
PITCHD = ['f0_std', 'f0_range_semitones', 'f0_jitter']
CLEAN = PITCHD + VRI + HNR + ['long_note_stability']
LOW = ['rms_mean', 'rms_std', 'energy_dynamic', 'spectral_flatness_mean', 'spectral_flatness_std', 'duration']
MFCC = [f'mfcc_{i}_{s}' for i in range(1, 14) for s in ('mean', 'std')]
FULL = LOW + MFCC + PITCHD + ['f0_mean', 'f0_min', 'f0_max'] + VRI + HNR + ['long_note_stability']
SETS = {'FULL': FULL, 'CLEAN': CLEAN, 'HNR': HNR, 'VRI': VRI}

# attack -> 范式 (据 REF_ctrsvdd_system_vocoder.md)
SVS = {'A01', 'A02', 'A03', 'A04', 'A05'}
SVC = {'A06', 'A07', 'A08'}


def fit_eval(tr, te, cols):
    cols = [c for c in cols if c in tr.columns]
    Xtr = tr[cols].replace([np.inf, -np.inf], np.nan).fillna(0); ytr = tr['_y'].values
    Xte = te[cols].replace([np.inf, -np.inf], np.nan).fillna(0); yte = te['_y'].values
    m = Pipeline([('s', StandardScaler()),
                  ('c', RandomForestClassifier(n_estimators=300, random_state=42,
                                               class_weight='balanced', n_jobs=-1))]).fit(Xtr, ytr)
    p = m.predict_proba(Xte)[:, 1]
    eer, _ = compute_eer(yte.astype(float), p)
    return round(eer * 100, 2), round(roc_auc_score(yte, p), 3)


def main():
    df = pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    df['paradigm'] = df['attack'].map(lambda a: 'SVS' if a in SVS else ('SVC' if a in SVC else 'bonafide'))
    df['cell'] = df['paradigm'] + '×' + df['vocoder_group']
    bona = df[df.label == 'real'].copy(); spoof = df[df.label == 'fake'].copy()

    # (1) 列联表
    print('=== (1) 范式 × 声码器 列联表 (spoof 样本数) ===')
    ct = pd.crosstab(spoof['paradigm'], spoof['vocoder_group'])
    print(ct.to_string())
    print('cells:', sorted(spoof['cell'].unique()))
    print('注: SVC 仅与 nsf-hifigan 共现, hifigan/ddsp 仅与 SVS → 两轴嵌套混杂; nsf 是唯一跨范式声码器。\n')

    # 固定 bonafide 50/50
    rng = np.random.RandomState(42); idx = rng.permutation(len(bona)); half = len(bona) // 2
    b_tr = bona.iloc[idx[:half]].copy(); b_te = bona.iloc[idx[half:]].copy()
    b_tr['_y'] = 0; b_te['_y'] = 0

    cells = sorted(spoof['cell'].unique())

    # (2) LOCO: in-dist(含该格) vs held-out(去该格), 同一 test
    print('=== (2) Leave-One-Cell-Out: EER% (in-dist | held-out | ratio) ===')
    rows = []
    for cell in cells:
        s_te = spoof[spoof.cell == cell].copy(); s_te['_y'] = 1
        te = pd.concat([b_te, s_te])
        for fs, cols in SETS.items():
            # in-dist: 训练含该格
            s_in = spoof.copy(); s_in['_y'] = 1
            eer_in, auc_in = fit_eval(pd.concat([b_tr, s_in]), te, cols)
            # held-out: 训练去该格
            s_out = spoof[spoof.cell != cell].copy(); s_out['_y'] = 1
            eer_out, auc_out = fit_eval(pd.concat([b_tr, s_out]), te, cols)
            ratio = round(eer_out / max(eer_in, 1e-6), 2)
            rows.append(dict(cell=cell, n_test=len(s_te), feature_set=fs,
                             eer_indist=eer_in, eer_heldout=eer_out, ratio=ratio,
                             auc_indist=auc_in, auc_heldout=auc_out))
    loco = pd.DataFrame(rows)
    loco.to_csv('outputs/ctrsvdd_factorial_loco.csv', index=False)
    for fs in SETS:
        sub = loco[loco.feature_set == fs]
        print(f'\n  [{fs}]')
        print(sub[['cell', 'n_test', 'eer_indist', 'eer_heldout', 'ratio']].to_string(index=False))

    # (3) 轴控解耦
    print('\n=== (3) 轴控解耦: 单独隔离声码器/范式效应 ===')
    def cellset(name):
        s = spoof[spoof.cell == name].copy(); s['_y'] = 1; return s
    C1 = 'SVS×hifigan'; C2 = 'SVS×ddsp'; C3 = 'SVS×nsf-hifigan'; C4 = 'SVC×nsf-hifigan'
    contrasts = [
        # (描述, 固定轴, train cells, test cell, shift类型)
        ('voc-shift|para=SVS', 'paradigm=SVS', [C1], C2, 'held-out vocoder (ddsp), 范式同'),
        ('voc-shift|para=SVS', 'paradigm=SVS', [C1], C3, 'held-out vocoder (nsf), 范式同'),
        ('para-shift|voc=nsf', 'vocoder=nsf',  [C4], C3, 'held-out paradigm (SVS), 声码器同'),
        ('para-shift|voc=nsf', 'vocoder=nsf',  [C3], C4, 'held-out paradigm (SVC), 声码器同'),
    ]
    arows = []
    for desc, fixed, trcells, tecell, note in contrasts:
        s_te = cellset(tecell); te = pd.concat([b_te, s_te])
        s_tr = pd.concat([cellset(c) for c in trcells])
        # in-dist 参考: 训练含目标格(目标格自身 + train cells)
        s_in = pd.concat([s_tr, cellset(tecell)]).drop_duplicates()
        for fs, cols in SETS.items():
            eer_out, auc_out = fit_eval(pd.concat([b_tr, s_tr]), te, cols)
            eer_in, auc_in = fit_eval(pd.concat([b_tr, s_in]), te, cols)
            arows.append(dict(contrast=desc, fixed=fixed, train='+'.join(trcells), test=tecell,
                              note=note, feature_set=fs, eer_indist=eer_in, eer_heldout=eer_out,
                              ratio=round(eer_out / max(eer_in, 1e-6), 2)))
    axis = pd.DataFrame(arows)
    axis.to_csv('outputs/ctrsvdd_factorial_axis.csv', index=False)
    for fs in SETS:
        sub = axis[axis.feature_set == fs]
        print(f'\n  [{fs}] EER% (in-dist | held-out | ratio)')
        print(sub[['contrast', 'train', 'test', 'eer_indist', 'eer_heldout', 'ratio']].to_string(index=False))

    print('\nSaved -> outputs/ctrsvdd_factorial_{loco,axis}.csv')


if __name__ == '__main__':
    main()
