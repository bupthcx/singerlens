"""诚实评估：按特征家族拆分 + 交叉验证，识别制作链路泄露 vs 真实风格信号。

动机：直接训练时 baseline 即达 AUC=1.0，疑似 real(原生16k/Demucs) 与 fake(44.1k/Seed-VC)
的制作链路差异造成捷径。本脚本将特征分为'低级/制作相关'与'可解释风格'两类，
分别用分层K折交叉验证评估，检验核心论点：仅凭风格特征能否区分真假。
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

FAMILIES = {
    'lowlevel': ['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration'],
    'mfcc': [f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')],
    'pitch': ['f0_mean','f0_std','f0_min','f0_max','f0_range_semitones','f0_jitter'],
    'pitch_dynamics': ['f0_std','f0_range_semitones','f0_jitter'],  # 八度不变
    'vri': ['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score'],
    'voice_quality': ['hnr_mean','hnr_std','hnr_low_ratio'],
    'stability': ['long_note_stability'],
}
# 可解释风格组合（剔除 mfcc 与低级能量/时长等制作相关特征）
STYLE_ONLY = FAMILIES['pitch'] + FAMILIES['vri'] + FAMILIES['voice_quality'] + FAMILIES['stability']
# 干净集：剔除绝对音高、energy_dynamic、mfcc、原始 rms 等被产线/数据污染的特征
CLEAN = FAMILIES['pitch_dynamics'] + FAMILIES['vri'] + FAMILIES['voice_quality'] + FAMILIES['stability']

def avail(cols, df):
    return [c for c in cols if c in df.columns]

def evaluate(df, cols, y, n_splits=5):
    cols = avail(cols, df)
    if not cols: return None
    X = df[cols].replace([np.inf,-np.inf], np.nan).fillna(0.0)
    model = Pipeline([('scaler', StandardScaler()),
                      ('clf', RandomForestClassifier(n_estimators=300, random_state=42, class_weight='balanced'))])
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    pred = cross_val_predict(model, X, y, cv=skf, method='predict')
    prob = cross_val_predict(model, X, y, cv=skf, method='predict_proba')[:,1]
    return len(cols), accuracy_score(y,pred), f1_score(y,pred,zero_division=0), roc_auc_score(y,prob)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--features', default='outputs/features.csv')
    ap.add_argument('--output-dir', default='outputs')
    args = ap.parse_args()
    df = pd.read_csv(args.features)
    y = df['label'].map({'real':0,'fake':1}).astype(int)
    groups = dict(FAMILIES)
    groups['STYLE_ONLY'] = STYLE_ONLY
    groups['CLEAN'] = CLEAN
    groups['FULL'] = sum(FAMILIES.values(), [])
    rows=[]
    for name, cols in groups.items():
        res = evaluate(df, cols, y)
        if res is None: continue
        n,acc,f1,auc = res
        rows.append({'group':name,'n_features':n,'cv_accuracy':round(acc,4),'cv_f1':round(f1,4),'cv_auc':round(auc,4)})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    out.to_csv(Path(args.output_dir)/'honest_eval_cv.csv', index=False)
    print('\nSaved -> outputs/honest_eval_cv.csv')

if __name__ == '__main__':
    main()
