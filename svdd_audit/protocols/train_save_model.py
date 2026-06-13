"""训练并保存 Demo 用检测器(CLEAN 诚实特征) + 雷达图参考统计。"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from singerlens.profile import fit_reference

CLEAN = ['f0_std','f0_range_semitones','f0_jitter',
         'vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std',
         'periodicity_score','micro_variation','vri_score',
         'hnr_mean','hnr_std','hnr_low_ratio','long_note_stability']

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--features',default='outputs/features_fixed.csv')
    ap.add_argument('--out-dir',default='outputs')
    a=ap.parse_args()
    df=pd.read_csv(a.features)
    cols=[c for c in CLEAN if c in df.columns]
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    y=df['label'].map({'real':0,'fake':1}).astype(int)
    model=Pipeline([('scaler',StandardScaler()),
                    ('clf',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
    model.fit(X,y)
    Path(a.out_dir).mkdir(parents=True,exist_ok=True)
    joblib.dump({'model':model,'features':cols}, Path(a.out_dir)/'detector.joblib')
    ref=fit_reference(df)
    json.dump(ref, open(Path(a.out_dir)/'profile_reference.json','w'), ensure_ascii=False, indent=2)
    print('saved detector.joblib (%d features) + profile_reference.json' % len(cols))
    print('train fit acc=%.3f' % model.score(X,y))

if __name__=='__main__': main()
