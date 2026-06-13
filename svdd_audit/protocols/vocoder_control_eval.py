"""声码器对齐对照评估：对比 (real原始 vs fake) 与 (real_vocoded vs fake) 两种条件下
各特征组的 CV/LOSO AUC。若某组 AUC 在 vocoded 条件下大幅下降，说明该组主要捕捉
'声码器产线伪迹'而非真实演唱差异。"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

FAM={
 'lowlevel':['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration'],
 'mfcc_mean':[f'mfcc_{i}_mean' for i in range(1,14)],
 'mfcc':[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')],
 'pitch_dynamics':['f0_std','f0_range_semitones','f0_jitter'],
 'vri':['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score'],
 'voice_quality':['hnr_mean','hnr_std','hnr_low_ratio'],
}
FAM['CLEAN']=FAM['pitch_dynamics']+FAM['vri']+FAM['voice_quality']+['long_note_stability']
FAM['FULL']=FAM['lowlevel']+FAM['mfcc']+FAM['pitch_dynamics']+FAM['vri']+FAM['voice_quality']+['long_note_stability']

def cv_auc(df,cols):
    cols=[c for c in cols if c in df.columns]
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0); y=df['label'].map({'real':0,'fake':1}).astype(int)
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
    skf=StratifiedKFold(5,shuffle=True,random_state=42)
    pb=cross_val_predict(m,X,y,cv=skf,method='predict_proba')[:,1]
    return roc_auc_score(y,pb)

def loso_auc(df,cols):
    cols=[c for c in cols if c in df.columns]
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0); y=df['label'].map({'real':0,'fake':1}).astype(int)
    aucs=[]
    for h in sorted(df.singer_id.unique()):
        tr=df.singer_id!=h; te=df.singer_id==h
        m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
        m.fit(X[tr],y[tr]); aucs.append(roc_auc_score(y[te],m.predict_proba(X[te])[:,1]))
    return np.mean(aucs)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--orig',default='outputs/features_fixed.csv')
    ap.add_argument('--vocoded',default='outputs/features_vocoded.csv')
    ap.add_argument('--output-dir',default='outputs')
    a=ap.parse_args()
    do=pd.read_csv(a.orig); dv=pd.read_csv(a.vocoded)
    print('%-16s | %-15s | %-15s' % ('feature_group','real_orig vs fake','real_VOCODED vs fake'))
    print('%-16s | %7s %7s | %7s %7s' % ('','CV_auc','LOSO','CV_auc','LOSO'))
    out=[]
    for g,cols in FAM.items():
        co,lo=cv_auc(do,cols),loso_auc(do,cols)
        cv,lv=cv_auc(dv,cols),loso_auc(dv,cols)
        print('%-16s | %7.3f %7.3f | %7.3f %7.3f' % (g,co,lo,cv,lv))
        out.append({'group':g,'orig_cv':round(co,4),'orig_loso':round(lo,4),'voc_cv':round(cv,4),'voc_loso':round(lv,4)})
    pd.DataFrame(out).to_csv(a.output_dir+'/vocoder_control.csv',index=False)
    print('Saved -> outputs/vocoder_control.csv')

if __name__=='__main__': main()
