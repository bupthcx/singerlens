"""Cross-dataset transfer: CtrSVDD <-> SingerLens。
A->A / B->B(域内5折CV) 对照 A->B / B->A(跨数据集训练-测试)。
证明 standard(域内) split 高估泛化。特征 FULL/CLEAN/HNR/VRI,指标 EER/AUC/F1。"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0,'/home/admin2/xf/CtrSVDD_Utils')
from eer import compute_eer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score

HNR=['hnr_mean','hnr_std','hnr_low_ratio']
VRI=['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score']
PITCHD=['f0_std','f0_range_semitones','f0_jitter']
CLEAN=PITCHD+VRI+HNR+['long_note_stability']
LOW=['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration']
MFCC=[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]
FULL=LOW+MFCC+PITCHD+['f0_mean','f0_min','f0_max']+VRI+HNR+['long_note_stability']
SETS={'FULL':FULL,'CLEAN':CLEAN,'HNR':HNR,'VRI':VRI}

def metrics(y,score):
    eer,_=compute_eer(y.astype(float),score)
    try: auc=roc_auc_score(y,score)
    except Exception: auc=float('nan')
    f1=f1_score(y,(score>=0.5).astype(int),zero_division=0)
    return round(eer*100,2),round(auc,3),round(f1,3)

def mk(df,cols):
    X=df[[c for c in cols if c in df.columns]].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    y=(df['label']=='fake').astype(int).values
    return X,y

def within(df,cols):
    X,y=mk(df,cols)
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
    sc=cross_val_predict(m,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]
    return metrics(y,sc)

def cross(tr,te,cols):
    Xtr,ytr=mk(tr,cols); Xte,yte=mk(te,cols)
    common=[c for c in Xtr.columns if c in Xte.columns]
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))]).fit(Xtr[common],ytr)
    sc=m.predict_proba(Xte[common])[:,1]
    return metrics(yte,sc)

def main():
    ctr=pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    sl=pd.read_csv('outputs/features_fixed.csv')
    print('CtrSVDD: real=%d fake=%d | SingerLens: real=%d fake=%d'%((ctr.label=='real').sum(),(ctr.label=='fake').sum(),(sl.label=='real').sum(),(sl.label=='fake').sum()))
    rows=[]
    for fs,cols in SETS.items():
        for name,fn in [('A->A(CtrSVDD CV)',lambda:within(ctr,cols)),
                        ('B->B(SingerLens CV)',lambda:within(sl,cols)),
                        ('A->B(train Ctr,test SL)',lambda:cross(ctr,sl,cols)),
                        ('B->A(train SL,test Ctr)',lambda:cross(sl,ctr,cols))]:
            e,a,f=fn(); rows.append({'feature_set':fs,'protocol':name,'EER':e,'AUC':a,'F1':f})
    r=pd.DataFrame(rows); r.to_csv('outputs/cross_dataset_transfer.csv',index=False)
    print('\n=== EER% (越低越好) ==='); print(r.pivot(index='protocol',columns='feature_set',values='EER').to_string())
    print('\n=== AUC ==='); print(r.pivot(index='protocol',columns='feature_set',values='AUC').to_string())
    print('Saved -> outputs/cross_dataset_transfer.csv')

if __name__=='__main__': main()
