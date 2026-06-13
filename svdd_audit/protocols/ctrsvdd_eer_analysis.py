"""E1/E2: 用我们的 FULL/CLEAN/HNR/VRI 在 CtrSVDD 子集上算 EER。
5折CV得分数 -> overall + by_attack + by_vocoder_group EER。复用 CtrSVDD eer.py。"""
from __future__ import annotations
import sys, argparse
import numpy as np, pandas as pd
sys.path.insert(0,'/home/admin2/xf/CtrSVDD_Utils')
from eer import compute_eer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

HNR=['hnr_mean','hnr_std','hnr_low_ratio']
VRI=['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score']
PITCHD=['f0_std','f0_range_semitones','f0_jitter']
CLEAN=PITCHD+VRI+HNR+['long_note_stability']
LOW=['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration']
MFCC=[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]
FULL=LOW+MFCC+PITCHD+['f0_mean','f0_min','f0_max']+VRI+HNR+['long_note_stability']
SETS={'FULL':FULL,'CLEAN':CLEAN,'HNR':HNR,'VRI':VRI}

def scores_cv(df,cols):
    cols=[c for c in cols if c in df.columns]
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    y=(df['label']=='fake').astype(int).values
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
    skf=StratifiedKFold(5,shuffle=True,random_state=42)
    return cross_val_predict(m,X,y,cv=skf,method='predict_proba')[:,1], y

def eer_pct(y,sc): 
    e,_=compute_eer(y.astype(float),sc); return round(e*100,2)

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',required=True); ap.add_argument('--out',default='outputs/ctrsvdd_eer.csv')
    a=ap.parse_args(); df=pd.read_csv(a.features)
    print('样本: bonafide=%d spoof=%d'%((df.label=='real').sum(),(df.label=='fake').sum()))
    rows=[]
    for fs,cols in SETS.items():
        sc,y=scores_cv(df,cols)
        d=df.reset_index(drop=True); d['_sc']=sc; d['_y']=y
        bf=d[d.label=='real']
        rows.append(dict(feature_set=fs,group='overall',eer=eer_pct(y,sc)))
        for g in sorted(d[d.label=='fake'].vocoder_group.unique()):
            sub=pd.concat([bf,d[(d.label=='fake')&(d.vocoder_group==g)]])
            rows.append(dict(feature_set=fs,group='voc:'+g,eer=eer_pct(sub._y.values,sub._sc.values)))
        for at in sorted(d[d.label=='fake'].attack.unique()):
            sub=pd.concat([bf,d[(d.label=='fake')&(d.attack==at)]])
            rows.append(dict(feature_set=fs,group='atk:'+at,eer=eer_pct(sub._y.values,sub._sc.values)))
    r=pd.DataFrame(rows); r.to_csv(a.out,index=False)
    piv=r.pivot(index='group',columns='feature_set',values='eer')
    print(piv.to_string())
    print('Saved ->',a.out)

if __name__=='__main__': main()
