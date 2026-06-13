"""Leave-One-Vocoder-Out: 留出一个声码器家族的spoof测试,其余训练。
检验检测器是否依赖训练中见过的生成/声码器家族特征(而非通用真伪)。"""
from __future__ import annotations
import sys
import numpy as np, pandas as pd
sys.path.insert(0,'/home/admin2/xf/CtrSVDD_Utils')
from eer import compute_eer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score

HNR=['hnr_mean','hnr_std','hnr_low_ratio']
VRI=['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score']
PITCHD=['f0_std','f0_range_semitones','f0_jitter']
CLEAN=PITCHD+VRI+HNR+['long_note_stability']
LOW=['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration']
MFCC=[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]
FULL=LOW+MFCC+PITCHD+['f0_mean','f0_min','f0_max']+VRI+HNR+['long_note_stability']
SETS={'FULL':FULL,'CLEAN':CLEAN,'HNR':HNR,'VRI':VRI}

def evalset(tr,te,cols):
    cols=[c for c in cols if c in tr.columns]
    Xtr=tr[cols].replace([np.inf,-np.inf],np.nan).fillna(0); ytr=tr['_y'].values
    Xte=te[cols].replace([np.inf,-np.inf],np.nan).fillna(0); yte=te['_y'].values
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))]).fit(Xtr,ytr)
    p=m.predict_proba(Xte)[:,1]; pred=(p>=0.5).astype(int)
    eer,_=compute_eer(yte.astype(float),p)
    return round(eer*100,2),round(roc_auc_score(yte,p),3),round(f1_score(yte,pred,zero_division=0),3)

def main():
    df=pd.read_csv('outputs/ctrsvdd_features_e1.csv')
    bona=df[df.label=='real'].copy(); spoof=df[df.label=='fake'].copy()
    # 固定bonafide 50/50,跨折一致
    rng=np.random.RandomState(42); idx=rng.permutation(len(bona)); half=len(bona)//2
    b_tr=bona.iloc[idx[:half]].copy(); b_te=bona.iloc[idx[half:]].copy()
    b_tr['_y']=0; b_te['_y']=0
    groups=['hifigan','nsf-hifigan','ddsp']
    rows=[]
    for held in groups:
        s_tr=spoof[spoof.vocoder_group!=held].copy(); s_tr['_y']=1
        s_te=spoof[spoof.vocoder_group==held].copy(); s_te['_y']=1
        tr=pd.concat([b_tr,s_tr]); te=pd.concat([b_te,s_te])
        for fs,cols in SETS.items():
            eer,auc,f1=evalset(tr,te,cols)
            rows.append(dict(held_out=held,n_test_spoof=len(s_te),feature_set=fs,EER=eer,AUC=auc,F1=f1))
    r=pd.DataFrame(rows); r.to_csv('outputs/ctrsvdd_lovo.csv',index=False)
    print('=== LOVO: EER% (越高=越依赖见过的声码器) ===')
    print(r.pivot(index='held_out',columns='feature_set',values='EER').to_string())
    print('\n=== LOVO: AUC ===')
    print(r.pivot(index='held_out',columns='feature_set',values='AUC').to_string())
    print('\n=== 对照: E1 in-distribution by_vocoder EER(FULL) ===')
    e1=pd.read_csv('outputs/ctrsvdd_eer_e1.csv'); print(e1[(e1.group.str.startswith('voc:'))][['group','FULL','CLEAN','HNR','VRI']].to_string(index=False))
    print('Saved -> outputs/ctrsvdd_lovo.csv')

if __name__=='__main__': main()
