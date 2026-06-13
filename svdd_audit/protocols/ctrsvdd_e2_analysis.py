"""E2: 声码器对照。三组二分类 EER + QC(无新混杂)。
组1 bonafide vs spoof(NSF); 组2 bonafide_vocoded vs spoof(NSF); 组3 bonafide vs bonafide_vocoded。"""
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

def eer_of(df,cols):
    cols=[c for c in cols if c in df.columns]
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); y=df['_y'].values
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
    sc=cross_val_predict(m,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]
    e,_=compute_eer(y.astype(float),sc); return round(e*100,2)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--e1-features',required=True); ap.add_argument('--voc-features',required=True); ap.add_argument('--out',default='outputs/ctrsvdd_e2.csv')
    a=ap.parse_args()
    e1=pd.read_csv(a.e1_features); voc=pd.read_csv(a.voc_features)
    bona=e1[e1.label=='real'].copy()
    spoof_nsf=e1[(e1.label=='fake')&(e1.vocoder_group=='nsf-hifigan')].copy()
    voc=voc.copy()  # bonafide_vocoded
    print('bonafide=%d spoof(NSF)=%d bonafide_vocoded=%d'%(len(bona),len(spoof_nsf),len(voc)))
    comps={'G1_bona_vs_spoofNSF':(bona,0,spoof_nsf,1),
           'G2_bonaVOC_vs_spoofNSF':(voc,0,spoof_nsf,1),
           'G3_bona_vs_bonaVOC':(bona,0,voc,1)}
    rows=[]
    for name,(A,ya,B,yb) in comps.items():
        A=A.copy(); A['_y']=ya; B=B.copy(); B['_y']=yb; d=pd.concat([A,B]).reset_index(drop=True)
        rec={'comparison':name,'n_pos':len(B),'n_neg':len(A)}
        for fs,cols in SETS.items(): rec[fs]=eer_of(d,cols)
        rows.append(rec); print(rec,flush=True)
    res=pd.DataFrame(rows); res.to_csv(a.out,index=False)
    # QC: 三组特征分布(无新混杂)
    qc=[]
    for name,g in [('bonafide',bona),('bonafide_vocoded',voc),('spoof_NSF',spoof_nsf)]:
        qc.append({'group':name,'n':len(g),
                   'native_sr_mode':g['native_sr'].mode().iloc[0] if 'native_sr' in g else 'NA',
                   'dur_med':round(g['native_dur'].median(),2) if 'native_dur' in g else round(g['duration'].median(),2),
                   'rms_mean':round(g['rms_mean'].mean(),4),'f0_mean':round(g['f0_mean'].mean(),1),
                   'f0_std':round(g['f0_std'].mean(),1),'hnr_low_ratio':round(g['hnr_low_ratio'].mean(),3)})
    qcdf=pd.DataFrame(qc); qcdf.to_csv('outputs/E2_qc.csv',index=False)
    print('\n=== E2 EER ==='); print(res.to_string(index=False))
    print('\n=== E2_qc (检查无新混杂) ==='); print(qcdf.to_string(index=False))
    print('Saved -> outputs/ctrsvdd_e2.csv + E2_qc.csv')

if __name__=='__main__': main()
