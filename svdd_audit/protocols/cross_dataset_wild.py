# 第三数据集 cross-dataset: train CtrSVDD / SingerLens (RF, 各特征集) -> test WildSVDD(野生)
import sys, numpy as np, pandas as pd
sys.path.insert(0,'/home/admin2/xf/CtrSVDD_Utils'); from eer import compute_eer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score
HNR=['hnr_mean','hnr_std','hnr_low_ratio']
VRI=['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score']
CLEAN=['f0_std','f0_range_semitones','f0_jitter']+VRI+HNR+['long_note_stability']
LOW=['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration']
MFCC=[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]
FULL=LOW+MFCC+['f0_mean','f0_min','f0_max']+CLEAN
SETS={'FULL':FULL,'CLEAN':CLEAN,'HNR':HNR,'VRI':VRI}
def met(y,sc):
    e,_=compute_eer(y.astype(float),sc)
    try: a=roc_auc_score(y,sc)
    except Exception: a=float('nan')
    return round(e*100,2),round(a,3)
def mk(df,cols):
    X=df[[c for c in cols if c in df.columns]].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return X,(df['label']=='fake').astype(int).values
def cross(tr,te,cols):
    Xtr,ytr=mk(tr,cols); Xte,yte=mk(te,cols); cm=[c for c in Xtr.columns if c in Xte.columns]
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))]).fit(Xtr[cm],ytr)
    return met(yte,m.predict_proba(Xte[cm])[:,1])
def within(df,cols):
    X,y=mk(df,cols)
    sc=cross_val_predict(Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))]),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]
    return met(y,sc)
ctr=pd.read_csv('outputs/ctrsvdd_features_e1.csv'); sl=pd.read_csv('outputs/features_fixed.csv'); wild=pd.read_csv('outputs/wildsvdd_features.csv')
print('WildSVDD: real=%d fake=%d (%d songs)'%((wild.label=='real').sum(),(wild.label=='fake').sum(),wild.idx.nunique() if 'idx' in wild else -1))
rows=[]
for fs,cols in SETS.items():
    rows.append({'feature_set':fs,'protocol':'WildSVDD within-CV',**dict(zip(['EER','AUC'],within(wild,cols)))})
    rows.append({'feature_set':fs,'protocol':'CtrSVDD -> WildSVDD',**dict(zip(['EER','AUC'],cross(ctr,wild,cols)))})
    rows.append({'feature_set':fs,'protocol':'SingerLens -> WildSVDD',**dict(zip(['EER','AUC'],cross(sl,wild,cols)))})
r=pd.DataFrame(rows); r.to_csv('outputs/cross_dataset_wild.csv',index=False)
print(); print(r.pivot(index='protocol',columns='feature_set',values='EER').to_string())
print('\nAUC:'); print(r.pivot(index='protocol',columns='feature_set',values='AUC').to_string())
print('Saved -> outputs/cross_dataset_wild.csv')
