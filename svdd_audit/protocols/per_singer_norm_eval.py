"""per-singer 标准化改进：对每位歌手做组内z-score(用该歌手全部片段均值/方差,无标签),
剥离歌手身份成分。对比 raw vs per-singer-z 在 Leave-One-Singer-Out 下的表现。
归一化合法性:测试时对未见歌手的一批片段用其自身统计归一,不使用标签,无泄露。"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

HNR=['hnr_mean','hnr_std','hnr_low_ratio']
VRI=['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score']
CLEAN=['f0_std','f0_range_semitones','f0_jitter']+VRI+HNR+['long_note_stability']
LOW=['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration']
MFCC=[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]
FULL=LOW+MFCC+['f0_mean','f0_min','f0_max']+CLEAN
SETS={'FULL':FULL,'CLEAN':CLEAN,'HNR':HNR,'VRI':VRI}

def per_singer_z(df, cols):
    out=df.copy()
    for c in cols:
        if c not in df.columns: continue
        g=df.groupby('singer_id')[c]
        out[c]=(df[c]-g.transform('mean'))/(g.transform('std')+1e-9)
    return out

def loso(df, cols):
    cols=[c for c in cols if c in df.columns]
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0); y=df['label'].map({'real':0,'fake':1}).astype(int)
    aucs=[];f1s=[];accs=[]
    for h in sorted(df.singer_id.unique()):
        tr=df.singer_id!=h; te=df.singer_id==h
        m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
        m.fit(X[tr],y[tr]); pb=m.predict_proba(X[te])[:,1]; pr=m.predict(X[te])
        aucs.append(roc_auc_score(y[te],pb)); f1s.append(f1_score(y[te],pr,zero_division=0)); accs.append(accuracy_score(y[te],pr))
    return np.mean(aucs),np.mean(f1s),np.mean(accs)

def main():
    df=pd.read_csv('outputs/features_fixed.csv')
    allcols=sorted(set(sum(SETS.values(),[])))
    dz=per_singer_z(df, allcols)
    print('%-8s | %-22s | %-22s' % ('','raw (原始)','per-singer-z (改进)'))
    print('%-8s | %6s %6s %6s | %6s %6s %6s' % ('set','AUC','F1','Acc','AUC','F1','Acc'))
    out=[]
    for s,cols in SETS.items():
        ra,rf,rc=loso(df,cols); za,zf,zc=loso(dz,cols)
        print('%-8s | %6.3f %6.3f %6.3f | %6.3f %6.3f %6.3f  (ΔAUC %+.3f)' % (s,ra,rf,rc,za,zf,zc,za-ra))
        out.append({'feature_set':s,'raw_auc':round(ra,3),'raw_f1':round(rf,3),'z_auc':round(za,3),'z_f1':round(zf,3),'delta_auc':round(za-ra,3)})
    pd.DataFrame(out).to_csv('outputs/per_singer_norm.csv',index=False)
    print('Saved -> outputs/per_singer_norm.csv')

if __name__=='__main__': main()
