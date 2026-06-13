"""跨歌手泛化（Leave-One-Singer-Out）：训练 2 位歌手，测试留出的第 3 位。
检验检测器学到的是'通用 AI 痕迹'还是'某歌手的个人风格'。"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

FAM = {
 'lowlevel':['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration'],
 'mfcc':[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')],
 'pitch_dynamics':['f0_std','f0_range_semitones','f0_jitter'],
 'vri':['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score'],
 'voice_quality':['hnr_mean','hnr_std','hnr_low_ratio'],
 'stability':['long_note_stability'],
}
CLEAN = FAM['pitch_dynamics']+FAM['vri']+FAM['voice_quality']+FAM['stability']
FULL  = sum(FAM.values(), [])
GROUPS = {'voice_quality':FAM['voice_quality'],'vri':FAM['vri'],'CLEAN':CLEAN,'FULL':FULL}

def avail(cols,df): return [c for c in cols if c in df.columns]

def run(df, cols):
    cols=avail(cols,df)
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    y=df['label'].map({'real':0,'fake':1}).astype(int)
    singers=sorted(df['singer_id'].unique())
    rows=[]
    for held in singers:
        tr=df['singer_id']!=held; te=df['singer_id']==held
        if y[te].nunique()<2: continue
        m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
        m.fit(X[tr],y[tr])
        pred=m.predict(X[te]); prob=m.predict_proba(X[te])[:,1]
        rows.append((held, accuracy_score(y[te],pred), f1_score(y[te],pred,zero_division=0), roc_auc_score(y[te],prob)))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',default='outputs/features_fixed.csv')
    ap.add_argument('--output-dir',default='outputs'); a=ap.parse_args()
    df=pd.read_csv(a.features)
    out=[]
    print('%-16s %-10s %8s %8s %8s' % ('feature_group','test_on','acc','f1','auc'))
    for gname,cols in GROUPS.items():
        rows=run(df,cols); aucs=[r[3] for r in rows]
        for held,acc,f1,auc in rows:
            print('%-16s %-10s %8.3f %8.3f %8.3f' % (gname,held,acc,f1,auc))
            out.append({'feature_group':gname,'test_on':held,'accuracy':round(acc,4),'f1':round(f1,4),'auc':round(auc,4)})
        print('%-16s %-10s %8s %8s %8.3f' % (gname,'MEAN','','',np.mean(aucs)))
        out.append({'feature_group':gname,'test_on':'MEAN','accuracy':'','f1':'','auc':round(np.mean(aucs),4)})
        print('-'*54)
    pd.DataFrame(out).to_csv(a.output_dir+'/cross_singer_loso.csv',index=False)
    print('Saved -> outputs/cross_singer_loso.csv')

if __name__=='__main__': main()
