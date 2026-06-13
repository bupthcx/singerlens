"""Leave-One-Song-Out（按歌手内留一首歌）：对每位歌手，训练其其余歌曲、测试留出的那首，
检验检测器对'同一歌手的未见歌曲'的泛化。与 Leave-One-Singer-Out 互补。"""
from __future__ import annotations
import argparse
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

FAM={
 'vri':['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score'],
 'voice_quality':['hnr_mean','hnr_std','hnr_low_ratio'],
}
FAM['CLEAN']=['f0_std','f0_range_semitones','f0_jitter']+FAM['vri']+FAM['voice_quality']+['long_note_stability']
FAM['FULL']=FAM['CLEAN']+['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration','f0_mean','f0_min','f0_max']+[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]

def evalgroup(df, cols):
    cols=[c for c in cols if c in df.columns]
    y=df['label'].map({'real':0,'fake':1}).astype(int)
    X=df[cols].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    rows=[]
    for singer in sorted(df.singer_id.unique()):
        songs=sorted(df[df.singer_id==singer].song_id.unique())
        if len(songs)<2: continue
        for held in songs:
            tr=(df.singer_id==singer)&(df.song_id!=held)
            te=(df.singer_id==singer)&(df.song_id==held)
            if y[tr].nunique()<2 or y[te].nunique()<2: continue
            m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
            m.fit(X[tr],y[tr]); pb=m.predict_proba(X[te])[:,1]; pr=m.predict(X[te])
            rows.append((singer,held,accuracy_score(y[te],pr),f1_score(y[te],pr,zero_division=0),roc_auc_score(y[te],pb)))
    return rows

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--features',default='outputs/features_fixed.csv'); ap.add_argument('--output-dir',default='outputs')
    a=ap.parse_args(); df=pd.read_csv(a.features); out=[]
    for g,cols in FAM.items():
        rows=evalgroup(df,cols)
        if not rows: continue
        print('==== %s ====' % g)
        print('%-10s %-16s %6s %6s %6s'%('singer','held_song','acc','f1','auc'))
        for s,h,ac,f1,au in rows:
            print('%-10s %-16s %6.3f %6.3f %6.3f'%(s,h,ac,f1,au)); out.append({'group':g,'singer':s,'held_song':h,'acc':round(ac,4),'f1':round(f1,4),'auc':round(au,4)})
        print('%-10s %-16s %6s %6s %6.3f'%('','MEAN','','',np.mean([r[4] for r in rows])))
        out.append({'group':g,'singer':'','held_song':'MEAN','acc':'','f1':'','auc':round(np.mean([r[4] for r in rows]),4)}); print()
    pd.DataFrame(out).to_csv(a.output_dir+'/cross_song_loso.csv',index=False); print('Saved -> outputs/cross_song_loso.csv')

if __name__=='__main__': main()
