"""难度分层:用 CLEAN 模型(real vs fake,5折CV)对每个 fake 的预测概率分 easy/medium/hard,
分析不同难度下的歌曲构成与特征表现,揭示'什么样的AI翻唱最难检测'。"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict

CLEAN=['f0_std','f0_range_semitones','f0_jitter','vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score','hnr_mean','hnr_std','hnr_low_ratio','long_note_stability']
KEY=['hnr_low_ratio','hnr_mean','vri_score','f0_jitter','long_note_stability','vibrato_depth_mean']

def main():
    df=pd.read_csv('outputs/features_fixed.csv').reset_index(drop=True)
    y=df['label'].map({'real':0,'fake':1}).astype(int)
    X=df[CLEAN].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])
    prob=cross_val_predict(m,X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]
    df['fake_prob']=prob
    fk=df[df.label=='fake'].copy()
    # 难度: prob越低=越像真=越难检测
    def tier(p): return 'hard(像真)' if p<0.4 else ('medium' if p<0.6 else 'easy(易检)')
    fk['难度']=fk['fake_prob'].map(tier)
    real=df[df.label=='real']
    rmean={c:real[c].mean() for c in KEY}; rstd={c:real[c].std()+1e-9 for c in KEY}
    print('=== fake 难度分布 (CLEAN模型预测) ===')
    print(fk['难度'].value_counts().reindex(['hard(像真)','medium','easy(易检)']).to_string())
    print('\n=== 各难度的歌手/歌曲构成 ===')
    for t in ['hard(像真)','medium','easy(易检)']:
        sub=fk[fk['难度']==t]
        comp=sub.groupby(['singer_id','song_id']).size().sort_values(ascending=False)
        print('[%s] n=%d  top: %s' % (t,len(sub),', '.join('%s/%s:%d'%(s,so,n) for (s,so),n in comp.head(3).items())))
    print('\n=== 各难度下关键特征均值 (括号=相对真人均值的z) ===')
    print('%-22s %10s %10s %10s' % ('feature','hard','medium','easy'))
    for c in KEY:
        vals=[]
        for t in ['hard(像真)','medium','easy(易检)']:
            sub=fk[fk['难度']==t]
            mv=sub[c].mean(); z=(mv-rmean[c])/rstd[c]
            vals.append('%.3f(z%+.1f)'%(mv,z))
        print('%-22s %12s %12s %12s' % (c,vals[0],vals[1],vals[2]))
    fk[['file_path','singer_id','song_id','fake_prob','难度']+KEY].to_csv('outputs/difficulty_fakes.csv',index=False)
    print('\nSaved -> outputs/difficulty_fakes.csv')

if __name__=='__main__': main()
