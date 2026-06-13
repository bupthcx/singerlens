"""归因与错误分析：3组二分类(real-vs-fake / real-vs-real_vocoded / real_vocoded-vs-fake)
× 4特征集(FULL/CLEAN/HNR/VRI)的 AUC/F1/Acc；FP/FN案例导出；
每特征 Cohen's d + 方差归因(eta^2 by label vs by singer)。"""
from __future__ import annotations
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

HNR=['hnr_mean','hnr_std','hnr_low_ratio']
VRI=['vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score']
PITCHD=['f0_std','f0_range_semitones','f0_jitter']
CLEAN=PITCHD+VRI+HNR+['long_note_stability']
LOW=['rms_mean','rms_std','energy_dynamic','spectral_flatness_mean','spectral_flatness_std','duration']
MFCC=[f'mfcc_{i}_{s}' for i in range(1,14) for s in ('mean','std')]
FULL=LOW+MFCC+PITCHD+['f0_mean','f0_min','f0_max']+VRI+HNR+['long_note_stability']
SETS={'FULL':FULL,'CLEAN':CLEAN,'HNR':HNR,'VRI':VRI}

def load():
    ff=pd.read_csv('outputs/features_fixed.csv'); fv=pd.read_csv('outputs/features_vocoded.csv')
    real=ff[ff.source_type=='bilibili_separated'].copy(); real['group']='real'
    fake=ff[ff.source_type=='seed_vc'].copy(); fake['group']='fake'
    rvoc=fv[fv.source_type=='real_vocoded'].copy(); rvoc['group']='real_vocoded'
    return real,fake,rvoc

def model():
    return Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))])

def cvpred(df,cols,y):
    X=df[[c for c in cols if c in df.columns]].replace([np.inf,-np.inf],np.nan).fillna(0.0)
    skf=StratifiedKFold(5,shuffle=True,random_state=42)
    prob=cross_val_predict(model(),X,y,cv=skf,method='predict_proba')[:,1]
    pred=(prob>=0.5).astype(int)
    return prob,pred

def main():
    real,fake,rvoc=load()
    setups={'real_vs_fake':(pd.concat([real,fake]),'real','fake'),
            'real_vs_realvoc':(pd.concat([real,rvoc]),'real','real_vocoded'),
            'realvoc_vs_fake':(pd.concat([rvoc,fake]),'real_vocoded','fake')}
    # ---- 1-3: 分类指标 ----
    metrics=[]
    for sname,(df,neg,pos) in setups.items():
        y=(df['group']==pos).astype(int).values
        for fs,cols in SETS.items():
            prob,pred=cvpred(df,cols,y)
            metrics.append({'setup':sname,'pos_class':pos,'feature_set':fs,
                'auc':round(roc_auc_score(y,prob),3),'f1':round(f1_score(y,pred,zero_division=0),3),
                'acc':round(accuracy_score(y,pred),3)})
    md=pd.DataFrame(metrics); md.to_csv('outputs/attribution_classification.csv',index=False)
    print('===== 1-3. 三组二分类 × 四特征集 ====='); print(md.to_string(index=False))

    # ---- 4: FP/FN 案例 (用 CLEAN 模型) ----
    print('\n===== 4. FP/FN 案例 (CLEAN) =====')
    real_ref=pd.concat([real,fake]); 
    rmean={c:real[c].mean() for c in CLEAN}; rstd={c:real[c].std()+1e-9 for c in CLEAN}
    fpfn=[]
    for sname,(df,neg,pos) in setups.items():
        df=df.reset_index(drop=True); y=(df['group']==pos).astype(int).values
        prob,pred=cvpred(df,CLEAN,y)
        fp_idx=[i for i in range(len(y)) if pred[i]==1 and y[i]==0]
        fn_idx=[i for i in range(len(y)) if pred[i]==0 and y[i]==1]
        fp_idx=sorted(fp_idx,key=lambda i:-prob[i])[:2]      # 最自信的FP
        fn_idx=sorted(fn_idx,key=lambda i:prob[i])[:2]       # 最自信的FN
        for kind,idxs in [('FP',fp_idx),('FN',fn_idx)]:
            for i in idxs:
                row=df.iloc[i]
                z={c:(row[c]-rmean[c])/rstd[c] for c in CLEAN}
                topdev=sorted(z.items(),key=lambda kv:-abs(kv[1]))[:3]
                rec={'setup':sname,'kind':kind,'file':os.path.basename(row['file_path']),
                     'true_group':row['group'],'pred_prob':round(float(prob[i]),3),
                     'hnr_mean':round(row['hnr_mean'],2),'hnr_low_ratio':round(row['hnr_low_ratio'],3),
                     'vri_score':round(row['vri_score'],3),'f0_jitter':round(row['f0_jitter'],4),
                     'anomaly_dims':'; '.join('%s(z=%+.1f)'%(k,v) for k,v in topdev)}
                fpfn.append(rec)
                print('[%s/%s] %s true=%s prob=%.2f | %s' % (sname,kind,rec['file'],row['group'],prob[i],rec['anomaly_dims']))
    import os as _o
    pd.DataFrame(fpfn).to_csv('outputs/attribution_fp_fn.csv',index=False)

    # ---- 5: Cohen's d + 方差归因 ----
    print('\n===== 5. 特征归因 (real vs fake 数据) =====')
    rf=pd.concat([real,fake])
    def cohen(a,b): 
        p=np.sqrt((a.var()+b.var())/2)+1e-12; return (a.mean()-b.mean())/p
    def eta2(df,feat,grp):
        grand=df[feat].mean(); sst=((df[feat]-grand)**2).sum()
        ssb=sum(len(g)*(g.mean()-grand)**2 for _,g in df.groupby(grp)[feat])
        return ssb/sst if sst>0 else 0
    feats=CLEAN+['energy_dynamic','spectral_flatness_mean','mfcc_2_mean','f0_mean']
    stats=[]
    for c in feats:
        if c not in rf.columns: continue
        d=cohen(rf[rf.group=='real'][c],rf[rf.group=='fake'][c])
        el=eta2(rf,c,'label'); es=eta2(rf,c,'singer_id')
        kind='真伪特征' if el>es else '歌手身份特征'
        stats.append({'feature':c,'cohen_d':round(d,2),'eta2_label':round(el,3),'eta2_singer':round(es,3),'ratio_L/S':round(el/(es+1e-9),2),'判定':kind})
    sd=pd.DataFrame(stats).sort_values('ratio_L/S',ascending=False); sd.to_csv('outputs/attribution_feature_stats.csv',index=False)
    print(sd.to_string(index=False))
    print('\nSaved -> outputs/attribution_{classification,fp_fn,feature_stats}.csv')

import os
if __name__=='__main__': main()
