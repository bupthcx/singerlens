# WavLM frozen embedding + RF: within-CV / cross-dataset / CtrSVDD-LOVO
import sys, re, numpy as np, pandas as pd
sys.path.insert(0,'/home/admin2/xf/CtrSVDD_Utils'); from eer import compute_eer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline; from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import roc_auc_score
ctr=pd.read_csv('/home/admin2/xf/wavlm/ctr_emb.csv'); sl=pd.read_csv('/home/admin2/xf/wavlm/sl_emb.csv'); wild=pd.read_csv('/home/admin2/xf/wavlm/wild_emb.csv')
W=[c for c in ctr.columns if re.match(r'w\d+$',c)]
print('WavLM dim=%d | ctr=%d sl=%d wild=%d'%(len(W),len(ctr),len(sl),len(wild)))
def met(y,sc): e,_=compute_eer(y.astype(float),sc); return round(e*100,2),round(roc_auc_score(y,sc),3)
def Xy(df): return df[W].replace([np.inf,-np.inf],np.nan).fillna(0).values,(df['label']=='fake').astype(int).values
def within(df):
    X,y=Xy(df); sc=cross_val_predict(Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))]),X,y,cv=StratifiedKFold(5,shuffle=True,random_state=42),method='predict_proba')[:,1]; return met(y,sc)
def cross(tr,te):
    Xtr,ytr=Xy(tr); Xte,yte=Xy(te); m=Pipeline([('s',StandardScaler()),('c',RandomForestClassifier(n_estimators=300,random_state=42,class_weight='balanced'))]).fit(Xtr,ytr); return met(yte,m.predict_proba(Xte)[:,1])
rows=[]
def add(p,e_a): rows.append({'protocol':p,'EER':e_a[0],'AUC':e_a[1]})
add('CtrSVDD within',within(ctr)); add('SingerLens within',within(sl)); add('WildSVDD within',within(wild))
add('CtrSVDD->SingerLens',cross(ctr,sl)); add('SingerLens->CtrSVDD',cross(sl,ctr))
add('CtrSVDD->WildSVDD',cross(ctr,wild)); add('SingerLens->WildSVDD',cross(sl,wild))
# LOVO on CtrSVDD
bona=ctr[ctr.label=='real']; spoof=ctr[ctr.label=='fake']
rng=np.random.RandomState(42); idx=rng.permutation(len(bona)); h=len(bona)//2; b_tr=bona.iloc[idx[:h]]; b_te=bona.iloc[idx[h:]]
for held in ['hifigan','nsf-hifigan','ddsp']:
    tr=pd.concat([b_tr,spoof[spoof.vocoder_group!=held]]); te=pd.concat([b_te,spoof[spoof.vocoder_group==held]])
    add('LOVO-%s'%held,cross(tr,te))
r=pd.DataFrame(rows); r.to_csv('outputs/wavlm_results.csv',index=False)
print(); print(r.to_string(index=False)); print('Saved -> outputs/wavlm_results.csv')
