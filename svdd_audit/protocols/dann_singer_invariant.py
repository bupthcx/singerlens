"""对抗式去歌手身份(DANN):编码器+真伪头+歌手判别头(经梯度反转层GRL)。
让表示对歌手不变、对真伪可分,提升跨歌手(Leave-One-Singer-Out)泛化。
对比:普通MLP(lambda=0) vs DANN(对抗)。多种子平均降方差。"""
from __future__ import annotations
import numpy as np, pandas as pd
import torch, torch.nn as nn
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score

CLEAN=['f0_std','f0_range_semitones','f0_jitter','vibrato_rate_mean','vibrato_rate_std','vibrato_depth_mean','vibrato_depth_std','periodicity_score','micro_variation','vri_score','hnr_mean','hnr_std','hnr_low_ratio','long_note_stability']
dev=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class GRL(torch.autograd.Function):
    @staticmethod
    def forward(ctx,x,lambd): ctx.lambd=lambd; return x.view_as(x)
    @staticmethod
    def backward(ctx,g): return -ctx.lambd*g, None

class DANN(nn.Module):
    def __init__(self,d_in,n_dom,hid=32,rep=16):
        super().__init__()
        self.enc=nn.Sequential(nn.Linear(d_in,hid),nn.ReLU(),nn.Dropout(0.3),nn.Linear(hid,rep),nn.ReLU())
        self.label=nn.Linear(rep,2)
        self.domain=nn.Sequential(nn.Linear(rep,hid),nn.ReLU(),nn.Linear(hid,n_dom))
    def forward(self,x,lambd=0.0):
        r=self.enc(x); return self.label(r), self.domain(GRL.apply(r,lambd))

def train_eval(Xtr,ytr,dtr,Xte,yte,n_dom,adv,seed,epochs=200):
    torch.manual_seed(seed); np.random.seed(seed)
    m=DANN(Xtr.shape[1],n_dom).to(dev)
    opt=torch.optim.Adam(m.parameters(),lr=1e-3,weight_decay=1e-4)
    ce=nn.CrossEntropyLoss()
    Xtr_t=torch.tensor(Xtr,dtype=torch.float32,device=dev); ytr_t=torch.tensor(ytr,device=dev); dtr_t=torch.tensor(dtr,device=dev)
    Xte_t=torch.tensor(Xte,dtype=torch.float32,device=dev)
    # class weights for label imbalance
    cw=torch.tensor([len(ytr)/(2*(ytr==0).sum()), len(ytr)/(2*(ytr==1).sum())],dtype=torch.float32,device=dev)
    ce_y=nn.CrossEntropyLoss(weight=cw)
    m.train()
    for ep in range(epochs):
        p=ep/epochs; lambd=(2.0/(1.0+np.exp(-10*p))-1.0) if adv else 0.0
        opt.zero_grad()
        ly,ld=m(Xtr_t,lambd)
        loss=ce_y(ly,ytr_t)+(ce(ld,dtr_t) if adv else 0.0)
        loss.backward(); opt.step()
    m.eval()
    with torch.no_grad():
        prob=torch.softmax(m(Xte_t,0.0)[0],1)[:,1].cpu().numpy()
    pred=(prob>=0.5).astype(int)
    return roc_auc_score(yte,prob),f1_score(yte,pred,zero_division=0),accuracy_score(yte,pred)

def main():
    df=pd.read_csv('outputs/features_fixed.csv')
    y=df['label'].map({'real':0,'fake':1}).astype(int).values
    singers=sorted(df.singer_id.unique()); sidx={s:i for i,s in enumerate(singers)}
    res={'plain':[], 'dann':[]}
    for held in singers:
        tr=df.singer_id!=held; te=df.singer_id==held
        sc=StandardScaler().fit(df.loc[tr,CLEAN].replace([np.inf,-np.inf],np.nan).fillna(0))
        Xtr=sc.transform(df.loc[tr,CLEAN].replace([np.inf,-np.inf],np.nan).fillna(0))
        Xte=sc.transform(df.loc[te,CLEAN].replace([np.inf,-np.inf],np.nan).fillna(0))
        ytr=y[tr.values]; yte=y[te.values]
        tr_singers=sorted(df.loc[tr,'singer_id'].unique()); dmap={s:i for i,s in enumerate(tr_singers)}
        dtr=df.loc[tr,'singer_id'].map(dmap).values
        for name,adv in [('plain',False),('dann',True)]:
            aucs=[train_eval(Xtr,ytr,dtr,Xte,yte,len(tr_singers),adv,seed)[0] for seed in range(5)]
            res[name].append((held,np.mean(aucs)))
    print('%-10s %8s %8s' % ('test_on','plain','DANN'))
    for i,held in enumerate(singers):
        print('%-10s %8.3f %8.3f' % (held,res['plain'][i][1],res['dann'][i][1]))
    pm=np.mean([a for _,a in res['plain']]); dm=np.mean([a for _,a in res['dann']])
    print('%-10s %8.3f %8.3f  (ΔAUC %+.3f)' % ('MEAN',pm,dm,dm-pm))
    pd.DataFrame({'test_on':singers,'plain_auc':[round(a,3) for _,a in res['plain']],'dann_auc':[round(a,3) for _,a in res['dann']]}).to_csv('outputs/dann_singer_invariant.csv',index=False)
    print('参考: RF+raw LOSO=0.679, RF+per-singer-z LOSO=0.745')
    print('Saved -> outputs/dann_singer_invariant.csv')

if __name__=='__main__': main()
