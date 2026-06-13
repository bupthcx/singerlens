"""AASIST(rawnet) 训练/评估驱动,用显式 train/test CSV(filepath,label[1=bonafide,0=spoof],vocoder_group)。
复用官方 SVDDModel + BinaryFocalLoss。输出 EER/AUC/F1 + score。"""
import argparse, os, sys, numpy as np, pandas as pd, librosa, torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
sys.path.insert(0,os.path.dirname(os.path.abspath(__file__)))
from models.model import SVDDModel
from utils import compute_eer, set_seed
from sklearn.metrics import roc_auc_score, f1_score

def pad_random(x,max_len=64000):
    if x.shape[0]>max_len:
        s=np.random.randint(x.shape[0]-max_len); return x[s:s+max_len]
    return pad_random(np.tile(x,int(max_len/x.shape[0])+1),max_len)

class DS(Dataset):
    def __init__(self,csv): self.df=pd.read_csv(csv).reset_index(drop=True)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i]
        x,_=librosa.load(r['filepath'],sr=16000,mono=True)
        x=librosa.util.normalize(pad_random(x))
        return x.astype('float32'), int(r['label']), i

class FocalLoss(nn.Module):
    def __init__(self,g=2.0,a=0.25): super().__init__(); self.g=g; self.a=a
    def forward(self,logits,t):
        bce=F.binary_cross_entropy_with_logits(logits,t,reduction='none'); pt=torch.exp(-bce)
        return (self.a*(1-pt)**self.g*bce).mean()

def run_eval(model,loader,dev):
    model.eval(); preds=[]; labs=[]
    with torch.no_grad():
        for x,label,_ in loader:
            _,p=model(x.to(dev)); preds.append(p.cpu().numpy().ravel()); labs.append(label.numpy())
    preds=np.concatenate(preds); labs=np.concatenate(labs)  # label1=bonafide
    eer,_=compute_eer(preds[labs==1],preds[labs==0])        # target=bonafide
    prob_fake=1/(1+np.exp(preds))                            # sigmoid(-logit)=P(spoof)
    yf=(labs==0).astype(int)                                 # fake=spoof=1
    auc=roc_auc_score(yf,prob_fake); f1=f1_score(yf,(prob_fake>=0.5).astype(int),zero_division=0)
    return round(eer*100,2),round(auc,3),round(f1,3),preds,labs

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--train-csv',required=True); ap.add_argument('--test-csv',required=True)
    ap.add_argument('--epochs',type=int,default=15); ap.add_argument('--bs',type=int,default=32)
    ap.add_argument('--gpu',type=int,default=0); ap.add_argument('--out',required=True); ap.add_argument('--tag',default='run'); ap.add_argument('--extra-tests',default='')
    a=ap.parse_args(); set_seed(42)
    dev=torch.device(f'cuda:{a.gpu}' if torch.cuda.is_available() else 'cpu')
    tr=DataLoader(DS(a.train_csv),batch_size=a.bs,shuffle=True,num_workers=6,drop_last=True)
    te=DataLoader(DS(a.test_csv),batch_size=a.bs,shuffle=False,num_workers=6)
    model=SVDDModel(dev,frontend='rawnet').to(dev)
    opt=optim.AdamW(model.parameters(),lr=1e-3,weight_decay=1e-9,betas=(0.9,0.999))
    sch=optim.lr_scheduler.CosineAnnealingLR(opt,T_max=10,eta_min=1e-6); crit=FocalLoss()
    for ep in range(a.epochs):
        model.train(); tot=0
        for x,label,_ in tr:
            x=x.to(dev); sl=(label.float()*0.9+0.05).to(dev).unsqueeze(1)
            _,p=model(x); loss=crit(p,sl); loss.backward(); opt.step(); opt.zero_grad(); tot+=loss.item()
        sch.step()
        if (ep+1)%5==0 or ep==a.epochs-1:
            eer,auc,f1,_,_=run_eval(model,te,dev)
            print('[%s] ep%d loss=%.3f  test EER=%.2f AUC=%.3f F1=%.3f'%(a.tag,ep+1,tot/len(tr),eer,auc,f1),flush=True)
    eer,auc,f1,preds,labs=run_eval(model,te,dev)
    rows=[{'tag':a.tag,'EER':eer,'AUC':auc,'F1':f1,'n_test':len(labs)}]
    print('FINAL [%s] EER=%.2f AUC=%.3f F1=%.3f'%(a.tag,eer,auc,f1),flush=True)
    if a.extra_tests:
        for spec in a.extra_tests.split(','):
            nm,pth=spec.split('='); el=DataLoader(DS(pth),batch_size=a.bs,shuffle=False,num_workers=6)
            ee,ea,ef,_,_=run_eval(model,el,dev); rows.append({'tag':a.tag+'__'+nm,'EER':ee,'AUC':ea,'F1':ef,'n_test':len(DS(pth))})
            print('EXTRA [%s -> %s] EER=%.2f AUC=%.3f F1=%.3f'%(a.tag,nm,ee,ea,ef),flush=True)
    os.makedirs(os.path.dirname(a.out),exist_ok=True)
    pd.DataFrame(rows).to_csv(a.out,index=False)

if __name__=='__main__': main()
