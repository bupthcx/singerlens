"""WavLM frozen embedding 提取:ModelScope WavLM -> 均值池化 -> 768维 embedding。
输入CSV(filepath,label,vocoder_group,attack),输出 embedding csv 供 LOVO(RF on embedding)。"""
import argparse, os, sys, numpy as np, pandas as pd, torch, librosa
def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--csv',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--model',default='AI-ModelScope/wav2vec2-base-960h'); a=ap.parse_args()
    from modelscope import snapshot_download
    from transformers import AutoModel
    mp=snapshot_download(a.model); dev='cuda' if torch.cuda.is_available() else 'cpu'
    m=AutoModel.from_pretrained(mp).to(dev).eval()
    df=pd.read_csv(a.csv); embs=[]; keep=[]
    with torch.no_grad():
        for i,r in df.iterrows():
            try:
                x,_=librosa.load(r['filepath'],sr=16000,mono=True)
                if len(x)>16000*8: x=x[:16000*8]
                t=torch.tensor(x)[None].float().to(dev)
                h=m(t).last_hidden_state.mean(1).squeeze(0).cpu().numpy()
                embs.append(h); keep.append(i)
            except Exception: pass
            if (i+1)%500==0: print('  %d/%d'%(i+1,len(df)),flush=True)
    E=np.stack(embs); sub=df.loc[keep].reset_index(drop=True)
    out=pd.concat([sub[['label','vocoder_group','attack']].reset_index(drop=True),
                   pd.DataFrame(E,columns=[f'w{i}' for i in range(E.shape[1])])],axis=1)
    out.to_csv(a.out,index=False); print('DONE rows=%d dim=%d -> %s'%(len(out),E.shape[1],a.out))
if __name__=='__main__': main()
