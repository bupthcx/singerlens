"""E2 step6: 对 E1 用过的 bonafide 子集做 NSF-HiFiGAN copy-synthesis -> 降回16k -> bonafide_vocoded wav。
处理链与 So-VITS-SVC spoof 一致(content->NSF-HiFiGAN@44.1k->downsample16k),避免引入采样率混杂。"""
import sys, argparse, os
import numpy as np, torch, librosa, soundfile as sf, pandas as pd
sys.path.insert(0,'/home/admin2/xf/so-vits-svc2')
from diffusion.vocoder import Vocoder

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--e1-features',required=True); ap.add_argument('--meta',required=True)
    ap.add_argument('--base',required=True); ap.add_argument('--ckpt',required=True); ap.add_argument('--outdir',required=True)
    a=ap.parse_args()
    os.makedirs(a.outdir,exist_ok=True)
    feats=pd.read_csv(a.e1_features); meta=pd.read_csv(a.meta)
    split_of=dict(zip(meta.filename,meta.split))
    bona=feats[feats.label=='real']['filename'].tolist()
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    voc=Vocoder('nsf-hifigan',a.ckpt,device=dev); sr_v=voc.vocoder_sample_rate; hop=voc.vocoder_hop_size
    print('bonafide待处理:%d  vocoder sr=%d'%(len(bona),sr_v),flush=True)
    done=0; err=0
    for i,fn in enumerate(bona):
        sp=split_of.get(fn); 
        fp=os.path.join(a.base,sp+'_set',fn+'.flac')
        out=os.path.join(a.outdir,fn+'_voc.wav')
        try:
            y,sr=librosa.load(fp,sr=None,mono=True)
            yt=torch.tensor(y)[None].float().to(dev)
            mel=voc.extract(yt,sr); nf=mel.size(1)
            y_v=librosa.resample(y,orig_sr=sr,target_sr=sr_v) if sr!=sr_v else y
            f0,_,_=librosa.pyin(y_v,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'),sr=sr_v,hop_length=hop)
            f0=np.nan_to_num(f0,nan=0.0); f0=np.pad(f0,(0,max(0,nf-len(f0))))[:nf]
            with torch.no_grad():
                audio=voc.infer(mel,torch.tensor(f0)[None,:,None].float().to(dev)).squeeze().cpu().numpy()
            audio16=librosa.resample(audio,orig_sr=sr_v,target_sr=16000)   # 降回16k对齐
            sf.write(out,audio16,16000); done+=1
        except Exception as e:
            err+=1
        if (i+1)%300==0: print('  %d/%d done=%d err=%d'%(i+1,len(bona),done,err),flush=True)
    print('DONE generated=%d err=%d -> %s'%(done,err,a.outdir))

if __name__=='__main__': main()
