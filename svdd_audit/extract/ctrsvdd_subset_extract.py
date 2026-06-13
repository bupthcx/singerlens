"""CtrSVDD 子集提特征: 读 ctrsvdd_meta.csv, 分层子采样(每attack N spoof + M bonafide,仅存在文件),
记录原生采样率/时长(QC) + 提取 SingerLens 特征 -> csv。"""
from __future__ import annotations
import sys, argparse, os
from pathlib import Path
import numpy as np, pandas as pd, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from singerlens.features import extract_clip_features

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--meta',required=True); ap.add_argument('--base',required=True,help='含 train_set/ dev_set/')
    ap.add_argument('--out',required=True)
    ap.add_argument('--n-per-attack',type=int,default=300); ap.add_argument('--n-bonafide',type=int,default=2400)
    ap.add_argument('--seed',type=int,default=42)
    a=ap.parse_args()
    m=pd.read_csv(a.meta)
    def path(r): return os.path.join(a.base, r['split']+'_set', r['filename']+'.flac')
    m['fp']=m.apply(path,axis=1); m['exists']=m['fp'].apply(os.path.exists)
    m=m[m.exists].copy()
    rng=np.random.RandomState(a.seed)
    parts=[]
    for at,g in m[m.label=='spoof'].groupby('attack'):
        parts.append(g.sample(min(a.n_per_attack,len(g)),random_state=a.seed))
    bf=m[m.label=='bonafide']
    parts.append(bf.sample(min(a.n_bonafide,len(bf)),random_state=a.seed))
    sub=pd.concat(parts).reset_index(drop=True)
    print('子集: %d (spoof=%d bonafide=%d)'%(len(sub),(sub.label=='spoof').sum(),(sub.label=='bonafide').sum()),flush=True)
    out=[]; err=0
    for i,r in sub.iterrows():
        try:
            info=sf.info(r['fp']); 
            feats=extract_clip_features(r['fp'])
        except Exception as e:
            err+=1; continue
        out.append({'filename':r['filename'],'source':r['source'],'singer_id':r['singer_id'],
                    'attack':r['attack'],'vocoder_group':r['vocoder_group'],
                    'label':'fake' if r['label']=='spoof' else 'real',
                    'native_sr':info.samplerate,'native_dur':round(info.frames/info.samplerate,3), **feats})
        if (i+1)%300==0: print('  %d/%d (err=%d)'%(i+1,len(sub),err),flush=True)
    pd.DataFrame(out).to_csv(a.out,index=False)
    print('DONE rows=%d err=%d -> %s'%(len(out),err,a.out))

if __name__=='__main__': main()
