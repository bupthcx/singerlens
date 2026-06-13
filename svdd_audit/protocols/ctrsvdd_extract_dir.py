"""对一个目录的 wav 批量提 SingerLens 特征(用于 bonafide_vocoded)。label/vocoder_group 可配。"""
from __future__ import annotations
import sys, argparse, os, glob
from pathlib import Path
import pandas as pd, soundfile as sf
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from singerlens.features import extract_clip_features

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--dir',required=True); ap.add_argument('--out',required=True)
    ap.add_argument('--label',default='real'); ap.add_argument('--vocoder-group',default='bonafide_vocoded')
    a=ap.parse_args()
    fs=sorted(glob.glob(os.path.join(a.dir,'*.wav')))
    out=[]; err=0
    for i,fp in enumerate(fs):
        try:
            info=sf.info(fp); feats=extract_clip_features(fp)
        except Exception: err+=1; continue
        fn=os.path.basename(fp).replace('_voc.wav','')
        out.append({'filename':fn,'attack':'-','vocoder_group':a.vocoder_group,'label':a.label,
                    'native_sr':info.samplerate,'native_dur':round(info.frames/info.samplerate,3), **feats})
        if (i+1)%300==0: print('  %d/%d err=%d'%(i+1,len(fs),err),flush=True)
    pd.DataFrame(out).to_csv(a.out,index=False)
    print('DONE rows=%d err=%d -> %s'%(len(out),err,a.out))

if __name__=='__main__': main()
