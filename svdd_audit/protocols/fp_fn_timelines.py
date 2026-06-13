"""为 FP/FN 案例生成片段内逐窗 AI 概率时间轴 + 汇总。展示模型在片段哪段最'犹豫/犯错'。"""
from __future__ import annotations
import os, sys, json
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf, matplotlib
matplotlib.use('Agg'); import matplotlib.pyplot as plt
sys.path.insert(0,'src')
import joblib
from singerlens.features import extract_clip_features
B=joblib.load('outputs/detector.joblib'); MODEL=B['model']; FEATS=B['features']

def fullpath(fname, grp):
    d={'real':'data/demo_data/real','fake':'data/demo_data/fake','real_vocoded':'data/demo_data/real_vocoded'}[grp]
    return os.path.join(d,fname)

def pred(feat): 
    x=pd.DataFrame([{f:feat.get(f,0.0) for f in FEATS}]).replace([np.inf,-np.inf],np.nan).fillna(0.0)
    return float(MODEL.predict_proba(x)[0,1])

def timeline(path, win=3.0, hop=1.0):
    y,sr=librosa.load(path,sr=16000,mono=True); dur=len(y)/sr
    import tempfile; tmp=Path(tempfile.mkdtemp()); ts=[];ps=[]
    for k,st in enumerate(np.arange(0,max(dur-win,0.01),hop)):
        seg=y[int(st*sr):int((st+win)*sr)]
        if len(seg)<sr*1.5: continue
        p=tmp/f'w{k}.wav'; sf.write(p,seg,sr)
        try: ps.append(pred(extract_clip_features(p))); ts.append(st+win/2)
        except Exception: pass
    return ts,ps

def main():
    fp=pd.read_csv('outputs/attribution_fp_fn.csv')
    outdir=Path('outputs/fp_fn_timelines'); outdir.mkdir(parents=True,exist_ok=True)
    rows=[]
    fig,axes=plt.subplots(len(fp),1,figsize=(8,1.5*len(fp))); 
    for i,(_,r) in enumerate(fp.iterrows()):
        path=fullpath(r['file'],r['true_group'])
        if not os.path.exists(path): print('缺失',path); continue
        ts,ps=timeline(path)
        rows.append({'setup':r['setup'],'kind':r['kind'],'file':r['file'],'true_group':r['true_group'],
                     'clip_pred':r['pred_prob'],'win_mean':round(np.mean(ps),3) if ps else None,
                     'win_min':round(min(ps),3) if ps else None,'win_max':round(max(ps),3) if ps else None})
        ax=axes[i]; ax.plot(ts,ps,'-o',ms=3,color='#B3261E'); ax.axhline(0.5,ls='--',c='gray',lw=0.8)
        ax.set_ylim(0,1); ax.set_ylabel('p(fake)',fontsize=7)
        ax.set_title('%s/%s  %s  true=%s clip_p=%.2f'%(r['setup'],r['kind'],r['file'],r['true_group'],r['pred_prob']),fontsize=7)
        ax.tick_params(labelsize=6)
    plt.tight_layout(); plt.savefig(outdir/'all_cases_timeline.png',dpi=130); plt.close()
    pd.DataFrame(rows).to_csv('outputs/fp_fn_timeline_summary.csv',index=False)
    print(pd.DataFrame(rows).to_string(index=False))
    print('\nSaved -> outputs/fp_fn_timelines/all_cases_timeline.png + fp_fn_timeline_summary.csv')

if __name__=='__main__': main()
