"""WildSVDD 情感一致性 AUC(Step2)。复用 emotion_consistency 的 valence 逻辑。
per-song: 歌词情感(structbert) vs 演唱情感(emotion2vec) -> consistency[0,1]。
score=-consistency(越低越像 fake),对 WildSVDD real-vs-fake 算 AUC。
对比: SingerLens 情感 AUC 基线 + AASIST/RF 跨域 Wild AUC -> 降幅。
"""
from __future__ import annotations
import json, glob, os, re, sys, tempfile
import numpy as np, pandas as pd, librosa, soundfile as sf
from sklearn.metrics import roc_auc_score

CREDIT = re.compile(r'(作[词詞曲]|编曲|編曲|制作|製作|混音|监制|監制|母带|母帶|配唱|和声|和聲|录音|錄音|出品|策划|企[划劃]|词曲|詞曲|演唱:|作曲:)')
def clean_lyrics(segs):
    seen=set(); out=[]
    for s in segs:
        t=re.sub(r'\s+','',s['text']).strip()
        if not t or CREDIT.search(t) or len(set(t))<=1 or t in seen: continue
        seen.add(t); out.append(t)
    return out

TEXT_POS={'高兴','喜好'}; TEXT_NEG={'悲伤','愤怒','恐惧','厌恶'}
SER_POS={'happy'}; SER_NEG={'sad','angry','fearful','disgusted'}
def valence(labels, scores, POS, NEG):
    p=sum(s for l,s in zip(labels,scores) if l.split('/')[-1] in POS)
    n=sum(s for l,s in zip(labels,scores) if l.split('/')[-1] in NEG)
    return float(p-n)


def main():
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks
    from funasr import AutoModel
    txtclf=pipeline(Tasks.text_classification, model='iic/nlp_structbert_emotion-classification_chinese-base')
    ser=AutoModel(model='iic/emotion2vec_plus_large', disable_update=True)
    tmp=tempfile.mkdtemp()

    def lyrics_val(lines):
        vals=[valence(txtclf(input=t)['labels'], txtclf(input=t)['scores'], TEXT_POS, TEXT_NEG) for t in lines]
        return float(np.mean(vals)) if vals else None
    def singing_val(voc):
        v,_=librosa.load(voc, sr=16000, mono=True); vals=[]
        for st in np.arange(0, max(len(v)/16000-10,0.01), 12)[:4]:   # 至多4个10s片段
            ch=v[int(st*16000):int((st+10)*16000)]
            if len(ch)<16000*5: continue
            p=os.path.join(tmp,'c.wav'); sf.write(p,ch,16000)
            r=ser.generate(p, granularity='utterance', extract_embedding=False)
            vals.append(valence(r[0]['labels'], r[0]['scores'], SER_POS, SER_NEG))
        return float(np.mean(vals)) if vals else None

    sub=pd.read_csv('outputs/wild_emotion_subsample.csv')
    rows=[]
    for _,r in sub.iterrows():
        idx=r['idx']; lj=f'outputs/wild_lyrics/{idx}.json'
        voc=glob.glob(f'/home/admin2/xf/wildsvdd/demucs_out/{idx}/htdemucs/*/vocals.wav')
        if not os.path.exists(lj) or not voc: continue
        lines=clean_lyrics(json.load(open(lj))['segments'])
        lv=lyrics_val(lines); sv=singing_val(voc[0])
        if lv is None or sv is None: continue
        cons=round(1-abs(lv-sv)/2, 3)
        rows.append(dict(idx=idx, label=r['label'], n_lines=len(lines),
                         lyric_val=round(lv,3), sing_val=round(sv,3), consistency=cons))
        print(f"{idx} {r['label']} lines={len(lines)} lyr={lv:+.2f} sing={sv:+.2f} cons={cons}", flush=True)
    df=pd.DataFrame(rows); df.to_csv('outputs/wild_emotion_consistency.csv', index=False)

    y=(df['label']=='fake').astype(int).values
    score=-df['consistency'].values            # 低一致性 -> fake
    wild_auc=roc_auc_score(y, score) if len(set(y))>1 else float('nan')
    rmean=df[df.label=='real']['consistency'].mean(); fmean=df[df.label=='fake']['consistency'].mean()

    # SingerLens 基线 AUC (现有 csv: 9 real + 9 fake consistency)
    sl=pd.read_csv('outputs/emotion_consistency.csv')
    sl_y=np.array([0]*len(sl)+[1]*len(sl)); sl_s=np.concatenate([-sl['consistency_real'].values, -sl['consistency_fake'].values])
    sl_auc=roc_auc_score(sl_y, sl_s)

    print('\n===== 情感一致性 AUC 对比 =====')
    print(f'WildSVDD (n={len(df)}, real {int((y==0).sum())}/fake {int((y==1).sum())}): consistency real={rmean:.3f} fake={fmean:.3f}  AUC={wild_auc:.3f}')
    print(f'SingerLens 基线 (9+9): AUC={sl_auc:.3f}')
    print(f'情感降幅 (SingerLens->WildSVDD): {sl_auc:.3f} -> {wild_auc:.3f}  (ratio {wild_auc/sl_auc:.2f})')
    print('--- 对照 trained 检测器跨域到 Wild (AUC) ---')
    print('  AASIST CtrSVDD->Wild 0.748 (within 0.987, ratio 0.76)')
    print('  RF-FULL CtrSVDD->Wild 0.520 ; SingerLens->Wild 0.447')
    print('  wav2vec2-ft cross-dataset 0.34-0.43')
    pd.DataFrame([dict(method='emotion_consistency', dataset='WildSVDD', AUC=round(wild_auc,3), n=len(df)),
                  dict(method='emotion_consistency', dataset='SingerLens(base)', AUC=round(sl_auc,3), n=2*len(sl))]
                 ).to_csv('outputs/wild_emotion_auc.csv', index=False)
    open('outputs/.wild_emotion.done','w').write('ok\n')
    print('Saved -> outputs/wild_emotion_consistency.csv + wild_emotion_auc.csv')


if __name__=='__main__': main()
