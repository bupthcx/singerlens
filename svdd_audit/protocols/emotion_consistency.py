"""情感一致性分析(方案模块四,案例解释用,非主检测特征)。
歌词情感(ModelScope中文情感分类) vs 演唱情感(emotion2vec SER) -> 一致性得分。
对每首歌分别计算真唱/AI翻唱的歌词-演唱情感一致性,检验AI是否保持情感契合。"""
from __future__ import annotations
import json, glob, os, re, sys
import numpy as np

CREDIT = re.compile(r'(作[词詞曲]|编曲|編曲|制作|製作|混音|监制|監制|母带|母帶|配唱|和声|和聲|录音|錄音|出品|策划|企[划劃]|词曲|詞曲|演唱:|作曲:)')
def clean_lyrics(segs):
    seen=set(); out=[]
    for s in segs:
        t=re.sub(r'\s+','',s['text']).strip()
        if not t or CREDIT.search(t): continue
        if len(set(t))<=1: continue              # 单字重复幻觉
        if t in seen: continue                   # 整行去重
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

    def lyrics_val(lines):
        vals=[]
        for t in lines:
            r=txtclf(input=t); vals.append(valence(r['labels'],r['scores'],TEXT_POS,TEXT_NEG))
        return float(np.mean(vals)) if vals else 0.0
    def singing_val(clips):
        vals=[]
        for c in clips:
            r=ser.generate(c, granularity='utterance', extract_embedding=False)
            vals.append(valence(r[0]['labels'],r[0]['scores'],SER_POS,SER_NEG))
        return float(np.mean(vals)) if vals else 0.0, len(vals)

    rows=[]
    for lj in sorted(glob.glob('outputs/lyrics/*.json')):
        key=os.path.basename(lj).replace('.json','')      # singer_x_song
        t=key.split('_'); singer=f'{t[0]}_{t[1]}'; song='_'.join(t[2:])
        lines=clean_lyrics(json.load(open(lj))['segments'])
        lv=lyrics_val(lines)
        reals=sorted(glob.glob(f'data/demo_data/real/{key}_*.wav'))[:12]
        fakes=sorted(glob.glob(f'data/demo_data/fake/{key}_*.wav'))[:12]
        rv,rn=singing_val(reals); fv,fn=singing_val(fakes)
        cons=lambda a,b: round(1-abs(a-b)/2,3)            # 一致性[0,1]
        rows.append((key,len(lines),round(lv,3),round(rv,3),round(fv,3),cons(lv,rv),cons(lv,fv)))
        print('%-26s lines=%3d lyr_val=%+.2f real_val=%+.2f fake_val=%+.2f | cons_real=%.2f cons_fake=%.2f'
              % (key,len(lines),lv,rv,fv,cons(lv,rv),cons(lv,fv)), flush=True)
    import csv
    with open('outputs/emotion_consistency.csv','w',newline='') as f:
        w=csv.writer(f); w.writerow(['song','lyric_lines','lyric_valence','real_sing_valence','fake_sing_valence','consistency_real','consistency_fake'])
        w.writerows(rows)
    rc=np.mean([r[5] for r in rows]); fc=np.mean([r[6] for r in rows])
    print('\n平均一致性: 真唱=%.3f  AI翻唱=%.3f' % (rc,fc))
    print('Saved -> outputs/emotion_consistency.csv')

if __name__=='__main__': main()
