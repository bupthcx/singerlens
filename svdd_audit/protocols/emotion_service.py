"""常驻情感推理服务(qwen3-asr环境,端口7861)。预加载 Whisper(medium)+emotion2vec+中文文本情感,
GET /emotion?path=<wav> 返回 JSON: 演唱情感/歌词/歌词情感/valence/一致性。供 Gradio Demo 调用。"""
import json, re, sys, urllib.parse
from http.server import BaseHTTPRequestHandler, HTTPServer
import numpy as np

print('[emotion_service] loading models...', flush=True)
import whisper
_wh = whisper.load_model('medium')
from funasr import AutoModel
_ser = AutoModel(model='iic/emotion2vec_plus_large', disable_update=True)
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
_txt = pipeline(Tasks.text_classification, model='iic/nlp_structbert_emotion-classification_chinese-base')
print('[emotion_service] models loaded.', flush=True)

CREDIT=re.compile(r'(作[词詞曲]|编曲|編曲|制作|製作|混音|监制|監制|配唱|和[声聲]|录音|錄音|出品|词曲|詞曲)')
TEXT_POS={'高兴','喜好'}; TEXT_NEG={'悲伤','愤怒','恐惧','厌恶'}
SER_POS={'happy'}; SER_NEG={'sad','angry','fearful','disgusted'}
def val(labels,scores,POS,NEG):
    p=sum(s for l,s in zip(labels,scores) if l.split('/')[-1] in POS)
    n=sum(s for l,s in zip(labels,scores) if l.split('/')[-1] in NEG)
    return float(p-n)

def analyze(path):
    # 演唱情感
    r=_ser.generate(path, granularity='utterance', extract_embedding=False)
    sl,ss=r[0]['labels'],r[0]['scores']
    sing_top=sorted(zip([l.split('/')[-1] for l in sl],ss),key=lambda x:-x[1])[:3]
    sing_val=val(sl,ss,SER_POS,SER_NEG)
    # 歌词
    tr=_wh.transcribe(path, language='zh', verbose=False)
    lines=[re.sub(r'\s+','',s['text']) for s in tr.get('segments',[])]
    lines=[t for t in lines if t and not CREDIT.search(t) and len(set(t))>1]
    seen=set(); uniq=[x for x in lines if not (x in seen or seen.add(x))]
    lyrics=''.join(uniq)
    if uniq:
        vals=[val(_txt(input=t)['labels'],_txt(input=t)['scores'],TEXT_POS,TEXT_NEG) for t in uniq]
        lyr_val=float(np.mean(vals))
        lr=_txt(input=lyrics[:120]); lyr_top=list(zip(lr['labels'][:3],[round(s,2) for s in lr['scores'][:3]]))
    else:
        lyr_val=0.0; lyr_top=[]
    cons=round(1-abs(lyr_val-sing_val)/2,3)
    return {'singing_top':[(l,round(s,2)) for l,s in sing_top],'singing_valence':round(sing_val,2),
            'lyrics':lyrics[:200],'lyrics_top':lyr_top,'lyrics_valence':round(lyr_val,2),
            'consistency':cons,'n_lines':len(uniq)}

class H(BaseHTTPRequestHandler):
    def log_message(self,*a): pass
    def do_GET(self):
        q=urllib.parse.urlparse(self.path)
        if q.path!='/emotion': self.send_response(404); self.end_headers(); return
        path=urllib.parse.parse_qs(q.query).get('path',[''])[0]
        try: out=analyze(path)
        except Exception as e: out={'error':str(e)}
        body=json.dumps(out,ensure_ascii=False).encode('utf-8')
        self.send_response(200); self.send_header('Content-Type','application/json'); self.end_headers(); self.wfile.write(body)

if __name__=='__main__':
    HTTPServer(('127.0.0.1',7861),H).serve_forever()
