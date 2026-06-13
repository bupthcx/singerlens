"""扫描 real/ 与 fake/ 目录，质量过滤 fake，重建 metadata.csv（支持多歌手）。
文件名规范：singer_<X>_<song>_<NNN>.wav（real）/ ..._fake.wav（fake）。
fake 质量门槛：duration>=8s, voiced_ratio>=0.25, rms_mean>=0.01（剔除弱源产生的垃圾）。"""
from __future__ import annotations
import csv, glob, os
import librosa, numpy as np

REAL_DIR='data/demo_data/real'; FAKE_DIR='data/demo_data/fake'
OUT='data/demo_data/metadata.csv'
FIELDS=['file_path','label','singer_id','song_id','source_type']

def parse(name):
    base=os.path.basename(name).replace('.wav','')
    base=base[:-5] if base.endswith('_fake') else base
    t=base.split('_')                       # [singer, X, song, NNN]
    return f'{t[0]}_{t[1]}', t[2]           # singer_id, song_id

def fake_ok(p):
    y,sr=librosa.load(p,sr=16000,mono=True)
    dur=librosa.get_duration(y=y,sr=sr)
    rms=float(np.sqrt(np.mean(y**2)))
    f0,_,_=librosa.pyin(y,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'),sr=sr,frame_length=1024,hop_length=256)
    vr=float(np.mean(np.isfinite(f0))) if f0.size else 0.0
    return (dur>=8 and vr>=0.25 and rms>=0.01), dur, vr, rms

def main():
    rows=[]; 
    for p in sorted(glob.glob(f'{REAL_DIR}/*.wav')):
        sid,song=parse(p)
        rows.append({'file_path':p,'label':'real','singer_id':sid,'song_id':song,'source_type':'bilibili_separated'})
    dropped=[]
    for p in sorted(glob.glob(f'{FAKE_DIR}/*.wav')):
        ok,dur,vr,rms=fake_ok(p)
        if not ok:
            dropped.append((os.path.basename(p),dur,vr,rms)); continue
        sid,song=parse(p)
        rows.append({'file_path':p,'label':'fake','singer_id':sid,'song_id':song,'source_type':'seed_vc'})
    with open(OUT,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=FIELDS); w.writeheader(); w.writerows(rows)
    # 统计
    from collections import Counter
    cnt=Counter((r['singer_id'],r['label']) for r in rows)
    print('=== metadata 写入 %d 条 -> %s ===' % (len(rows),OUT))
    for k in sorted(cnt): print('  %-10s %-5s %d' % (k[0],k[1],cnt[k]))
    print('=== 丢弃的低质 fake: %d ===' % len(dropped))
    for name,dur,vr,rms in dropped: print('  DROP %s dur=%.1f voiced=%.2f rms=%.4f' % (name,dur,vr,rms))

if __name__=='__main__':
    main()
