"""在 CtrSVDD 上复用 SingerLens 特征(FULL/CLEAN/HNR/VRI)。
解析协议 -> 对每条 flac 提特征 -> 输出 CSV(含 source/singer/system/vocoder/label + 全部特征)。
缺失文件(授权未下的bonafide)自动跳过。协议格式: 源 歌手ID 文件名 攻击类型 攻击ID 标签
"""
from __future__ import annotations
import argparse, os, sys, csv
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from singerlens.features import extract_clip_features

# 系统->声码器映射(已据 CtrSVDD 论文 arXiv:2406.02438 §2.2 核对)
# 确认: A01/A05/A12=HiFi-GAN; A03=DDSP; A04/A06=source-filter HiFi-GAN;
#       A02/A07-A11/A13=Soft-VITS-SVC/VISinger 端到端VITS(集成GAN解码器,无独立声码器);
#       A14=ACE-Studio 商用专有. 存疑(待生成配置核实): A09可能用sf-HiFiGAN; A07-A11/A13的SSL先验各异.
SYS2VOC={'A01':'hifigan','A02':'hifigan','A03':'ddsp','A04':'nsf-hifigan','A05':'hifigan',
         'A06':'nsf-hifigan','A07':'nsf-hifigan','A08':'nsf-hifigan','A09':'nsf-hifigan','A10':'nsf-hifigan',
         'A11':'nsf-hifigan','A12':'hifigan','A13':'nsf-hifigan','A14':'proprietary','-':'bonafide'}

def parse_protocol(path):
    rows=[]
    for line in open(path):
        p=line.split()
        if len(p)<6: continue
        src,singer,fname,atk,atkid,label=p[0],p[1],p[2],p[3],p[4],p[5]
        rows.append(dict(source=src,singer_id=singer,filename=fname,attack=atkid if atkid!='-' else atk,label=label))
    return rows

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--protocol',required=True)
    ap.add_argument('--audio-dir',required=True,help='含 *.flac 的目录(如 train_set)')
    ap.add_argument('--output',required=True)
    ap.add_argument('--limit',type=int,default=0,help='调试用,只处理前N条')
    args=ap.parse_args()
    rows=parse_protocol(args.protocol)
    if args.limit: rows=rows[:args.limit]
    out=[]; miss=0; err=0
    for i,r in enumerate(rows):
        fp=os.path.join(args.audio_dir, r['filename']+'.flac')
        if not os.path.exists(fp): miss+=1; continue
        try:
            feats=extract_clip_features(fp)
        except Exception as e:
            err+=1; continue
        atk=r['attack']
        rec={'filename':r['filename'],'source':r['source'],'singer_id':r['singer_id'],
             'system':atk,'attack_type':SYS2TYPE.get(atk,'?'),'vocoder':SYS2VOC.get(atk,'?'),
             'label':'fake' if r['label']=='deepfake' else 'real', **feats}
        out.append(rec)
        if (i+1)%500==0: print('  processed %d/%d (miss=%d err=%d)'%(i+1,len(rows),miss,err),flush=True)
    if out:
        import pandas as pd
        pd.DataFrame(out).to_csv(args.output,index=False)
    print('DONE rows=%d miss=%d err=%d -> %s'%(len(out),miss,err,args.output))

if __name__=='__main__': main()
