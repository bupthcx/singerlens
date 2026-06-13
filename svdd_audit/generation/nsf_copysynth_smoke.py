"""NSF-HiFiGAN copy-synthesis smoke test:wav -> mel+F0 -> NSF-HiFiGAN -> 重合成wav。
为 E2 声码器对照(bonafide_vocoded)验证管线。"""
import sys, argparse, numpy as np, torch, librosa, soundfile as sf
sys.path.insert(0,'/home/admin2/xf/so-vits-svc2')
from diffusion.vocoder import Vocoder

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--ckpt',required=True); ap.add_argument('--wav',required=True); ap.add_argument('--out',required=True)
    a=ap.parse_args()
    dev='cuda' if torch.cuda.is_available() else 'cpu'
    voc=Vocoder('nsf-hifigan', a.ckpt, device=dev)
    sr_v=voc.vocoder_sample_rate; hop=voc.vocoder_hop_size
    print('vocoder sr=%d hop=%d mel_bins=%d'%(sr_v,hop,voc.dimension))
    y,sr=librosa.load(a.wav,sr=None,mono=True)
    yt=torch.tensor(y)[None].float().to(dev)
    mel=voc.extract(yt,sr)                 # B,n_frames,bins
    nf=mel.size(1)
    # F0 在 vocoder 采样率/hop 上提取,对齐到 mel 帧
    y_v=librosa.resample(y,orig_sr=sr,target_sr=sr_v) if sr!=sr_v else y
    f0,vf,_=librosa.pyin(y_v,fmin=librosa.note_to_hz('C2'),fmax=librosa.note_to_hz('C7'),sr=sr_v,hop_length=hop)
    f0=np.nan_to_num(f0,nan=0.0)
    if len(f0)<nf: f0=np.pad(f0,(0,nf-len(f0)))
    else: f0=f0[:nf]
    f0_t=torch.tensor(f0)[None,:,None].float().to(dev)
    with torch.no_grad():
        audio=voc.infer(mel,f0_t).squeeze().cpu().numpy()
    sf.write(a.out,audio,sr_v)
    print('OK 输出 %s  时长%.2fs  采样率%d'%(a.out,len(audio)/sr_v,sr_v))

if __name__=='__main__': main()
