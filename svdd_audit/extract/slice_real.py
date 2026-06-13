"""通用真实歌声切片：Demucs 人声 -> 10s 片段(hop 8s, 有声比例>=0.5, 16kHz)。
复刻 slice_dearfriend_real.py 的逻辑，参数化以支持多歌手多歌曲。"""
from __future__ import annotations
import argparse, os
import librosa, numpy as np, soundfile as sf

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--audio', required=True, help='demucs vocals.wav 路径')
    ap.add_argument('--singer', required=True, help='如 singer_b')
    ap.add_argument('--song', required=True, help='如 kaishidongle')
    ap.add_argument('--out', default='data/demo_data/real')
    ap.add_argument('--segment-len', type=float, default=10.0)
    ap.add_argument('--hop', type=float, default=8.0)
    ap.add_argument('--min-voiced', type=float, default=0.5)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    y, sr = librosa.load(args.audio, sr=16000)
    segments = []
    for start in np.arange(0, len(y)/sr - args.segment_len, args.hop):
        s = int(start*sr); e = int((start+args.segment_len)*sr)
        chunk = y[s:e]
        _, voiced_flag, _ = librosa.pyin(chunk, fmin=librosa.note_to_hz('C2'),
                                         fmax=librosa.note_to_hz('C7'), sr=sr)
        vr = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
        if vr >= args.min_voiced:
            segments.append((start, chunk, vr))
    print('有效片段数：%d' % len(segments))
    for i,(start,chunk,vr) in enumerate(segments):
        fname = '%s_%s_%03d.wav' % (args.singer, args.song, i+1)
        sf.write(os.path.join(args.out, fname), chunk, sr)
        print('  保存 %s 起始%.1fs 有声%.3f' % (fname, start, vr))
    print('共 %d 个片段 -> %s/' % (len(segments), args.out))

if __name__ == '__main__':
    main()
