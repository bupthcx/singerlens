"""P3: human listening pilot 准备包(仅准备，不招人)。
从 WildSVDD 选 4 类各 10 条: easy real / easy fake / model-failed fake(FN) / model-FP real。
重建 10s 音频、匿名打乱、写盲 manifest + key + README。
"""
from __future__ import annotations
import os, glob, shutil
from pathlib import Path
import numpy as np, pandas as pd, librosa, soundfile as sf

SR = 16000; N = 10
ROOT = '/home/admin2/xf/wildsvdd'
OUT = Path('outputs/human_pilot'); AUD = OUT / 'audio'
AUD.mkdir(parents=True, exist_ok=True)


def wild_slice(idx, ci):
    """复现 wild_process 切法，返回第 ci 个 10s 片段。"""
    voc = glob.glob(f'{ROOT}/demucs_out/{idx}/htdemucs/*/vocals.wav')
    if not voc:
        return None
    v, _ = librosa.load(voc[0], sr=SR, mono=True)
    c = 0
    for st in np.arange(0, max(len(v) / SR - 10, 0.01), 8):
        ch = v[int(st * SR):int((st + 10) * SR)]
        if len(ch) < SR * 5:
            continue
        if c == ci:
            return ch
        c += 1
    return None


def main():
    cs = pd.read_csv('outputs/score_clip_scores.csv')
    wmeta = pd.read_csv('outputs/window_features_wild.csv')
    wmeta = wmeta.drop_duplicates('clip_id').set_index('clip_id')[['singer_id', 'model']].to_dict('index')

    within = cs[cs.protocol == 'Wild_all_within']
    cross = cs[cs.protocol == 'CtrSVDD->Wild_all']
    used = set(); picks = []

    def take(df, cond, by_asc, cat, k=N):
        d = df[cond].sort_values('score', ascending=by_asc)
        cnt = 0
        for _, r in d.iterrows():
            if r['clip_id'] in used:
                continue
            used.add(r['clip_id']); picks.append((cat, r['clip_id'], r['label'], round(r['score'], 3)))
            cnt += 1
            if cnt >= k:
                break

    take(within, within.y == 0, True, 'easy_real')          # 域内自信判真(低分)
    take(within, within.y == 1, False, 'easy_fake')         # 域内自信判假(高分)
    take(cross, (cross.y == 1) & (cross.score < 0.5), True, 'model_failed_fake')   # 跨域 FN
    take(cross, (cross.y == 0) & (cross.score > 0.5), False, 'model_fp_real')      # 跨域 FP

    # 重建音频 + 匿名打乱
    rng = np.random.RandomState(7)
    order = list(range(len(picks))); rng.shuffle(order)
    key_rows = []; man_rows = []; ok = 0
    for anon_i, pi in enumerate(order, start=1):
        cat, cid, label, score = picks[pi]
        idx, ci = cid.rsplit('_c', 1); ci = int(ci)
        seg = wild_slice(idx, ci)
        if seg is None:
            print('MISS audio', cid); continue
        anon = f'HP{anon_i:03d}'
        sf.write(AUD / f'{anon}.wav', seg, SR)
        md = wmeta.get(cid, {})
        man_rows.append(dict(anon_id=anon, audio_path=f'audio/{anon}.wav', duration=round(len(seg) / SR, 1)))
        key_rows.append(dict(anon_id=anon, category=cat, true_label=label, model_score=score,
                             source_clip=cid, singer=md.get('singer_id', ''), model=md.get('model', '')))
        ok += 1
    man = pd.DataFrame(man_rows).sort_values('anon_id')
    # 盲 manifest: 加空标注列
    man['human_label'] = ''; man['confidence_1to5'] = ''; man['suspected_reason'] = ''
    man.to_csv(OUT / 'human_pilot_manifest.csv', index=False)
    pd.DataFrame(key_rows).sort_values('anon_id').to_csv(OUT / 'human_pilot_key.csv', index=False)
    print('wrote', ok, 'clips. category counts:')
    print(pd.DataFrame(key_rows)['category'].value_counts().to_string())
    print('Saved ->', OUT)


if __name__ == '__main__':
    main()
