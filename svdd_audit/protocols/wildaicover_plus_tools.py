from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shlex
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path("/home/admin2/xf/SingerLens")
BASE = ROOT / "data" / "wildaicover_plus"
OUT = ROOT / "outputs"

FAKE_QUERIES = [
    ("RVC", "AI翻唱 RVC"),
    ("RVC", "RVC AI cover 中文"),
    ("RVC", "AI孙燕姿 RVC 翻唱"),
    ("RVC", "AI陈奕迅 RVC 翻唱"),
    ("RVC", "AI周杰伦 RVC 翻唱"),
    ("So-VITS-SVC", "AI翻唱 so-vits-svc"),
    ("So-VITS-SVC", "so-vits-svc AI cover 中文"),
    ("So-VITS-SVC", "AI邓紫棋 so-vits-svc 翻唱"),
    ("So-VITS-SVC", "AI王菲 so-vits-svc 翻唱"),
    ("Diff-SVC", "AI翻唱 Diff-SVC"),
    ("Diff-SVC", "Diff-SVC AI cover"),
    ("GPT-SoVITS", "GPT-SoVITS AI翻唱"),
    ("GPT-SoVITS", "GPT-SoVITS cover 歌曲"),
    ("DDSP-SVC", "DDSP-SVC AI翻唱"),
    ("DDSP-SVC", "DDSP-SVC AI cover"),
    ("unknown", "AI翻唱 bilibili"),
    ("unknown", "AI cover 中文 翻唱"),
    ("unknown", "AI歌手 翻唱"),
]

REAL_QUERIES = [
    ("real", "孙燕姿 现场 翻唱"),
    ("real", "陈奕迅 现场 演唱"),
    ("real", "周杰伦 现场 演唱"),
    ("real", "邓紫棋 现场 演唱"),
    ("real", "王菲 现场 演唱"),
    ("real", "林俊杰 现场 演唱"),
    ("real", "张学友 现场 演唱"),
    ("real", "单依纯 现场 演唱"),
    ("real", "毛不易 现场 演唱"),
    ("real", "薛之谦 现场 演唱"),
    ("real", "张杰 现场 演唱"),
    ("real", "汪苏泷 现场 演唱"),
    ("real", "华晨宇 现场 演唱"),
    ("real", "五月天 现场 演唱"),
    ("real", "林忆莲 现场 演唱"),
]


def run(cmd: list[str], cwd: Path | None = None, timeout: int | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=timeout)


def yt_json_lines(url: str, limit: int) -> list[dict]:
    cmd = ["yt-dlp", "--flat-playlist", "--dump-json", f"bilisearch{limit}:{url}"]
    p = run(cmd, cwd=ROOT, timeout=120)
    rows = []
    for line in p.stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return rows


def normalize_url(item: dict) -> str:
    url = item.get("url") or item.get("webpage_url") or item.get("original_url") or ""
    if url.startswith("http"):
        return url
    vid = item.get("id")
    if vid:
        return f"https://www.bilibili.com/video/av{vid}"
    return url


def infer_title(url: str) -> str:
    try:
        p = run(["yt-dlp", "--skip-download", "--print", "%(title)s", url], cwd=ROOT, timeout=30)
    except subprocess.TimeoutExpired:
        return ""
    title = p.stdout.strip().splitlines()
    return title[-1] if title else ""


def clean_text(x: str) -> str:
    return re.sub(r"\s+", " ", str(x or "")).strip()


def collect_candidates(per_query: int, max_per_label: int, infer_titles: bool) -> pd.DataFrame:
    rows = []
    seen = set()
    for label, queries in [("fake", FAKE_QUERIES), ("real", REAL_QUERIES)]:
        for generator_hint, query in queries:
            for item in yt_json_lines(query, per_query):
                url = normalize_url(item)
                if not url or url in seen:
                    continue
                seen.add(url)
                title = clean_text(item.get("title") or "")
                if infer_titles and not title:
                    title = infer_title(url)
                rows.append(
                    {
                        "sample_id": f"wacp_{len(rows):04d}",
                        "label": label,
                        "platform": "bilibili",
                        "url": url,
                        "song": "",
                        "singer": "" if label == "fake" else query.split()[0],
                        "claimed_target_singer": query.split()[0].replace("AI", "") if label == "fake" else "",
                        "generator_hint": generator_hint if label == "fake" else "human",
                        "source_type": "public_ai_cover" if label == "fake" else "public_real_performance",
                        "duration": "",
                        "path": "",
                        "query": query,
                        "title": title,
                        "download_status": "pending",
                        "demucs_status": "pending",
                        "clip_count": 0,
                    }
                )
                if sum(r["label"] == label for r in rows) >= max_per_label:
                    break
            if sum(r["label"] == label for r in rows) >= max_per_label:
                break
    df = pd.DataFrame(rows)
    BASE.mkdir(parents=True, exist_ok=True)
    out = BASE / "candidates.csv"
    df.to_csv(out, index=False)
    return df


def download_audio(candidates: Path, max_per_label: int) -> pd.DataFrame:
    df = pd.read_csv(candidates)
    raw = BASE / "raw"
    raw.mkdir(parents=True, exist_ok=True)
    done = {"fake": 0, "real": 0}
    for i, r in df.iterrows():
        label = r["label"]
        if done.get(label, 0) >= max_per_label:
            continue
        sid = r["sample_id"]
        outtmpl = str(raw / f"{sid}.%(ext)s")
        cmd = [
            "yt-dlp",
            "-x",
            "--audio-format",
            "wav",
            "--no-playlist",
            "--cookies",
            str(ROOT / "cookies" / "bilibili.txt"),
            "-o",
            outtmpl,
            str(r["url"]),
        ]
        p = run(cmd, cwd=ROOT, timeout=300)
        wav = raw / f"{sid}.wav"
        if wav.exists() and wav.stat().st_size > 10000:
            df.at[i, "download_status"] = "ok"
            df.at[i, "path"] = str(wav)
            done[label] += 1
        else:
            df.at[i, "download_status"] = "fail"
            df.at[i, "download_error"] = clean_text(p.stderr)[-300:]
        df.to_csv(BASE / "download_status.csv", index=False)
    return df


def demucs_vocals(status_csv: Path, max_items: int | None) -> pd.DataFrame:
    df = pd.read_csv(status_csv)
    demucs_root = BASE / "demucs"
    demucs_root.mkdir(parents=True, exist_ok=True)
    n = 0
    for i, r in df[df["download_status"].eq("ok")].iterrows():
        if max_items is not None and n >= max_items:
            break
        sid = r["sample_id"]
        src = Path(r["path"])
        cmd = ["python", "-m", "demucs", "-n", "htdemucs", "--two-stems", "vocals", "-o", str(demucs_root), str(src)]
        p = run(cmd, cwd=ROOT, timeout=900)
        candidates = list(demucs_root.glob(f"**/{src.stem}/vocals.wav"))
        if candidates:
            df.at[i, "demucs_status"] = "ok"
            df.at[i, "path"] = str(candidates[0])
            n += 1
        else:
            df.at[i, "demucs_status"] = "fail"
            df.at[i, "demucs_error"] = clean_text(p.stderr)[-300:]
        df.to_csv(BASE / "demucs_status.csv", index=False)
    return df


def slice_clips(status_csv: Path, segment_len: float, hop: float, min_voiced: float) -> pd.DataFrame:
    import librosa
    import soundfile as sf

    df = pd.read_csv(status_csv)
    clips = BASE / "clips"
    clips.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, r in df[df["demucs_status"].eq("ok")].iterrows():
        y, sr = librosa.load(r["path"], sr=16000, mono=True)
        dur = librosa.get_duration(y=y, sr=sr)
        clip_count = 0
        for j, start in enumerate(np.arange(0, max(0.0, dur - segment_len), hop)):
            s = int(start * sr)
            e = int((start + segment_len) * sr)
            chunk = y[s:e]
            if chunk.size < int(segment_len * sr * 0.8):
                continue
            _, voiced_flag, _ = librosa.pyin(
                chunk,
                fmin=librosa.note_to_hz("A2"),
                fmax=librosa.note_to_hz("A5"),
                sr=sr,
            )
            vr = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
            if vr < min_voiced:
                continue
            clip_id = f"{r['sample_id']}_c{j:03d}"
            path = clips / f"{clip_id}.wav"
            sf.write(path, chunk, sr)
            rec = r.to_dict()
            rec.update({"clip_id": clip_id, "duration": segment_len, "path": str(path), "start_time": round(float(start), 2)})
            rows.append(rec)
            clip_count += 1
            if clip_count >= 4:
                break
    clips_df = pd.DataFrame(rows)
    clips_df.to_csv(BASE / "metadata_clips.csv", index=False)
    return clips_df


def export_feature_metadata(clips_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(clips_csv)
    meta = pd.DataFrame(
        {
            "file_path": df["path"],
            "label": df["label"],
            "singer_id": df["singer"].fillna("").replace("", "unknown"),
            "song_id": df["song"].fillna("").replace("", df["sample_id"]),
            "source_type": df["source_type"],
            "sample_id": df["sample_id"],
            "platform": df["platform"],
            "url": df["url"],
            "song": df["song"],
            "singer": df["singer"],
            "claimed_target_singer": df["claimed_target_singer"],
            "generator_hint": df["generator_hint"],
            "duration": df["duration"],
            "path": df["path"],
        }
    )
    meta.to_csv(BASE / "metadata_features.csv", index=False)
    return meta


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("collect")
    c.add_argument("--per-query", type=int, default=12)
    c.add_argument("--max-per-label", type=int, default=120)
    c.add_argument("--infer-titles", action="store_true")
    d = sub.add_parser("download")
    d.add_argument("--candidates", default=str(BASE / "candidates.csv"))
    d.add_argument("--max-per-label", type=int, default=80)
    m = sub.add_parser("demucs")
    m.add_argument("--status-csv", default=str(BASE / "download_status.csv"))
    m.add_argument("--max-items", type=int, default=None)
    s = sub.add_parser("slice")
    s.add_argument("--status-csv", default=str(BASE / "demucs_status.csv"))
    s.add_argument("--segment-len", type=float, default=8.0)
    s.add_argument("--hop", type=float, default=8.0)
    s.add_argument("--min-voiced", type=float, default=0.35)
    e = sub.add_parser("feature-meta")
    e.add_argument("--clips-csv", default=str(BASE / "metadata_clips.csv"))
    args = ap.parse_args()
    if args.cmd == "collect":
        df = collect_candidates(args.per_query, args.max_per_label, args.infer_titles)
    elif args.cmd == "download":
        df = download_audio(Path(args.candidates), args.max_per_label)
    elif args.cmd == "demucs":
        df = demucs_vocals(Path(args.status_csv), args.max_items)
    elif args.cmd == "slice":
        df = slice_clips(Path(args.status_csv), args.segment_len, args.hop, args.min_voiced)
    elif args.cmd == "feature-meta":
        df = export_feature_metadata(Path(args.clips_csv))
    print(df.shape)
    print(df.head().to_string(index=False))


if __name__ == "__main__":
    main()
