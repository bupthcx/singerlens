from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import parselmouth
from scipy.signal import medfilt

from .vri import compute_vri


@dataclass(frozen=True)
class FeatureConfig:
    sample_rate: int = 16000
    hop_length: int = 256
    frame_length: int = 1024
    # 收紧到女声流行音域，减小亚谐波/超谐波跳变空间（原为 C2-C6）
    fmin: float = librosa.note_to_hz("A2")   # ~110 Hz
    fmax: float = librosa.note_to_hz("A5")   # ~880 Hz
    # 响度归一化目标（dBFS）。统一响度可消除 real/fake 之间的能量泄露捷径
    target_dbfs: float = -20.0
    loudness_normalize: bool = True
    # F0 八度跳变校正
    octave_correct: bool = True


def load_audio(path: str | Path, config: FeatureConfig = FeatureConfig()) -> tuple[np.ndarray, int]:
    y, sr = librosa.load(path, sr=config.sample_rate, mono=True)
    if y.size == 0:
        raise ValueError(f"Empty audio file: {path}")
    if config.loudness_normalize:
        rms = float(np.sqrt(np.mean(y ** 2)))
        if rms > 1e-8:
            gain = (10.0 ** (config.target_dbfs / 20.0)) / rms
            y = y * gain
            peak = float(np.max(np.abs(y)))
            if peak > 0.99:                     # 防削波
                y = y * (0.99 / peak)
    return y, sr


def _octave_correct(f0: np.ndarray) -> np.ndarray:
    """消除片段内的八度跳变：用中值滤波估计稳健参考音高，
    再将每帧按整数八度折叠到参考 ±0.5 八度内。
    只移除突跳，保留 VRI 所需的亚八度自然微扰。"""
    voiced_mask = f0 > 0
    if int(voiced_mask.sum()) < 5:
        return f0
    out = f0.astype(float).copy()
    vals = out[voiced_mask]
    log = np.log2(vals)
    k = 5 if log.size >= 5 else (log.size // 2) * 2 + 1
    ref = float(np.median(medfilt(log, kernel_size=k)))
    shift = np.round(log - ref)               # 每帧最近的整数八度偏移
    out[voiced_mask] = np.power(2.0, log - shift)
    return out


def extract_f0(y: np.ndarray, config: FeatureConfig = FeatureConfig()) -> np.ndarray:
    f0, _, _ = librosa.pyin(
        y,
        fmin=config.fmin,
        fmax=config.fmax,
        sr=config.sample_rate,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    )
    f0 = np.nan_to_num(f0, nan=0.0)
    if config.octave_correct:
        f0 = _octave_correct(f0)
    return f0


def _safe_ratio(a: float, b: float, default: float = 0.0) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or abs(b) < 1e-12:
        return default
    return float(a / b)


def _f0_features(f0: np.ndarray) -> dict[str, float]:
    voiced = f0[f0 > 0]
    if voiced.size < 3:
        return {
            "f0_mean": 0.0,
            "f0_std": 0.0,
            "f0_min": 0.0,
            "f0_max": 0.0,
            "f0_range_semitones": 0.0,
            "f0_jitter": 0.0,
        }

    diffs = np.diff(voiced)
    return {
        "f0_mean": float(np.mean(voiced)),
        "f0_std": float(np.std(voiced)),
        "f0_min": float(np.min(voiced)),
        "f0_max": float(np.max(voiced)),
        "f0_range_semitones": float(12.0 * np.log2(np.max(voiced) / np.min(voiced))),
        "f0_jitter": float(np.mean(np.abs(diffs)) / max(np.mean(voiced), 1e-12)),
    }


def _hnr_features(y: np.ndarray, sr: int) -> dict[str, float]:
    try:
        sound = parselmouth.Sound(y, sampling_frequency=sr)
        harmonicity = sound.to_harmonicity_cc()
        values = harmonicity.values
        valid = values[np.isfinite(values) & (values > -200)]
        if valid.size == 0:
            return {"hnr_mean": 0.0, "hnr_std": 0.0, "hnr_low_ratio": 0.0}
        return {
            "hnr_mean": float(np.mean(valid)),
            "hnr_std": float(np.std(valid)),
            "hnr_low_ratio": float(np.mean(valid < 10.0)),
        }
    except Exception:
        return {"hnr_mean": 0.0, "hnr_std": 0.0, "hnr_low_ratio": 0.0}


def _long_note_stability(f0: np.ndarray, min_frames: int = 16) -> float:
    voiced = f0 > 0
    if voiced.size == 0:
        return 0.0

    scores: list[float] = []
    start = None
    for idx, is_voiced in enumerate(np.r_[voiced, False]):
        if is_voiced and start is None:
            start = idx
        elif not is_voiced and start is not None:
            segment = f0[start:idx]
            start = None
            if segment.size >= min_frames and np.mean(segment) > 0:
                cents = 1200.0 * np.log2(segment / np.mean(segment))
                scores.append(float(np.std(cents)))

    if not scores:
        return 0.0
    drift = float(np.mean(scores))
    return float(1.0 / (1.0 + drift / 50.0))


def extract_clip_features(
    audio_path: str | Path,
    config: FeatureConfig = FeatureConfig(),
) -> dict[str, float]:
    y, _ = load_audio(audio_path, config)
    f0 = extract_f0(y, config)
    rms = librosa.feature.rms(
        y=y,
        frame_length=config.frame_length,
        hop_length=config.hop_length,
    )[0]
    flatness = librosa.feature.spectral_flatness(y=y)[0]
    mfcc = librosa.feature.mfcc(y=y, sr=config.sample_rate, n_mfcc=13)

    rms_nonzero = rms[rms > 1e-8]
    energy_dynamic = _safe_ratio(float(np.max(rms_nonzero)) if rms_nonzero.size else 0.0, float(np.min(rms_nonzero)) if rms_nonzero.size else 0.0)

    features: dict[str, float] = {
        "duration": float(librosa.get_duration(y=y, sr=config.sample_rate)),
        "rms_mean": float(np.mean(rms)),
        "rms_std": float(np.std(rms)),
        "energy_dynamic": float(energy_dynamic),
        "spectral_flatness_mean": float(np.mean(flatness)),
        "spectral_flatness_std": float(np.std(flatness)),
        "long_note_stability": _long_note_stability(f0),
    }
    features.update(_f0_features(f0))
    features.update(_hnr_features(y, config.sample_rate))
    features.update(compute_vri(f0, sr=config.sample_rate, hop_length=config.hop_length))

    for idx, values in enumerate(mfcc, start=1):
        features[f"mfcc_{idx}_mean"] = float(np.mean(values))
        features[f"mfcc_{idx}_std"] = float(np.std(values))

    return features


def extract_dataset_features(metadata_csv: str | Path, output_csv: str | Path | None = None) -> pd.DataFrame:
    metadata_path = Path(metadata_csv)
    metadata = pd.read_csv(metadata_path)
    rows = []
    for row in metadata.to_dict(orient="records"):
        audio_path = Path(row["file_path"])
        if not audio_path.is_absolute():
            audio_path = audio_path if audio_path.exists() else metadata_path.parent / audio_path
        features = extract_clip_features(audio_path)
        rows.append({**row, **features})

    df = pd.DataFrame(rows)
    if output_csv is not None:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
    return df
