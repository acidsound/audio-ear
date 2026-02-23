#!/usr/bin/env python3
"""
WAV Ear Compare

A/B WAV 파일을 시간-주파수 기반으로 비교해 "귀"처럼 차이를 수치화한다.
외부 의존성 없이(Python 표준 라이브러리만) 실행되도록 작성했다.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import struct
import sys
import wave
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

EPS = 1e-12


@dataclass
class SignalData:
    path: str
    sample_rate: int
    samples: List[float]


@dataclass
class AnalysisConfig:
    fft_size: int
    hop_size: int
    max_seconds: Optional[float]


@dataclass
class AnalysisResult:
    sample_rate: int
    frames: int
    duration_seconds: float
    peak_dbfs: float
    rms_dbfs: float
    attack_seconds: Optional[float]
    low_decay_t20_seconds: Optional[float]
    low_decay_t40_seconds: Optional[float]
    full_decay_t20_seconds: Optional[float]
    full_decay_t40_seconds: Optional[float]
    spectral_centroid_hz: float
    spectral_rolloff_hz: float
    spectral_flatness: float
    spectral_tilt_db_per_decade: float
    resonance_prominence_db: float
    ring_ratio: float
    median_f0_hz: Optional[float]
    f0_drift_cents_per_sec: Optional[float]
    click_rms_dbfs: Optional[float]
    body_rms_dbfs: Optional[float]
    click_to_body_db: Optional[float]
    early_high_low_ratio_db: Optional[float]
    harmonic_h1_db: Optional[float]
    harmonic_h2_db: Optional[float]
    harmonic_h3_db: Optional[float]
    harmonic_h15_db: Optional[float]
    harmonic_h2_to_h1_db: Optional[float]
    harmonic_h3_to_h1_db: Optional[float]
    harmonic_h15_to_h1_db: Optional[float]
    formant_like_peaks_hz: List[float]
    band_levels_db: Dict[str, float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare reference and test WAV by FFT/STFT metrics and output actionable suggestions."
    )
    parser.add_argument("--ref", required=True, help="Reference WAV path (target sound A)")
    parser.add_argument("--test", required=True, help="Test WAV path (current implementation B)")
    parser.add_argument("--fft-size", type=int, default=2048, help="FFT size (power of two)")
    parser.add_argument("--hop-size", type=int, default=512, help="Hop size in samples")
    parser.add_argument(
        "--max-seconds",
        type=float,
        default=4.0,
        help="Analyze at most this many seconds after onset alignment (default: 4.0)",
    )
    parser.add_argument(
        "--no-align-onset",
        action="store_true",
        help="Disable onset alignment",
    )
    parser.add_argument(
        "--no-loudness-match",
        action="store_true",
        help="Disable RMS loudness matching before spectral comparison",
    )
    parser.add_argument("--json-out", help="Write machine-readable JSON report")
    parser.add_argument("--md-out", help="Write Markdown report")
    parser.add_argument(
        "--title",
        default="WAV Ear Compare Report",
        help="Report title for markdown output",
    )
    return parser.parse_args()


def is_power_of_two(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def db20(x: float) -> float:
    return 20.0 * math.log10(max(x, EPS))


def db10(x: float) -> float:
    return 10.0 * math.log10(max(x, EPS))


def rms(samples: Sequence[float]) -> float:
    if not samples:
        return 0.0
    s = 0.0
    for v in samples:
        s += v * v
    return math.sqrt(s / len(samples))


def mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return sum(values) / len(values)


def median(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    m = n // 2
    if n % 2 == 1:
        return sorted_vals[m]
    return 0.5 * (sorted_vals[m - 1] + sorted_vals[m])


def read_wav_mono(path: str) -> SignalData:
    if not os.path.exists(path):
        raise FileNotFoundError(path)

    with wave.open(path, "rb") as wf:
        n_channels = wf.getnchannels()
        sample_rate = wf.getframerate()
        sample_width = wf.getsampwidth()
        n_frames = wf.getnframes()
        comp_type = wf.getcomptype()
        if comp_type != "NONE":
            raise ValueError(f"Unsupported WAV compression in {path}: {comp_type}")
        raw = wf.readframes(n_frames)

    total = n_frames * n_channels
    if sample_width == 1:
        ints = raw
        pcm = [((b - 128) / 128.0) for b in ints]
    elif sample_width == 2:
        ints = struct.unpack("<" + "h" * total, raw)
        pcm = [v / 32768.0 for v in ints]
    elif sample_width == 3:
        pcm = []
        for i in range(0, len(raw), 3):
            b0 = raw[i]
            b1 = raw[i + 1]
            b2 = raw[i + 2]
            val = b0 | (b1 << 8) | (b2 << 16)
            if val & 0x800000:
                val -= 0x1000000
            pcm.append(val / 8388608.0)
    elif sample_width == 4:
        ints = struct.unpack("<" + "i" * total, raw)
        pcm = [v / 2147483648.0 for v in ints]
    else:
        raise ValueError(f"Unsupported sample width in {path}: {sample_width} bytes")

    if n_channels == 1:
        mono = pcm
    else:
        mono = []
        for i in range(0, len(pcm), n_channels):
            mono.append(sum(pcm[i : i + n_channels]) / n_channels)

    return SignalData(path=path, sample_rate=sample_rate, samples=mono)


def resample_linear(samples: Sequence[float], src_rate: int, dst_rate: int) -> List[float]:
    if src_rate == dst_rate:
        return list(samples)
    if not samples:
        return []

    ratio = src_rate / dst_rate
    out_len = max(1, int(round(len(samples) * dst_rate / src_rate)))
    out = [0.0] * out_len
    for i in range(out_len):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx
        if idx >= len(samples) - 1:
            out[i] = samples[-1]
        else:
            out[i] = (samples[idx] * (1.0 - frac)) + (samples[idx + 1] * frac)
    return out


def detect_onset(samples: Sequence[float]) -> int:
    if not samples:
        return 0
    peak = max(abs(v) for v in samples)
    if peak <= EPS:
        return 0

    threshold = max(peak * 0.06, 1e-5)
    win = 128
    acc = 0.0
    abs_vals = [abs(v) for v in samples]
    for i, val in enumerate(abs_vals):
        acc += val
        if i >= win:
            acc -= abs_vals[i - win]
        if i >= win and (acc / win) >= threshold:
            return max(0, i - win)
    return 0


def align_and_prepare(
    ref: SignalData,
    test: SignalData,
    align_onset: bool,
    loudness_match: bool,
    max_seconds: Optional[float],
) -> Tuple[List[float], List[float], int, int, float, int]:
    target_sr = max(ref.sample_rate, test.sample_rate)
    ref_samples = resample_linear(ref.samples, ref.sample_rate, target_sr)
    test_samples = resample_linear(test.samples, test.sample_rate, target_sr)

    ref_onset = detect_onset(ref_samples) if align_onset else 0
    test_onset = detect_onset(test_samples) if align_onset else 0

    ref_trim = ref_samples[ref_onset:]
    test_trim = test_samples[test_onset:]

    n = min(len(ref_trim), len(test_trim))
    ref_trim = ref_trim[:n]
    test_trim = test_trim[:n]

    if max_seconds is not None:
        max_len = int(target_sr * max_seconds)
        ref_trim = ref_trim[:max_len]
        test_trim = test_trim[:max_len]

    if not ref_trim or not test_trim:
        raise ValueError("Aligned signals are empty. Check input files.")

    gain_offset_db = 0.0
    if loudness_match:
        ref_rms = rms(ref_trim)
        test_rms = rms(test_trim)
        if test_rms > EPS:
            gain = ref_rms / test_rms
            gain_offset_db = db20(gain)
            test_trim = [v * gain for v in test_trim]

    return ref_trim, test_trim, ref_onset, test_onset, gain_offset_db, target_sr


def build_fft_plan(n: int) -> Tuple[List[int], List[Tuple[int, int, List[complex]]]]:
    if not is_power_of_two(n):
        raise ValueError(f"FFT size must be power of two, got {n}")

    levels = n.bit_length() - 1
    bitrev = [0] * n
    for i in range(n):
        x = i
        y = 0
        for _ in range(levels):
            y = (y << 1) | (x & 1)
            x >>= 1
        bitrev[i] = y

    stages = []
    size = 2
    while size <= n:
        half = size // 2
        phase = -2.0 * math.pi / size
        twiddles = [complex(math.cos(phase * k), math.sin(phase * k)) for k in range(half)]
        stages.append((size, half, twiddles))
        size *= 2

    return bitrev, stages


def fft_real(frame: Sequence[float], bitrev: Sequence[int], stages: Sequence[Tuple[int, int, Sequence[complex]]]) -> List[complex]:
    n = len(frame)
    out = [0j] * n
    for i, v in enumerate(frame):
        out[bitrev[i]] = complex(v, 0.0)

    for size, half, twiddles in stages:
        for start in range(0, n, size):
            for k in range(half):
                a = start + k
                b = a + half
                t = twiddles[k] * out[b]
                u = out[a]
                out[a] = u + t
                out[b] = u - t

    return out[: (n // 2) + 1]


def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> Tuple[float, float]:
    if len(xs) != len(ys) or len(xs) < 2:
        return 0.0, 0.0
    x_mean = mean(xs)
    y_mean = mean(ys)
    num = 0.0
    den = 0.0
    for x, y in zip(xs, ys):
        dx = x - x_mean
        num += dx * (y - y_mean)
        den += dx * dx
    if den <= EPS:
        return 0.0, y_mean
    slope = num / den
    intercept = y_mean - slope * x_mean
    return slope, intercept


def decay_time_seconds(env_db: Sequence[float], peak_idx: int, drop_db: float, frame_dt: float) -> Optional[float]:
    if not env_db or peak_idx < 0 or peak_idx >= len(env_db):
        return None
    target = env_db[peak_idx] - drop_db
    for i in range(peak_idx + 1, len(env_db)):
        if env_db[i] <= target:
            return (i - peak_idx) * frame_dt
    return None


def summarize_formant_peaks(avg_spec: Sequence[float], freqs: Sequence[float]) -> List[float]:
    peaks: List[Tuple[float, float]] = []
    for i in range(1, len(avg_spec) - 1):
        f = freqs[i]
        if f < 200.0 or f > 5000.0:
            continue
        if avg_spec[i] > avg_spec[i - 1] and avg_spec[i] >= avg_spec[i + 1]:
            peaks.append((avg_spec[i], f))

    if not peaks:
        return []

    peaks.sort(key=lambda x: x[0], reverse=True)
    top = [p[1] for p in peaks[:3]]
    return top


def peak_db_near_frequency(
    avg_spec_lin: Sequence[float],
    freqs: Sequence[float],
    target_hz: Optional[float],
    window_cents: float = 120.0,
) -> Optional[float]:
    if target_hz is None or target_hz <= 0.0 or not avg_spec_lin or not freqs:
        return None
    span_ratio = math.pow(2.0, window_cents / 1200.0)
    lo = target_hz / span_ratio
    hi = target_hz * span_ratio
    indices = [i for i, f in enumerate(freqs) if lo <= f <= hi]
    if not indices:
        idx = min(range(len(freqs)), key=lambda i: abs(freqs[i] - target_hz))
        indices = [idx]
    peak_lin = max(avg_spec_lin[i] for i in indices)
    return db10(peak_lin)


def analyze_signal(samples: Sequence[float], sample_rate: int, cfg: AnalysisConfig) -> AnalysisResult:
    n_fft = cfg.fft_size
    hop = cfg.hop_size

    if len(samples) < n_fft:
        padded = list(samples) + [0.0] * (n_fft - len(samples))
    else:
        padded = list(samples)

    n_frames = 1 + max(0, (len(padded) - n_fft) // hop)
    if n_frames <= 0:
        n_frames = 1

    window = [0.5 - 0.5 * math.cos((2.0 * math.pi * i) / (n_fft - 1)) for i in range(n_fft)]
    bitrev, stages = build_fft_plan(n_fft)

    freqs = [(sample_rate * i) / n_fft for i in range((n_fft // 2) + 1)]

    band_defs = [
        ("sub", 20.0, 60.0),
        ("bass", 60.0, 150.0),
        ("lowmid", 150.0, 600.0),
        ("mid", 600.0, 2000.0),
        ("highmid", 2000.0, 6000.0),
        ("air", 6000.0, 16000.0),
        ("low_all", 20.0, 200.0),
        ("high_all", 2000.0, 8000.0),
    ]
    band_bins: Dict[str, List[int]] = {}
    for name, lo, hi in band_defs:
        bins = [i for i, f in enumerate(freqs) if lo <= f < hi]
        if not bins:
            closest = min(range(len(freqs)), key=lambda idx: abs(freqs[idx] - (0.5 * (lo + hi))))
            bins = [closest]
        band_bins[name] = bins

    pitch_bins = [i for i, f in enumerate(freqs) if 40.0 <= f <= 2000.0]
    if not pitch_bins:
        pitch_bins = [1]

    resonance_bins = [i for i, f in enumerate(freqs) if 200.0 <= f <= 8000.0]
    if not resonance_bins:
        resonance_bins = pitch_bins

    frame_rms_db: List[float] = []
    band_env_lin: Dict[str, List[float]] = {name: [] for name, _, _ in band_defs}
    centroids: List[float] = []
    rolloffs: List[float] = []
    flatnesses: List[float] = []
    resonances: List[float] = []
    fluxes: List[float] = []
    avg_spec_lin = [0.0] * len(freqs)

    prev_mag: Optional[List[float]] = None
    pitch_candidates: List[Tuple[float, float, float]] = []  # (time_sec, frame_db, f0_hz)

    for fi in range(n_frames):
        start = fi * hop
        frame = padded[start : start + n_fft]
        if len(frame) < n_fft:
            frame = frame + [0.0] * (n_fft - len(frame))

        windowed = [frame[i] * window[i] for i in range(n_fft)]
        spec = fft_real(windowed, bitrev, stages)

        mag = []
        for c in spec:
            mag.append((c.real * c.real) + (c.imag * c.imag))

        for i in range(len(avg_spec_lin)):
            avg_spec_lin[i] += mag[i]

        frame_energy = 0.0
        for v in frame:
            frame_energy += v * v
        frame_db = db20(math.sqrt(frame_energy / len(frame)))
        frame_rms_db.append(frame_db)

        total_pow = sum(mag) + EPS
        centroid = 0.0
        for i, power in enumerate(mag):
            centroid += freqs[i] * power
        centroid /= total_pow
        centroids.append(centroid)

        roll_target = 0.95 * total_pow
        roll_cum = 0.0
        roll_freq = freqs[-1]
        for i, power in enumerate(mag):
            roll_cum += power
            if roll_cum >= roll_target:
                roll_freq = freqs[i]
                break
        rolloffs.append(roll_freq)

        log_sum = 0.0
        for p in mag:
            log_sum += math.log(p + EPS)
        gm = math.exp(log_sum / len(mag))
        am = total_pow / len(mag)
        flatnesses.append(gm / (am + EPS))

        section = [mag[i] for i in resonance_bins]
        section_mean = mean(section)
        section_peak = max(section)
        resonances.append(db10((section_peak + EPS) / (section_mean + EPS)))

        if prev_mag is not None:
            f = 0.0
            for i in range(len(mag)):
                d = math.sqrt(mag[i] + EPS) - math.sqrt(prev_mag[i] + EPS)
                if d > 0:
                    f += d
            fluxes.append(f / len(mag))
        prev_mag = mag

        for band_name, _, _ in band_defs:
            bins = band_bins[band_name]
            bp = 0.0
            for idx in bins:
                bp += mag[idx]
            band_env_lin[band_name].append(bp / len(bins))

        p_idx = max(pitch_bins, key=lambda idx: mag[idx])
        delta = 0.0
        if 1 <= p_idx < len(mag) - 1:
            a = mag[p_idx - 1]
            b = mag[p_idx]
            c = mag[p_idx + 1]
            denom = (a - 2.0 * b + c)
            if abs(denom) > EPS:
                delta = 0.5 * (a - c) / denom
                delta = max(-0.5, min(0.5, delta))
        f0 = (p_idx + delta) * sample_rate / n_fft
        t = fi * hop / sample_rate
        pitch_candidates.append((t, frame_db, f0))

    avg_spec_lin = [v / n_frames for v in avg_spec_lin]
    avg_spec_db = [db10(v) for v in avg_spec_lin]

    peak_idx = max(range(len(frame_rms_db)), key=lambda i: frame_rms_db[i])
    peak_db = frame_rms_db[peak_idx]
    onset_idx = 0
    for i, val in enumerate(frame_rms_db):
        if val >= (peak_db - 30.0):
            onset_idx = i
            break

    frame_dt = hop / sample_rate
    attack_seconds = max(0.0, (peak_idx - onset_idx) * frame_dt)

    low_env_db = [db10(v) for v in band_env_lin["low_all"]]
    # 전체 감쇄는 프레임 RMS 기반으로 계산한다.
    full_env_db = frame_rms_db

    low_decay_t20 = decay_time_seconds(low_env_db, peak_idx, 20.0, frame_dt)
    low_decay_t40 = decay_time_seconds(low_env_db, peak_idx, 40.0, frame_dt)
    full_decay_t20 = decay_time_seconds(full_env_db, peak_idx, 20.0, frame_dt)
    full_decay_t40 = decay_time_seconds(full_env_db, peak_idx, 40.0, frame_dt)

    hi_lin = band_env_lin["high_all"]
    early_frames = max(1, int(round(0.08 / frame_dt)))
    tail_start = peak_idx + max(1, int(round(0.10 / frame_dt)))
    tail_end = peak_idx + max(2, int(round(0.50 / frame_dt)))
    early = mean(hi_lin[peak_idx : min(len(hi_lin), peak_idx + early_frames)])
    tail = mean(hi_lin[tail_start : min(len(hi_lin), tail_end)]) if tail_start < len(hi_lin) else 0.0
    ring_ratio = tail / (early + EPS)

    onset_window_frames = max(1, int(round(0.03 / frame_dt)))
    onset_end = min(len(band_env_lin["low_all"]), onset_idx + onset_window_frames)
    early_high_low_ratio_db: Optional[float] = None
    if onset_end > onset_idx:
        early_low = mean(band_env_lin["low_all"][onset_idx:onset_end])
        early_high = mean(band_env_lin["high_all"][onset_idx:onset_end])
        early_high_low_ratio_db = db10((early_high + EPS) / (early_low + EPS))

    energy_gate = peak_db - 35.0
    pitch_values: List[Tuple[float, float]] = []
    for t, level_db, f0 in pitch_candidates:
        if level_db >= energy_gate and 20.0 <= f0 <= 5000.0:
            pitch_values.append((t, f0))

    median_f0 = median([f for _, f in pitch_values])
    drift_cents_per_sec: Optional[float] = None
    if len(pitch_values) >= 3:
        t0, f_ref = pitch_values[0]
        xs: List[float] = []
        ys: List[float] = []
        for t, f in pitch_values:
            xs.append(t - t0)
            ys.append(1200.0 * math.log2(max(f, EPS) / max(f_ref, EPS)))
        slope, _ = linear_regression(xs, ys)
        drift_cents_per_sec = slope

    click_len = max(1, int(round(sample_rate * 0.008)))
    body_start = max(click_len, int(round(sample_rate * 0.020)))
    body_end = min(len(samples), max(body_start + 1, int(round(sample_rate * 0.120))))
    click_rms_dbfs: Optional[float] = None
    body_rms_dbfs: Optional[float] = None
    click_to_body_db: Optional[float] = None
    if len(samples) >= click_len:
        click_rms_dbfs = db20(rms(samples[:click_len]))
    if body_end > body_start:
        body_rms_dbfs = db20(rms(samples[body_start:body_end]))
    if click_rms_dbfs is not None and body_rms_dbfs is not None:
        click_to_body_db = click_rms_dbfs - body_rms_dbfs

    h1_db = peak_db_near_frequency(avg_spec_lin, freqs, median_f0, window_cents=120.0)
    h2_db = peak_db_near_frequency(
        avg_spec_lin, freqs, None if median_f0 is None else median_f0 * 2.0, window_cents=120.0
    )
    h3_db = peak_db_near_frequency(
        avg_spec_lin, freqs, None if median_f0 is None else median_f0 * 3.0, window_cents=120.0
    )
    h15_db = peak_db_near_frequency(
        avg_spec_lin, freqs, None if median_f0 is None else median_f0 * 1.5, window_cents=120.0
    )
    h2_to_h1_db = None if h1_db is None or h2_db is None else h2_db - h1_db
    h3_to_h1_db = None if h1_db is None or h3_db is None else h3_db - h1_db
    h15_to_h1_db = None if h1_db is None or h15_db is None else h15_db - h1_db

    formant_peaks = summarize_formant_peaks(avg_spec_lin, freqs)

    slope_x: List[float] = []
    slope_y: List[float] = []
    for i, f in enumerate(freqs):
        if 60.0 <= f <= 12000.0:
            slope_x.append(math.log10(f))
            slope_y.append(avg_spec_db[i])
    spectral_tilt, _ = linear_regression(slope_x, slope_y)

    band_levels_db = {}
    for band_name, _, _ in band_defs:
        if band_name in ("low_all", "high_all"):
            continue
        band_levels_db[band_name] = db10(mean(band_env_lin[band_name]))

    duration_seconds = len(samples) / sample_rate

    return AnalysisResult(
        sample_rate=sample_rate,
        frames=n_frames,
        duration_seconds=duration_seconds,
        peak_dbfs=peak_db,
        rms_dbfs=db20(rms(samples)),
        attack_seconds=attack_seconds,
        low_decay_t20_seconds=low_decay_t20,
        low_decay_t40_seconds=low_decay_t40,
        full_decay_t20_seconds=full_decay_t20,
        full_decay_t40_seconds=full_decay_t40,
        spectral_centroid_hz=mean(centroids),
        spectral_rolloff_hz=mean(rolloffs),
        spectral_flatness=mean(flatnesses),
        spectral_tilt_db_per_decade=spectral_tilt,
        resonance_prominence_db=mean(resonances),
        ring_ratio=ring_ratio,
        median_f0_hz=median_f0,
        f0_drift_cents_per_sec=drift_cents_per_sec,
        click_rms_dbfs=click_rms_dbfs,
        body_rms_dbfs=body_rms_dbfs,
        click_to_body_db=click_to_body_db,
        early_high_low_ratio_db=early_high_low_ratio_db,
        harmonic_h1_db=h1_db,
        harmonic_h2_db=h2_db,
        harmonic_h3_db=h3_db,
        harmonic_h15_db=h15_db,
        harmonic_h2_to_h1_db=h2_to_h1_db,
        harmonic_h3_to_h1_db=h3_to_h1_db,
        harmonic_h15_to_h1_db=h15_to_h1_db,
        formant_like_peaks_hz=formant_peaks,
        band_levels_db=band_levels_db,
    )


def f0_delta_cents(ref_hz: Optional[float], test_hz: Optional[float]) -> Optional[float]:
    if ref_hz is None or test_hz is None:
        return None
    return 1200.0 * math.log2(max(test_hz, EPS) / max(ref_hz, EPS))


def decay_delta_seconds(ref: Optional[float], test: Optional[float]) -> Optional[float]:
    if ref is None or test is None:
        return None
    return test - ref


def paired_formant_shift(ref_peaks: Sequence[float], test_peaks: Sequence[float]) -> List[float]:
    n = min(len(ref_peaks), len(test_peaks))
    return [test_peaks[i] - ref_peaks[i] for i in range(n)]


def build_suggestions(
    ref: AnalysisResult,
    test: AnalysisResult,
    gain_offset_db: float,
) -> List[Dict[str, object]]:
    suggestions: List[Dict[str, object]] = []

    low_t20_delta = decay_delta_seconds(ref.low_decay_t20_seconds, test.low_decay_t20_seconds)
    if low_t20_delta is not None and low_t20_delta < -0.020:
        suggestions.append(
            {
                "focus": "초기 저역 감쇄",
                "finding": "B의 저역(20-200Hz) 감쇄가 A보다 빠릅니다.",
                "evidence": f"low_decay_t20 delta = {low_t20_delta * 1000:.1f} ms",
                "actions": [
                    "Amp envelope decay/release를 저역 경로에서 약 10~30% 연장",
                    "필터 envelope amount 또는 HPF cutoff를 낮춰 초기 저역 소실 완화",
                    "drive/saturation이 저역을 깎는 구조라면 pre-filter drive를 줄이고 post-filter 보정 EQ 적용",
                ],
            }
        )

    low_t40_delta = decay_delta_seconds(ref.low_decay_t40_seconds, test.low_decay_t40_seconds)
    if low_t40_delta is not None and low_t40_delta > 0.040:
        suggestions.append(
            {
                "focus": "저역 tail 과다",
                "finding": "B의 저역 tail이 A보다 길어 붕붕거림 가능성이 있습니다.",
                "evidence": f"low_decay_t40 delta = {low_t40_delta * 1000:.1f} ms",
                "actions": [
                    "저역 전용 감쇄 곡선을 더 가파르게 설정",
                    "필터 Q 또는 feedback 경로의 저역 이득을 미세 하향",
                ],
            }
        )

    centroid_delta = test.spectral_centroid_hz - ref.spectral_centroid_hz
    if centroid_delta > 180.0:
        suggestions.append(
            {
                "focus": "밝기 과다",
                "finding": "B가 A보다 고역 에너지가 상대적으로 높습니다.",
                "evidence": f"spectral_centroid delta = {centroid_delta:.1f} Hz",
                "actions": [
                    "필터 cutoff 초기값 또는 envelope amount를 소폭 하향",
                    "노이즈/하모닉 생성량(osc mix, waveshaper drive)을 줄임",
                ],
            }
        )
    elif centroid_delta < -180.0:
        suggestions.append(
            {
                "focus": "밝기 부족",
                "finding": "B가 A보다 어둡게 들릴 가능성이 높습니다.",
                "evidence": f"spectral_centroid delta = {centroid_delta:.1f} Hz",
                "actions": [
                    "필터 cutoff 또는 envelope amount를 소폭 증가",
                    "고역 damping이 과하면 damping/LPF 기울기 완화",
                ],
            }
        )

    resonance_delta = test.resonance_prominence_db - ref.resonance_prominence_db
    if resonance_delta > 2.5:
        suggestions.append(
            {
                "focus": "공진 과다",
                "finding": "B의 공진 피크가 A보다 두드러집니다.",
                "evidence": f"resonance_prominence delta = {resonance_delta:.2f} dB",
                "actions": [
                    "Filter Q를 낮추거나 key tracking을 완만하게 조정",
                    "필터 self-oscillation 성분이 있으면 제한(clamp) 적용",
                ],
            }
        )
    elif resonance_delta < -2.5:
        suggestions.append(
            {
                "focus": "공진 부족",
                "finding": "B의 공진 캐릭터가 A보다 덜 살아있습니다.",
                "evidence": f"resonance_prominence delta = {resonance_delta:.2f} dB",
                "actions": [
                    "Filter Q를 소폭 올리고 cutoff envelope와 동조",
                    "필터 앞단 하모닉(soft clip) 추가로 공진 지각 강화",
                ],
            }
        )

    ring_delta = test.ring_ratio - ref.ring_ratio
    if ring_delta > 0.10:
        suggestions.append(
            {
                "focus": "링잉 과다",
                "finding": "B의 고역 tail이 A보다 오래 남습니다.",
                "evidence": f"ring_ratio delta = {ring_delta:.3f}",
                "actions": [
                    "필터/딜레이 feedback을 낮추고 damping을 증가",
                    "트랜지언트 이후 릴리즈 구간에서 고역만 빠르게 감쇄",
                ],
            }
        )
    elif ring_delta < -0.10:
        suggestions.append(
            {
                "focus": "링잉 부족",
                "finding": "B의 잔향/울림 tail이 A보다 짧습니다.",
                "evidence": f"ring_ratio delta = {ring_delta:.3f}",
                "actions": [
                    "feedback 또는 공진 릴리즈를 소폭 증가",
                    "고역 감쇄가 과하면 damping 강도를 줄임",
                ],
            }
        )

    cents = f0_delta_cents(ref.median_f0_hz, test.median_f0_hz)
    if cents is not None and abs(cents) > 8.0:
        direction = "높습니다" if cents > 0 else "낮습니다"
        suggestions.append(
            {
                "focus": "피치 정합",
                "finding": f"B의 중심 피치가 A 대비 {direction}",
                "evidence": f"median_f0 delta = {cents:.2f} cents",
                "actions": [
                    "oscillator base tuning 또는 sample playback rate 보정",
                    "resample/interpolation 경로의 비율 계산(특히 44.1/48k 변환) 재검토",
                ],
            }
        )

    if ref.f0_drift_cents_per_sec is not None and test.f0_drift_cents_per_sec is not None:
        drift_delta = test.f0_drift_cents_per_sec - ref.f0_drift_cents_per_sec
        if abs(drift_delta) > 3.0:
            suggestions.append(
                {
                    "focus": "피치 드리프트",
                    "finding": "시간축 피치 변화량이 A/B 간 다릅니다.",
                    "evidence": f"f0_drift delta = {drift_delta:.2f} cents/s",
                    "actions": [
                        "envelope가 oscillator pitch에 미치는 모듈레이션 양 확인",
                        "필터/드라이브 비선형으로 인한 지각 피치 이동 여부 점검",
                    ],
                }
            )

    if ref.click_to_body_db is not None and test.click_to_body_db is not None:
        click_delta = test.click_to_body_db - ref.click_to_body_db
        if click_delta < -2.0:
            suggestions.append(
                {
                    "focus": "클릭 어택 부족",
                    "finding": "B의 초기 클릭(어택) 존재감이 A보다 약합니다.",
                    "evidence": f"click_to_body delta = {click_delta:.2f} dB",
                    "actions": [
                        "click/noise 경로 레벨 또는 cutoff를 소폭 상승",
                        "초기 5~15ms 구간의 high-shelf 또는 transient shaper 양을 미세 증가",
                    ],
                }
            )
        elif click_delta > 2.0:
            suggestions.append(
                {
                    "focus": "클릭 어택 과다",
                    "finding": "B의 초기 클릭(어택)이 A보다 과하게 두드러집니다.",
                    "evidence": f"click_to_body delta = {click_delta:.2f} dB",
                    "actions": [
                        "click 경로 레벨/decay를 소폭 감소",
                        "초기 고역 부스트 양을 줄여 body 대비 균형 맞춤",
                    ],
                }
            )

    if ref.harmonic_h15_to_h1_db is not None and test.harmonic_h15_to_h1_db is not None:
        h15_delta = test.harmonic_h15_to_h1_db - ref.harmonic_h15_to_h1_db
        if h15_delta < -1.5:
            suggestions.append(
                {
                    "focus": "OSC2(1.5x) 존재감 부족",
                    "finding": "B의 1.5x 부분음이 A보다 약해 중고역 바디가 덜 단단하게 들릴 수 있습니다.",
                    "evidence": f"h1.5_to_h1 delta = {h15_delta:.2f} dB",
                    "actions": [
                        "OSC2 level/drive를 소폭 상승",
                        "OSC2 중심 대역(약 1.3x~1.8x f0) peaking EQ를 미세 보강",
                    ],
                }
            )
        elif h15_delta > 1.5:
            suggestions.append(
                {
                    "focus": "OSC2(1.5x) 과다",
                    "finding": "B의 1.5x 부분음이 A보다 강해 피치 중심이 위로 당겨 들릴 수 있습니다.",
                    "evidence": f"h1.5_to_h1 delta = {h15_delta:.2f} dB",
                    "actions": [
                        "OSC2 level/drive를 소폭 감소",
                        "OSC2 decay를 약간 줄여 루트 대비 균형 회복",
                    ],
                }
            )

    if abs(gain_offset_db) > 1.5:
        suggestions.append(
            {
                "focus": "레벨 매칭",
                "finding": "비교 전 RMS 레벨 차이가 커서 지각 평가를 왜곡할 수 있습니다.",
                "evidence": f"auto gain match applied = {gain_offset_db:+.2f} dB",
                "actions": [
                    "A/B 렌더 단계에서 LUFS 또는 RMS 기준 정규화",
                    "분석 전 gain-match를 고정 파이프라인으로 포함",
                ],
            }
        )

    if not suggestions:
        suggestions.append(
            {
                "focus": "요약",
                "finding": "핵심 지표에서 큰 편차가 관찰되지 않았습니다.",
                "evidence": "current thresholds not exceeded",
                "actions": [
                    "더 미세한 차이는 동일 MIDI/velocity로 더 긴 샘플(>=8초) 비교",
                    "필요 시 밴드 경계와 임계치(스크립트 파라미터) 조정",
                ],
            }
        )

    return suggestions


def result_to_dict(result: AnalysisResult) -> Dict[str, object]:
    return {
        "sample_rate": result.sample_rate,
        "frames": result.frames,
        "duration_seconds": result.duration_seconds,
        "peak_dbfs": result.peak_dbfs,
        "rms_dbfs": result.rms_dbfs,
        "attack_seconds": result.attack_seconds,
        "low_decay_t20_seconds": result.low_decay_t20_seconds,
        "low_decay_t40_seconds": result.low_decay_t40_seconds,
        "full_decay_t20_seconds": result.full_decay_t20_seconds,
        "full_decay_t40_seconds": result.full_decay_t40_seconds,
        "spectral_centroid_hz": result.spectral_centroid_hz,
        "spectral_rolloff_hz": result.spectral_rolloff_hz,
        "spectral_flatness": result.spectral_flatness,
        "spectral_tilt_db_per_decade": result.spectral_tilt_db_per_decade,
        "resonance_prominence_db": result.resonance_prominence_db,
        "ring_ratio": result.ring_ratio,
        "median_f0_hz": result.median_f0_hz,
        "f0_drift_cents_per_sec": result.f0_drift_cents_per_sec,
        "click_rms_dbfs": result.click_rms_dbfs,
        "body_rms_dbfs": result.body_rms_dbfs,
        "click_to_body_db": result.click_to_body_db,
        "early_high_low_ratio_db": result.early_high_low_ratio_db,
        "harmonic_h1_db": result.harmonic_h1_db,
        "harmonic_h2_db": result.harmonic_h2_db,
        "harmonic_h3_db": result.harmonic_h3_db,
        "harmonic_h15_db": result.harmonic_h15_db,
        "harmonic_h2_to_h1_db": result.harmonic_h2_to_h1_db,
        "harmonic_h3_to_h1_db": result.harmonic_h3_to_h1_db,
        "harmonic_h15_to_h1_db": result.harmonic_h15_to_h1_db,
        "formant_like_peaks_hz": result.formant_like_peaks_hz,
        "band_levels_db": result.band_levels_db,
    }


def format_opt(value: Optional[float], unit: str = "", digits: int = 3) -> str:
    if value is None:
        return "n/a"
    return f"{value:.{digits}f}{unit}"


def to_markdown(
    title: str,
    ref_path: str,
    test_path: str,
    ref: AnalysisResult,
    test: AnalysisResult,
    gain_offset_db: float,
    ref_onset: int,
    test_onset: int,
    sample_rate: int,
    suggestions: Sequence[Dict[str, object]],
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- Reference A: `{ref_path}`")
    lines.append(f"- Test B: `{test_path}`")
    lines.append(f"- Sample Rate (analysis): `{sample_rate} Hz`")
    lines.append(f"- Onset alignment (samples): A=`{ref_onset}`, B=`{test_onset}`")
    lines.append(f"- Gain match applied to B: `{gain_offset_db:+.2f} dB`")
    lines.append("")
    lines.append("## 핵심 지표")
    lines.append("")
    lines.append("| Metric | A (ref) | B (test) | Delta (B-A) |")
    lines.append("|---|---:|---:|---:|")

    def row(name: str, a: Optional[float], b: Optional[float], unit: str = "", digits: int = 3) -> None:
        if a is None or b is None:
            lines.append(f"| {name} | {format_opt(a, unit, digits)} | {format_opt(b, unit, digits)} | n/a |")
        else:
            delta = b - a
            lines.append(
                f"| {name} | {a:.{digits}f}{unit} | {b:.{digits}f}{unit} | {delta:+.{digits}f}{unit} |"
            )

    row("RMS", ref.rms_dbfs, test.rms_dbfs, " dBFS", 2)
    row("Peak", ref.peak_dbfs, test.peak_dbfs, " dBFS", 2)
    row("Attack", ref.attack_seconds, test.attack_seconds, " s", 4)
    row("Low decay T20", ref.low_decay_t20_seconds, test.low_decay_t20_seconds, " s", 4)
    row("Low decay T40", ref.low_decay_t40_seconds, test.low_decay_t40_seconds, " s", 4)
    row("Centroid", ref.spectral_centroid_hz, test.spectral_centroid_hz, " Hz", 1)
    row("Rolloff(95%)", ref.spectral_rolloff_hz, test.spectral_rolloff_hz, " Hz", 1)
    row("Flatness", ref.spectral_flatness, test.spectral_flatness, "", 4)
    row("Spectral tilt", ref.spectral_tilt_db_per_decade, test.spectral_tilt_db_per_decade, " dB/dec", 3)
    row("Resonance prominence", ref.resonance_prominence_db, test.resonance_prominence_db, " dB", 2)
    row("Ring ratio", ref.ring_ratio, test.ring_ratio, "", 4)
    row("Median F0", ref.median_f0_hz, test.median_f0_hz, " Hz", 2)
    row("F0 drift", ref.f0_drift_cents_per_sec, test.f0_drift_cents_per_sec, " cents/s", 2)
    row("Click RMS", ref.click_rms_dbfs, test.click_rms_dbfs, " dBFS", 2)
    row("Body RMS", ref.body_rms_dbfs, test.body_rms_dbfs, " dBFS", 2)
    row("Click/Body", ref.click_to_body_db, test.click_to_body_db, " dB", 2)
    row("Early High/Low", ref.early_high_low_ratio_db, test.early_high_low_ratio_db, " dB", 2)
    row("H2/H1", ref.harmonic_h2_to_h1_db, test.harmonic_h2_to_h1_db, " dB", 2)
    row("H3/H1", ref.harmonic_h3_to_h1_db, test.harmonic_h3_to_h1_db, " dB", 2)
    row("H1.5/H1", ref.harmonic_h15_to_h1_db, test.harmonic_h15_to_h1_db, " dB", 2)

    lines.append("")
    lines.append("## 대역 레벨")
    lines.append("")
    lines.append("| Band | A (dB) | B (dB) | Delta |")
    lines.append("|---|---:|---:|---:|")
    for band in ["sub", "bass", "lowmid", "mid", "highmid", "air"]:
        a = ref.band_levels_db.get(band, 0.0)
        b = test.band_levels_db.get(band, 0.0)
        lines.append(f"| {band} | {a:.2f} | {b:.2f} | {b - a:+.2f} |")

    lines.append("")
    lines.append("## Formant-like Peaks")
    lines.append("")
    lines.append(f"- A: {', '.join(f'{x:.1f}Hz' for x in ref.formant_like_peaks_hz) if ref.formant_like_peaks_hz else 'n/a'}")
    lines.append(f"- B: {', '.join(f'{x:.1f}Hz' for x in test.formant_like_peaks_hz) if test.formant_like_peaks_hz else 'n/a'}")

    lines.append("")
    lines.append("## 개선 제안")
    lines.append("")
    for i, item in enumerate(suggestions, start=1):
        lines.append(f"{i}. **{item.get('focus', 'Focus')}**")
        lines.append(f"   - Finding: {item.get('finding', '')}")
        lines.append(f"   - Evidence: {item.get('evidence', '')}")
        actions = item.get("actions", [])
        if isinstance(actions, list):
            for action in actions:
                lines.append(f"   - Action: {action}")

    lines.append("")
    lines.append("## LLM 리뷰 프롬프트 연결")
    lines.append("")
    lines.append(
        "`references/analysis_prompt_ko.md`에 JSON 리포트를 붙여 넣으면, 지표 기반으로 DSP 수정 우선순위를 자동 제안받을 수 있습니다."
    )

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    if not is_power_of_two(args.fft_size):
        print("--fft-size must be a power of two", file=sys.stderr)
        return 2
    if args.hop_size <= 0:
        print("--hop-size must be positive", file=sys.stderr)
        return 2

    try:
        ref_data = read_wav_mono(args.ref)
        test_data = read_wav_mono(args.test)

        cfg = AnalysisConfig(
            fft_size=args.fft_size,
            hop_size=args.hop_size,
            max_seconds=args.max_seconds,
        )

        ref_samples, test_samples, ref_onset, test_onset, gain_offset_db, sample_rate = align_and_prepare(
            ref_data,
            test_data,
            align_onset=(not args.no_align_onset),
            loudness_match=(not args.no_loudness_match),
            max_seconds=args.max_seconds,
        )

        ref_result = analyze_signal(ref_samples, sample_rate, cfg)
        test_result = analyze_signal(test_samples, sample_rate, cfg)

        formant_shift = paired_formant_shift(ref_result.formant_like_peaks_hz, test_result.formant_like_peaks_hz)
        diff = {
            "rms_db_delta": test_result.rms_dbfs - ref_result.rms_dbfs,
            "peak_db_delta": test_result.peak_dbfs - ref_result.peak_dbfs,
            "attack_delta_seconds": decay_delta_seconds(ref_result.attack_seconds, test_result.attack_seconds),
            "low_decay_t20_delta_seconds": decay_delta_seconds(
                ref_result.low_decay_t20_seconds, test_result.low_decay_t20_seconds
            ),
            "low_decay_t40_delta_seconds": decay_delta_seconds(
                ref_result.low_decay_t40_seconds, test_result.low_decay_t40_seconds
            ),
            "centroid_delta_hz": test_result.spectral_centroid_hz - ref_result.spectral_centroid_hz,
            "rolloff_delta_hz": test_result.spectral_rolloff_hz - ref_result.spectral_rolloff_hz,
            "flatness_delta": test_result.spectral_flatness - ref_result.spectral_flatness,
            "spectral_tilt_delta_db_per_decade": (
                test_result.spectral_tilt_db_per_decade - ref_result.spectral_tilt_db_per_decade
            ),
            "resonance_delta_db": test_result.resonance_prominence_db - ref_result.resonance_prominence_db,
            "ring_ratio_delta": test_result.ring_ratio - ref_result.ring_ratio,
            "median_f0_delta_cents": f0_delta_cents(ref_result.median_f0_hz, test_result.median_f0_hz),
            "f0_drift_delta_cents_per_sec": (
                None
                if ref_result.f0_drift_cents_per_sec is None or test_result.f0_drift_cents_per_sec is None
                else test_result.f0_drift_cents_per_sec - ref_result.f0_drift_cents_per_sec
            ),
            "click_to_body_delta_db": (
                None
                if ref_result.click_to_body_db is None or test_result.click_to_body_db is None
                else test_result.click_to_body_db - ref_result.click_to_body_db
            ),
            "early_high_low_ratio_delta_db": (
                None
                if ref_result.early_high_low_ratio_db is None or test_result.early_high_low_ratio_db is None
                else test_result.early_high_low_ratio_db - ref_result.early_high_low_ratio_db
            ),
            "harmonic_h2_to_h1_delta_db": (
                None
                if ref_result.harmonic_h2_to_h1_db is None or test_result.harmonic_h2_to_h1_db is None
                else test_result.harmonic_h2_to_h1_db - ref_result.harmonic_h2_to_h1_db
            ),
            "harmonic_h3_to_h1_delta_db": (
                None
                if ref_result.harmonic_h3_to_h1_db is None or test_result.harmonic_h3_to_h1_db is None
                else test_result.harmonic_h3_to_h1_db - ref_result.harmonic_h3_to_h1_db
            ),
            "harmonic_h15_to_h1_delta_db": (
                None
                if ref_result.harmonic_h15_to_h1_db is None or test_result.harmonic_h15_to_h1_db is None
                else test_result.harmonic_h15_to_h1_db - ref_result.harmonic_h15_to_h1_db
            ),
            "formant_peak_shift_hz": formant_shift,
        }

        suggestions = build_suggestions(ref_result, test_result, gain_offset_db)

        report = {
            "config": {
                "fft_size": args.fft_size,
                "hop_size": args.hop_size,
                "max_seconds": args.max_seconds,
                "align_onset": not args.no_align_onset,
                "loudness_match": not args.no_loudness_match,
            },
            "inputs": {
                "reference": args.ref,
                "test": args.test,
                "analysis_sample_rate": sample_rate,
                "onset_offset_samples": {"reference": ref_onset, "test": test_onset},
                "gain_match_applied_db": gain_offset_db,
            },
            "reference": result_to_dict(ref_result),
            "test": result_to_dict(test_result),
            "difference": diff,
            "suggestions": suggestions,
        }

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        md_text = to_markdown(
            title=args.title,
            ref_path=args.ref,
            test_path=args.test,
            ref=ref_result,
            test=test_result,
            gain_offset_db=gain_offset_db,
            ref_onset=ref_onset,
            test_onset=test_onset,
            sample_rate=sample_rate,
            suggestions=suggestions,
        )

        if args.md_out:
            with open(args.md_out, "w", encoding="utf-8") as f:
                f.write(md_text)

        # stdout에도 요약을 출력해 터미널만으로 확인 가능하게 한다.
        print(md_text)

        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
