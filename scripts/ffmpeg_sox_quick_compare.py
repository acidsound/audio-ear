#!/usr/bin/env python3
"""
FFmpeg/SoX Quick Compare

FFT 정밀 분석 전에 빠르게 A/B 차이를 스캔하는 경량 프로파일러.
- ffprobe: 포맷/샘플레이트/채널 메타
- ffmpeg astats: RMS/Peak/crest/entropy 등 전체 통계
- ffmpeg aspectralstats: centroid/flatness/flux/slope/rolloff 프레임 통계
- sox stat (옵션): rough frequency, amplitude 기반 보조 지표
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Quick A/B WAV compare using ffmpeg + sox")
    parser.add_argument("--ref", required=True, help="Reference WAV path (A)")
    parser.add_argument("--test", required=True, help="Test WAV path (B)")
    parser.add_argument("--json-out", help="Write JSON report")
    parser.add_argument("--md-out", help="Write Markdown report")
    parser.add_argument("--title", default="FFmpeg/SoX Quick Audio Compare")
    return parser.parse_args()


def require_bin(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise RuntimeError(f"Required binary not found: {name}")
    return path


def run(cmd: List[str], allow_fail: bool = False) -> subprocess.CompletedProcess:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0 and not allow_fail:
        raise RuntimeError(
            f"Command failed ({proc.returncode}): {' '.join(cmd)}\nSTDERR:\n{proc.stderr.strip()}"
        )
    return proc


def parse_last_float(label: str, text: str) -> Optional[float]:
    pattern = re.compile(rf"{re.escape(label)}:\s*([-+]?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)")
    vals = [float(m.group(1)) for m in pattern.finditer(text)]
    return vals[-1] if vals else None


def parse_ffprobe(path: str) -> Dict[str, object]:
    proc = run([
        "ffprobe",
        "-v",
        "error",
        "-show_streams",
        "-show_format",
        "-of",
        "json",
        path,
    ])
    data = json.loads(proc.stdout)

    stream = None
    for st in data.get("streams", []):
        if st.get("codec_type") == "audio":
            stream = st
            break

    fmt = data.get("format", {})
    return {
        "format_name": fmt.get("format_name"),
        "duration": _as_float(fmt.get("duration")),
        "bit_rate": _as_float(fmt.get("bit_rate")),
        "sample_rate": _as_float(stream.get("sample_rate")) if stream else None,
        "channels": stream.get("channels") if stream else None,
        "bits_per_sample": stream.get("bits_per_sample") if stream else None,
        "codec_name": stream.get("codec_name") if stream else None,
    }


def _as_float(v: object) -> Optional[float]:
    if v is None:
        return None
    try:
        return float(v)
    except Exception:
        return None


def parse_astats(path: str) -> Dict[str, Optional[float]]:
    proc = run(
        [
            "ffmpeg",
            "-hide_banner",
            "-nostats",
            "-i",
            path,
            "-af",
            "astats=metadata=1:reset=0",
            "-f",
            "null",
            "-",
        ],
        allow_fail=False,
    )
    log = proc.stderr

    return {
        "peak_level_db": parse_last_float("Peak level dB", log),
        "rms_level_db": parse_last_float("RMS level dB", log),
        "crest_factor": parse_last_float("Crest factor", log),
        "dynamic_range": parse_last_float("Dynamic range", log),
        "zero_crossing_rate": parse_last_float("Zero crossings rate", log),
        "entropy": parse_last_float("Entropy", log),
    }


def parse_aspectralstats(path: str) -> Dict[str, Optional[float]]:
    with tempfile.NamedTemporaryFile(prefix="aspectral_", suffix=".txt", delete=False) as tmp:
        meta_file = tmp.name

    try:
        filt = (
            "aspectralstats=win_size=2048:overlap=0.75:"
            "measure=centroid+flatness+flux+slope+rolloff,"
            f"ametadata=mode=print:file={meta_file}"
        )
        run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                path,
                "-af",
                filt,
                "-f",
                "null",
                "-",
            ],
            allow_fail=False,
        )

        metrics: Dict[str, List[float]] = {
            "centroid": [],
            "flatness": [],
            "flux": [],
            "slope": [],
            "rolloff": [],
        }
        with open(meta_file, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line.startswith("lavfi.aspectralstats.1."):
                    continue
                key_val = line.split("=", 1)
                if len(key_val) != 2:
                    continue
                key = key_val[0].split(".")[-1]
                if key not in metrics:
                    continue
                try:
                    metrics[key].append(float(key_val[1]))
                except ValueError:
                    continue

        out: Dict[str, Optional[float]] = {}
        for key, vals in metrics.items():
            if vals:
                out[f"{key}_mean"] = statistics.fmean(vals)
                out[f"{key}_median"] = statistics.median(vals)
                out[f"{key}_p95"] = _percentile(vals, 95.0)
            else:
                out[f"{key}_mean"] = None
                out[f"{key}_median"] = None
                out[f"{key}_p95"] = None
        out["frames"] = len(metrics["centroid"])
        return out
    finally:
        try:
            os.unlink(meta_file)
        except OSError:
            pass


def _percentile(vals: List[float], q: float) -> Optional[float]:
    if not vals:
        return None
    sorted_vals = sorted(vals)
    if len(sorted_vals) == 1:
        return sorted_vals[0]
    pos = (len(sorted_vals) - 1) * (q / 100.0)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_vals[lo]
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def parse_sox_stat(path: str) -> Dict[str, Optional[float]]:
    if not shutil.which("sox"):
        return {"available": False}

    proc = run(["sox", path, "-n", "stat"], allow_fail=False)
    log = proc.stderr

    return {
        "available": True,
        "max_amplitude": _parse_sox_value("Maximum amplitude", log),
        "min_amplitude": _parse_sox_value("Minimum amplitude", log),
        "rms_amplitude": _parse_sox_value("RMS     amplitude", log),
        "rough_frequency": _parse_sox_value("Rough   frequency", log),
        "volume_adjustment": _parse_sox_value("Volume adjustment", log),
    }


def _parse_sox_value(label: str, text: str) -> Optional[float]:
    pattern = re.compile(rf"^{re.escape(label)}:\s*([-+]?\d+(?:\.\d+)?)", re.MULTILINE)
    m = pattern.search(text)
    return float(m.group(1)) if m else None


def analyze_one(path: str) -> Dict[str, object]:
    return {
        "path": path,
        "ffprobe": parse_ffprobe(path),
        "astats": parse_astats(path),
        "aspectralstats": parse_aspectralstats(path),
        "sox": parse_sox_stat(path),
    }


def delta(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return b - a


def build_insights(ref: Dict[str, object], test: Dict[str, object]) -> List[str]:
    insights: List[str] = []

    r_ast = ref["astats"]  # type: ignore[index]
    t_ast = test["astats"]  # type: ignore[index]
    r_spec = ref["aspectralstats"]  # type: ignore[index]
    t_spec = test["aspectralstats"]  # type: ignore[index]

    rms_delta = delta(r_ast.get("rms_level_db"), t_ast.get("rms_level_db"))
    if rms_delta is not None and abs(rms_delta) > 1.0:
        insights.append(f"RMS 레벨 차이 큼: B-A {rms_delta:+.2f} dB (레벨 매칭 후 비교 권장)")

    centroid_delta = delta(r_spec.get("centroid_mean"), t_spec.get("centroid_mean"))
    if centroid_delta is not None:
        if centroid_delta > 200:
            insights.append(f"B가 더 밝음(centroid +{centroid_delta:.1f}Hz): cutoff/drive 과다 가능성")
        elif centroid_delta < -200:
            insights.append(f"B가 더 어두움(centroid {centroid_delta:.1f}Hz): cutoff/envelope 양 부족 가능성")

    flux_delta = delta(r_spec.get("flux_mean"), t_spec.get("flux_mean"))
    if flux_delta is not None and abs(flux_delta) > 0.003:
        if flux_delta > 0:
            insights.append("B의 스펙트럴 변화량(flux)이 더 큼: 트랜지언트가 거칠거나 모듈레이션이 큼")
        else:
            insights.append("B의 스펙트럴 변화량(flux)이 더 작음: 트랜지언트가 둔할 수 있음")

    crest_delta = delta(r_ast.get("crest_factor"), t_ast.get("crest_factor"))
    if crest_delta is not None and abs(crest_delta) > 1.0:
        if crest_delta > 0:
            insights.append("B의 crest factor가 높음: 피크가 더 튀고 바디가 얇을 수 있음")
        else:
            insights.append("B의 crest factor가 낮음: 컴프레션/포화로 다이내믹이 눌렸을 수 있음")

    r_sox = ref.get("sox", {})  # type: ignore[assignment]
    t_sox = test.get("sox", {})  # type: ignore[assignment]
    if isinstance(r_sox, dict) and isinstance(t_sox, dict):
        if r_sox.get("available") and t_sox.get("available"):
            rf_delta = delta(r_sox.get("rough_frequency"), t_sox.get("rough_frequency"))
            if rf_delta is not None and abs(rf_delta) > 20:
                insights.append(
                    f"SoX rough frequency 차이 {rf_delta:+.1f}Hz: 피치/공진 중심 이동 가능성"
                )

    if not insights:
        insights.append("큰 편차 없음. 다음 단계로 정밀 FFT 비교(wav_ear_compare.py) 권장")
    return insights


def fmt(v: Optional[float], digits: int = 3) -> str:
    if v is None:
        return "n/a"
    return f"{v:.{digits}f}"


def render_md(title: str, ref: Dict[str, object], test: Dict[str, object], insights: List[str]) -> str:
    r_ast = ref["astats"]  # type: ignore[index]
    t_ast = test["astats"]  # type: ignore[index]
    r_spec = ref["aspectralstats"]  # type: ignore[index]
    t_spec = test["aspectralstats"]  # type: ignore[index]

    lines: List[str] = [f"# {title}", ""]
    lines.append(f"- A: `{ref['path']}`")
    lines.append(f"- B: `{test['path']}`")
    lines.append("")

    lines.append("## Quick Metrics")
    lines.append("")
    lines.append("| Metric | A | B | Delta (B-A) |")
    lines.append("|---|---:|---:|---:|")

    def row(name: str, a: Optional[float], b: Optional[float], digits: int = 3):
        d = delta(a, b)
        d_text = "n/a" if d is None else f"{d:+.{digits}f}"
        lines.append(f"| {name} | {fmt(a, digits)} | {fmt(b, digits)} | {d_text} |")

    row("RMS level dB", r_ast.get("rms_level_db"), t_ast.get("rms_level_db"), 2)
    row("Peak level dB", r_ast.get("peak_level_db"), t_ast.get("peak_level_db"), 2)
    row("Crest factor", r_ast.get("crest_factor"), t_ast.get("crest_factor"), 2)
    row("Entropy", r_ast.get("entropy"), t_ast.get("entropy"), 4)
    row("Centroid mean", r_spec.get("centroid_mean"), t_spec.get("centroid_mean"), 1)
    row("Rolloff mean", r_spec.get("rolloff_mean"), t_spec.get("rolloff_mean"), 1)
    row("Flatness mean", r_spec.get("flatness_mean"), t_spec.get("flatness_mean"), 4)
    row("Flux mean", r_spec.get("flux_mean"), t_spec.get("flux_mean"), 5)
    row("Slope mean", r_spec.get("slope_mean"), t_spec.get("slope_mean"), 6)

    lines.append("")
    lines.append("## Insights")
    lines.append("")
    for i, ins in enumerate(insights, 1):
        lines.append(f"{i}. {ins}")

    lines.append("")
    lines.append("## Next")
    lines.append("")
    lines.append("정밀 분석은 `wav_ear_compare.py`로 이어서 실행하세요.")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    try:
        require_bin("ffprobe")
        require_bin("ffmpeg")

        ref = analyze_one(args.ref)
        test = analyze_one(args.test)
        insights = build_insights(ref, test)

        report = {
            "reference": ref,
            "test": test,
            "insights": insights,
        }

        if args.json_out:
            with open(args.json_out, "w", encoding="utf-8") as f:
                json.dump(report, f, ensure_ascii=False, indent=2)

        md = render_md(args.title, ref, test, insights)
        if args.md_out:
            with open(args.md_out, "w", encoding="utf-8") as f:
                f.write(md)

        print(md)
        return 0
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
