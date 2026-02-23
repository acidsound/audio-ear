#!/usr/bin/env python3
"""
Preset Effective Compare

두 프리셋 JSON을 비교하되, 엔진의 실제 처리 우선순위를 반영해
"실제로 들리는 차이"와 "엔진에서 사실상 무시되는 차이"를 분리해서 보여준다.
"""

from __future__ import annotations

import argparse
import json
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

EPS = 1e-12


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Compare two preset JSONs by engine-effective parameters")
    p.add_argument("--a", required=True, help="Preset A JSON path (before)")
    p.add_argument("--b", required=True, help="Preset B JSON path (after)")
    p.add_argument("--json-out", help="Write JSON report")
    p.add_argument("--md-out", help="Write Markdown report")
    p.add_argument("--title", default="Preset Effective Compare")
    return p.parse_args()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"Preset root must be object: {path}")
    # Allow wrapper objects produced by tooling: { "preset": { ... }, "mod": { ... } }
    if isinstance(data.get("preset"), dict):
        return data["preset"]
    return data


def _norm_wave(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    w = value.strip().lower()
    if w in ("tri", "triangle"):
        return "triangle"
    if w in ("sqr", "square"):
        return "square"
    if w in ("sin", "sine"):
        return "sine"
    return w


def _norm_filter(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    t = value.strip().lower()
    if t in ("lpf", "lowpass"):
        return "lowpass"
    if t in ("hpf", "highpass"):
        return "highpass"
    if t in ("bpf", "bandpass"):
        return "bandpass"
    return t


def _num(v: Any) -> Optional[float]:
    if isinstance(v, (int, float)) and math.isfinite(v):
        return float(v)
    return None


def _bool(v: Any) -> Optional[bool]:
    if isinstance(v, bool):
        return v
    return None


def _get(d: Dict[str, Any], path: str) -> Any:
    cur: Any = d
    for key in path.split("."):
        if not isinstance(cur, dict) or key not in cur:
            return None
        cur = cur[key]
    return cur


def _extract_effective(preset: Dict[str, Any]) -> Tuple[Dict[str, Any], List[Dict[str, str]]]:
    eff: Dict[str, Any] = {}
    ignored_fields: List[Dict[str, str]] = []

    for osc_id in ("osc1", "osc2", "osc3", "osc4"):
        osc = preset.get(osc_id)
        if not isinstance(osc, dict):
            continue

        eff[f"{osc_id}.enabled"] = _bool(osc.get("enabled"))
        eff[f"{osc_id}.wave"] = _norm_wave(osc.get("wave"))
        eff[f"{osc_id}.freq"] = _num(osc.get("freq"))
        eff[f"{osc_id}.level"] = _num(osc.get("level"))
        eff[f"{osc_id}.a_decay"] = _num(osc.get("a_decay"))
        eff[f"{osc_id}.drive"] = _num(osc.get("drive"))
        eff[f"{osc_id}.staticLevel"] = _bool(osc.get("staticLevel"))
        eff[f"{osc_id}.noAttack"] = _bool(osc.get("noAttack"))

        penv = osc.get("pitchEnv")
        if isinstance(penv, dict):
            # Engine behavior: pitchEnv exists -> schedulePitch() uses pitchEnv branch,
            # so startFreq/endFreq/p_decay are effectively ignored for pitch trajectory.
            eff[f"{osc_id}.pitch.mode"] = "pitchEnv"
            eff[f"{osc_id}.pitch.startMultiplier"] = _num(penv.get("startMultiplier"))
            eff[f"{osc_id}.pitch.cvTargetRatio"] = _num(penv.get("cvTargetRatio"))
            eff[f"{osc_id}.pitch.cvDecay"] = _num(penv.get("cvDecay"))
            eff[f"{osc_id}.pitch.dropDelay"] = _num(penv.get("dropDelay"))
            eff[f"{osc_id}.pitch.dropRatio"] = _num(penv.get("dropRatio"))
            eff[f"{osc_id}.pitch.dropTime"] = _num(penv.get("dropTime"))
            for field in ("startFreq", "endFreq", "p_decay"):
                ignored_fields.append(
                    {
                        "path": f"{osc_id}.{field}",
                        "reason": "pitchEnv is present, so this field is bypassed in _schedulePitch()",
                    }
                )
        else:
            eff[f"{osc_id}.pitch.mode"] = "legacy"
            eff[f"{osc_id}.pitch.startFreq"] = _num(osc.get("startFreq"))
            eff[f"{osc_id}.pitch.endFreq"] = _num(osc.get("endFreq"))
            eff[f"{osc_id}.pitch.p_decay"] = _num(osc.get("p_decay"))

    for sec in ("click", "snap", "noise", "noise2", "masterLowShelf", "masterPeak", "masterHighShelf", "masterEnv", "shaper", "tomMacros"):
        obj = preset.get(sec)
        if not isinstance(obj, dict):
            continue
        for k in sorted(obj.keys()):
            path = f"{sec}.{k}"
            v = obj[k]
            if k == "filter_type":
                eff[path] = _norm_filter(v)
            elif k == "wave":
                eff[path] = _norm_wave(v)
            elif isinstance(v, bool):
                eff[path] = v
            elif isinstance(v, (int, float)) and math.isfinite(v):
                eff[path] = float(v)
            elif isinstance(v, str):
                eff[path] = v

    if isinstance(preset.get("vol"), (int, float)):
        eff["vol"] = float(preset["vol"])
    if isinstance(preset.get("accent"), (int, float)):
        eff["accent"] = float(preset["accent"])

    return eff, ignored_fields


def _value_changed(a: Any, b: Any, tol: float = 1e-9) -> bool:
    if a is None and b is None:
        return False
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return abs(float(a) - float(b)) > tol
    return a != b


def _num_delta(a: Any, b: Any) -> Optional[float]:
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        return float(b) - float(a)
    return None


def _weight_for_key(key: str) -> float:
    weighted = {
        ".enabled": 8.0,
        ".wave": 4.0,
        ".freq": 6.0,
        ".level": 6.0,
        ".a_decay": 5.0,
        ".drive": 4.0,
        "pitch.startMultiplier": 8.0,
        "pitch.cvDecay": 6.0,
        "pitch.dropRatio": 6.0,
        "pitch.dropTime": 5.0,
        "click.level": 5.0,
        "click.decay": 4.0,
        "noise.level": 4.0,
        "noise.cutoff": 4.0,
        "masterPeak.gain": 4.0,
        "masterPeak.Q": 3.0,
        "masterHighShelf.gain": 3.0,
        "masterLowShelf.gain": 3.0,
        "vol": 4.0,
    }
    for tail, w in weighted.items():
        if key.endswith(tail) or tail in key:
            return w
    return 1.0


def _audible_score(changes: Sequence[Dict[str, Any]]) -> float:
    score = 0.0
    for c in changes:
        key = c["key"]
        a = c.get("a")
        b = c.get("b")
        w = _weight_for_key(key)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            base = max(abs(float(a)), 1.0)
            mag = min(1.0, abs(float(b) - float(a)) / base)
            score += w * mag
        else:
            score += w
    return score


def _compare_raw_ignored(a_raw: Dict[str, Any], b_raw: Dict[str, Any], ignored: Sequence[Dict[str, str]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    seen = set()
    for item in ignored:
        path = item["path"]
        if path in seen:
            continue
        seen.add(path)
        va = _get(a_raw, path)
        vb = _get(b_raw, path)
        if _value_changed(va, vb):
            out.append(
                {
                    "key": path,
                    "a": va,
                    "b": vb,
                    "delta": _num_delta(va, vb),
                    "reason": item["reason"],
                }
            )
    return out


def _to_markdown(
    title: str,
    a_path: str,
    b_path: str,
    changes: Sequence[Dict[str, Any]],
    ignored_changes: Sequence[Dict[str, Any]],
    score: float,
) -> str:
    lines: List[str] = []
    lines.append(f"# {title}")
    lines.append("")
    lines.append(f"- A: `{a_path}`")
    lines.append(f"- B: `{b_path}`")
    lines.append(f"- Effective audible change score: `{score:.2f}`")
    lines.append("")

    if score < 1.0:
        lines.append("요약: 실효 파라미터 차이가 매우 작아 청감 차이가 미미할 가능성이 큽니다.")
    elif score < 4.0:
        lines.append("요약: 실효 파라미터 차이가 작아서 미세한 차이만 날 가능성이 큽니다.")
    else:
        lines.append("요약: 실효 파라미터 차이가 충분해 청감 차이가 날 가능성이 큽니다.")
    lines.append("")

    lines.append("## Effective Changes")
    lines.append("")
    if changes:
        lines.append("| Key | A | B | Delta |")
        lines.append("|---|---:|---:|---:|")
        for c in changes:
            d = c.get("delta")
            delta_txt = f"{d:+.6g}" if isinstance(d, (int, float)) else "-"
            lines.append(f"| {c['key']} | {c.get('a')} | {c.get('b')} | {delta_txt} |")
    else:
        lines.append("- None")
    lines.append("")

    lines.append("## Ignored-Likely Changes")
    lines.append("")
    if ignored_changes:
        lines.append("| Key | A | B | Why ignored |")
        lines.append("|---|---:|---:|---|")
        for c in ignored_changes:
            lines.append(f"| {c['key']} | {c.get('a')} | {c.get('b')} | {c.get('reason')} |")
    else:
        lines.append("- None")

    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()

    a_raw = _load_json(args.a)
    b_raw = _load_json(args.b)

    a_eff, a_ignored = _extract_effective(a_raw)
    b_eff, b_ignored = _extract_effective(b_raw)
    ignored_catalog = a_ignored + b_ignored

    keys = sorted(set(a_eff.keys()) | set(b_eff.keys()))
    changes: List[Dict[str, Any]] = []
    for key in keys:
        va = a_eff.get(key)
        vb = b_eff.get(key)
        if _value_changed(va, vb):
            changes.append(
                {
                    "key": key,
                    "a": va,
                    "b": vb,
                    "delta": _num_delta(va, vb),
                }
            )

    ignored_changes = _compare_raw_ignored(a_raw, b_raw, ignored_catalog)
    score = _audible_score(changes)

    report = {
        "inputs": {"a": args.a, "b": args.b},
        "effective_change_score": score,
        "effective_changes": changes,
        "ignored_likely_changes": ignored_changes,
    }

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)

    md = _to_markdown(args.title, args.a, args.b, changes, ignored_changes, score)
    if args.md_out:
        with open(args.md_out, "w", encoding="utf-8") as f:
            f.write(md)

    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
