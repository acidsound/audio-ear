# audio-ear

Reference/test WAV A/B comparison toolkit for drum and synth sound design.

`audio-ear` helps you quantify what your ears hear:
- transient click/body balance
- low-end decay mismatch
- pitch center and pitch drift
- resonance/ring/formant shifts
- harmonic balance (including 1.5x partial often tied to OSC2-like character)

## Requirements

- Python 3.9+
- `ffmpeg`, `ffprobe`, `sox` in PATH (quick analysis)

## Quick Compare

```bash
python3 scripts/ffmpeg_sox_quick_compare.py \
  --ref /abs/path/A.wav \
  --test /abs/path/B.wav \
  --json-out /tmp/quick.json \
  --md-out /tmp/quick.md
```

## Detailed Compare

```bash
python3 scripts/wav_ear_compare.py \
  --ref /abs/path/A.wav \
  --test /abs/path/B.wav \
  --max-seconds 4.0 \
  --fft-size 2048 \
  --hop-size 512 \
  --json-out /tmp/detail.json \
  --md-out /tmp/detail.md
```

## Preset Effective Compare

```bash
python3 scripts/preset_effective_compare.py \
  --a /abs/path/preset_before.json \
  --b /abs/path/preset_after.json \
  --json-out /tmp/preset_effective.json \
  --md-out /tmp/preset_effective.md
```

## One-shot Pipeline

```bash
bash scripts/run_ear_pipeline.sh /abs/path/A.wav /abs/path/B.wav /tmp/ear-run \
  --preset-a /abs/path/preset_before.json \
  --preset-b /abs/path/preset_after.json
```

## Using With LLM Agents (Codex/Claude)

You can use `audio-ear` as an "objective ear" inside an agent loop:
1. compare reference vs current render
2. extract measurable gaps
3. modify synth params or DSP
4. re-render and re-compare

### Agent Prompt Template

```text
Use audio-ear to compare:
- ref: /abs/path/ref.wav
- test: /abs/path/current.wav
- preset before: /abs/path/preset_before.json
- preset after: /abs/path/preset_after.json

Tasks:
1) Run scripts/run_ear_pipeline.sh and collect quick/detail/preset_effective reports.
2) Summarize the top 5 actionable mismatches with exact metrics.
3) Propose concrete parameter changes (which field, direction, and expected acoustic effect).
4) Keep tonal balance priority over loudness.
```

### Example: Codex/Claude Iteration Prompt

```text
Compare /tmp/ref_tom_h.wav and /tmp/acidbros_ht_factory.wav using audio-ear.
Then tune HT to match these goals:
- stronger F-like body (1.5x partial) but avoid octave-up dominance
- keep wood/ring character
- reduce dull/dark feel without harsh click

Output:
- a short metric table (before/after)
- exact preset changes you applied
- paths to generated WAV and report files
```

### Example: Stop Conditions For Auto-Tuning

```text
Repeat render+compare up to 10 iterations, stop early when all are true:
- |median_f0_delta_cents| <= 15
- click_to_body_delta_db between -1.0 and +1.0
- |harmonic_h15_to_h1_delta_db| <= 3.0
- air band delta >= -3 dB
```
