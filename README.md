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
