# audio-ear

언어: [English](README.md) | **한국어**

드럼/신스 사운드 디자인을 위한 레퍼런스/테스트 WAV A/B 비교 툴킷입니다.

`audio-ear`는 귀로 느끼는 차이를 수치로 정리해줍니다.
- 트랜지언트 click/body 밸런스
- 저역 감쇄(Decay) 불일치
- 피치 중심과 시간축 피치 드리프트
- resonance/ring/formant 변화
- 부분음 밸런스(특히 OSC2 성격과 연결되는 1.5배 성분)

## 요구 사항

- Python 3.9+
- `ffmpeg`, `ffprobe`, `sox` 설치 및 PATH 등록(Quick 분석에 필요)

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

## LLM Agent 연동 (Codex/Claude)

`audio-ear`를 에이전트 루프의 "객관적 귀"로 사용할 수 있습니다.
1. 레퍼런스와 현재 렌더 비교
2. 정량 지표로 차이 추출
3. 파라미터/DSP 수정
4. 재렌더 후 재비교

### 에이전트 프롬프트 템플릿

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

### 예시: Codex/Claude 반복 튜닝 프롬프트

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

### 예시: 자동 최적화 중단 조건

```text
Repeat render+compare up to 10 iterations, stop early when all are true:
- |median_f0_delta_cents| <= 15
- click_to_body_delta_db between -1.0 and +1.0
- |harmonic_h15_to_h1_delta_db| <= 3.0
- air band delta >= -3 dB
```
