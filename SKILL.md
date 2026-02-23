---
name: audio-ear
description: Compare reference/test WAV files for synth or sampler tuning and produce actionable DSP fixes. Use when asked to diagnose low-end decay mismatch, pitch drift, resonance, ringing, formant shifts, or overall timbre gaps between A and B. Includes a fast ffmpeg/sox profile and a detailed FFT/STFT comparison pipeline.
---

# Audio Ear Skill

레퍼런스 A와 구현 B의 사운드를 수치 비교해 개선 포인트를 뽑는다.

## Workflow

1. 빠른 1차 진단(FFmpeg/SoX)
2. 정밀 2차 분석(FFT/STFT)
3. JSON 결과를 프롬프트에 넣어 수정 우선순위 도출

## 1) Quick Compare (FFmpeg/SoX)

```bash
python3 scripts/ffmpeg_sox_quick_compare.py \
  --ref /abs/path/A.wav \
  --test /abs/path/B.wav \
  --json-out /tmp/quick.json \
  --md-out /tmp/quick.md
```

용도:
- 레벨/밝기/트랜지언트 차이를 빠르게 확인
- 큰 편차가 있는지 먼저 걸러내기

## 2) Detailed Compare (FFT/STFT)

```bash
python3 scripts/wav_ear_compare.py \
  --ref /abs/path/A.wav \
  --test /abs/path/B.wav \
  --max-seconds 4.0 \
  --fft-size 2048 \
  --hop-size 512 \
  --json-out /tmp/ear_detail.json \
  --md-out /tmp/ear_detail.md
```

핵심 지표:
- 저역 감쇄(T20/T40)
- 스펙트럼 중심(centroid), rolloff, flatness, tilt
- resonance prominence, ring ratio
- median F0, F0 drift
- click/body 비율(초기 어택 강도)
- H2/H1, H3/H1, H1.5/H1(부분음/OSC2 유사 성분)
- formant-like peak

## 3) LLM 리뷰 연결

`references/analysis_prompt_ko.md` 템플릿에 `/tmp/quick.json`, `/tmp/ear_detail.json` 내용을 넣고 개선안을 요청한다.

## 4) Preset 실효 비교 (중요)

파라미터가 달라 보여도 엔진 우선순위 때문에 실제 음색 변화가 거의 없을 수 있다.
특히 `pitchEnv`가 있으면 `startFreq/endFreq/p_decay`는 `_schedulePitch()`에서 우선되지 않는다.

```bash
python3 scripts/preset_effective_compare.py \
  --a /abs/path/preset_before.json \
  --b /abs/path/preset_after.json \
  --json-out /tmp/preset_effective.json \
  --md-out /tmp/preset_effective.md
```

출력:
- Effective Changes: 실제 렌더에 반영되는 차이
- Ignored-Likely Changes: 바뀌어도 청감 영향이 작은/없는 필드

## 권장 캡처 규칙

- A/B를 동일 MIDI, 동일 길이, 동일 velocity로 렌더
- 가능한 dry 신호 기준으로 먼저 비교
- 레벨 편차가 크면 gain-match 후 해석
- 같은 샘플레이트를 권장(정밀 스크립트는 필요 시 리샘플링)

## 빠른 일괄 실행

```bash
bash scripts/run_ear_pipeline.sh /abs/path/A.wav /abs/path/B.wav /tmp/ear-run \
  --preset-a /abs/path/preset_before.json \
  --preset-b /abs/path/preset_after.json
```

출력:
- `quick.md`, `quick.json`
- `detail.md`, `detail.json`
- (옵션) `preset_effective.md`, `preset_effective.json`
