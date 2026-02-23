# Audio Ear Review Prompt (KO)

아래 JSON 리포트를 기반으로, 레퍼런스 A와 구현 B의 음색 차이를 DSP 수정 계획으로 변환하라.

## 입력

- `quick_report_json`: `ffmpeg_sox_quick_compare.py` 결과
- `detail_report_json`: `wav_ear_compare.py` 결과
- (선택) 관련 코드 파일/파라미터 정의

## 출력 형식

1. **핵심 편차 3개 이내**
- 각 항목에 수치 근거를 붙인다. (예: `low_decay_t20 delta = -38ms`)
- 지표 간 충돌이 있으면 신뢰도 낮음으로 표시한다.

2. **원인 가설 (우선순위 순)**
- 엔벨로프, 필터, 공진, 비선형(드라이브), 리샘플링/튜닝 중 어디가 가장 가능성이 높은지 제시한다.
- 가설마다 "왜 그렇게 보는지"를 숫자와 함께 1문장으로 적는다.

3. **수정안 (작게 시작)**
- 각 가설에 대해 최대 2개의 최소 변경안을 제시한다.
- 파라미터 변경은 범위로 제시한다. (예: `decay +10~20%`, `Q -0.1~-0.25`)
- 구현 변경이 필요하면 pseudocode 5~10줄 이내로 제시한다.

4. **검증 루프**
- 다음 비교에서 확인할 지표와 통과 기준을 수치로 제시한다.
- 예: `|low_decay_t20 delta| < 15ms`, `|centroid delta| < 120Hz`

## 제약

- 모호한 표현("조금", "느낌상") 금지.
- 반드시 수치 -> 가설 -> 수정안을 연결한다.
- 빠른 실험 순서로 정렬한다.

## 실행용 템플릿

```text
[Context]
- 목표 사운드: (예: TR-909 clap 레퍼런스)
- 현재 문제: (예: B가 초반 저역이 빨리 죽음)

[quick_report_json]
{{quick_report_json}}

[detail_report_json]
{{detail_report_json}}

[Request]
위 형식(핵심 편차/원인/수정안/검증 루프)으로 답하라.
```
