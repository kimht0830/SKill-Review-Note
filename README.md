# Contrastive Skill Analysis

같은 task에서 나온 **실패 trajectory와 성공 trajectory를 대조**해 성공/실패를 가른 원인을 찾고, 그로부터 skill 규칙을 뽑는 실험 코드입니다.
현재 단계는 **쌍 하나를 분석하는 analyst 프롬프트**가 Qwen3.5-9B에서 어떤 결과를 내는지 확인하는 것입니다.

## 구성

```
prompts/contrastive_analyst.md      analyst system prompt
scripts/build_pairs.py              (로컬) baseline 결과 → data/pairs/*.jsonl, data/images/
scripts/run_analyst.py              (Colab) vLLM 서버에 analyst 프롬프트 적용 → outputs/<tag>/
notebooks/run_contrastive_analyst.ipynb   Colab 실행 노트북
data/pairs/<benchmark>.jsonl        미리 렌더링된 analyst 입력 (쌍 하나당 한 줄)
data/images/                        DocVQA 쌍의 문서 이미지
```

## 쌍 정의

no-skill baseline(no-PI)과 reference answer를 준 run(PI)을 task 단위로 비교해, 결과가 다른 task를 쌍으로 만듭니다.

| source | 실패 trajectory | 성공 trajectory |
|---|---|---|
| `hindsight` | no-PI 실패 | PI 성공 (정답을 보고 풂) |
| `natural` | PI 실패 | no-PI 성공 (추가 정보 없음) |

현재 쌍 수 (`data/pairs/summary.json`):

| benchmark | hindsight | natural |
|---|---|---|
| SearchQA | 339 | 7 |
| DocVQA | 46 | 3 |
| OfficeQA | 49 | 4 |
| LiveMath | 65 | 2 |
| SpreadsheetBench | 49 | 29 |

## analyst 입력 형식

```
## Current Skill            (지금은 empty)
## Task                     두 trajectory가 공통으로 받은 입력 (reference answer 헤더 제거)
## FAILED trajectory        최종 답 / WRONG, [F1] [F2] ... step
## SUCCESSFUL trajectory    source 태그, [S1] [S2] ... step
```

- step 단위: tool 호출 1회 = 1 step, 긴 텍스트 응답은 문단 단위 (250자 미만 문단은 병합)
- tool output은 앞 800자 + 뒤 400자만 남김
- QA 벤치마크의 채점 메시지(gold answer 포함)는 제거
- SpreadsheetBench 실행 피드백은 **틀린 셀 위치와 실제 출력값만** 남기고 기대값은 `<hidden>` 처리 (`--ss-feedback full`로 기대값 포함 가능)
- DocVQA는 문서 이미지를 함께 입력

## 실행

1. 로컬에서 쌍 생성 (raw 결과가 바뀌었을 때만):
   ```bash
   python scripts/build_pairs.py --results-root "<...>/MyExperiment/results/no-skill_baseline"
   ```
2. 레포를 GitHub에 push
3. Colab에서 `notebooks/run_contrastive_analyst.ipynb`를 열고 1번 셀의 `REPO_URL`을 수정한 뒤 순서대로 실행 (A100 권장)

## 출력

`outputs/<tag>/<benchmark>.jsonl`에 쌍마다 `raw`, `parsed`, `json_ok`, 토큰 수가 기록되고, `summary.md`에 벤치마크와 source별 JSON 파싱률, recoverable 비율, 규칙 예시가 정리됩니다.
