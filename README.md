# Contrastive Skill Analysis

같은 task에서 나온 **실패 trajectory와 성공 trajectory를 대조**해 성공/실패를 가른 원인을 찾고, 그로부터 skill 규칙(rule)을 뽑은 뒤, 그 rule이 타당한지 확인하는 실험 코드입니다.
target(문제를 푸는 agent)과 analyst(rule을 만드는 optimizer)는 모두 `Qwen/Qwen3.5-9B`입니다.

## 파이프라인 (`notebooks/run_contrastive_analyst.ipynb`, Colab)

| 단계 | 스크립트 | 내용 |
|---|---|---|
| 1 | `scripts/run_rollouts.py` | SkillOpt train split을 빈 skill로 풂: no-PI / PI(정답을 user turn 앞에 줌) |
| 2 | `scripts/build_pairs.py` | 두 결과가 갈린 문항을 쌍으로 묶고 analyst 입력을 렌더링 |
| 3 | `scripts/run_analyst.py` | 쌍마다 `prompts/contrastive_analyst.md`로 원인 분석 + rule 생성 |
| 4 | `scripts/validate_rules.py` | 실패했던 쪽과 같은 조건에서 system prompt에 rule만 추가해 다시 풀고, 빈 skill 대조군보다 많이 맞히는지 확인 |

결과는 모두 Google Drive `contrastive_skill/<EXP>/`에 저장되고, 각 단계는 끝난 문항/쌍/job을 건너뛰며 이어서 실행됩니다.
`scripts/render_html.py`는 analyst 결과를 `report.html` 뷰어로 만듭니다.

## 쌍 정의

| source | 실패 trajectory | 성공 trajectory | validity check 조건 |
|---|---|---|---|
| `hindsight` | no-PI 실패 | PI 성공 (정답을 보고 풂) | no-PI + rule |
| `natural` | PI 실패 | no-PI 성공 (추가 정보 없음) | PI + rule |

## analyst 입력 형식

```
## Current Skill            (지금은 empty)
## Agent instructions       agent의 system prompt (두 trajectory 공통)
## Task                     두 trajectory가 공통으로 받은 입력 (PI 정답 헤더 제거)
## FAILED trajectory        최종 답 / WRONG, [F1] [F2] ... step
## SUCCESSFUL trajectory    source 태그, [S1] [S2] ... step
```

- step 단위: tool 호출 1회 = 1 step, 긴 텍스트 응답은 문단 단위 (250자 미만 문단은 병합)
- tool output은 앞 800자 + 뒤 400자만 남김
- QA 벤치마크의 채점 메시지(gold answer 포함)는 제거
- SpreadsheetBench 실행 피드백은 **틀린 셀 위치와 실제 출력값만** 남기고 기대값은 `<hidden>` 처리 (`--ss-feedback full`로 기대값 포함 가능)
- DocVQA는 문서 이미지를 함께 입력

## 구성

```
prompts/contrastive_analyst.md      analyst system prompt
scripts/run_rollouts.py             1단계 rollout (skillopt_harness 실행)
scripts/build_pairs.py              2단계 쌍 생성
scripts/run_analyst.py              3단계 analyst (vLLM OpenAI 호환 서버)
scripts/validate_rules.py           4단계 rule validity check
scripts/render_html.py              analyst 결과 → report.html
notebooks/run_contrastive_analyst.ipynb   Colab 실행 노트북
skillopt_harness/                   SkillOpt(commit fa4ca18, MIT)에서 평가에 필요한 부분만 옮겨 온 코드
  skillopt/, configs/, data/*_id_split/, scripts/eval_only.py   원본 그대로
  scripts/materialize_*.py          데이터 준비 (searchqa는 원본, 나머지는 이전 baseline 노트북에서 옮겨 옴)
  scripts/eval_only_pi.py, pi_config.json   PI 주입 wrapper와 템플릿 (이전 PI baseline 노트북에서 옮겨 옴)
  scripts/eval_jobs.py              eval_only.py를 (task, skill) job 단위로 돌리는 wrapper (validity check용)
```

## 실행

1. 레포를 GitHub에 push
2. Colab에서 `notebooks/run_contrastive_analyst.ipynb`를 열고 1번 셀의 `REPO_URL`을 수정한 뒤 순서대로 실행 (A100 권장, OfficeQA용 `HF_TOKEN` Secret 필요)

이전 baseline 결과(로컬, test split)로 쌍을 만들 때:
```bash
python scripts/build_pairs.py --out-dir <out> \
  --nopi-root "<...>/no-skill_baseline/qwen3.5-9B/run1" --nopi-fmt "{prefix}_noskill_full" \
  --pi-root "<...>/no-skill_baseline/qwen3.5-9B-PI/run1" --pi-fmt "{prefix}_noskill_pi_full"
```
