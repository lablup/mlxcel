# PR #2553: Activity decode 오버헤드를 5쌍이 아니라 구간으로 판정

**날짜**: 2026-10-23
**상태**: PR 열림; Closes #1925

## 요약

`WebUI installed artifact` 잡의 Activity 성능 게이트는 모드마다 off/on을 번갈아 5쌍 측정하고, 쌍별 성능 저하의 중앙값이 2% 예산을 넘거나 baseline CV가 5%를 넘으면 그 모드를 실패시켰다. 자체 호스팅 GB10 러너 `lablup-dgxspark21`에서는 쌍별 노이즈(SD 3.6 포인트, kurtosis 12~25) 때문에 코드가 바뀌지 않아도 두 조건이 모두 발동한다. #1925와 그 코멘트에는 decode에 영향을 줄 수 없는 PR에서 실패한 시도 10건이 기록되어 있고, 그중 하나는 Activity arm이 6.76% 더 빨랐는데도 CV 때문에 실패했다. PR #2553은 이 규칙을 순차적이고 분포 가정이 없는 검정으로 바꾼다. 점추정은 쌍별 저하의 중앙값, 구간은 98% sign-test 구간 `[x(k), x(n+1-k)]`이다. 각 모드는 12쌍에서 시작해 2쌍씩 최대 40쌍까지 늘리며, 구간이 `investigate`(구간 전체가 2% 위), `within-target`(상한이 4% 이하) 중 하나로 결정될 때까지 측정한다. 상한에 도달해도 결정되지 않으면 `inconclusive`가 되고, 측정 노이즈를 명시하는 메시지와 함께 실패한다. 모든 상수는 JavaScript 하네스와 Python 포트가 함께 읽는 정책 파일 하나로 옮겼다. verifier는 원시 샘플에서 모든 모드의 판정을 다시 도출하고, 두 언어 구현은 기록된 샘플 벡터로 만든 공유 fixture 하나로 테스트된다. 기록된 636개 실행을 리샘플링한 추정에서 실제 변화가 0%일 때 오탐 실패율은 모드당 0.21%, 실제 저하가 5% 이상일 때 검출률은 95% 이상이다. 러너에서는 변경되지 않은 트리로 다섯 번 연속 통과했고, 일회용 PR에 주입한 slowdown은 두 모드 모두 `investigate`로 실패했다. 이슈가 제안한 주입 방식(CPU burst, Activity poll에 묶은 지연)은 GB10에서 decode에 전혀 닿지 않았다. 15개 파일, +3995/-120이며 그중 3278줄이 fixture다.

## 1. 문제 정의

게이트의 판정은 `webui/scripts/activity-performance.mjs`에 있었다. 모드당 5쌍, `status: degradation > 2 || cv > 5 ? 'investigate' : 'within-target'`. 같은 규칙이 임계값과 개수를 각자 복사한 채 두 군데에서 더 적용되었다. `scripts/webui/verify_activity_performance.py`는 `paired_runs == 5`를 요구하고 `degradation > 2 or cv > 5`를 다시 적용했으며 모드별 샘플 수와 preflight 수를 하드코딩했다. `scripts/webui/summarize_activity_evidence.py`는 status, `paired_runs == 5`, `<= 2`, `<= 5` 검사를 반복했다.

이 규칙에서 두 가지 결함이 나왔다.

- **5쌍으로는 2%를 구분할 수 없다.** 실패한 실행들의 쌍별 범위는 6~21 포인트였다. 변화가 없는 변경도 중앙값 임계를 넘었다. PR #1920(표시 이름 문자열만 변경)은 one-visible +3.38%로, PR #2279(YAML 파서만 변경)는 두 번째 시도에서 one-visible +4.65%로 실패했다.
- **CV는 오버헤드가 아니라 drift를 측정했다.** baseline CV는 대부분 블록 전체에 걸친 느린 drift를 기록하는데, 번갈아 측정하는 쌍 구조는 바로 이 drift를 상쇄하기 위한 것이다. run 35425050365 attempt 1에서는 two-visible 블록 동안 `off` 속도가 171.4에서 141.7 tok/s로 떨어졌고, Activity arm이 6.76% 더 빨랐는데도 CV 7.45 때문에 실패했다. PR #2279 attempt 1은 모든 쌍이 더 빨랐는데 CV 5.17로 실패했다.

단일 GB10 러너에서 재실행 한 번에 10~40분이 든다. 이슈는 2%나 5% 임계값을 올리는 방안을 기각했다. 노이즈와 실제 회귀를 똑같이 가리기 때문이다. 이슈의 maintainer 코멘트는 호스트 수준 경합(같은 머신의 동시 컴파일로, GPU 프로세스 검사에는 보이지 않는다)도 점수를 움직였다고 덧붙였다(예: PR #1944, #1946). 그 부분은 #1949(host-quiet precheck)로 분리되었고, 이 PR은 판정만 다룬다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `webui/scripts/activity-performance-policy.json` (신규) | 상수의 유일한 사본: `budget_percent` 2, `precision_ceiling_percent` 4, `tail_alpha` 0.01, `min_pairs` 12, `pair_step` 2, `max_pairs` 40 |
| `webui/scripts/activity-performance-config.mjs` | 순수 판정 함수: `validatePolicy`, `verdictPolicy`(import 시 로드 및 검증), `signTestRank`, `pairDegradation`, `medianInterval`, `lookVerdict`, `sequentialVerdict`, `verdictRule`, `modeSummary`, `pairsFromSamples`; `performanceCompletion`에 `inconclusive` 추가 |
| `webui/scripts/activity-performance.mjs` | `lookVerdict`가 `continue`를 벗어날 때까지 측정 루프 반복; 모드마다 `mode: status: rule` 출력; 출력에 `verdict_policy` 기록; 기존 `median` 헬퍼(n이 짝수면 위쪽 가운데 원소를 취함) 제거 |
| `scripts/webui/activity_performance_verdict.py` (신규) | 같은 함수들을 연산 단위로 옮긴 Python 포트, 그리고 `check_summary_consistency`, `failure_message`, `expected_counts` |
| `scripts/webui/verify_activity_performance.py` | `samples`에서 모드별 판정을 재도출; `status`, `paired_runs`, 중앙값, 구간은 정확히 일치해야 하고 CV와 paired range는 상대 오차 1e-9 허용; 샘플 수와 preflight 수를 `paired_runs`에서 도출해 하드코딩된 두 개수 표를 대체 |
| `scripts/webui/summarize_activity_evidence.py` | 샘플이 없으므로 정책 기준의 구조와 내부 일관성만 검사; `verdict_policy`가 저장소의 정책과 같아야 함; `interval_percent`와 `verdict_policy`를 출력; CV 상한 제거 |
| `webui/scripts/activity-performance-fixtures.json` (신규) | 22개 케이스: 기록된 실패 시도 10건, 이어 붙인 null 시퀀스 2건, 합성 slowdown, speedup, drift, inconclusive, 경계 케이스, 거부 케이스 4건 |
| `webui/scripts/activity-performance-verdict.spec.mjs` (신규), `scripts/webui/activity_performance_verdict_tests.py` (신규) | 둘 다 모든 fixture 케이스를 평가해 `expected`를 정확히 재현해야 함; JS는 sign-test rank, look 일정, 정책 검증, `performanceCompletion`도 테스트 |
| `webui/scripts/activity-stream.spec.mjs` | verdict spec을 import(unit 파일 목록을 가진 `package.json`이 WebUI 번들 source digest에 포함되기 때문); visible-only 테스트가 `inconclusive`도 다룸 |
| `scripts/webui/*_tests.py` (verifier, summarizer) | 가변 `paired_runs`와 정책에 맞춰 재작성 |
| `.github/workflows/ci.yml`, `Makefile` | 새 Python 테스트를 helper 테스트 목록에 추가; 타임아웃 변경 없음 |
| `docs/webui-integration-matrix.md` | Performance 행에 새 판정 설명 |

커밋: `744c11f4`(본 변경), `a7cc03fa`(lint용 `/* global URL */`, 테스트 스크립트 실행 권한, 빈 줄 하나 제거).

## 3. 기술적 선택과 그 이유

### 3.1 중앙값과 sign-test 구간

이 러너의 쌍별 노이즈는 꼬리가 두껍고(636개 실행에서 kurtosis 10 초과) 치우쳐 있다(skew -0.7). sign-test 구간은 분산 추정도, 무작위 bootstrap도 필요 없다. 입력은 `P(Binomial(n, 1/2) <= k - 1) <= alpha`를 만족하는 가장 큰 rank `k` 하나뿐이다. 꼬리당 `alpha` 0.01에서 n = 12이면 `k`는 2이고(`P(<= 1) = 13/4096`이므로 98% 구간은 `[x(2), x(11)]`), n = 40이면 13이다. n = 6에서는 조건을 만족하는 순서 통계량이 없으므로 `validatePolicy`는 `tail_alpha`에 비해 너무 작은 `min_pairs`를 거부한다. 두 언어가 같은 double을 정렬하고 이항 합은 n = 50까지 double에서 정확하므로(그래서 검증에 `max_pairs <= 50`이 있다) JavaScript와 Python 결과가 비트 단위로 같다. 그 덕에 verifier는 중앙값과 구간 경계를 허용 오차 없이 `!=`로 비교한다. 기각한 대안: t-구간을 쓰는 평균(두꺼운 꼬리에 취약)과 Hodges-Lehmann(데이터에 없는 대칭성을 가정). 둘 다 검정력은 비슷했다.

### 3.2 세 가지 결과와 4% 상한의 의미

- `investigate`: 구간 전체가 2% 예산보다 위에 있다. 2.0에 닿기만 하는 구간은 위에 있는 것이 아니다(fixture의 `[2.0, 2.0]` 케이스는 `within-target`).
- `within-target`: `investigate`가 아니고 상한이 4% 이하다.
- 상한 전에는 `continue`, 40쌍에서는 `inconclusive`.

2% 예산은 그대로다. 4% 상한은 두 번째 예산이 아니라 정밀도 요건이다. 통과하려면 2%를 충분히 넘는 slowdown을 볼 수 있을 만큼 측정이 정밀했어야 한다. 리샘플링 모델에서 4%는 null `inconclusive`가 모드당 0.15%이고, 3%는 2.4%로 1% 오탐 목표를 깬다. 구간 전체가 2% 아래에 있을 때만 통과시키는 방안은 훨씬 많은 쌍이 필요해서 기각했고, 이 트레이드오프는 이슈에 이미 기록되어 있었다.

이 비대칭의 대가는 PR 자체 표에 나온다. 실제 2% slowdown은 81.6% 확률로, 실제 4% slowdown은 12.9% 확률로 통과한다(78.6%는 `investigate`). 게이트는 5% 이상 효과(96.0%, 6%에서 99.67%)를 잡도록 설계되었고, 2%와 5% 사이에서는 부분적인 검출기다. 이슈가 주입 slowdown에 대해 95% 이상 검출을 요구했기 때문에 acceptance 주입을 약 6%로 잡았다.

### 3.3 리샘플링 추정

노이즈 모델: 이 러너의 완전한 diagnostics artifact 636개 전체(2026-09-15~2026-10-22, 6,360쌍)를 실제 변화 0%로 중심화하고, 그 풀에서 iid로 쌍을 뽑은 뒤 실행별 shift N(0, 0.83^2)를 더했다. 0.83은 #2119 이후 실행에서 추정한 실행 간 SD이고, 한 실행 안의 one-visible 평균과 two-visible 평균은 상관이 없다(r = -0.07). 행마다 20,000회 시뮬레이션, 순차 look 포함:

| 실제 저하 | investigate | inconclusive | within-target | 평균 쌍 수 |
|---|---|---|---|---|
| 0% | 0.06% | 0.15% | 99.79% | 13.5 |
| 2% | 11.4% | 7.0% | 81.6% | 20.4 |
| 4% | 78.6% | 8.5% | 12.9% | 21.6 |
| 5% | 96.0% | 2.3% | 1.6% | 17.1 |
| 6% | 99.67% | 0.27% | 0.07% | 14.1 |
| 8% | 100% | 0% | 0% | 12.4 |

실제 변화 0%에서 오탐 실패는 모드당 0.21%, CI 잡당(두 모드) 약 0.4%, `full` 모드에서 약 0.6%다. 쌍이 iid이고 실행 shift가 없으면 0.01% 미만이다. 지배적인 요인은 실행 shift다. shift SD가 1.5이면 오탐 `investigate`가 2.4%로 오른다.

### 3.4 순차 look, 그리고 validator의 재생

look은 `min_pairs + k * pair_step`(12, 14, ..., 40)에서만 일어나고, 그 외의 개수에서 `lookVerdict`는 `continue`를 반환한다. 하네스는 첫 결정에서 멈춘다. `sequentialVerdict`는 기록된 시퀀스에 대해 모든 look을 다시 재생하고, 결정 이후에도 계속 측정했거나("continued after the verdict") 결정 없이 멈춘("without a verdict") 기록을 거부한다. 리샘플링 추정이 순차 look을 포함하므로 0.21%에는 반복 검정 비용이 이미 반영되어 있다.

### 3.5 정책 하나, 집행자 셋

하네스, verifier, summarizer는 원래 각자 임계값을 갖고 있었다. 이제 정책 파일이 유일한 사본이고 두 언어에서 같은 방식으로 검증된다(정확한 키 집합, `ceiling >= budget`, `0 < alpha < 0.5`, 짝수 `min_pairs`와 `pair_step`, `(max - min) % step == 0`). 하네스는 출력에 정책을 기록하고, 두 Python validator는 다른 정책으로 만든 증거를 거부한다. 샘플을 가진 verifier는 Python 포트로 모드마다 판정을 재도출한다. 요약만 보는 summarizer는 status가 기록된 구간에서 따라 나오는지, 중앙값이 구간 안에 있는지, `inconclusive`가 상한에서만 나오는지를 검사한다. 샘플 수와 preflight 수는 `expected_counts(paired_runs)`에서 나온다. warmup 1개, 쌍마다 `off` 샘플 하나와 모드 샘플 하나, preflight 기록은 모드의 클라이언트 수(one-visible과 hidden은 1, two-visible은 2)에 `1 + paired_runs`를 곱한 값이다.

### 3.6 보고 방식과 완료 상태

실패한 모드는 `mode: status (slowdown | measurement noise): 중앙값, 98% 구간, 쌍 수, 규칙` 형식으로 하네스 stdout과 verifier assertion 양쪽에 보고된다. 그래서 artifact를 내려받지 않고도 잡 로그에서 `investigate`와 `inconclusive`를 구별할 수 있다. `full` 모드에서 `performanceCompletion`은 `investigate`, `inconclusive`, `within-target` 순으로 우선한다. CI가 돌리는 `visible-only-headed`에서는 최상위 status가 `incomplete`로 유지되고, 잡을 실패시키는 것은 모드별 판정이다.

### 3.7 Warmup과 소요 시간

데이터에는 첫 쌍 편향이 없으므로(pair 0 평균 -0.50 대 전체 -0.48, 순서 효과 없음) 앞쪽 쌍을 버리지 않는다. 기존 warmup 요청은 첫 baseline보다 1.6% 느리며 이미 제외되어 있었다. 쌍당 one-visible은 중앙값 7.2초(최대 7.7초), two-visible은 8.4초(최대 9.2초)가 걸리고, 첫 쌍 전에 약 34초가 더해진다. CI에서 모드당 40쌍인 최악의 경우는 약 12분, 예상 null 실행은 약 4분(이전에는 약 2분)이다. `full` 모드의 최악은 약 20분으로 1800초 activity 타임아웃 안이다. 이슈의 최악 목표는 15분 미만이었다.

## 4. 검증

PR test plan의 로컬 검사: `pnpm --dir webui run unit`과 `lint`, `make verify-webui-helper-tests`(`activity_performance_verdict_tests.py` 포함), verifier와 summarizer 테스트 스크립트, `make verify-versions verify-env-var-docs verify-python-tooling`.

Fixture: #1925와 그 코멘트에 나온 모든 실패 시도(35423171315 a1; 35425297407 a1, a2, a3; 35425050365 a1; 35421166203 a1; #1944의 35567777700 a1; #1946의 35588958676 a1; #2279의 38265665905 a1, a2)의 샘플별 로그를 artifact가 만료되기 전에 복사했다. 실행 순서대로 이어 붙이면 두 모드 모두 12쌍에서 `within-target`으로 결정된다(one-visible 중앙값 -1.31%, 구간 [-4.50%, 1.83%]; two-visible -0.15%, [-4.99%, 2.45%]). 같은 쌍에서 Activity arm을 6% 느리게 하면 14쌍에서 `investigate`, baseline CV 12.19%인 drift 케이스는 `within-target`으로 결정된다.

`lablup-dgxspark21`에서:

| 실행 | 결과 |
|---|---|
| 39853154267 (test plan) | one-visible `within-target`(중앙값 -0.15%, [-1.83%, 1.25%]), two-visible `within-target`(-0.76%, [-2.64%, 0.47%]), 각 12쌍; verifier와 summarizer가 증거 수용; verifier 소요 3분 42초 |
| 39853154267, attempt 1~5 (job 120084449655, 120097378392, 120110916929, 120116442394, 120121248222) | head `a7cc03fa`에서 `WebUI installed artifact` 다섯 번 연속 통과 |
| draft PR #2554의 39870659579 (닫힘, 머지되지 않음) | 주입 slowdown: one-visible `investigate`(중앙값 4.58%, [3.89%, 5.68%]), two-visible `investigate`(8.05%, [7.11%, 9.62%]), 각 12쌍 |

주입은 열린 `/ui-api/v1/events` 스트림마다 decode tick당 400 us를 sleep한다. Activity 페이지는 보이는 동안에만 정확히 이 스트림을 열어 둔다.

GB10에서 작동하지 않은 주입:

- poll마다 80개 스레드를 돌리는 CPU burst(run 39862888632)는 visible arm의 runnable task를 46~87로 올렸지만 decode는 두 arm 모두 152 tok/s에 머물렀다.
- Activity poll에 묶은 지연(run 39866915190, 39868894929)은 0%로 측정되었다. 계측한 실행을 보면 operations poll이 decode 구간 안에 들어온 적이 한 번도 없었다. 각 poll은 요청이 끝나는 순간에 응답되었다.

## 5. 미검증 및 미해결 항목

- **주입의 실제 효과를 미리 측정하지 않았다.** 이슈의 기준은 "실제 효과를 미리 4~8%로 측정한" 주입이었다. 공유 CI 호스트에서 로컬 빌드가 허용되지 않아 효과는 게이트 자체 출력으로만 특성화되었다. one-visible(4.58%, [3.89%, 5.68%])은 4~8% 안이고, two-visible의 중앙값(8.05%)과 상한은 그보다 조금 위다.
- **기록된 fixture 케이스는 구조상 `investigate`를 반환할 수 없다.** 기록된 5쌍에서는 n이 `min_pairs`보다 작고 n = 5의 sign-test rank가 0이라, 모두 구간이 null인 `continue`를 반환한다. 따라서 "여섯 실행 중 어느 것도 `investigate`를 반환하지 않는다"는 acceptance 항목은 구조적으로 충족된다. 실질적인 null 증거는 이어 붙인 12쌍 시퀀스와 리샘플링 표다.
- **보장은 실행 수준 shift에 의존한다.** 0.21% 오탐 수치는 실행 간 shift가 측정값 0.83 포인트 근처에 머무는 동안만 성립하고, 1.5에서는 2.4%다. 호스트 경합을 막는 것은 이 판정이 아니라 #1949 host-quiet precheck의 몫이다.
- **2~5% slowdown은 자주 통과한다**(실제 2%에서 81.6%, 4%에서 12.9%). 3.2 참고.
- **`full` 모드는 실행되지 않았다.** native hidden acceptance는 여전히 사용자 보류 상태이고, 20분 최악 시간은 쌍당 시간에서 낸 추정이지 측정값이 아니다.
- **null 실행의 CI 소요 시간이 대략 두 배가 된다**(예상 약 4분, 이전 약 2분). 측정된 acceptance 실행에서 verifier는 3분 42초 걸렸다. `ci.yml` 타임아웃은 그대로다.
- **규칙 문자열 포맷.** Python의 `f"{value:.2f}"`가 JavaScript의 `toFixed(2)`를 "여기서 보이는 크기 범위에서" 대신한다. verifier는 `rule` 문자열을 비교하지 않으므로 차이가 나도 표시상의 문제에 그친다.

## 6. 학습 포인트

- 작은 예산에 대해 고정 개수로 판정하는 게이트는 설계상 노이즈로 실패한다. 측정된 노이즈에서 표본 크기를 정하고, 언제 멈출지는 구간이 결정하게 하라. "결정할 수 없음"은 이름을 가진 별도의 실패로 두어 누구도 노이즈를 회귀로 읽지 않게 한다.
- 두 구현이 똑같이 계산할 수 있는 통계량을 고르라. 정확한 이항 rank를 쓰는 순서 통계량은 무작위성도 허용 오차도 필요 없어서, Python verifier가 JavaScript 판정을 정확한 일치로 재도출할 수 있다.
- 한 규칙이 여러 곳에서 집행되면 상수를 파일 하나로 옮기고, 생산자가 출력에 정책을 찍게 하고, 소비자는 다른 정책으로 만든 증거를 거부하게 하라.
- 주입한 slowdown은 측정 대상과 측정 순간에 같은 자원을 공유해야 한다. GB10에서 CPU 압박은 decode를 늦추지 않았고, Activity poll은 decode 구간이 아니라 요청 사이에 응답되어 poll에 묶은 지연은 0%였다. 페이지가 decode 동안 열어 두는 이벤트 스트림에 묶은 지연은 decode에 닿았다.
- fixture가 의존하는 증거는 CI artifact가 만료되기 전에 저장소로 복사하라.
