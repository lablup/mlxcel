# PR #2828: headed WebKit 내비게이션 피드백 예산 보정

**Date**: 2026-11-02
**Status**: Completed (PR open; closes #2821)

## 요약

PR #2828은 제품 예산을 근거 없이 올리는 대신 측정으로 원인을 정해 빨간 `WebUI bundle` 작업을 해소한다. 이제 headed WebKit 프로젝트는 내비게이션 피드백에 150 ms를 허용하고, Chromium과 Firefox는 100 ms를 유지한다. 근거는 Playwright 1.64.1과 함께 온 WebKit 27.2다. event-to-semantic-DOM 구간은 30~34 ms에 머문 반면, 두 번의 rAF paint 꼬리는 36~57 ms에서 74~80 ms로 움직였고, 같은 트리인 95ea652a가 어떤 실행에서는 46 ms로 통과하고 다른 실행에서는 105~110 ms로 실패했다. React 19.3.1은 별도 대조군으로 측정했고 원인이 아니다.

## 1. 문제 정의

CI 실행 41321895260의 `webkit-headed-performance`에서 `tests/performance.spec.ts:290`이 세 번 모두 실패했다. 108, 110, 105 ms였고 예산은 100 ms였다. 이 작업은 #2819가 Playwright 1.63.0을 1.64.1로, React 19.3.0을 19.3.1로 올리기 전 main에서 통과하고 있었다. 원인 없이 선택지는 매번 실패하는 예산을 유지하거나 근거 없이 올리는 것뿐이었다.

## 2. 근거

| 팔 | 결과 |
|---|---|
| 95ea652a main CI (run 41322479478) | 46 ms로 통과 |
| #2819 PR 실행 41321895260 | 108, 110, 105 ms로 실패 |
| #2819 이전, WebKit 26.6 / v2359, 러너 샘플 7개 | 합계 62, 64, 64, 87, 89, 90, 98 ms; event to predicate 중앙값 33 ms; predicate to paint 36~57 ms |
| #2819 이후, WebKit 27.2 / v2370, 시도 7회 | 합계 46, 69, 105, 107, 108, 108, 110 ms; event to predicate 중앙값 31 ms; 실패한 모든 시도에서 predicate to paint 74~80 ms |
| 로컬 Chromium, React 19.3.0 | 31.5~31.7 ms |
| 로컬 Chromium, React 19.3.1 | 28.6~35.4 ms, event to predicate는 동일하게 5.9~6.3 ms |

통과한 실행과 실패한 실행 사이에 애플리케이션 절반은 변하지 않았다. 같은 트리가 통과하고 실패하며, 움직인 것은 paint 꼬리뿐이다. 원인은 엔진이다. React는 통제된 Chromium 팔로 배제했다.

## 3. 변경 요약

| 파일 | 변경 |
|---|---|
| `webui/tests/performance.spec.ts` | `WEBKIT_HEADED_NAVIGATION_FEEDBACK_BUDGET_MS = 150`; 단언과 evidence 필드 `feedback_budget_ms`가 프로젝트별 값을 사용; 주석에 전후 수치와 95ea652a 대조 기록 |
| `webui/playwright.engines.config.ts` | 엔진 보정 WebKit 예외를 명시하고 기존 재시도 근거를 유지 |

## 4. 기술적 선택

### 4.1 예외는 제품이 아니라 엔진에 한정

Chromium과 Firefox는 100 ms 예산을 유지한다. headed WebKit 프로젝트만 150 ms를 받고, 실제 적용값을 evidence 파일에 기록해 이후 독자가 어느 예산으로 나온 샘플인지 알 수 있게 한다. 전역 인상은 모든 엔진에서 진짜 내비게이션 회귀를 가렸을 것이다.

### 4.2 수치는 임계값 옆에 둔다

주석에 엔진 버전, event-to-DOM과 paint의 분리, 같은 트리의 통과·실패 합계, 이슈 링크를 담는다. 측정 없이 바뀐 임계값은 우회와 구분되지 않고, 다음 의존성 범프에서 같은 조사를 다시 열게 된다.

### 4.3 React는 원인이 아니며, 그것도 측정했다

통제된 Chromium 팔에서 React 19.3.0과 19.3.1의 event-to-predicate가 같았다. 이로써 #2819의 나머지 절반을 커밋된 lockfile을 건드리지 않고 배제했다.

## 5. 검증

| 검사 | 결과 |
|---|---|
| 수정 스펙, 로컬 Chromium 3회 | 통과, evidence의 `feedback_budget_ms=100` |
| `pnpm --dir webui run typecheck` | 통과 |
| `pnpm --dir webui run lint` | 통과 |
| PR CI run 41410786629, `WebUI bundle` 작업 125163183335 | 150 ms WebKit 예산으로 성공 |
| `pnpm install --frozen-lockfile` 복원 | 통과, package.json과 lockfile 변경 없음 |

## 6. 미검증

이 Apple Silicon 호스트에서는 로컬 WebKit을 실행할 수 없다. `browserType.launch`가 headed와 headless, 샌드박스 사용과 미사용 모두에서 NSXPC 리스너 오류로 180초 타임아웃된다. 따라서 Playwright 팔은 통제된 로컬 이분 탐색이 아니라 CI 러너 샘플이다. 마지막 인수 조건인 main의 `WebUI bundle` 통과는 이 머지 이후에만 확인할 수 있다.

## 7. 학습 포인트

같은 트리에서 나온 CI 결과 두 개가 커밋 두 개보다 많은 것을 알려준다. 하나는 통과하고 하나는 실패했으므로 차이는 diff가 아니라 환경이나 엔진에 있어야 한다. event-to-DOM과 paint 절반을 나눠 측정한 것이 움직임을 paint 스케줄링으로 좁혔다.

## 8. 위험

낮음. 테스트 하나와 설정 주석 하나만 바꾼다. headed WebKit 팔의 150 ms가 지나치게 관대할 위험은 있지만, event-to-DOM 절반은 30~34 ms에 머물고 다른 엔진은 100 ms를 유지하므로 내비게이션 회귀는 여전히 의미 있는 문턱을 넘어야 한다.
