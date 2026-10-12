# 기술 보고서: PR #2274 - 서버 프롬프트 조회 손익분기 거버너

**작성일**: 2026-10-12
**상태**: CUDA(GB10)에서 구현 및 검증 완료
**언어**: Rust, Python, shell
**PR**: [#2274](https://github.com/lablup/mlxcel/pull/2274)
**위험도**: 중간

## 요약

이슈 #2264는 서버 전용 프롬프트 조회 손익분기 거버너를 추가한다. CUDA 서버는 출력 토큰 하한 4개, shadow 확인 길이 7개, 제안이 없는 첫 요청 이후 즉시 re-prime, paged 진입 한도 4608을 사용한다. CLI 기본값, graded adaptation, 비적응형 동작은 유지된다.

보정 커밋 `b274abf0`는 의도적으로 적용 범위가 좁은 정책을 구현한다. 완전 실패 라운드는 갱신 전 accepted-token EMA가 4 이상일 때만 한 번 더 probe한다. 부분 저수익 라운드는 이 grace를 받지 않고, 연속 완전 실패는 계속 멈추며, 완전 수락 또는 하한을 만족하는 라운드는 low-yield streak을 초기화한다. Release build, 전체 CUDA suite, 집중 테스트, contract 검사, 변경하지 않은 attempt-4 matrix가 모두 통과했다.

## 1. 문제 정의

기존 서버 워크로드는 약한 프롬프트 조회 일치가 lookahead를 반복해서 중단하고 동기 검증 비용을 지불하여 일반 채팅 성능이 3.49% 하락했다. 같은 기능은 복사 중심 프롬프트에서 큰 이득을 제공하므로 전체 비활성화는 이슈 요구사항을 만족하지 못한다.

수용 기준은 채팅 median −1% 이상, 모든 기존 복사 이득의 90% 이상 보존, 동시성 2에서 서로 겹치지 않는 성능 하락 없음, paged 8192 control 차단이다.

## 2. 평가 중인 최종 정책

검증 라운드의 출력량은 `accepted + 1` 토큰이다. 서버는 최소 두 번의 검증 표본이 쌓인 뒤 emitted-token EMA가 4보다 작고 현재 제안도 불완전하며 같은 하한보다 낮을 때 멈춘다. 현재 라운드가 하한을 만족하면 EMA 부채를 갚을 수 있고, 짧은 제안을 완전히 수락하면 제안 토큰을 거부한 것이 아니므로 계속 실행한다.

Drafter는 실제 제안 길이를 전달한다. 이에 `2/2`와 `2/7`을 구분하여 전자는 너비 7로 확장할 수 있고 후자는 부분 저수익 근거로 처리한다. 이 수정은 실제 상태 모델 결함을 바로잡았지만 그 자체만으로 측정된 짧은 복사 이득을 회복하지 못했다.

선택적 경계 규칙은 현재 갱신 전 accepted-token EMA를 사용한다.

- `accepted == 0`이고 갱신 전 accepted EMA가 4 이상이면 한 번 더 probe한다.
- `1/7`, `2/7` 같은 부분 저수익은 emitted EMA가 하한보다 낮아지면 경계 grace를 받지 않는다.
- 두 번째 연속 완전 실패에서는 멈춘다.
- 완전 수락 또는 하한 만족 라운드는 low-yield streak을 초기화한다.
- Shadow 확인 후 재개하면 accepted EMA, emitted EMA, sample count, low-yield 이력을 초기화한다.

이 zero-versus-partial 구분은 관찰된 abrupt boundary에 한정한 휴리스틱이며, 수락 토큰 0개가 일반적으로 수익성이 있다는 뜻이 아니다. 수집한 복사 trace에서 완전 실패 직전 accepted EMA는 4.25, 4.981, 6.982였다. 채팅의 완전 실패는 q3의 copy-identical `2/2 → 7/7 → 0/7` prefix를 제외하면 4 미만에서 발생했다. 이 prefix는 올바르게 grace를 받아 제안 7개, 수락 0개, 검증 라운드 1개를 추가했고, 그 밖의 관찰된 high-EMA 채팅 저수익은 부분 수락이었다.

## 3. 근거가 된 기술적 선택

### 실제 제안 길이

보정 `550107f56c6c570b0cd6419c12700d1d8b3429b4`는 짧은 완전 수락 제안을 보존했다. Attempt 2는 이 수정이 benchmark의 근본 원인을 해결하지 못했음을 보여 주었다. 고립된 넓은 완전 실패가 여전히 수익성 높은 복사 실행을 끝냈다.

### 컨텍스트 진입

명목상 4096-token 워크로드의 실제 프롬프트 길이는 4099, 4101, 4102 토큰이었다. 관찰한 최대 prompt-plus-continuation 범위는 4358 토큰이었다. 따라서 paged 한도를 이 범위를 포함하는 가장 작은 512-token 경계인 4608로 올리고 4609와 8192 control은 계속 제외한다. Dense storage에는 이 한도를 적용하지 않는다.

### 복구 경로

기존 거버너의 일시정지 상태는 파이프라인 실행을 다시 허용하면서도 이후 복사 구간을 찾는 shadow probe를 유지한다. Drafter 호출을 건너뛰면 출력량 근거와 복구 근거를 모두 수집하지 못하므로 제외했다.

## 4. 회귀 테스트 범위

현재 테스트는 다음을 다룬다.

- accepted-EMA grace 하한 바로 아래와 정확히 4인 경우
- 완전 실패와 부분 `1/7`, `2/7`의 차이
- trace에서 관찰한 `2/2 → 7/7 → 0/7` 복사 경계
- 기본 너비와 큰 draft 너비에서 연속 완전 실패
- 하한 만족 라운드의 EMA 부채 상환과 짧은 완전 수락 보존
- shadow 재개 상태 초기화와 작은 너비의 floor cap
- CLI 기본값, graded 정책, 비적응형 동작 유지
- 스케줄러 파이프라인 복구, 동시성 2 격리, re-prime 주기, 컨텍스트 경계

Exact-counter 진단은 선택적 정책을 검증했다. 정확한 8개 채팅 워크로드에서 제안 173개, 수락 75개, 검증 라운드 32개를 기록하여 넓은 grace의 226/83/43보다 작업량을 줄였다. 복사 trajectory는 정확히 239/214/35로 유지됐다. 채팅 합계에는 q3의 예상된 copy-identical 경계 동작이 포함된다.

## 5. 성능 근거

각 시도는 소스, binary, 환경, host gate, 원시 출력, checksum 근거와 함께 별도로 보존한다.

- **Attempt 1:** 채팅과 동시성은 통과했으나 짧은 복사와 paged 1024/2048 보존 기준에 실패했다.
- **Attempt 2:** 짧은 완전 수락 보정 후에도 paired short-copy round 5회가 실패하여 나머지 matrix 전에 중단했다.
- **Attempt 3:** 넓게 적용한 isolated-miss grace가 복사 성능을 회복했다. 짧은 복사 이득은 +142.132%, +151.215%, +145.537%, +146.083%였다. 모든 복사, 동시성, gate-control 기준은 통과했지만 채팅은 median −1.419%로 실패했다.
- **Attempt 4:** 선택적 grace가 전체 matrix를 통과했다. 채팅 median은 −0.275%였다. 짧은 복사 이득은 +159.283%, +157.037%, +151.289%, +148.602%였다. Paged ratio는 2.7349배, 2.8189배, 1.5221배였고 dense 8192는 2.3475배였다. 동시성 2가 통과했고 paged 8192는 계속 차단됐다.

Attempt 3은 넓은 grace가 기존 동작보다 일반 채팅에 제안 60개, 수락 토큰 8개, 검증 라운드 12개를 추가함을 보여 주었다. 선택적 갱신 전 accepted-EMA 조건은 복사 trajectory를 유지하면서 이 채팅 작업량을 줄였다.

## 6. 최종 검증

보정 커밋 `b274abf0`는 recovery와 floor boundary를 포함한 governor/drafter 테스트 54개와 scheduler 테스트 20개를 통과했다. Release build는 6분 16초에 통과했다. Test build, clippy, formatting, contract 검사 3개도 통과했다. 전체 CUDA suite는 root 테스트 8,936개와 core 테스트 2,093개를 integration failure 없이 통과했다.

Attempt 4 분석기는 실패 없이 종료 코드 0을 반환했다. Binary SHA-256은 `2388e9f8ef914e11ca0e621801a0cad7bf4101fc359a572a49d6a352afae9485`, native MLX pin은 `81ba1c6a0e50a9268b931579c2d4f1158b9aab5a`였고 host gate는 실행 중인 CI 작업이 없음을 기록했다. 최종 build 전에 임시 runtime tracing을 모두 제거했다.

## 근거 위치

- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/STATUS.md`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-1.md`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-2/`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-3/summary-corrected.txt`
- `docs/benchmark_results/data/server-prompt-lookup-break-even-gb10-2026-10-11/attempt-4/summary.txt`
- `docs/benchmark_results/server-prompt-lookup-break-even-gb10-2026-10-11.md`
