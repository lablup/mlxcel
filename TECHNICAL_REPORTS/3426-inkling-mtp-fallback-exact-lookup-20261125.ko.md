# PR #3426: Inkling MTP classic decode fallback, exact prompt lookup 처리량, Inkling 서버 lookup 테스트

**날짜**: 2026-11-25

**상태**: draft PR(#3425를 닫는다). M3 Ultra(512 GB)에서 검증했다. 좁은 lib 테스트, ignored 서버 e2e 테스트, clippy, contract check를 통과했다. 항목 3의 수용 기준 하나는 충족하지 못했다.

## 요약

이슈 #3425는 speculative exactness 작업의 후속 세 가지를 묶었다.

- 항목 1: #3421이 추가한 네 계열에서 `MLXCEL_MLA_ROW_INVARIANT=1`일 때의 prompt lookup 처리량 측정
- 항목 2: 1세대 Inkling의 서버 prompt lookup 테스트
- 항목 3: Inkling-Small MTP를 classic decode보다 느리지 않게 만들기

항목 1과 2는 완료했다. 항목 3에서는 acceptance를 보고 classic decode로 넘어가는 fallback을 넣었다. 기본 MTP 폭은 classic 대비 0.27x에서 0.81x로, 폭 2는 0.90x로 올랐다. 이슈가 요구한 0.97x에는 닿지 못했다.

## 1. Exact prompt lookup 처리량

이 스위치는 acceptance가 낮으면 거의 비용이 없고, 행 단위 블록이 batched kernel을 대신하는 곳에서 비용이 가장 크다.

- copy 위주 응답에서 줄어드는 처리량: OLMoE 34%(702.3 대비 464.6 tok/s), DeepSeek V2 12%, Arcee 9%, K2 Horizon 5%
- 그래도 exact lookup은 이 프롬프트에서 plain decode보다 1.09x~3.23x 빠르다.
- prose에서는 acceptance가 0.1 미만이다. 스위치를 켜든 끄든 lookup이 plain decode보다 느리다: DeepSeek V2 0.84x~0.86x, OLMoE 0.85x
- 스위치와 `MLXCEL_QMV_WIDE=0`을 함께 쓰면 체크포인트와 프롬프트 8쌍 모두에서 텍스트가 plain decode와 같다.

## 2. Inkling 서버 prompt lookup

e2e 테스트는 plain 서버와 lookup 서버를 함께 띄운다. 두 서버에 같은 요청을 보내 텍스트가 같아야 통과한다.

- 요청 종류: completion과 chat, stream과 non-stream
- 프롬프트: copy 프롬프트와 prose 프롬프트
- 마지막으로 동시 요청 한 쌍을 보낸다.
- copy 응답 중 하나 이상이 `timings.draft_n > 0`을 보고해야 한다. 그래서 verify round가 한 번도 돌지 않으면 실패한다.

이 체크포인트는 대부분의 프롬프트에 외운 문단 하나로 답한다. 그래서 copy 프롬프트는 그 문단 뒤에 문단의 첫머리를 한 번 더 붙인 형태로 만들었다. 처음 고른 프롬프트들은 응답이 1~9 토큰으로 끝나 proposal이 하나도 나오지 않았고, 위 단언이 이를 잡아냈다. tiny 모델 scheduler 테스트는 한 행만 돌 때와 두 행이 한 batch로 돌 때를 다룬다. 두 테스트 모두 결함을 찾지 못했다.

## 3. Inkling-Small MTP

**프로파일**

- exact 2행 round는 36~37 ms가 걸리고, decode step은 21.7 ms다.
- round가 이득이 되려면 폭 2에서 acceptance가 약 0.70은 되어야 하는데, 실제로는 0.35~0.51이다.
- round를 exact하게 만드는 row-invariant 곱과 grouped attention이 바로 그 비용이다.
- acceptance는 mlx-vlm과 같다. 그러니 원인은 포팅이 아니라 head다.

**메커니즘**

drafting이 손해일 때는 drafting을 멈춘다.

- 폭 선택: `ExpectedThroughputWidth`를 MTP round loop에서 재사용한다. 폭 1이면 새 `MtpTarget::classic_step`으로 pipelined classic step을 돌린다.
- drafter 따라잡기: 건너뛴 위치는 다음 round 전에 `Drafter::extend_history`로 한 번의 forward에 넘긴다.
- 재시도 주기: classic step 8번마다 drafter를 다시 시도한다. 재시도가 지면 주기가 두 배가 되어 최대 64까지 늘고, 이기면 처음으로 돌아간다.
- opt-in: Inkling MTP target과 head만 opt in한다. 다른 MTP 계열과 DFlash loop(Inkling 2 DSpark)는 바뀌지 않는다.

**결과**

128 토큰 기준이다. 모든 arm에서 출력이 classic decode와 같다.

| 프롬프트 | Classic | 폭 10 고정 | 폭 10 기본 | 폭 2 고정 | 폭 2 기본 |
|---|---|---|---|---|---|
| prose | 46.18 | 12.41 | 37.83 | 39.61 | 41.42 |
| code | 46.05 | 12.59 | 37.40 | 41.63 | 41.16 |

**남은 차이**

- MTP 세션 안의 classic step은 23.7 ms로, 엔진의 21.7 ms보다 느리다.
- 세션 안에서 classic만 돌려도 40.4 tok/s에 그친다(엔진은 45.8).
- 시도한 변형 11가지는 차이를 줄이지 못했다. 원인은 아직 밝히지 못했다.

## 4. 위험

- target과 drafter가 둘 다 opt in하지 않으면 fallback은 아무것도 바꾸지 않는다.
- 폭 10에서는 첫 두 round가 상한 폭으로 돈다. 세션마다 약 450 ms가 든다.
