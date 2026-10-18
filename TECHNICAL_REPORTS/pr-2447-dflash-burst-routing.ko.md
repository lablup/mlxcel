# PR #2447: 큐에 쌓인 DFlash 선두 요청을 speculative burst로 라우팅

**작성일**: 2026-10-18
**상태**: #2442의 항목 2, 3을 구현했고 GB10(CUDA)에서 실제 체크포인트로 검증함. 항목 1(행별 accept)은 #2442에 열려 있음

## 요약

#2442 참조 (항목 2, 3만 해당). 기본값 `--max-batch-prefill 4`에서 큐에 요청이 2개 이상 쌓여도 더 이상 speculative decoding을 잃지 않는다. burst가 처리할 수 있는 선두 요청은 이제 `execute_prefill`을 타고, 이 경로가 `try_speculative_burst`에 도달한다. GB10에서 배치 DFlash 윈도우의 출력이 temperature 0에서 순차 요청과 달라졌기 때문에, 배치 DFlash 윈도우는 `MLXCEL_ENABLE_DFLASH_BATCH`로 켜야 하는 옵트인(기본 꺼짐)이 된다. `run_dflash_burst_batched`는 drafter를 로드하기 전에 B = 1 경로와 같은 exactness probe를 적용한다.

## 1. 문제 정의

### 1.1 라우팅 (항목 2)

`execute_prefill_action`은 기본값 `--max-batch-prefill 4`에서 큐에 2개 이상이 있으면 `execute_batched_prefill`로 보냈다. `try_speculative_burst`를 호출하는 곳은 `execute_prefill`뿐이므로, 큐에 쌓인 요청은 B = 1이든 배치든 speculative decoding을 전혀 받지 못했다. burst에 도달하려면 `--max-batch-prefill 1`이 필요했다.

이력을 보면 의도된 동작이 아니다. 배치 prefill(`60e50b0c`)이 burst(`6a816947`)보다 먼저 존재했고, burst의 윈도우 수집기는 바로 이런 큐 행을 가져가며, `docs/benchmark_results/gemma4-mtp-batched-gb10-2026-10-07.md`가 이 누락을 기존 공백으로 적고 우회했다.

### 1.2 Exactness probe (항목 3)

`run_dflash_burst`는 drafter 로드 전에 `exactness_allows(bs)`로 검사한다. `run_dflash_burst_batched`는 `supports_batched()`만 확인했다. 문서 주석은 `dflash_exactness_allows`가 이미 probe를 수행하는데도 Qwen 3.5를 permissive라고 적고 있었다.

### 1.3 검증 중 발견한 배치 burst 불일치

라우팅 수정만 적용하면 기본 플래그의 동시 요청이 배치 DFlash burst에 도달하고, GB10에서 그 출력은 temperature 0에서 순차 요청과 달랐다. 이 불일치는 기존부터 있었다. 수정하지 않은 main 바이너리에 `--max-batch-prefill 1`을 주면 같은 어긋난 바이트가 나온다. 어떤 라운드도 clamp되지 않는 동일 프롬프트 2개의 윈도우도 어긋나므로, 배치 verify forward 자체가 단일 행 forward와 바이트 단위로 같지 않다. B = 1 exactness probe는 이를 감지하지 못한다.

## 2. 기술적 선택과 그 이유

### 2.1 burst 대상 선두 요청을 `execute_prefill`로 라우팅

classic 행이 디코딩 중이지 않은 동안, burst가 처리할 선두 요청은 `execute_prefill`을 탄다. 새 함수 `queued_head_takes_speculative_burst`는 burst의 게이트를 부작용 없이 그대로 따른다: 비어 있는 active batch, 공유 KV 예산, `should_burst_for_sequence`, 사용 중인 slice slot, 그리고 DFlash의 경우 `dflash_burst_would_engage`의 drafter 이전 게이트(adopted prefix, target variant, exactness verdict, greedy 전용 drafter). burst가 거절할 선두 요청은 기존처럼 배치 prefill을 유지한다.

- 적격 여부는 선두 요청만으로 정해진다. 거절된 선두 뒤의 행은 이전처럼 선두와 함께 배치 prefill로 처리된다.
- MTP는 `MLXCEL_ENABLE_MTP_BATCH`가 설정되지 않으면 classic 배치 prefill을 유지한다. 기존 윈도우 거절과 같은 동작이다.
- drafter가 없는 서버는 변하지 않는다.
- 이 결정은 `execute_prefill_action` 문서 주석에 기록했다.

### 2.2 active batch가 비어 있을 것을 요구

run-to-completion burst는 active batch에 아무것도 넣지 않고, tick 정책은 batch에 자리가 있고 큐가 비어 있지 않은 한 계속 Prefill을 고른다. classic 행이 디코딩 중인데 큐의 모든 선두를 burst로 보내면, 큐가 비지 않는 동안 그 행들의 디코딩이 멈춘다. classic 행이 active이면 큐의 행은 기존처럼 배치 prefill을 쓴다. 읽기 전용 리뷰에서 나온 지적이며 커밋 `9aeffffd`에서 추가했다.

### 2.3 배치 DFlash 윈도우는 옵트인

`MLXCEL_ENABLE_DFLASH_BATCH`의 기본값은 꺼짐이다. 기본적으로 큐의 DFlash 선두 요청은 tick마다 하나씩 정확한 B = 1 burst를 실행한다. 플래그를 설정하면 배치 윈도우가 복원된다. B = 1만 지원하는 DFlash 계열(LFM2, Muse Glimmer, Laguna)도 거절만 할 수 있는 윈도우를 더 이상 수집하지 않으므로, 선두 요청이 B = 1 burst를 받는다. PR은 이 결정이 이슈 본문의 범위를 넘으며 리뷰가 필요한 유일한 결정이라고 밝힌다.

### 2.4 공유 exactness 헬퍼

`run_dflash_burst_batched`는 `ensure_loaded` 이전에 `run_dflash_burst`와 같은 `exactness_allows(bs)` 게이트와 거절 경고를 적용하며, 공유 헬퍼 `dflash_target_exactness_allows`를 통해 수행한다. 배치 variant 게이트는 `dflash_target_supports_batched`를 쓴다. 오래된 문서는 수정했다.

## 3. 변경 사항

- `src/server/batch/scheduler/request_memory.rs`: `execute_prefill_action`의 라우팅과 문서 주석.
- `src/server/batch/scheduler/speculative_finalize.rs`, `src/server/batch/queue.rs`, `src/server/batch/scheduler/mod.rs`: 적격성 검사와 연결.
- `src/server/batch/speculative_burst.rs`, `src/server/batch/dflash_target.rs`: 공유 exactness 헬퍼, 배치 exactness 게이트, 옵트인 윈도우 정책.
- `src/models/speculative_exactness.rs`: `#[cfg(test)] seed_decision_for_test`. 메모이즈된 verdict를 직접 기록하며 probe, `qmv_wide` 재시도, `MLXCEL_MTP_ALLOW_INEXACT`를 거치지 않는다.
- `src/models/qwen3_5_tests.rs`, `src/models/mod.rs`: `qwen3_5_tests::two_layer_model`을 `pub(crate)`로 변경.
- `docs/environment-variables.md`: `MLXCEL_ENABLE_DFLASH_BATCH` 행 추가, `MLXCEL_ENABLE_MTP_BATCH` 행에서 윈도우에 `--max-batch-prefill 1`이 필요하다는 문구 삭제. `docs/CONTINUOUS_BATCHING.md` 갱신.

## 4. 검증

### 4.1 테스트

`scheduler_dflash_routing_tests`(신규)는 2-layer 합성 Qwen 3.5 위에서 기본 `max_batch_prefill`, 길이가 같은 큐 요청 2개, 그리고 drafter 경로가 존재하지 않는 DFlash dispatch로 스케줄러를 실행한다(drafter 로드에 도달한 burst는 행을 실패시키므로 어느 burst가 돌았는지 알 수 있다):

- `queued_equal_length_dflash_requests_reach_the_batched_burst_as_one_window`: 배치 윈도우를 켜면 두 행이 한 tick에 하나의 윈도우로 `try_run_burst_batched`에 도달한다.
- `queued_dflash_head_runs_the_b1_burst_by_default`: 선두가 B = 1 burst를 실행하고 두 번째 요청은 큐에 남는다.
- `queued_dflash_requests_keep_batched_prefill_when_the_burst_would_decline`, `queued_dflash_requests_keep_batched_prefill_while_classic_rows_decode`(mirostat 행이 디코딩 중), `queued_requests_without_speculative_dispatch_keep_batched_prefill`: 행들은 여전히 배치 prefill을 쓴다.

`qwen35_batched_dflash_declines_a_failing_exactness_verdict_before_drafter_io`(신규): 실패 verdict는 윈도우를 그대로 돌려주고, 클라이언트 이벤트를 보내지 않으며, drafter를 로드하지 않은 채로 둔다.

되돌림 검사:

- 라우팅 조건을 제거하면 앞의 두 라우팅 테스트가 실패한다(`no row may take the classic prefill: 2 active`).
- 배치 exactness 게이트를 제거하면 exactness 테스트가 실패한다(`Err` 대신 drafter 로드 오류를 동반한 `Ok`).
- DFlash 윈도우 정책을 강제로 켜면 `queued_dflash_head_runs_the_b1_burst_by_default`가 실패한다.
- active batch 조건을 제거하면 `queued_dflash_requests_keep_batched_prefill_while_classic_rows_decode`가 실패한다.

수정 후에는 모두 통과한다.

로컬 실행: `cargo fmt --all -- --check`, `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `server::batch::`로 필터한 lib 테스트 바이너리(551 통과, 9 무시), `speculative_exactness`(16 통과), `qwen3_5_tests`(20 통과). 모두 `gpu-lock` 아래에서 `--test-threads=1`로 실행했다. 다섯 가지 contract 검사도 통과한다.

### 4.2 실제 체크포인트 (GB10, CUDA)

target `~/models_archive/mlx/qwen3.5-4b-4bit`, drafter `~/models_archive/mlx/qwen3.5-4b-dflash`, `MLXCEL_SDPA_DETERMINISTIC=1`, `RUST_LOG=info,mlxcel::server::batch=debug`, test-fast `mlxcel-server --model <target> --model-draft <drafter> --draft-kind dflash`, 나머지는 기본 플래그(`max_batch_size=4`, `max_batch_prefill=4`, prompt cache 켜짐, 측정된 sm_121 기본값에서 나온 block 4). 드라이버: primer 요청 하나, 이어서 0.25초 뒤에 길이가 같은(프롬프트 23 토큰) temperature 0 `/v1/chat/completions` 2개를 `seed: 1`, `max_tokens: 96`으로 보내 함께 큐에 쌓이게 하고, 그다음 같은 2개를 하나씩 보낸다. 비교 대상 텍스트는 reasoning과 content를 합친 것이다. 정확성만 확인했고 처리량 수치는 없다.

| 바이너리, 플래그 | 동시 행을 처리한 경로 | 동시 == 순차 |
|---|---|---|
| main 74f9ae1d, 기본 | classic 배치 prefill 및 디코드, burst 로그 없음 | 예, 예 |
| main, `--max-batch-prefill 1` | `DFlash batched burst ... batch_size=2` 한 번 | 아니오 (행 0이 약 230자 뒤에 어긋남), 예 |
| main, `--max-batch-prefill 1 --no-prompt-cache` | 배치 burst | 아니오, 예 |
| main, `--max-batch-prefill 1 --draft-block-size 16` | 배치 burst | 아니오, 예 |
| main, `--max-batch-prefill 1`, 동일 프롬프트 2개 | 배치 burst, `clamped_rounds=0` | 아니오, 아니오 |
| 이 PR, 기본 | `DFlash accept lengths` B = 1 burst 두 번 | 예, 예 |
| 이 PR, 기본 + `MLXCEL_ENABLE_DFLASH_BATCH=1` | `DFlash batched burst ... batch_size=2` 한 번 | 아니오, 예 (`--max-batch-prefill 1`의 main과 같은 바이트) |
| 이 PR, drafter 없음 | classic | 예, 예 |

이 PR과 기본 플래그에서는 동시 행이 speculative decoding을 받으며, 그 텍스트는 순차 요청, main 바이너리의 classic 출력, drafter 없는 서버의 출력과 같다. drafter 없는 서버의 동시 및 순차 텍스트는 main과 바이트 단위로 같다. 두 검사 모두 최종 head(리뷰 후속 반영 후)에서 같은 결과로 반복했다.

## 5. 리뷰 메모

읽기 전용 리뷰는 게이트 미러링과 B = 1 리팩터링이 올바르다고 판단했다. 리뷰에서 반영한 변경: active batch 조건(2.2절), 테스트 seed 헬퍼가 더 이상 프로덕션 게이트를 실행하지 않음, 문서 주석에 선두 요청 기준 결정과 MTP 동작을 명시. 그대로 둔 항목:

- 기본 설정의 동시 DFlash 트래픽은 이제 classic 배치 디코드가 아니라 연속된 B = 1 burst로 처리된다. 따라서 나중 요청은 앞선 요청을 기다리고 자기 burst가 끝날 때 토큰을 받는다. 이는 이슈가 큐의 요청에 요구한 speculative decoding이며 `--max-batch-prefill 1`이 이미 하던 동작이다. classic 대안도 정확하다. 이 거래가 원치 않는 경우, 큐 길이 2 이상에서 main의 classic 동작을 유지하려면 `queued_head_takes_speculative_burst`의 DFlash 분기에서 `dispatch_runs_batched_window()`를 요구하도록 한 줄을 바꾸면 된다. 공유 장비라 지연 시간과 처리량 비교는 하지 않았다.
- `MLXCEL_ENABLE_MTP_BATCH=1`에서 윈도우가 무너지거나 거절되면 선두는 classic 단일 prefill로 가고 나머지는 큐 뒤로 다시 들어간다. 이는 기존 `try_speculative_burst` 동작(전에도 `--max-batch-prefill 1`로 도달 가능)이며 여기서 바꾸지 않았다.

## 6. 다루지 않은 항목

- #2442의 항목 1(행별 accept, B = 4 대 B = 1 parity 게이트, 처리량)은 열려 있다. 1.3절의 불일치 때문에 이 게이트는 현재 이 호스트의 해당 프롬프트에서 clamp와 무관하게 실패한다. 배치 exactness 검사가 통과하게 되면 옵트인 플래그를 없애야 한다.
- tick당 B = 1 burst 하나와 classic 배치 디코드의 처리량 비교는 하지 않았다.

## 7. 관련 항목

- 이슈 #2442: #2353 및 PR #2440(`TECHNICAL_REPORTS/pr-2440-dflash-batched-uniform-accept.ko.md`)의 후속. 해당 이슈 코멘트는 배치 burst 불일치를 항목 1과 관련된 발견으로 기록한다.

## 8. 변경 요약

13개 파일 변경, 727줄 추가, 62줄 삭제.

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `a6c0dd7e` | fix | route queued DFlash heads to the burst (#2442) |
| `9aeffffd` | fix | keep batched prefill while classic rows decode |
