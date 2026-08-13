# 기술 보고서: PR #1130 - fix(core): stop the paged budget test underflowing past the v2 reserve

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust
**위험도**: Low (테스트 기대값과 운영 로그 문구만 바뀐다. 어떤 예산도 이전과 다른 블록 수로 해석되지 않는다)

---

## 요약

`execution::memory_estimate::tests::resolve_block_budget_explicit_bytes_floors_to_block_count`가 GB10 / CUDA sm_121 / Linux aarch64에서 0.00초 만에 `assertion failed: shortfall < 100`으로 결정적으로 실패했다. 간헐적 실패도 아니고 이 장비에서 흔한 TF32 수치 편차도 아니다. 테스트가 기대값을 `(per_block * 100 - workspace) / per_block`으로 직접 계산했고, 이는 장비에서 유도된 값을 고정값에서 부호 없이 빼는 식이다.

이슈는 결함이 테스트에만 있다고 보았고, 수용 기준은 그것을 가정하지 말고 확인하거나 반박하라고 요구했다. 확인된다. 예약분이 예산과 만나는 모든 지점을 빠짐없이 훑었고, 구현을 예약분 아래 예산으로 직접 찔러 보는 회귀 테스트를 새로 넣었다. 구현은 처음부터 옳았다.

부수적으로 하나 더 나왔다. `Some(0)`은 Metal이 아닌 모든 호스트에서 실제로 도달 가능하고 호출자도 제대로 처리하지만, 그때 남기는 경고가 엉뚱한 원인을 지목하고 있었다. 이제 예약분을 이름으로 밝힌다.

---

## 1. 문제 정의

### 1.1 배경

#899는 fused paged decode v2 워크스페이스를 KV 바이트 예산에 먼저 청구한 뒤 남은 몫을 블록으로 나눈다. 그래야 admission이 실제로 메모리로 뒷받침할 수 있는 블록만 내준다. 예약분은 `paged_v2_workspace_reserve_bytes`가 계산하며 지배항은 `2 * device_target_ctas() * n_rep * (head_dim + 1) * 4`다.

`device_target_ctas()`(`src/lib/mlxcel-core/src/paged_v2/plan.rs`)는 `metal_is_available()`이 거짓이면 `DEFAULT_TARGET_CTAS = 512`를 돌려준다. Linux와 CUDA 호스트는 전부 여기 해당한다. Apple 부품에서는 `gpu_core_count * 8`이고 하한이 64다. 즉 예약분은 테스트가 임시 디렉터리에 써 넣은 모델 기하가 아니라 테스트를 돌리는 기계의 성질이다.

### 1.2 기존 문제점

- **테스트가 호스트에 대한 가정을 상수로 박아 두었다.** minimal config에서 `per_block`은 131072이므로 시험 대상 예산은 `per_block * 100 = 13107200`바이트다. `PAGED_V2_MAX_N_REP = 16`, `PAGED_V2_CONCURRENT_LAUNCHES = 2`에 512 CTA 목표를 넣으면 예약분은 17040384바이트가 된다. 예약분이 예산 전체를 3933184바이트 넘어서면서 `u64` 뺄셈이 감싸 돌았고 `shortfall`은 `2^47 - 31`이 되었다. 64비트 호스트에서는 이 값도 `usize::try_from`을 멀쩡히 통과한다. 감싸 돈 값이 그대로 `assert_eq!`에 들어가는 것을 막아 준 것은 `assert!(shortfall < 100)` 하나뿐이었다.
- **경계가 Apple 구간 안에 있어 macOS CI도 믿을 수 없다.** 보고 코어 수 48에서는 예약분이 12813312바이트라 통과하고, 50에서는 13341696바이트라 실패한다. 테스트의 정오가 어느 맥에서 돌았느냐에 달려 있었다.
- **실패한 단언이 정작 중요한 단언이 아니었다.** `shortfall < 100`은 테스트가 자기 산술을 검산하는 문장이다. 고정할 값어치가 있는 성질, 즉 워크스페이스를 빼먹은 예산은 정확히 예약분만큼 모자라게 돌아온다는 계약은 그 아래 `assert_eq!`가 혼자 지고 있었다.
- **`Some(0)` 계약은 문서에만 있고 덮여 있지 않았다.** `resolve_paged_block_budget`은 한 블록 아래로 내려간 예산이 `Some(0)`이 된다고 적어 두었고, 예약분보다 작은 예산이 `usize::MAX`가 아니라 거기로 떨어지게 만드는 것이 `saturating_sub`다. 그런데 예약분보다 작은 예산을 요청하는 테스트가 하나도 없었으므로 그 포화 연산은 곁다리로만 덮여 있었다.

### 1.3 위험 평가

| 위험 | 영향 | 가능성 |
|---|---|---|
| 실패가 이 장비 특유의 수치 편차로 치부되어 모듈 전체가 신뢰를 잃는다 | Medium | High |
| 나중에 누가 구현의 `saturating_sub`를 `-`로 되돌려 작은 예산에 `usize::MAX` 블록을 발급한다 | High | Low |
| 운영자가 낮은 `--kv-cache-budget`을 걸고 무제한 풀을 받은 뒤, 모델 크기를 지목하는 경고를 읽는다 | Medium | Medium |

---

## 2. 기술 검토

### 2.1 언더플로가 있는 곳: 테스트뿐, 확인 완료

감사 범위가 작아서 전수로 끝난다. `paged_v2_workspace_reserve_bytes` 호출부는 트리 전체에 넷이다.

- `src/execution/memory_estimate.rs:875`, 구현. `saturating_sub`로 뺀다.
- `src/execution/memory_estimate.rs:1697`, 실패하던 테스트. `-`로 뺐다.
- `the_v2_workspace_reserve_is_small_and_scales_with_the_head_dim` 안의 둘. 예약분을 재기만 하고 빼지 않는다.

비테스트 코드에서 예약분을 예산에서 빼는 지점은 정확히 하나이고 이미 포화 연산이었다. 다만 결론을 grep에만 기대게 두지는 않았다. 새로 넣은 `resolve_block_budget_below_the_workspace_reserve_is_zero_blocks`는 `0`, `1`, `workspace / 2`, `workspace - 1`, `workspace` 예산으로 실제 함수를 호출해 각각 `Some(0)`을 요구한다. 뺄셈이 감싸 돌았다면 뒤의 셋은 `Some(usize::MAX)`가 된다. 이 테스트가 통과하므로 구현이 깨끗하다는 것은 눈으로 읽은 결론이 아니라 측정된 결론이다.

### 2.2 환경변수 고정을 택하지 않은 이유

테스트 안에서 `MLXCEL_PAGED_DECODE_V2_TARGET_CTAS`를 고정하면 예약분이 결정적이 된다. 이슈가 이 선택지를 맨 뒤에 둔 이유가 있다. `device_target_ctas()`는 `OnceLock`으로 메모이즈한다. 테스트 바이너리를 공유하는 상황에서는 먼저 부른 쪽이 이기므로, 재정의가 먹히는지가 테스트 순서에 달린다. 결정적 실패를 순서 의존 실패로 바꾸는 셈이라 원래보다 나빠진다.

### 2.3 장비에 기대지 않고 계약을 진술하기

수정은 기대값을 구현이 계산하는 방식 그대로 쓰고, 그 결과 성질을 두 번 진술한다. 어느 호스트에서도 둘 중 하나는 공허해지지 않게 하기 위해서다.

- 상한 있는 진술: `100 - shortfall == min(ceil(workspace / per_block), 100)`. `workspace <= 100 * per_block`이면 `floor((100 * pb - w) / pb) == 100 - ceil(w / pb)`이므로 항등식이다. 그보다 크면 포화가 양변을 100에 고정한다.
- 상한 없는 진술: `per_block * (100 + ceil(workspace / per_block))`바이트를 요청하면 정확히 100블록이 나온다. 예약분을 뺀 나머지가 `100 * per_block + (ceil(w / pb) * pb - w)`이고 괄호 항이 `[0, per_block)`에 있어 내림에서 사라진다.

GB10에서는 상한 있는 진술이 포화 분기로 들어간다. 상한 없는 진술을 함께 둔 이유가 그것이다. 예약분 산술 자체를 CTA 목표가 큰 장비에서도 실행시켜, 작은 Apple 부품에서만 덮이는 경로로 남기지 않는다.

### 2.4 `Some(0)`은 도달 가능하고 처리된다

이슈는 문서상의 `Some(0)` 동작이 실제로 도달 가능하며 호출자가 처리하는지 가정하지 말고 확인하라고 했다. 양쪽 다 확인된다. `resolve_worker_paged_block_budget`(`src/server/model_worker.rs`)은 `Some(n) if n > 0`을 먼저 잡고 `Some(_)`로 떨어뜨려 경고를 남긴 뒤 `None`을 돌려주며 풀을 무제한으로 둔다. 모든 요청을 막아 버릴 0 예산을 설치하지 않는다.

확인되지 않은 것은 진단 쪽이다. 경고는 "model too large for a meaningful paged budget at this batch / available memory"였다. `Auto` 경로에는 맞는 설명이지만 명시적 바이트 예산에는 틀린 설명이다. Metal이 아닌 호스트에서는 대략 16.25 MiB 미만의 `--kv-cache-budget`이면 모델 크기와 무관하게 여기로 온다.

---

## 3. 기술적 결정

### 3.1 다시 유도하지 말고 구현을 그대로 반영한다

기대값은 이제 구현과 같은 `saturating_sub`를 쓴다. 다른 산술로 기대값을 다시 유도한 것이 애초에 결함을 만들었다. 0 아래에서 무슨 일이 일어나는지에 대해 테스트와 코드의 답이 달랐고, 틀린 쪽은 테스트였다.

### 3.2 100블록이라는 표현은 유지한다

테스트 전체를 측정된 예약분 기준으로 다시 쓰는 것이 이슈의 두 번째 선택지였다. 앞의 세 단언에는 적용하지 않았다. 블록 단위로 읽는 편이 자연스럽고, 요청에 예약분을 더해 두었으므로 예약분 크기에 영향받지 않는다. 예약분을 일부러 빼먹는 네 번째 경우만 장비 독립적인 처리가 필요했다.

### 3.3 기존 테스트에 케이스를 더하지 않고 회귀 테스트를 따로 둔다

예약분 미만 경로는 "바이트 예산이 블록 수로 내림된다"와 다른 계약이고, `saturating_sub`를 고정하는 쪽은 이쪽이다. 그래서 테스트를 따로 두고 경계를 양쪽에서 못 박았다. `workspace + per_block - 1`은 `Some(0)`, `workspace + per_block`은 `Some(1)`이다.

### 3.4 경고에 예약분을 적는다

메시지는 이제 예약분을 바이트로, 그리고 그 위에 한 블록이라는 임계를 함께 보고한다. `Auto` 경로에도 맞는 문장이다. 예약분에 블록 하나를 더한 값에 못 미치는 예산이라는 것이 두 경로가 함께 부딪힌 상황 그대로이기 때문이다. 해석 로직은 바뀌지 않았다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 3 |
| 프로덕션 동작 변경 | 1 (로그 문구만) |
| 추가 테스트 | 1 |
| 수리한 테스트 | 1 |
| 수리한 테스트에 추가된 단언 | 2 |

### 영역별 변경

**`src/execution/memory_estimate.rs`**
- `resolve_paged_block_budget`: 왜 이 뺄셈이 포화여야 하는지, Metal이 아닌 호스트에서 예약분 미만 예산이 어떻게 도달 가능한지를 숫자와 함께 기록한 주석.
- `resolve_block_budget_explicit_bytes_floors_to_block_count`: 감싸 도는 `-`를 `saturating_sub`로 바꾸고, 산술 검산인 `shortfall < 100`을 상한 있는 예약분 항등식으로 교체하고, 상한 없는 경우를 추가.
- `resolve_block_budget_below_the_workspace_reserve_is_zero_blocks`: 신규 회귀 테스트.

**`src/server/model_worker.rs`**
- `resolve_worker_paged_block_budget`: 0블록 경고가 모델 크기 대신 워크스페이스 예약분을 지목한다.

**`CHANGELOG.md`**
- `## [Unreleased]` / `### Fixed`에 경고 문구 항목 하나. 어떤 예산도 다른 블록 수로 해석되지 않는다는 점을 명시했다.

---

## 5. 검증과 후속

### 통과

- GB10에서 `cargo test --profile test-fast --features cuda --lib -- --exact execution::memory_estimate::tests::resolve_block_budget_explicit_bytes_floors_to_block_count`. 이전에 실패하던 바로 그 케이스다.
- GB10에서 `cargo test --profile test-fast --features cuda --lib execution::memory_estimate`: 36개 통과, 0개 실패.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`.
- `cargo fmt --all -- --check`.

### 덮지 않은 것

- Apple의 작은 CTA 분기는 하드웨어에서 실행하지 않았다. 실행이 아니라 구성으로 덮었다. 모든 단언이 `workspace`에 대한 항등식이거나 거기서 계산한 요청이며, 상한 있는 진술은 경계 양쪽에서 정확하다.
- `--test-threads=1`은 쓰지 않았다. 이 CUDA 호스트에서는 전체 lib 스위트가 병렬 실행 중 이 변경과 무관한 이유로 중단되며, 직렬 전체 실행은 이 장비가 내줄 수 있는 게이트가 아니다. 위의 모듈 필터가 동등한 증거다.
- 예약분 미만 `--kv-cache-budget`으로 실제 모델을 띄운 종단 실행은 없다. 경로는 단위 수준에서 덮었고 경고 문구는 서버 로그에서 관찰한 것이 아니라 읽어서 확인했다.

### 후속

- `PAGED_V2_MAX_N_REP = 16`은 흔한 경우 기준으로 대략 4배 과다 예약이다(Llama 3은 4). 예약분을 넉넉하게 잡은 것은 의도지만, 이제는 작은 명시적 예산에 비해 충분히 커져서 이 값을 조이면 쓸 만한 블록 수로 해석되는 예산 범위가 넓어진다. 이번 범위 밖이고 따로 볼 값어치가 있다.
