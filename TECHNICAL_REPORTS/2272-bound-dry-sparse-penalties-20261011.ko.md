# 기술 보고서: PR #2272 - DRY 및 sparse penalty 단계 비용 제한

**날짜**: 2026-10-11
**상태**: CUDA(GB10)에서 구현 및 검증 완료
**언어**: Rust
**위험 수준**: 중간

## 요약

이 변경은 sampling에서 요청 크기에 따라 증폭되던 비용 두 가지를 제거합니다. DRY suffix matching은 reverse Z를 사용하고 breaker 판별은 reversed-token Aho-Corasick matcher를 사용합니다. 따라서 선택된 history 길이 `N`과 설정된 breaker pattern token 수 `P`에 대해 한 단계의 host 작업이 quadratic suffix scan이나 `N × P` breaker 검사 대신 expected `O(N + P)`가 됩니다. DRY와 rebuild frequency/presence update는 vocabulary 크기의 host buffer를 할당하고 upload하지 않고, 변경되는 token ID만 scatter합니다.

구현은 기존 경로의 dtype과 element 값을 정확히 보존합니다. 이를 위해 이슈의 원래 가정 대신 고정된 CUDA MLX promotion table을 따랐습니다. CUDA에서는 bf16/f32 mixed binary arithmetic 결과가 bf16으로 남을 수 있습니다. Scalar zero 연산으로 sparse scatter 전에 같은 backend별 dtype을 결정하고, 기존 incremental frequency/presence 경로의 명시적 f32 promotion도 유지합니다.

## 1. 문제 정의

기존 DRY 구현은 token position을 index한 뒤 현재 token의 이전 occurrence마다 뒤로 걸으며 비교했습니다. Period `p`, full-history window 길이 `N`인 stream에서는 한 단계가 약 `N² / (2p)`번 비교했습니다. Server request는 prompt를 포함한 full history를 선택할 수 있으므로 요청이 제어하는 history 길이에 따라 작업량이 증가했습니다.

Breaker 처리에도 명시적인 상한이 필요했습니다. 각 위치에서 보존된 scalar breaker predicate를 호출하면 exact-ID vector와 multi-token tail collection을 반복해서 scan하므로 history window와 breaker 설정 크기를 곱한 작업이 생깁니다. 이와 별도로 DRY와 rebuild frequency/presence penalty는 소수의 token ID만 바뀌어도 vocabulary 크기의 host vector를 만들었습니다.

## 2. 변경 요약

### 2.1 제한된 DRY matching

`dry_penalty_map`은 선택된 history를 뒤집고 Z algorithm을 한 번 실행합니다. 이전 occurrence `pos`에 대해 `Z[last_pos - pos]`가 기존 backward walk와 같은 common-suffix 길이를 제공합니다. Forward pass는 각 candidate보다 엄격히 앞에 있는 가장 최근 breaker를 저장하므로 candidate 위치 자체의 breaker가 match를 자르지 않는 규칙을 보존합니다.

Exact breaker는 singleton pattern이 됩니다. Forward multi-token breaker `[head] + tail`은 automaton에 `reverse(tail) + [head]`로 삽입합니다. Reversed position `r`에서 도달한 terminal은 언제나 원래 위치 `N - 1 - r`에 대응하므로 failure link를 따라 terminal state를 전파하면 모든 match를 나열하지 않고도 overlap 및 suffix pattern을 표시할 수 있습니다. Sparse hash transition을 사용할 때 trie 구성은 expected `O(P)`, scan은 expected `O(N)`입니다.

### 2.2 Sparse device update

DRY penalty는 next-token ID별 최대 penalty 하나로 축약한 뒤 `take_along_axis`, 산술 연산, `put_along_axis`로 모든 logits row에 scatter합니다. Rebuild frequency/presence 경로도 변경되는 ID와 값만 만들고 같은 sparse helper를 사용합니다. 어느 경로도 vocabulary 크기에 비례하는 host vector를 할당하지 않습니다.

### 2.3 정확한 dtype 보존

Dense DRY reference는 `add(logits, f32_buffer)`를 사용했고 rebuild frequency/presence reference는 `subtract(logits, f32_buffer)`를 사용했습니다. 고정된 CUDA MLX patch에서 bf16/f32 arithmetic은 bf16을 만들 수 있고, f16 및 patch되지 않은 backend의 promotion은 다릅니다. 각 rebuild 경로는 scalar f32 zero에 대응 연산을 먼저 적용하여 sparse scatter 전에 같은 platform result dtype과 변경되지 않은 값을 얻습니다.

Incremental `SamplerState` 경로는 이 변경 전부터 다르게 동작했습니다. 범위 안 ID가 있으면 logits를 명시적으로 f32로 cast하고 모든 ID가 범위 밖이면 변경되지 않은 copy를 반환했습니다. Shared sparse helper에는 promotion된 array를 전달하며 두 분기의 기존 dtype 동작을 모두 유지합니다.

## 3. 기술적 선택과 그 이유

### 3.1 Stateless per-step matching

Persistent DRY index는 반복 작업을 더 줄일 수 있지만 history divergence에서 rebuild하고 sliding window를 rebase하며 token이 추가될 때 결과가 달라지는 breaker tail을 다시 확인해야 합니다. Stateless reverse-Z와 automaton pass는 sampler-state synchronization contract를 추가하지 않고 요청 semantics와 제한된 작업량을 보존합니다.

### 3.2 반복 predicate 대신 compiled matcher

첫 linear DRY 초안은 여전히 모든 위치에서 `dry_breaks_at`을 호출하여 `O(N × P)` 설정 증폭을 남겼습니다. 최종 matcher는 단계마다 exact ID와 tail을 한 번 preprocess합니다. 변경하지 않은 scalar predicate와 위치별로 직접 비교하는 test가 negative ID, duplicate, empty tail, overlapping suffix, window 끝에 정확히 맞는 pattern, window 밖으로 넘어가는 tail을 검증합니다.

### 3.3 강제 f32 대신 backend가 결정하는 promotion

모든 sparse rebuild 경로를 f32로 강제하면 CUDA bf16 출력이 바뀝니다. Scalar zero arithmetic은 vocabulary 크기의 buffer를 다시 만들지 않고 기존 binary-operation contract를 유지합니다. 기존 incremental f32 동작은 rebuild semantics와 통합하지 않고 명시적으로 유지합니다.

## 4. 검증

측정 source commit은 base `73700aafa6a792f07830578c0c26137cea3a038b` 위의 `ecb89a308aecc468dd1886d9c9f4982f556f3c93`입니다. CUDA GB10 gate에서 core build는 26.74초, clippy는 9.08초에 끝났고 formatting과 Rust 검사가 통과했습니다. Sampling selection은 수동 `dry_step_cost_report`만 ignore한 상태로 210개 test가 통과했고 engine selection은 63개 test가 통과했습니다.

PR 공개 전 구현을 `3ae626c2a4d67e7628a4050f01cc62b7419296fb` 위로 rebase했으며 결과 checkout은 `327b6bf13a31d0f98e4cce717443d03b4be2cb73`입니다. `sampling.rs` blob은 `59fb6b02c6b54913aa549085fafc090e0b516d55`로 동일하고, 측정 checkout과 공개 checkout 사이에서 `src/lib/mlxcel-core`, `src/lib/mlx-cpp`, `Cargo.lock`을 비교한 diff는 비어 있습니다. Raw evidence package에는 공개된 측정 base에서 시작하는 원본 2-commit format patch와 hash 및 빈 diff 증명이 포함됩니다.

Sampling coverage는 window sentinel, 길이 2,048까지의 periodic 및 random history, batch width 1과 2, f32 및 bf16 logits, exact/empty-tail/nonempty-tail breaker를 조합한 deterministic parity case 512개를 포함합니다. Frequency/presence rebuild parity는 window 1과 64, batch width 1과 2, f32/f16/bf16 logits를 포함합니다. 별도 regression은 incremental bf16 경로가 계속 f32로 promote되고 모든 ID가 범위 밖일 때의 early return을 보존함을 확인합니다.

Instrumented complexity test는 reverse-Z 작업을 `8N` 아래로 제한하고 window를 4,096에서 8,192로 두 배 늘릴 때 작업량이 거의 두 배임을 확인합니다. 또한 exact breaker 2,048개의 build와 scan이 additive임을 검증하고, 긴 failure walk를 유발하는 128-pattern shared-trunk comb를 사용하여 구성이 `8P`, scan이 `3N` 아래임을 확인합니다.

독립 implementation review와 security review에서 남은 finding은 없었습니다.

## 5. Timing 증거

수동 test는 vocabulary 4, breaker 없음, period-3 history와 1,024, 4,096, 16,384 window에서 정확한 변경 전 reference와 새 구현을 모두 실행합니다. Reference iteration 수는 3, 3, 1이고 새 경로는 각 window에서 20회 실행합니다. `black_box`는 lazy MLX array를 평가하지 않고 받으므로 아래 수치는 GPU end-to-end latency가 아니라 host matching 및 MLX graph construction 시간입니다.

| Window | Reference µs/step | 새 구현 µs/step | Speedup |
|---:|---:|---:|---:|
| 1,024 | 619.751 | 23.607 | 26.25× |
| 4,096 | 4889.370 | 29.997 | 163.00× |
| 16,384 | 77199.293 | 111.433 | 692.79× |

Window가 16배 늘어나는 동안 reference 비용은 124.57배, 새 경로는 4.72배 증가했으며 16,384-token speedup은 692.79배였습니다.

Raw command output, 내부 및 외부 driver, environment, host-gate output, build/test log, checksum, binary/source provenance는 `docs/benchmark_results/data/dry-penalty-step-cost-gb10-2026-10-11/`에 공개합니다.

## 6. 호환성과 제한 사항

Request field, default, public API는 바뀌지 않습니다. Full-history, zero-window, short-history, low-base, breaker, exponent-cap, max-per-token semantics는 reference parity로 유지됩니다. XLA sampler와 windowed repetition penalty는 이 이슈의 범위 밖입니다.

Automaton은 단계마다 다시 구성하므로 전체 작업은 history 길이뿐 아니라 설정된 pattern token도 포함합니다. Hash-map transition을 사용하므로 명시한 상한은 deterministic worst-case가 아니라 expected bound입니다. Timing test는 quiet GB10 host 한 대의 scaling과 local cost에 대한 증거이며 cross-platform throughput 보장은 아닙니다.

## 7. 관련 기록

- Issue #2259: 구현 요구 사항
- Issue #2229 및 #2256: security review 배경
- `docs/benchmark_results/dry-penalty-step-cost-gb10-2026-10-11.md`: 날짜가 기록된 timing 결과
- `docs/benchmark_results/data/dry-penalty-step-cost-gb10-2026-10-11/`: raw evidence package
