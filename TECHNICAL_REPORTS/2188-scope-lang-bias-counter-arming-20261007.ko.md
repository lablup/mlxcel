# 기술 보고서: PR #2188 - Lang-Bias 카운터 활성화를 테스트 스레드로 한정

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 헤드 `c0a70f60`(origin/main `97f35bca` 위로 리베이스), PR 열림, 머지 대기. #2187을 닫음.

**언어**: Rust(mlxcel-core 신규 `lang_bias_counters.rs`, `lib.rs`, `sampling.rs`; mlxcel `sampling_observability_tests.rs`), Markdown(`docs/environment-variables.md`)

**위험도**: 낮음. 프로덕션 게이트의 의미는 그대로입니다. 같은 변수, 같은 파싱, 같은 프로세스 전역 `OnceLock`을 씁니다. 새로 생긴 런타임 동작은 테스트가 설정하지 않는 한 `None`인 스레드 로컬 오버라이드뿐이므로, 프로덕션은 캐시된 값 앞에서 스레드 로컬 `Cell` 하나를 더 읽습니다. 나머지 변경은 모두 테스트와 문서입니다.

## 요약

PR #2182가 Gemma 3 decode lookahead를 복원한 뒤, 그 PR의 스케줄러 테스트 다섯 개가 ROCm에서 단일 스레드로 `-p mlxcel --lib` 전체를 돌릴 때만 실패했습니다. 각 테스트는 단독으로도, 어느 모듈 하나의 테스트와 함께 돌려도 통과했으므로, `make verify-rocm`은 뚜렷한 원인 없이 5개 실패로 깨졌습니다.

피해 테스트보다 앞에 정렬되는 테스트 5,236개를 13단계로 이분 탐색한 결과 `sampling_observability_tests`에 도달했습니다. 이 파일의 모든 테스트는 `counter_lock()`을 호출하고, 이 함수가 `set_var("MLXCEL_LANG_BIAS_COUNTERS", "1")`로 opt-in lang-bias 억제 카운터를 켰습니다. 게이트는 이 변수를 프로세스 전역 `OnceLock`에 캐시하므로, 그 뒤로 프로세스 안의 모든 bias 행이 `row_supports_fused_batch_except_bias`를 통과하지 못했습니다. 피해 테스트는 `ignore_eos`를 설정하고, 이것이 토큰 bias를 설치하므로 `lookahead_params`가 `None`을 반환해 lookahead가 전혀 시작되지 않았습니다.

PR #2188은 게이트를 새 `mlxcel_core::lang_bias_counters` 모듈로 옮기고 스레드 로컬 RAII `scoped_override`를 추가합니다. observability 테스트는 본문 전체 동안 이 가드를 쥐고, `set_var`는 제거되었습니다. 문서화되지 않았던 이 변수는 이제 `docs/environment-variables.md`에 항목이 있습니다. 프로덕션 영향은 없습니다. 누수는 순수 Rust 프로세스 상태이므로 Metal과 CUDA의 lib 실행에서도 똑같이 발생할 수 있습니다.

오케스트레이터가 `c0a70f60`에서 실행한 `make verify-rocm`은 모든 단계를 통과했습니다. 11,993개 통과, 0개 실패, 383개 무시, smoke OK입니다. 이로써 #2182 이후 ROCm 게이트가 다시 0 실패가 되었습니다.

## 1. 문제 정의

### 1.1 증상

origin/main `9c0ae2e9`에서 `cargo test -p mlxcel --lib --profile test-fast --features rocm -- --test-threads=1`은 재현 가능하게 8,844개 통과, 5개 실패, 156개 무시를 보고했습니다. 다섯 실패는 모두 #2182가 `src/server/batch/scheduler/mod.rs`에 추가하거나 다시 켠 테스트입니다.

- `scheduler_completion_snapshot_tests::model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind`("Gemma 3 rewinds its own state, so it pipelines")
- `scheduler_model_owned_lookahead_tests::batched_lookahead_matches_force_sync_past_the_window`와 `single_sequence_lookahead_matches_force_sync_past_the_window`("the lookahead pipeline ran (0 ticks)")
- `scheduler_model_owned_lookahead_tests::failed_rewind_fails_the_request`("the request is still running")
- `scheduler_model_owned_lookahead_tests::lookahead_is_gated_on_the_rewind_capability`("Dense: FP16 Gemma 3 pipelines")

이 테스트들은 단독으로도, `models`, `execution`, `loading`, `server` 모듈 테스트만 함께 돌려도 통과했습니다. 이슈는 스케줄러를 건드리지 않는 PR #2186 검증 중에 등록되었고, 처음 가설(`MLXCEL_FORCE_SYNC`, 전역 KV 캐시 모드)은 추측이었습니다.

### 1.2 이분 탐색

단일 스레드 실행에서 피해 테스트는 늦게 돌기 때문에, 간섭하는 테스트는 그보다 앞에 정렬되어야 합니다. 그런 테스트 5,236개를 반복해서 반으로 나눴습니다. 각 단계는 미리 빌드된 test-fast 바이너리에서 한쪽 절반과 `lookahead_is_gated_on_the_rewind_capability`를 `--exact --test-threads=1`로 실행했으므로 재빌드가 필요 없었습니다. 13단계(5,236의 log2를 올림하면 13) 끝에 `sampling_observability_tests::byte_fragment_suppression_counter_tracks_opt_in_entries`에 도달했습니다.

이 테스트는 정렬 순서상 첫 번째였을 뿐입니다. 파일 안의 어느 테스트든 실패를 일으키는데, 모두 `counter_lock()`을 잡고 `counter_lock()`이 카운터를 켰기 때문입니다. 독립 확인으로, `MLXCEL_LANG_BIAS_COUNTERS=1`을 export한 상태에서 피해 테스트 하나만 돌려도 같은 실패가 재현됩니다. 이로써 원인이 observability 테스트의 다른 부작용이 아니라 이 변수임이 확인됩니다.

### 1.3 누수된 상태

B9 lang-bias observability 카운터는 두 종류입니다. `mlxcel_lang_bias_applied_total`은 비용이 없어 항상 집계됩니다. `mlxcel_lang_bias_tokens_suppressed_total`과 `mlxcel_lang_bias_byte_fragment_suppressions_total`은 bias 적용 전 argmax가 필요하고, 이를 읽으려면 bias가 있는 decode 스텝마다 `eval`과 디바이스에서 호스트로의 읽기가 필요합니다(decode의 13%~78%로 측정됨). 그래서 이 둘은 `MLXCEL_LANG_BIAS_COUNTERS`로 opt-in이며, 켜져 있는 동안 bias 행은 배치 fused 샘플러를 쓸 수 없습니다.

```rust
if !config.token_bias.is_empty() && lang_bias_counters_enabled() {
    return false;
}
```

`lang_bias_counters_enabled()`는 환경 변수를 한 번 읽어 `static OnceLock<bool>`에 담았습니다. observability 테스트는 카운터가 켜져 있어야 했으므로 `counter_lock()`이 `arm_counters()`를 호출했고, 이 함수가 `Once` 아래에서 `set_var("MLXCEL_LANG_BIAS_COUNTERS", "1")`을 실행했습니다. 주석에는 이 테스트 바이너리의 다른 어떤 코드도 비어 있지 않은 bias로 샘플링하지 않는다고 적혀 있었습니다. #2182가 이 전제를 깨뜨렸습니다.

피해 테스트에서의 연쇄는 다음과 같습니다.

1. 스케줄러 테스트는 `ignore_eos`로 요청을 만들고, 이것이 EOS 토큰에 bias를 설치하므로 `config.token_bias`가 비어 있지 않습니다.
2. `lookahead_params`(`src/server/batch/scheduler/decode_tick.rs`)는 `batched_decode_fused_params`를 호출하고, 이 함수는 `row_supports_fused_batch_except_bias`가 false인 행을 거부합니다.
3. 게이트가 `true`로 캐시된 상태에서는 모든 bias 행이 거부되고, `lookahead_params`가 `None`을 반환해 스케줄러가 동기 tick으로 폴백합니다. 테스트는 lookahead tick 0개를 보게 됩니다.

단일 스레드 실행에서는 `sampling_observability_tests`가 `server::batch::scheduler`보다 앞에 정렬되므로 활성화가 항상 먼저 일어났습니다. 병렬 실행에서는 어느 테스트가 게이트를 먼저 건드리느냐에 달려 있습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `mlxcel-core/src/lang_bias_counters.rs`(신규) | `enabled()`: 스레드 로컬 오버라이드가 우선하고, 없으면 `MLXCEL_LANG_BIAS_COUNTERS`를 프로세스 전역 `OnceLock`으로 읽음(파싱은 이전과 동일). `scoped_override(bool) -> ScopedOverride` RAII 가드, `#[must_use]`, `!Send`, `#[doc(hidden)]`. 단위 테스트 `override_is_scoped_nested_and_thread_local` |
| `mlxcel-core/src/lib.rs` | `pub mod lang_bias_counters;` |
| `mlxcel-core/src/sampling.rs` | `lang_bias_counters_enabled()` 제거. `apply_token_bias_stage`와 `row_supports_fused_batch_except_bias`가 `crate::lang_bias_counters::enabled()` 호출. `config_supports_fused_batch_false_for_token_bias`가 게이트를 끈 상태로 고정하고, 켠 상태에서는 bias 행이 fused 경로를 벗어남을 확인 |
| `src/sampling_observability_tests.rs` | `counter_lock()`이 `CounterScope { _counters: ScopedOverride, _lock: MutexGuard }` 반환. `arm_counters()`와 그 안의 `unsafe set_var` 제거. 모듈 문서 갱신 |
| `docs/environment-variables.md` | `MLXCEL_LANG_BIAS_COUNTERS` 항목 신설 |

커밋은 두 개입니다. 수정(`e4ddc1fc`)과, 술어 테스트에서 게이트를 고정하고 문서를 다듬은 리뷰 후속(`c0a70f60`)입니다. 파일 5개, 172줄 추가, 40줄 삭제.

## 3. 설계

### 3.1 리셋 가능한 캐시가 아니라 스레드 로컬 오버라이드

근본 문제는 테스트가 자신보다 오래 사는 프로세스 상태를 바꾼다는 점입니다. 가능한 수정은 여럿이었습니다.

- **테스트 후 변수 해제.** 효과가 없습니다. `OnceLock`이 이미 `true`를 캐시했습니다.
- **캐시를 리셋 가능하게 만들기**(리셋 훅이 있는 `AtomicU8`, 또는 호출마다 환경 변수 읽기). 프로덕션 동작을 바꾸고, 스텝마다 `env::var`를 추가하거나 테스트가 여전히 프로세스 전체에 대해 뒤집을 수 있는 atomic을 남기며, 병렬 테스트가 게이트를 켜는 것과 다른 스레드의 스케줄러 테스트 사이의 경쟁도 그대로 남습니다.
- **observability 테스트를 별도 테스트 바이너리로 분리.** 이번 경우는 격리되지만 다음에 이 변수를 설정하는 테스트에게 같은 함정을 남기고, 링크 비용도 늘어납니다.
- **스레드 로컬 오버라이드(채택).** 오버라이드는 `thread_local! { static OVERRIDE: Cell<Option<bool>> }`에 있습니다. `enabled()`는 오버라이드가 `Some`이면 그 값을, 아니면 캐시된 환경 값을 반환합니다. 호출 스레드 밖에서는 보이지 않으므로, 직렬이든 병렬이든 다른 테스트에 영향이 없습니다.

이 설계는 테스트로 확인되는 한 가지 사실에 기대고 있습니다. observability 테스트의 샘플링 호출은 테스트 스레드에서 실행됩니다. `sample_token_optimized`를 스케줄러 스레드를 거치지 않고 직접 호출하므로 테스트 스레드의 오버라이드가 적용됩니다. PR은 이 전제가 깨지면 크게 실패하도록 해 두었습니다. 변수가 설정되지 않은 상태에서 오버라이드가 샘플링 호출까지 닿지 않으면 `sampling_observability_tests`의 카운터 단언이 실패합니다.

### 3.2 RAII 가드

`scoped_override(enabled)`는 셀에 `Some(enabled)`를 넣고 이전 값을 저장한 가드를 반환하며, `Drop`이 이전 값을 복원합니다. 그래서 중첩이 자연스럽게 됩니다(바깥 `true` 안의 `scoped_override(false)`가 끝나면 `true`로 돌아감). 단위 테스트는 이것과 함께 스레드 격리(새로 띄운 스레드는 오버라이드가 아닌 기준값을 봄)를 확인합니다.

타입 수준의 선택 두 가지가 가드를 안전하게 유지합니다.

- **`!Send`**(`PhantomData<*const ()>`). 다른 스레드에서 가드를 drop하면 수정한 셀이 아니라 그 스레드의 셀을 복원하게 됩니다. 이제 컴파일러가 이동을 거부합니다.
- **`#[must_use]`.** `let _ = scoped_override(true);`는 즉시 drop되어 아무것도 켜지 않습니다.

순서가 뒤바뀐 drop은 오래된 값을 복원합니다. 스택을 두는 대신 문서 주석에 이를 명시했습니다. `#[doc(hidden)]`은 프로덕션이 환경 변수로만 카운터를 설정하므로 이 함수를 공개 문서에서 숨깁니다.

### 3.3 락과 오버라이드의 drop 순서

`CounterScope`는 오버라이드와 뮤텍스 가드를 담습니다. Rust는 구조체 필드를 선언 순서대로 drop하므로 `_counters`를 먼저 선언했습니다. 오버라이드가 락 해제 전에 지워지므로, 이 스레드가 카운터를 켜 둔 동안 다음 observability 테스트가 시작될 수 없습니다. 오버라이드가 스레드 로컬이라 지금은 정확성에 꼭 필요하지는 않지만, 나중에 테스트가 작업을 다른 스레드로 띄우더라도 락이 활성 구간 전체를 덮도록 합니다.

### 3.4 술어 테스트 고정

`config_supports_fused_batch_false_for_token_bias`는 bias 행이 fused 경로에 남는다고 단언했습니다. 개발자가 `MLXCEL_LANG_BIAS_COUNTERS=1`을 export해 두었다면 무관한 이유로 실패했을 것입니다. 이제 본문 동안 `scoped_override(false)`를 쥐고, 중첩된 `scoped_override(true)` 블록에서 행이 fused 경로를 벗어남을 확인합니다. 이 테스트는 게이트의 두 분기를 모두 다루며 셸 환경과 무관해졌습니다.

### 3.5 모듈을 옮긴 이유

게이트는 `sampling.rs` 안의 private 함수였고, `mlxcel` 크레이트의 테스트는 여기에 접근할 수 없습니다. 독립된 public 모듈로 옮기면서 `sampling_observability_tests`가 카운터를 켤 정식 방법이 생겼고, 게이트의 계약(캐시됨, opt-in, fused 경로를 벗어나는 이유, 테스트가 켜야 하는 방법)이 모듈 문서 한 곳에 모였습니다.

## 4. 프로덕션 영향

없습니다. 프로덕션에서 `MLXCEL_LANG_BIAS_COUNTERS`는 서버 시작 전에 설정되거나 설정되지 않으며, 이후 아무것도 바꾸지 않습니다. 따라서 한 모델 패밀리를 서빙하다 다른 패밀리를 서빙하는 서버도 내내 같은 설정을 봅니다. `scoped_override`는 프로덕션 코드에서 호출되지 않으므로 스레드 로컬은 항상 `None`이고, `enabled()`는 이전과 같은 캐시 값을 반환합니다.

비용은 호출마다 스레드 로컬 `Cell::get` 하나이며, 원래 bias가 있는 decode 스텝마다 실행되던 두 호출 지점에서만 발생합니다. 주변 샘플링 작업에 비하면 무시할 수 있습니다.

누수는 백엔드와 무관합니다. 커널이나 디바이스 경로가 아니라 Rust 프로세스 상태에 있으므로, Metal이나 CUDA에서 단일 스레드로 lib을 돌려도 같은 다섯 실패가 나오고, 어느 백엔드든 병렬 실행에서 observability 테스트가 스케줄러 테스트보다 먼저 게이트를 켜면 발생할 수 있습니다. ROCm에서만 드러난 것은 `make verify-rocm`이 lib 테스트를 `--test-threads=1`로 돌리는 게이트이기 때문입니다.

## 5. 문서

`MLXCEL_LANG_BIAS_COUNTERS`는 카운터가 opt-in이 된 뒤로 존재했지만 `docs/environment-variables.md`에 항목이 없었습니다. 새 항목은 다음을 기록합니다.

- 허용 값: `0`이 아닌 비어 있지 않은 값이면 켜짐. 기본값은 꺼짐.
- 집계 대상: 두 억제 카운터. `mlxcel_lang_bias_applied_total`은 설정과 무관하게 집계됨.
- 운영자가 알아야 할 비용과 부작용: 토큰 bias가 있는 요청(`logit_bias`, `ignore_eos`, `--lang-bias`)은 배치 fused 샘플러를 벗어나고, 따라서 lookahead decode 파이프라인도 벗어남. bias 없는 요청은 영향 없음.
- 프로세스당 한 번 읽으므로 서버 시작 전에 설정해야 함.
- 테스트는 변수를 설정하지 않고 `scoped_override`로 카운터를 켬(issue #2187).

lookahead에 대한 부작용은 억제 문제를 진단하려고 카운터를 켜는 사람이 가장 놀랄 만한 부분이었습니다.

## 6. 검증

gfx1151(`--features rocm`, test-fast 프로파일, `--test-threads=1`)에서 실행했습니다.

- **최소 쌍.** `byte_fragment_suppression_counter_tracks_opt_in_entries`와 `model_owned_sequence_primes_the_decode_lookahead_only_with_a_rewind`, `lookahead_is_gated_on_the_rewind_capability` 각각의 조합: 수정 전 1개 실패, 수정 후 2개 통과.
- **lib 전체 실행.** `cargo test -p mlxcel --lib --profile test-fast --features rocm --no-fail-fast -- --test-threads=1`: 8,849개 통과, 0개 실패, 156개 무시(수정 전: 8,844개 통과, 5개 실패).
- **대상 테스트.** `sampling_observability_tests`: 6개 통과. `lang_bias_counters::tests::override_is_scoped_nested_and_thread_local` 통과. `config_supports_fused_batch_false_for_token_bias`는 `MLXCEL_LANG_BIAS_COUNTERS=1`을 export했을 때와 안 했을 때 모두 통과.
- **유닛의 `make verify-rocm`**(`MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`): 스크립트 게이트, fmt, clippy `-D warnings`, 32토큰 smoke 통과. 테스트 바이너리 152개에서 11,987개 통과, 0개 실패, 382개 무시.
- **오케스트레이터의 `make verify-rocm`**(헤드 `c0a70f60`, `97f35bca` 위로 리베이스): 모든 단계 통과, 11,993개 통과, 0개 실패, 383개 무시, smoke OK. #2182 이후 처음으로 ROCm 게이트가 통과했습니다.

검증하지 않은 것: 이 호스트에 없는 Metal과 CUDA. Metal이나 CUDA 경로는 건드리지 않았고, 변경은 백엔드와 무관한 Rust 코드입니다.

## 7. 기술적 선택과 그 이유

- **오버라이드를 스레드로 한정.** 직렬이든 병렬이든 다른 테스트로 새어 나갈 수 없는 유일한 선택이며, 프로덕션 게이트는 그대로 둡니다.
- **프로세스 전역 `OnceLock` 유지.** decode 스텝마다 환경 변수를 읽거나 캐시를 가변으로 만들면 테스트 전용 필요 때문에 프로덕션이 바뀝니다.
- **RAII 가드, `!Send`, `#[must_use]`.** 누수를 다시 만들거나 조용히 아무것도 켜지 않는 오용을 타입으로 막습니다.
- **오버라이드가 샘플링 호출에 닿지 않으면 크게 실패.** 변수가 설정되지 않은 상태에서 오버라이드가 효과가 없으면 observability 단언이 실패하므로, 나중에 샘플링이 테스트 스레드 밖으로 옮겨지는 리팩터링이 드러납니다.
- **술어 테스트를 양쪽으로 고정.** 개발자의 셸에 더 이상 의존하지 않으며, 이번 문제를 일으킨 카운터 켜짐 분기도 테스트합니다.
- **변수를 부작용과 함께 문서화.** fused 경로와 lookahead에서 벗어나는 동작이 진단 스위치를 테스트 실패로 만든 원인이었고, 운영자도 같은 경고를 받습니다.

## 8. 남은 위험과 후속 작업

- **`metric_not_incremented_when_bias_empty`는 병렬 `cargo test`에서 불안정할 수 있음.** 이 테스트는 빈 bias로 샘플링한 뒤 `lang_bias_applied_total() == 0`을 단언합니다. 이 카운터는 항상 켜져 있고 프로세스 전역이며, `counter_lock()`은 이 파일의 테스트끼리만 직렬화합니다. 기본 병렬 하네스에서는 같은 순간 bias로 샘플링하는 다른 테스트(예: `ignore_eos`를 쓰는 스케줄러 테스트)가 `reset_counters()`와 단언 사이에 값을 올릴 수 있습니다. 이 PR 이전부터 있던 문제이고 PR이 바꾸지 않았으며, 단일 스레드 게이트에서는 나타나지 않습니다. 가능한 수정은 모든 bias 샘플러가 존중하는 락 아래에서 증분을 단언하거나, 카운터 단언도 스레드 단위로 만드는 것입니다.
- **다른 캐시된 환경 게이트.** 같은 패턴(`OnceLock`에 캐시되는 변수를 테스트가 `set_var`로 설정)은 다른 곳에도 있을 수 있습니다. 예를 들어 `MLXCEL_APC_TRACE` 등 `OnceLock`으로 읽는 스위치입니다. 이 PR은 게이트를 깨뜨린 하나를 고쳤고, 캐시된 변수에 대한 다른 `set_var` 호출은 감사하지 않았습니다.
- **프로세스 교훈: 공유 호스트에서의 `pkill -f`.** 이 유닛 작업 중에 넓은 패턴의 `pkill -f`가 공유 개발 호스트에서 사용되었습니다. 이 호스트에서는 다른 worktree와 에이전트가 각자의 cargo와 테스트 프로세스를 돌립니다. 패턴 매칭은 다른 실행의 프로세스까지 죽일 수 있습니다. 이제 오케스트레이터의 지침에 규칙으로 들어갔습니다. 자신이 시작한 프로세스만 PID로 종료합니다.

## 9. 학습 포인트

- **테스트 순서에 따른 실패는 누수된 프로세스 상태를 가리킵니다.** 단독으로도, 모듈 부분 집합과 함께여도 통과하지만 전체 직렬 실행에서만 실패하는 테스트는 거의 항상 앞선 어떤 테스트가 전역 상태를 바꾼 경우입니다. 앞선 테스트들을 이분 탐색하면 log2(n)번 실행으로 찾을 수 있습니다.
- **미리 빌드된 바이너리로 이분 탐색.** 이미 빌드된 테스트 바이너리에 절반씩 `--exact`로 실행하니 13단계 각각이 재빌드가 아니라 한 번의 실행이었습니다.
- **처음 걸린 것이 원인 전체는 아닙니다.** 이분 탐색은 테스트 하나를 가리켰지만, 메커니즘은 파일의 모든 테스트가 호출하는 헬퍼에 있었습니다. 환경 변수만으로 재현해 본 것이 메커니즘과 정렬상 처음이었던 테스트를 구분해 주었습니다.
- **`set_var`와 `OnceLock`의 조합은 되돌릴 수 없는 문입니다.** 프로세스 전역 캐시로 읽는 변수를 설정한 테스트는 바이너리의 나머지 전체에 대해 그 값을 바꾸며, 나중에 해제해도 소용없습니다. 테스트 전용 스위치는 범위가 한정된 스레드 로컬 오버라이드가 안전한 형태입니다.
- **다른 코드에 대한 불변식을 적은 주석은 낡습니다.** `arm_counters()`는 "이 테스트 바이너리의 다른 어떤 코드도 비어 있지 않은 bias로 샘플링하지 않는" 동안에만 안전했습니다. #2182는 이 파일을 건드리지 않고도 그 전제를 깨뜨렸습니다.
