# PR #3372: OpenXLA continuous batching의 슬롯별 순환 상태 전달, mlxcel-server로 Mamba2 서빙

**날짜**: 2026-11-24
**상태**: PR 열림(#3340을 닫음, epic #564 Phase 1 단위 U4). ROCm gfx1151에서 IREE `hip`으로 검증했다. attention이 없는 순환 모델만 해당한다. hybrid(attention과 순환 레이어 혼합)는 #3344까지 batch engine이 계속 거부하고, prompt cache 가드는 #3341이다.

## 요약

#3371(#3339) 이후 Mamba2는 xla 백엔드에서 단일 시퀀스로 생성됐지만, continuous batching 엔진은 슬롯별 KV만 알았다. 그래서 `IreeRaggedLlama::load`가 순환 config를 거부했고 `mlxcel-server`는 이를 서빙할 수 없었다. 이 PR은 ragged decode 그래프가 순환 상태 spec마다 `[B, ...dims]` 텐서 하나를 받고 돌려주게 하고, C shim에 KV와 같은 `rg_valid[slot]` 플래그 아래 저장되는 슬롯별 상태 집합 `sscache_b[]`를 두며, attention 없는 모델에 대한 로드 거부를 푼다. 이제 `MLXCEL_BACKEND=xla mlxcel-server -m <mamba2 checkpoint>`가 continuous batching으로 서빙하고, 배치된 모든 요청은 단일 시퀀스 출력과 토큰 단위로 같다.

두 번째 커밋은 #3371의 결정 하나를 뒤집는다. 순환 config는 이제 MLA config처럼 모든 장치에서, 단일 시퀀스와 배치 모두 f32로 고정된다. serve 게이트를 돌려 보니 예전 hip 기본값 f16에서 단일 시퀀스 Mamba2 decode가 일부 프롬프트에서 약 16 토큰 뒤 HF fp32 oracle에서 벗어났고, 배치 스트림을 그렇게 어긋나는 기준과 비교하고 있었다. 고정의 대가는 처리량이다. `mamba2-130m`의 serve 게이트는 f16에서 배치 285.6 tok/s, f32에서 123.7 tok/s였다.

#3371과 마찬가지로 설계와 리뷰의 대부분은 요청 간 격리에 쓰였다. 재사용된 슬롯, 찢어진 저장(torn store), Rust에서 거부된 요청, 취소되거나 연결이 끊긴 요청, 실패한 ragged decode 모두, 어떤 행도 자기가 끝내지 않은 상태를 이어가지 못하게 남겨야 한다.

## 1. 문제 정의

#3371은 상태 ABI의 단일 시퀀스 절반을 넣었고, 배치 절반은 모든 층에서 거부된 채로 남겼다.

- **Emitter.** `emit_decode_ragged_with_native`가 `c.reject_recurrent("ragged decode")`를 호출했고, ragged 인자 스키마는 항상 `kcache` / `vcache`를 선언했다. ragged 경로의 Mamba2 블록은 `unreachable!`였다.
- **Shim.** context에는 슬롯별 KV만 있었다(`kcache_b`, `vcache_b`; `xla_ensure_batch_kv`가 할당하고, `rg_valid` 아래에서 `xla_store_prefill_kv_slot`이 쓰고, ragged decode 뒤 교체). `xla_require_kv`는 `n_state > 0`인 context에서 `xla_llama_ragged_reset`, `prefill_slot_logits`, diagnostics와 embeddings 슬롯 prefill, ragged decode를 거부했다.
- **Runtime.** `IreeRaggedLlama::load`는 순환 상태가 있는 config를 모두 거부했고, 그 컴파일 단계는 항상 embeddings prefill을 emit했다. emitter는 순환 config에서 이 prefill에 `Err`를 돌려준다.
- **필수 모듈 entry.** shim은 `n_state > 0`이면 `decode_step.state_import`를 필수로 요구했다. ragged decode 모듈은 상태를 인자로 넘기므로 그런 entry가 없고, 다른 거부를 다 풀어도 ragged 순환 번들은 로드될 수 없었다.
- **서버 시작.** serve worker는 모든 OpenXLA 체크포인트에 Gemma3n audio tower가 있는지 묻는데, `Gemma3nXlaAudioConfig`는 `config.json`을 엄격하게 파싱했다. Mamba2 config는 `time_step_limit`에 따옴표 없는 `Infinity`를 담으므로, 배치 코드가 돌기도 전에 서버 시작이 `parse config JSON: expected value at line 25`로 실패했다.
- **#3339에서 이월된 격리 규칙**(#3340 이슈 코멘트): 거부되거나 실패한 prefill은 그 슬롯의 상태를 버린다. shim에 닿기 전에 Rust에서 거부된 요청이 슬롯의 이전 상태를 살려 두면 안 된다. 재사용된 슬롯은 이전 점유자의 상태를 보지 않는다. 실패한 ragged decode는 `rg_valid`처럼 살아 있는 모든 슬롯을 무효화해야 한다.

#3340은 두 가지 위험을 적었다. 하나의 유효성 플래그가 KV와 상태를 원자적으로 덮어야 한다(일부만 저장된 상태가 유효로 보이면 요청이 조용히 망가진다). 그리고 `B = 8`에서 ragged 그래프가 상태 입력과 출력만큼 커진다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `csrc/xla_iree.c` | `sscache_b[XLA_MAX_STATE]`; `xla_ensure_batch_state`(0으로 채운 `[rg_bsz, ...dims]`, 전부 아니면 없음, overflow 검사)와 `xla_release_batch_state`; `xla_validate_batch_state`; `xla_store_prefill_kv_slot`이 `xla_store_prefill_slot(c, slot, kc, vc, states)`로 바뀜; 새 export `xla_llama_slot_invalidate`; `xla_require_kv`가 slot을 받아 거부 시 무효화; `ragged_reset`이 상태 집합을 해제; 슬롯 prefill이 `state_zero[]`를 넣고 상태 결과를 꺼냄; ragged decode가 `sscache_b[]`를 넣고 교체; `xla_llama_decode_ragged_impl`이 새 `_inner`를 감싸고 상태가 있는 번들에서 실패 시 모든 `rg_valid` 바이트를 지움; `decode_step.state_import`는 더 이상 필수 entry가 아님; 테스트 카운터 `xla_test_batch_state_{allocs,bytes,reset_counters}` |
| `src/emitter/model.rs` | `RaggedArgs { kv: Option<(Val, Val)>, state: Vec<Val> }`; `has_kv_cache()`일 때만 KV 인자; KV 뒤에 상태 인자; KV가 없으면 RoPE 벡터와 key mask 없음; 결과를 목록으로 구성(토큰 또는 logits, KV가 있으면 K/V, 상태마다 `[B, ...]` 결과 하나); `reject_recurrent`는 hybrid에만 |
| `src/emitter/model_recurrent.rs` | `RecurrentLayers::split_rows` / `join_rows`(레이어별 slice, `dynamic_update_slice`로 되쓰기); ragged Mamba2 블록이 `[B, hidden]`에서 `ssm::mamba2_mixer_step` 실행; `take_batched_state_args` |
| `src/emitter/recurrent_config.rs`, `builder.rs`, `mod.rs` | `Config::pins_f32_precision`(MLA 또는 순환 상태); `resolve_pinned_precision_with_override` |
| `src/emitter/recurrent_ragged_tests.rs`(신규) | emitter 테스트 5개: B=4와 B=8에서 KV 없는 슬롯별 상태, sampling 포함 모든 정밀도에서 f32 상태, 상태별 레이어별 되쓰기 한 번, hybrid 거부, dense ragged 시그니처 불변 |
| `src/emitter/recurrent_tests.rs` | "상태 계약 없는 그래프는 Mamba2를 거부" 테스트에서 ragged 항목 제거; `recurrent_config_pins_f32_unless_precision_is_explicit` 추가 |
| `src/iree.rs` | `effective_precision_with_override`가 모든 dense config에 고정 규칙 적용; `IreeRaggedLlama::load`는 `has_recurrent_state() && has_kv()`만 거부; ragged 컴파일이 순환 config에서 embeddings prefill 생략; 슬롯 진입점 8곳 모두에 `begin_slot_prefill`(슬롯 검증 후 `xla_llama_slot_invalidate`); test-support 카운터 접근자; `recurrent_runtime_pins_f32_unless_precision_is_explicit` |
| `src/gemma3n_audio_config.rs` | 관대한 HF JSON 파서; `audio_config`가 있는 config에 따옴표 처리된 비유한 수가 있으면 거부 |
| `src/lib.rs` | `test-support`에서 batch 상태 카운터 재노출 |
| `tests/xla_ssm_state_carry.rs`(신규) | 합성 2레이어 Mamba2; 장치 테스트 5개(IREE 없으면 ignore) |
| `tests/xla_ssm_single_seq.rs` | 좁아진 `xla_require_kv` 범위에 맞춘 doc 주석 수정 |
| `scripts/xla/validate_arch.sh`, `README.md`, `FINDINGS_ssm.md`, `docs/supported-models.md`, `docs/environment-variables.md`, `CHANGELOG.md` | serve 게이트가 Mamba2에도 적용; 슬롯별 ABI 절; 측정과 정밀도 결정; Mamba2를 서빙 가능으로 표시; `MLXCEL_XLA_PRECISION` 행에 f32 고정; changelog 항목(#3339 항목도 이제 f32가 기본이라고 수정) |

19개 파일, +1430 / -224, 커밋 2개.

### ragged 그래프 계약

attention 없는 순환 config의 ragged `decode_step.main`은 `cache_len`까지 기존 입력을 받고, 이어서 `Config::state_tensors` 항목마다 `[B, ...dims]` 입력 하나(`ssm_h [B, L, H, P, N]`, `ssm_conv [B, L, K-1, conv_dim]`, f32)를 받으며, 토큰 또는 logits 뒤에 진행된 상태 텐서를 돌려준다. `kcache` / `vcache` 입력과 결과, RoPE gather, key mask는 없다. `Ragged` 레이아웃에는 읽히지 않는 스칼라 자리표시자가 들어간다. 행은 서로 독립이다. 비활성 행도 다른 행처럼 진행되고 shim이 그 결과를 버린다. 상태는 어떤 contraction 정밀도에서도 f32다(`mamba2_ragged_decode_samples_and_emits_at_every_precision`). 그래프에는 `scatter`, `stablehlo.while`, `stablehlo.custom_call`, `util.global`이 없고 gather는 정확히 하나(토큰 embedding)다.

`mamba2-130m`의 상태 레이아웃은 #3371에서 `[24, 24, 64, 128]`과 `[24, 3, 1792]`이므로, 슬롯 하나의 행은 18,874,368 + 516,096 바이트(약 18.5 MiB)이고 `B = 8` 집합은 버퍼 세대 하나당 약 148 MiB다. 이 수치는 레이아웃에서 계산한 값이지 측정값이 아니다.

dense ragged MLIR은 main과 바이트 단위로 같다. 검증 golden이 통과하고 `attention_ragged_decode_keeps_its_kv_results`가 dense 시그니처를 확인한다.

### shim 슬롯 수명 주기

| 이벤트 | shim 동작 |
|---|---|
| `xla_llama_ragged_reset` | 기존 KV 리셋과 함께 `sscache_b[]`를 해제한다. 다음 슬롯 저장이 새로 0으로 채운 집합을 할당하므로 리셋을 넘어 살아남는 행은 없다. |
| Rust 슬롯 진입(`begin_slot_prefill`) | 슬롯 범위를 검사한 뒤 다른 어떤 검사보다 먼저 `xla_llama_slot_invalidate`를 호출한다. Rust에서 거부된 요청(빈 프롬프트, 너무 큰 프롬프트, 잘못된 prepared payload)은 슬롯을 무효 상태로 남긴다. |
| `xla_llama_prefill_slot_logits` | `rg_valid[slot]`을 지우고, `state_zero[]`를 다시 0으로 채워 `real_len` 뒤에 넣고, `has_kv`일 때만 K/V를 꺼내고, 상태 결과마다 꺼내서 shape를 검증한 뒤 `xla_store_prefill_slot`을 호출한다. |
| `xla_store_prefill_slot` | 플래그를 지우고, KV와 모든 상태 결과를 검증하고, batch KV와 상태 집합이 있는지 보장하고, K, V(`has_kv`일 때)와 상태 행마다 `slot * row_bytes` 위치로 device 간 복사한 뒤, 마지막 복사가 끝나야 플래그를 세운다. 실패 시 롤백은 없고, 이후 prefill이 성공할 때까지 슬롯은 무효로 남는다. |
| ragged decode | 모든 `sscache_b[i]`가 있어야 진행한다. KV 뒤에 `sscache_b[]`를 넣고, 각 `[rg_bsz, ...dims]` 결과를 검증하고, 모든 결과가 검증된 뒤 KV와 상태를 함께 교체한다. |
| 실패한 ragged decode(호출 전, 호출, 검증, readback) | 아무것도 교체하지 않는다. `n_state > 0`인 번들에서는 모든 `rg_valid` 바이트를 지운다. attention 전용 번들은 기존처럼 플래그를 유지한다. |
| embeddings와 diagnostics 슬롯 prefill | 상태가 있는 context는 여전히 `xla_require_kv`가 거부한다. 이제 거부할 때 슬롯도 무효화한다. |
| `xla_llama_free` | `xla_release_state`가 슬롯별 집합도 해제한다. |

취소와 연결 끊김은 스케줄러 슬롯만 해제한다. 낡은 행을 지우지 않는 이유는, 그 슬롯에 다음으로 들어오는 요청이 0 상태 prefill로 행 전체를 덮어쓰고, 그때까지 플래그가 꺼져 있기 때문이다.

### Rust와 엔진

`XlaBatchEngine`의 `Slot`, `XlaBucketSet`, `src/server/batch/xla_worker.rs`에는 필드가 추가되지 않았다. 상태는 KV처럼 슬롯을 키로 장치에 있다. bucket context는 각자 `sscache_b` 집합을 가진다. `IreeRaggedLlama::load`는 이제 hybrid만 거부하고, 그 컴파일 단계는 단일 시퀀스 번들처럼 순환 config에서 embeddings prefill을 emit하지 않는다.

## 3. 기술적 선택과 그 이유

**ragged decode에서 상태는 global이 아니라 인자로 넘긴다.** 단일 시퀀스 decode는 `@state_import`로 채우는 모듈 global에 상태를 둔다. ragged 경로는 이미 KV를 입력으로 넘기고 출력으로 교체하므로 상태도 같은 패턴을 따른다. spec마다 입력 하나와 결과 하나이고 shim에서 교체한다. 이렇게 하면 배치에는 슬롯 주소 기반 저장 방식 하나(행으로 인덱싱하는 device 버퍼)만 남고, 슬롯 prefill은 device 복사 한 번으로 행 하나를 쓸 수 있으며, 슬롯별 import entry가 필요 없다. ragged 그래프에 `util.global`이 없다는 것을 테스트가 확인한다. 대가는 인자가 donate되지 않아 매 step마다 장치에 `B x row` 바이트짜리 새 상태 버퍼를 쓴다는 점인데, KV도 이미 같은 패턴이다.

**scatter가 아니라 레이어별 slice와 `dynamic_update_slice`.** `split_rows`는 순환 레이어마다 `[B, 1, ...]`을 잘라 `[B, ...]`로 reshape하고, `join_rows`는 진행된 레이어를 그 레이어 인덱스에 되쓴다. 매 step 배치의 모든 행을 쓰므로 행별 인덱스가 필요 없고, ragged 테스트가 금지하는 `scatter`가 그래프에 들어가지 않는다. `mamba2_ragged_decode_writes_every_layer_row_back`는 상태 텐서별 레이어별 쓰기 한 번이라는 개수를 고정한다.

**KV가 없으면 RoPE도 mask도 없다.** attention 없는 모델은 둘 다 읽지 않는다. emit하면 IREE가 접을 수도 있고 못 접을 수도 있는 gather와 compare가 늘어난다. 생략하면 부재를 검사할 수 있고(그래프 전체에 gather 하나), dense emit 경로는 건드리지 않으며, 바이트 동일 golden이 이를 확인한다.

**KV와 상태에 유효성 플래그 하나.** 이슈의 핵심 위험에 대한 답이다. 상태 플래그를 따로 두면 ragged decode가 KV는 유효하고 상태는 아닌 슬롯(또는 반대)을 볼 수 있다. 첫 전송 전에 지우고 마지막 전송 뒤에 세우는 플래그 하나면, `ssm_h` 뒤 `ssm_conv` 전에 실패한 저장은 통째로 무효다. 테스트 hook은 attention 없는 모델에서 첫 상태 전송 뒤에 실패하도록 옮겼는데, 이슈가 torn-store 테스트로 잡으라고 한 바로 그 경우다.

**모든 슬롯 prefill은 0 상태에서 시작한다.** 슬롯 prefill은 행의 현재 내용이 아니라 항상 `state_zero[]`를 넣고 행 전체를 덮어쓴다. 그래서 재사용된 슬롯은 이전 점유자의 상태를 읽을 수 없고, 축출(취소, 연결 끊김, 완료)에는 device 작업이 필요 없다. 축출 시 행을 0으로 채우는 대안은 지금은 아무 일도 하지 않는 경로에 device fill을 추가하고, 실패 뒤의 안전을 위해서는 여전히 입장 시 덮어쓰기 보장이 필요하다.

**어떤 검사보다 먼저 Rust에서 무효화.** #3371의 교훈을 슬롯에 적용한 것이다. Rust 엔진이 거부한 요청은 shim의 무효화에 닿지 않으므로, `begin_slot_prefill`은 슬롯 진입점 8곳 모두에서 범위 검사 직후 `xla_llama_slot_invalidate`를 호출한다. 이를 되돌리면 `refused_requests_and_failed_decodes_invalidate_slot_state`가 실패한다.

**실패한 decode는 상태가 있는 번들에서만 모든 슬롯을 무효화.** 실패한 step은 아무것도 교체하지 않으므로 두 경우 모두 상주 버퍼는 step 이전 내용을 유지한다. 그래도 shim은 일부만 실행된 순환 step이 믿을 만한 행별 상태를 남기지 않는다고 보고, 슬롯이 다시 prefill될 때까지 모든 행을 거부한다. #3339 보안 리뷰의 LOW 항목(재시도된 decode가 일부 진행된 상태에서 이어가는 문제)의 배치 버전이다. attention 전용 번들은 기존 동작을 유지하므로 dense 경로는 바뀌지 않는다. 운영 환경에서는 pump 오류가 이미 `fail_all` 뒤 serve worker를 끝내므로 이 규칙 단독의 효과는 작다. 직접 `IreeRaggedLlama`를 쓰는 호출자와 이후의 복구 경로에서 의미가 있다.

**지연 할당, 리셋 시 해제.** `xla_ensure_batch_state`는 `xla_ensure_batch_kv`처럼 리셋 뒤 첫 슬롯 저장에서 할당하고, 끝나면 집합 전체가 있거나 아예 없다. 운영 환경은 로드 시에만 리셋하므로 클라이언트가 재할당을 일으킬 수 없다. 리셋이 집합을 해제하지 않으면 `state_allocation_counters_show_one_allocation_per_reset`가 실패한다.

**상태 계약이 아직 없는 경로에는 `xla_require_kv` 유지.** embeddings와 diagnostics 슬롯 prefill에는 상태 입력과 출력이 없다. 이들은 상태가 있는 context를 계속 거부하고, 이제 슬롯도 무효화한다. emitter와 `IreeRaggedLlama::load`는 hybrid를 계속 거부하므로, KV와 상태를 함께 가진 모델이 한쪽에 대해서만 검증된 경로를 타지 않는다.

**`decode_step.state_import`는 선택 entry.** 단일 시퀀스 decode 모듈에만 있다. 필수로 두면 ragged 번들이 로드되지 않는다. 단일 시퀀스 경로는 이제 로드 시가 아니라 첫 import에서 거부한다(LOW, 리뷰에서 수용; import가 pending으로 남으므로 decode가 낡은 global로 실행되지 않는다).

**순환 config를 f32로 고정(#3371 결정 번복).** #3371은 40 토큰 프롬프트 하나가 token-exact였으므로 hip 기본값 f16을 유지했다. 이 PR의 serve 게이트는 f16 차이 두 건을 찾았다(64 토큰 프롬프트 `xla_batch_bench` B=4에서 7/8, 요청 6이 토큰 17부터 다름; 서버 검사 3/4). 두 작업 모두 f32에서는 정확했다. 다른 결과를 낸 서버 프롬프트("In computer science, a linked list is")에서 배치 f16 스트림은 HF fp32 oracle과 같았고, 16 토큰 뒤 벗어난 쪽은 단일 시퀀스 f16 decode였다. 차이는 `[B, hidden]` projection과 단일 행 projection 사이의 반올림이지 상태 처리 문제가 아니다. 선택지는 f16을 유지하고 어긋나는 기준을 받아들이기, ragged 번들만 고정하기(CLI가 oracle과 서버 양쪽에서 어긋난 채로 남음), 둘 다 고정하기였다. CLI, 서버, oracle이 모두 일치하는 선택은 둘 다 고정하는 것뿐이다. `Config::pins_f32_precision`이 이를 기존 MLA 규칙에 합치고, 명시적 `MLXCEL_XLA_PRECISION`은 측정 목적으로 여전히 고정을 덮어쓴다.

**Gemma3n audio probe에는 관대한 파서, 필요한 곳에서만 엄격하게.** 이 probe는 서버 시작 시 모든 OpenXLA 체크포인트에서 돌므로 Mamba2의 따옴표 없는 `Infinity`를 파싱해야 한다. 실제로 `audio_config`가 있는 config는 따옴표 처리가 필요했다면 여전히 거부한다. audio reader가 따옴표 처리된 수를 없는 필드로 보고 기본값을 쓰기 때문이다.

## 4. 리뷰 결과

**구현 리뷰**(orchestrator 필수 수정, 0807595c):

| 발견 | 심각도 | 상태 |
|---|---|---|
| hip 기본값 f16에서 단일 시퀀스 기준이 HF fp32 oracle에서 벗어나, serve 게이트(7/8)와 서버 검사(3/4)가 배치 출력을 틀린 기준과 비교함 | HIGH | 수정: 순환 config를 MLA처럼 f32로 고정, 명시적 `MLXCEL_XLA_PRECISION`으로 덮어쓰기 가능; 고정이 없으면 `recurrent_config_pins_f32_unless_precision_is_explicit` 실패; 새 기본값에서 모든 게이트 재실행 |
| `decode_step.state_import`가 없는 단일 시퀀스 번들이 로드 시가 아니라 첫 decode에서 실패 | LOW | 열림, 수용: 거부 메시지가 명확하고 import가 pending으로 남음 |
| #3344가 f16 Nemotron-H 실행을 계획함; `MLXCEL_XLA_PRECISION`을 설정하지 않으면 hybrid도 f32 고정을 물려받음 | 참고 | #3344용 |

**보안 및 성능 리뷰**(head 0807595c): CRITICAL, HIGH, MEDIUM 발견 없음, 커밋 없음.

확인한 것: 거부된 입장, torn store, 재사용된 슬롯, 취소, 연결 끊김, 실패한 ragged decode(호출 전, 호출, 검증, readback), bucket 전환, 종료에서의 요청 간 격리. 범위: 슬롯은 어떤 offset 계산보다 먼저 범위 검사된다. `rg_bsz * row`는 `xla_ensure_batch_state`에서 검사되고, `rg_bsz`를 바꾸는 리셋이 집합도 해제하므로 이후 저장에서도 그 경계가 유효하다. `state_desc`는 create 시에만 쓰이고(bucket에는 복사), prefill과 decode 결과는 복사나 교체 전에 그것과 shape를 대조한다. 자원 사용: 상태 집합은 bucket마다 `B x row`로 고정되고 리셋마다 한 번 할당된다. step마다 상태는 호스트 복사 없이 장치 안에서 넘어간다.

| 발견 | 심각도 | 상태 |
|---|---|---|
| 고정된 config(MLA, 순환)에서 인식되지 않는 `MLXCEL_XLA_PRECISION`(예: `fp8`, `F16`)은 무시되고 f32가 쓰이지만, dense config는 이를 거부함 | LOW | 열림: 안전한 방향이고 MLA는 이 PR 이전에도 그랬지만, 운영자의 오타가 보고되지 않음 |
| prefill이나 decode 오류 하나가 serve worker 전체를 멈추고, 실패한 pump에서 꺼낸 요청은 버려짐 | 기존 문제 | 범위 밖 |

**Finalize**: 변경 없음. 문서와 CHANGELOG에 이미 f32 기본값이 적혀 있었다.

## 5. 테스트와 검증

별도 표기가 없으면 모두 ROCm gfx1151, IREE `hip`, `AntonV/mamba2-130m-hf` @ `05e8773f`, 고정 이후 기본 정밀도(f32)에서 실행했다.

고정 이전(당시 hip 기본값 f16):

| 검사 | 정밀도 | 결과 |
|-------|-----------|--------|
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24` | f16 | PASS |
| `xla_batch_bench`, oracle 프롬프트, `--cancel 2` | f16 | PASS |
| `xla_batch_bench`, oracle 프롬프트, `--buckets 256,512` | f16 | PASS |
| `xla_batch_bench`, B=8, 요청 16개, 64 토큰 프롬프트 | f16 | PASS |
| `xla_batch_bench`, B=4, 요청 8개, 64 토큰 프롬프트 | f16 | 7/8(요청 6이 토큰 17부터 다름) |
| 같은 작업 | f32 | PASS |
| 서버, 동시 4개 temperature 0 `/v1/completions` 대 `mlxcel generate` | f16 | 3/4 |
| 같은 작업 | f32 | 4/4 |

고정 이후:

| 검사 | 결과 |
|-------|--------|
| `mlxcel generate`, greedy 40 토큰 대 HF fp32 oracle, 프롬프트 5개("The capital of France is", "In computer science, a linked list is", "Once upon a time, in a small village by the sea,", "The three primary colors are", "Water boils at a temperature of") | 5/5 token-exact; `xla_oracle_check` 5/5 |
| `validate_arch.sh --batch 4 --requests 8 --maxcap 24`(structural, token-exact, serve reference-exact) | PASS; 배치 123.7 tok/s 대 순차 75.0 |
| `xla_batch_bench`, B=4, 요청 8개, 64 토큰 프롬프트 | PASS(8/8); 122.8 tok/s |
| `xla_batch_bench`, oracle 프롬프트, `--cancel 2` | PASS(2/2 취소); 122.5 tok/s |
| `xla_batch_bench`, oracle 프롬프트, `--buckets 256,512` | PASS; 98.3 tok/s |
| `xla_batch_bench`, B=8, 요청 16개, 64 토큰 프롬프트 | PASS; 108.7 tok/s |
| `mlxcel-server`, 동시 4개 temperature 0 `/v1/completions` | 4/4 CLI와 동일 |

처리량 수치는 `FINDINGS_ssm.md`에 기록된 단일 실행 값이며 분산은 측정하지 않았다.

`tests/xla_ssm_state_carry.rs`(합성 2레이어 HF 레이아웃 Mamba2, capacity 32, B=4; 모든 배치 스트림을 `IreeLlama` 단일 시퀀스 기준과 비교):

- `mid_stream_admit_into_a_recycled_slot_matches_the_reference`
- `half_full_and_full_batches_give_identical_per_request_ids`
- `torn_state_store_leaves_the_slot_invalid_and_the_next_admit_recovers`(`IreeRaggedLlama::test_set_slot_store_failure` 사용)
- `refused_requests_and_failed_decodes_invalidate_slot_state`
- `state_allocation_counters_show_one_allocation_per_reset`

`cargo test --release --features xla-iree,test-utils --test xla_ssm_state_carry -- --ignored`로 실행: hip에서 5/5. `xla_ssm_single_seq` 6/6, `xla_ragged_hardening` 6/6. 각 전달 수정을 되돌리면 해당 테스트가 실패했다. Rust 슬롯 무효화와 decode 실패 시 전체 무효화를 되돌리면 `refused_requests_and_failed_decodes_invalidate_slot_state`, 상태 행 복사를 되돌리면 테스트 4개, 리셋 시 해제를 되돌리면 `state_allocation_counters_show_one_allocation_per_reset`가 실패한다.

추가된 단위 테스트: `recurrent_ragged_tests.rs` 5개, 정밀도 고정용으로 `recurrent_tests.rs` 1개와 `iree.rs` 1개(둘 다 고정이 없으면 실패), `gemma3n_audio_config.rs` 1개. `cargo test --profile test-fast -p mlxcel-xla --lib`: PR 본문 기준 PR head에서 520개 통과. 이 리포트를 쓸 때 브랜치가 이미 origin/main과 같았으므로 리포트를 위해 다시 돌리지는 않았다.

main(e062a10e) 대비 dense 회귀, hip: `xla_ragged_hardening` 6/6, `xla_greedy_timing` 3/3, `xla_decode_scalar_reuse` 4/4, `xla_prepared_prefill` 4/4로 양쪽 동일. `llama-3.2-1b` `xla_batch_bench` B=4, 요청 8개: 양쪽 REFERENCE-EXACT PASS, 이전 64.56 tok/s, 이후 63.53 tok/s(단일 실행).

정적 게이트: `cargo fmt --check`; `cargo clippy --features xla-iree --all-targets -- -D warnings`; `iree` 유무 양쪽의 mlxcel-xla clippy.

## 6. 남은 작업과 한계

- **hybrid는 거부된다.** shim의 저장과 decode는 이미 `has_kv`와 상태를 함께 처리한다. KV와 상태를 함께 가진 config에 대한 emitter의 `reject_recurrent`와 `IreeRaggedLlama::load`의 hybrid 거부가 #3344가 풀어야 할 지점이다. hybrid도 f32 고정을 물려받는다.
- **prompt cache.** 순환 상태는 토큰 prefix 해시로 다시 만들 수 없다. 순환 모델을 prompt cache와 APC에서 배제하는 가드는 이 PR이 아니라 #3341이다.
- **f16은 순환 모델에서 token-exact가 아니다.** `MLXCEL_XLA_PRECISION`으로 측정 목적에만 쓸 수 있다. 고정으로 `mamba2-130m` serve 게이트의 배치 처리량이 285.6에서 123.7 tok/s로 줄었다. 단일 시퀀스의 f32 대 f16 처리량은 측정하지 않았다.
- **실제 체크포인트는 하나.** token exactness의 근거는 `mamba2-130m`(`n_groups = 1`), 40 토큰 프롬프트 5개, 합성 fixture뿐이다. 더 큰 Mamba2 체크포인트는 돌리지 않았다.
- **상태 메모리와 그래프 크기.** `mamba2-130m`에서 슬롯별 집합은 슬롯당 약 18.5 MiB(계산값)이고, 인자가 donate되지 않으므로 step마다 새 집합을 쓴다. 이슈가 위험으로 적은 `B = 8`에서의 컴파일 시간과 그래프 크기는 따로 측정하지 않았다.
- **`state_import` 없는 단일 시퀀스 번들**은 로드 시가 아니라 첫 decode에서 실패한다(LOW, 수용).
- **고정된 config에서 인식되지 않는 정밀도 override**는 조용히 무시된다(LOW, MLA에는 이전부터 있던 동작).
- **serve worker 실패 방식**(prefill이나 decode 오류가 worker를 멈추고 꺼낸 요청이 버려짐)은 기존 문제이고 바뀌지 않았다.
- **CUDA와 Metal은 돌리지 않았다.** 각각 #3342와 #3343이다.

## 7. 학습 포인트

- 슬롯별 KV에 슬롯별 상태가 더해지면 같은 유효성 플래그 아래 두어야 한다. 플래그가 둘이면 어떤 테스트도 열거하지 않는 조합(KV는 유효, 상태는 아님)이 생긴다. 첫 복사 전에 지우고 마지막 복사 뒤에 세우는 플래그 하나면 모든 부분 저장이 구조적으로 무효다.
- 모든 입장을 알려진 0 상태에서 시작하면 축출에 비용이 들지 않는다. 입장 경로가 행의 이전 내용을 절대 읽지 않으면 취소, 연결 끊김, 완료에 device 작업이 필요 없고, 격리 논증은 모든 종료 경로가 아니라 코드 경로 하나에만 기대게 된다.
- #3371의 규칙은 일반화된다. 네이티브 층이 실패 시 무엇을 하든, 그 위에서 거부된 요청은 그 정리를 일으키지 않는다. 각 진입점 맨 위, 검증보다 먼저 무효화하는 것이 견고한 형태다.
- 같은 경로라도 상태 종류에 따라 복구 규칙이 다를 수 있다. KV 행은 append-only라 재시도된 step이 같은 위치를 다시 쓴다. 순환 행은 누적 요약이라, 진행 정도를 모르는 step에서 이어가는 것은 shape로 검사할 수 없다. 다시 입장할 때까지 모든 행을 거부하는 규칙은 실패한 step이 어디까지 갔는지 따질 필요가 없다.
- serve 게이트는 정밀도 게이트이기도 하다. 배치 출력을 단일 시퀀스 출력과 비교하면서 #3371의 프롬프트 하나짜리 oracle 검사보다 많은 프롬프트와 토큰을 다뤘고, f16에서는 단일 시퀀스 기준 자체가 틀렸다는 것이 드러났다. 맞은 쪽은 배치 스트림이었고 틀린 쪽은 기준이었다. 둘 다 fp32 oracle과 대조한 것이 이 둘을 갈라냈다.
- 모든 체크포인트에서 도는 시작 probe에는 어떤 reader보다 관대한 파서가 필요하고, 엄격한 검사는 그 결과를 실제로 쓰는 곳에만 둔다. Gemma3n audio probe는 Mamba2와 상관이 없으면서도 Mamba2 서빙을 막았다.
