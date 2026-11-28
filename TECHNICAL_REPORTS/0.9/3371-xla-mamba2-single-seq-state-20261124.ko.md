# PR #3371: OpenXLA shim의 단일 시퀀스 순환 상태 전달, xla에서 Mamba2 생성

**날짜**: 2026-11-24
**상태**: PR 열림(#3339를 닫음, epic #564 Phase 1 단위 U3). ROCm gfx1151에서 IREE `hip`과 `local-task`로 검증했다. 단일 시퀀스만 지원한다. 순환 모델의 continuous batching과 `mlxcel-server`는 #3340이다.

## 요약

#3370(#3338)은 순환 상태를 받고 돌려주는 Mamba2 prefill과 decode 그래프를 emit했지만, 이를 실행할 수 있는 곳이 없었다. C shim과 `IreeLlama`는 KV cache만 알았고, `compile_vmfbs`는 순환 config의 embeddings prefill에서 실패했으며, CLI는 `mamba2-130m`의 `config.json`조차 파싱하지 못했다. 이 PR은 shim에 일반화된 상태 텐서 목록과 선택적 KV cache를 도입하고, prefill 상태를 요청마다 한 번 `@state_import`로 decode 모듈의 global에 넣으며, 그 레이아웃을 `RuntimeConfig`, 번들 fingerprint, CLI까지 연결한다. 이제 `MLXCEL_BACKEND=xla mlxcel generate`가 HF `Mamba2ForCausalLM` 체크포인트를 단일 시퀀스로 실행한다.

`AntonV/mamba2-130m-hf` @ `05e8773f`는 HF fp32 greedy oracle 대비 40 토큰이 정확히 일치한다. `hip` 기본값 f16, `hip` f32, `local-task` f32 모두에서, CLI 경로(prefix prefill 후 decode)와 `xla_oracle_check`(전체 prefill 후 decode) 양쪽으로 확인했다. 새 상태 리셋 경로를 타는 한 토큰 프롬프트도 10 토큰 oracle과 일치한다. f16이 토큰 단위로 정확하므로 순환 config는 f32로 고정하지 않고 장치별 기본 정밀도를 유지한다.

리뷰의 대부분은 요청 간 격리에 쓰였다. 요청을 거부하거나 실패시키는 모든 경로(prefill 실패, 상태 리셋 실패, shim에 닿기 전에 Rust에서 거부된 프롬프트, 상태가 있는 번들에서 호출된 KV 전용 진입점)가 이제 저장된 상태를 버리므로, decode가 이전 요청을 조용히 이어가는 일이 없다.

## 1. 문제 정의

#3370 이후 emitter는 Mamba2 모델을 만들 수 있었지만, 그 주변 런타임은 모든 곳에서 attention을 가정했다.

- **shim의 상태는 KV cache 하나뿐이었다.** `xla_llama_create`는 KV 버퍼를 할당했고, prefill은 결과에서 `kcache`/`vcache`를 꺼냈고, decode는 `decode_step.kv_import`로 global을 채웠다. attention 없는 번들에는 KV 결과가 없고 대신 `decode_step.state_import` 진입점이 있으므로, shim은 엉뚱한 출력을 읽었을 것이다.
- **`compile_vmfbs`는 모든 dense config에 `prefill_embeddings`를 만들었다.** #3370이 이 emitter가 순환 config에 `Err`를 반환하도록 했으므로 Mamba2 번들은 컴파일되지 않았다.
- **`@state_import`는 한 번도 실행된 적이 없었다.** #3370의 모델 검사는 단일 진입점 보조 모듈과 0으로 초기화한 global을 썼으므로, 이 이슈의 핵심인 prefill에서 decode로의 인계는 검증되지 않았다.
- **`real_len = 0`이 허용됐다.** dense의 한 토큰 프롬프트 경로는 빈 prefix를 prefill하고 padding의 0번 행에서 logits를 읽는다. 순환 모델에서 그 행은 의미 있는 상태 경계가 아니다.
- **HF의 bare `Infinity`.** Mamba2 config에는 `"time_step_limit": [0.0, Infinity]`가 있다. emitter는 #3370부터 이를 파싱했지만 `RuntimeConfig::from_json`, `config_eos_token_ids`, CLI의 `xla_num_layers`는 엄격하게 파싱했다. 그래서 CLI는 런타임에 닿기 전에 `mamba2-130m`에서 실패했고, EOS 조회는 config의 EOS id를 조용히 잃었을 것이다.
- **#3339에 남긴 #3370 이월 코멘트의 열린 질문 두 개.** f16 오차(logits 최대 절대 오차 0.42)가 greedy 토큰을 바꾸는지, 그리고 HF torch 경로와 `mamba_ssm` 커널이 서로 다르게 처리하는 `n_groups > 1`의 gated RMSNorm 그룹화를 어떻게 다룰지다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `csrc/xla_iree.c` | `xla_state_desc`, `XLA_MAX_STATE = 8`, `XLA_MAX_STATE_RANK = 8`. context 필드 `has_kv`, `n_state`, `state_desc[]`, `sscache[]`, `state_import_pending`, `state_zero[]`, `has_prefill_embeddings`. `xla_set_state_contract`, `xla_validate_state`, `xla_state_zero_ensure`, `xla_drop_state`, `xla_state_import_globals`. 공개 함수 `xla_llama_state_reset`, `xla_llama_state_drop`. ragged, slot, embeddings 진입점의 `xla_require_kv` 가드. `xla_llama_create`와 bucket 생성이 상태 레이아웃을 받음. prefill이 0으로 채운 상태 입력을 넣고 상태 결과를 채택. `MLXCEL_XLA_TEST_SUPPORT` 아래의 테스트 hook |
| `src/state_ffi.rs` (신규) | `StateFfi`: `StateTensorSpec`을 `has_kv`, `n_state`, rank, dim, 원소 코드로 평탄화. shim의 상한을 그대로 반복. 번들 fingerprint용 `identity()` |
| `src/hf_json.rs` (신규, feature gate 없음) | `quote_non_finite_json`을 `emitter/recurrent_config.rs`에서 옮김. `parse_hf_config_json`과 `parse_hf_config_json_tracked`(엄격 파싱 먼저, 실패할 때만 따옴표 처리) |
| `src/iree.rs` | `RuntimeConfig::{state_tensors, has_kv, has_recurrent_state, state_ffi}`. 로드 시 `check_ssm_norm_grouping`. `CompiledBundle.prefill_embeddings_vmfb: Option`. `compile_vmfbs`가 순환 config의 embeddings prefill을 건너뜀. 상태 identity를 fingerprint에 해시. `IreeLlama.has_state`, 모든 prefill 진입점의 `invalidate_sequence_state`, 빈 prefix의 상태 리셋. `IreeRaggedLlama::load`가 순환 config를 거부. 테스트 지원 접근자 |
| `src/lib.rs` | `XlaInferenceSession::begin_request`(모든 세션 요청 진입점에서 상태를 버림). `config_eos_token_ids`가 관대한 파서를 사용. `parse_hf_config_json` 재공개 |
| `src/emitter/config.rs`, `recurrent_config.rs` | `from_json_str`이 추적 파서를 쓰고, Mamba2가 아닌 타입에서 따옴표 처리된 비유한 수를 거부. `Config::check_ssm_norm_grouping` |
| `src/commands/generate.rs` | `xla_num_layers`가 관대한 파서를 사용. `xla_model_has_no_kv_cache`가 xla의 Mamba2에서 `KV cache mode` 배너를 숨김 |
| `tests/xla_ssm_single_seq.rs` (신규) | 합성 2층 Mamba2. 장치 테스트 6개(IREE 없이는 ignore)와 장치 없는 테스트 1개 |
| `scripts/xla/validate_arch.sh` | `--single-only`: structural 사전 게이트와 token-exact 게이트 1만, serve 게이트 없음 |
| `README.md`, `FINDINGS_ssm.md`, `docs/supported-models.md`, `CHANGELOG.md` | 상태 ABI 절, 단일 시퀀스 결과, 지원 모델 메모, changelog 항목 |

파일 15개, +1687 / -209.

### shim 상태 계약

`xla_llama_create`에 `has_kv, n_state, state_ranks, state_dims, state_elems`가 추가된다. dim은 `Config::state_tensors` 순서로 텐서마다 이어 붙인다. 원소 코드 0은 f32이고, 현재 상태 원소 타입은 이것뿐이다. `xla_set_state_contract`는 {0,1} 밖의 `has_kv`, `[0, 8]` 밖의 `n_state`, KV도 상태도 없는 번들, `n_state > 0`인데 빠진 배열, `[1, 8]` 밖의 rank, f32가 아닌 코드, 그리고 양수가 아니거나 누적 바이트 곱을 `IREE_DEVICE_SIZE_MAX`나 `SIZE_MAX` 너머로 밀어내는 dim을 거부한다. bucket context는 base context에서 레이아웃을 복사하고 자기 상태 버퍼를 따로 할당한다. Rust 쪽 `StateFfi::new`도 같은 상한을 반복하므로, 표현할 수 없는 레이아웃은 그래프를 컴파일하기 전에 실패한다.

Mamba2의 레이아웃은 `ssm_h [L, H, P, N]` 다음 `ssm_conv [L, K-1, conv_dim]`이고, 둘 다 f32이며 `has_kv = 0`이다. `mamba2-130m`에서는 `[24, 24, 64, 128]`과 `[24, 3, 1792]`이다.

### 요청 수명 주기

| 단계 | shim 동작 |
|---|---|
| prefill, `real_len >= 1` | `xla_state_zero_ensure`가 각 `state_zero[i]`를 한 번 할당하고, 이후 매 prefill 전에 장치에서 다시 0으로 채운다. 0 버퍼는 `real_len` 뒤에 들어간다. KV는 `has_kv`일 때만 꺼낸다. 상태 결과마다 shape를 검증하고, 모두 유효할 때만 `sscache[]`를 교체하며 `state_import_pending = 1`이 된다. |
| 순환 context에서 prefill, `real_len = 0` | 거부. dense context는 기존 `real_len = 0` seeding을 유지한다. |
| 빈 프롬프트 prefix(한 토큰 프롬프트) | `IreeLlama::prefill_padded`가 prefill 대신 `xla_llama_state_reset`을 호출한다. `sscache[i]`가 0으로 채운 `state_zero[i]`를 가리키고 import가 대기 상태가 된다. |
| 그 뒤 첫 decode | `xla_state_import_globals`가 `sscache[]`로 `decode_step.state_import`를 한 번 호출하고, 성공했을 때만 플래그를 지운다. import가 실패하면 플래그가 남고 decode는 거부한다. 이후 decode는 global을 제자리에서 갱신한다. |
| prefill 실패나 거부, 리셋 실패, `xla_require_kv` 거부 | `xla_drop_state`: `sscache[]`를 해제하고 플래그를 지운다. 그러면 prefill이나 리셋이 성공할 때까지 decode가 거부한다. |
| Rust 요청 시작마다 | `IreeLlama::invalidate_sequence_state`(prepared, DeepStack, Gemma3n을 포함한 모든 prefill 진입점)와 `XlaInferenceSession::begin_request`(모든 세션 요청 진입점)가 어떤 검증보다 먼저 `xla_llama_state_drop`을 호출한다. 상태가 없으면 아무 일도 하지 않는다. |

### 번들 구성

순환 번들은 token prefill과 decode 모듈로 이뤄진다. `compile_vmfbs`는 `prefill_embeddings`를 건너뛰고, `CompiledBundle.prefill_embeddings_vmfb`는 `None`이며, C 경로에는 null 포인터가 넘어가고, shim은 `n_state > 0`일 때만 embeddings 모듈을 선택 사항으로 취급한다. 필수 진입점은 `prefill.main`, `decode_step.main`, embeddings 진입점(있을 때), 그리고 상태를 전달하는 번들이면 `decode_step.state_import`다. fingerprint는 `state-abi-v1:has_kv=..:ranks=..:dims=..:elems=..`를 해시하므로, 한 상태 레이아웃으로 컴파일된 캐시 번들이 다른 레이아웃에 로드되는 일은 없다.

### config 파싱

`hf_json.rs`는 `iree` feature 없이 컴파일되므로 `xla-backend`만 켠 CLI 빌드도 쓸 수 있다. 먼저 엄격한 `serde_json` 파싱을 하고 실패할 때만 따옴표 처리를 하므로, 일반 config는 원문 그대로 파싱된다. `parse_hf_config_json_tracked`는 따옴표 처리를 했는지도 알려준다. `Config::from_json_str`은 이를 이용해 `mamba2`가 아닌 모든 모델 타입에서 따옴표 처리된 `Infinity`/`NaN`을 거부한다. 다른 reader들은 숫자 자리에 들어온 문자열을 없는 선택 필드로 보고 기본값을 쓰기 때문이다.

## 3. 기술적 선택과 그 이유

**shim의 일반화된 상태 목록, KV는 선택.** #3338에 기록되고 #3370이 emitter에 구현한 epic 수준 결정을 따른다. shim은 이제 `has_kv = 0, n_state > 0`(Mamba2)과 `has_kv = 1, n_state = 0`(기존 모든 모델)을 별개의 경우로 다루고, 둘이 함께인 경우도 이후 hybrid(Nemotron-H #3344, Jamba)를 위해 표현할 수 있다. 대가는 shim의 모든 KV 사용처에 붙는 `has_kv` 검사다. 그 외 dense 경로는 바뀌지 않았고 dense 회귀 실행이 이를 확인한다.

**요청마다 한 번 import, `kv_import`와 같은 방식.** 대안은 매 decode step에 상태를 입력과 출력으로 넘기는 것이었다. 모듈 global 방식은 decode를 토큰당 호출 한 번, step별 상태 전송 없이 유지하고, shim에 이미 있던 KV용 패턴(`kv_import_pending`)을 재사용하므로 두 cache가 같은 수명 주기 규칙을 따른다.

**상태는 원자적으로 채택.** prefill은 모든 상태 결과가 검증된 뒤에만 `sscache[]`를 교체하고, KV와 상태를 함께 채택한다. 일부만 채택하면 decode가 새 텐서와 옛 텐서가 섞인 상태를 오류 없이 import하게 된다.

**`real_len = 0`은 거부하고 호스트에서 리셋.** dense의 한 토큰 경로는 빈 prefix를 prefill하고 padding의 0번 행을 읽는데, KV에서는 decode가 그 슬롯을 덮어쓰므로 해가 없다. 순환 모델에서는 그 prefill이 padding 토큰 위로 상태를 진행시킨다. shim은 순환 context에서만 `real_len = 0`을 거부하고, `IreeLlama`는 빈 prefix에 `xla_llama_state_reset`을 호출한다. dense 경로는 기존 seeding을 유지하므로 기존 동작은 바뀌지 않는다. 리셋 경로가 돌려주는 토큰은 seed-then-decode 루프가 버리는 자리표시자이고, 이 루프는 dense 경로의 0번 행 토큰도 원래 버린다.

**모든 거부에서 상태를 버리되, 두 층에서.** shim은 자기 실패에서 상태를 버린다. 보안 리뷰는 Rust에서 거부된 요청(너무 크거나 빈 프롬프트)은 shim에 닿지 않으므로, 이전 요청의 `sscache`와 decode global이 살아남아 이후 `decode`나 `decode_step`이 이전 시퀀스를 이어간다는 점을 찾았다. 모든 Rust 진입점 맨 앞, 검증보다 먼저 상태를 버리면 어떤 검사가 먼저 실패하는지와 무관하게 이 문제가 닫힌다. decode 안에서 상태 유효성을 검사하는 대안은 통하지 않는다. 낡은 상태도 shape로는 유효하기 때문이다.

**`xla_require_kv`는 `has_kv == 0`만이 아니라 상태가 있으면 거부.** 첫 버전은 ragged, slot, embeddings 진입점을 attention 없는 context에서만 거부했다. 그러면 이후 hybrid가 가드를 통과해 이전 요청의 상태 옆에 새 KV를 채택하게 된다. `n_state > 0`이면 거부하도록 하면, 누군가 가드를 조여야 한다는 사실을 기억하지 않아도 #3340과 #3344에서 가드가 올바르다. 지금은 `IreeRaggedLlama::load`가 순환 config를 거부하고 순환 번들에는 embeddings 모듈이 없으므로 도달할 수 없다.

**장치별 기본 정밀도 유지.** 이월 코멘트는 token-exact 게이트를 hip 기본값(f16)에서 돌리고, 실패할 때만 순환 config를 f32로 고정하라고 했다. 결과는 40/40이었고 hip f32와 local-task f32도 같았으므로 `effective_precision`은 바뀌지 않았다. 이 판단의 근거는 체크포인트 하나, 40 토큰 프롬프트 하나, 한 토큰 프롬프트 하나뿐이고, #3370에서 측정한 logits 오차 0.42는 여전히 있다.

**전체 폭 norm인 `n_groups > 1`은 거부.** HF torch 경로는 gated 출력을 `d_inner` 전체로 정규화하고, `mamba_ssm` 커널은 그룹별로 정규화한다. 둘은 `n_groups = 1`에서만 같다. `check_ssm_norm_grouping`은 런타임 로드 시 `n_groups > 1`이면서 `norm_group_size == d_inner`인 config를 거부하므로, Codestral Mamba는 조용히 다른 텍스트를 내지 않고 메시지와 함께 실패한다. `norm_group_size = d_inner / n_groups`로 두는 계열(Nemotron-H 방식)은 통과한다. emitter와 HF 모델 검사는 이런 config를 계속 받아들이므로, 나중에 기준 구현과 대조해 결론을 낼 수 있다.

**Mamba2 밖에서는 따옴표 처리된 비유한 수를 거부.** 관대한 파서를 공유하면 Llama config의 `"rope_theta": Infinity`가 문자열로 파싱되어 조용히 기본값으로 대체될 수 있다. 모든 선택 float reader에 따옴표 표기를 가르치는 것보다 거부가 안전하다.

**`MLXCEL_XLA_TEST_SUPPORT` 뒤의 테스트 hook.** `xla_test_set_state_zero_failure`, `xla_test_state_status`, `xla_test_require_kv`는 `test-support` feature에서만 존재한다. 리셋 실패 경로는 강제 실패 없이는 밖에서 일으킬 수 없고, 리뷰 수정에는 수정을 되돌리면 실패하는 테스트가 필요했다.

**모델 타입으로 KV 배너를 숨김.** `xla_model_has_no_kv_cache`는 배너가 세션 로드 전에 출력되므로 런타임에 묻지 않고 xla 백엔드에서 `model_type == "mamba2"`를 검사한다. hybrid에는 별도 검사가 필요하다고 문서화되어 있다.

## 4. 리뷰 결과

**구현 리뷰**(e6a03cdb):

| 발견 사항 | 심각도 | 상태 |
|---|---|---|
| 0 상태 준비가 실패하면 `xla_llama_state_reset`이 일찍 반환하고 이전 `sscache`를 남겨, 이어지는 decode가 이전 요청을 계속함 | MEDIUM | 수정: 실패한 prefill처럼 상태를 버림. `failed_state_reset_drops_the_previous_state`는 수정 없이 실패 |
| `SsmPrefillForm` 재공개 위의 불필요한 주석 | LOW | 수정 |
| `xla_require_kv`가 `has_kv == 0`만 거부 | LOW | e690800e에서 수정(아래) |
| xla의 Mamba2에서 CLI가 `KV cache mode: fp16 (applied to 24 of 24 layers)`를 출력 | LOW | 25eeb40f에서 수정 |
| `cargo clippy --features xla-iree,test-utils --all-targets`가 `IreeRaggedLlama` prepared-slot 메서드의 `private_interfaces`에서 실패 | LOW | 열림, 이 PR이 건드리지 않은 코드 |

**보안 및 성능 리뷰**(e690800e). 범위: C/FFI 상태 경계, 버퍼 수명, 요청 간 격리, 테스트 hook, HF JSON 관용 처리.

| 발견 사항 | 심각도 | 상태 |
|---|---|---|
| shim에 닿기 전 Rust에서 거부된 요청이 이전 상태를 `sscache`와 decode global에 남김 | MEDIUM | 수정: 모든 엔진 prefill 진입점과 세션 요청 진입점에서 `xla_llama_state_drop`. `request_refused_in_rust_drops_the_previous_state` |
| `xla_require_kv`가 hybrid(KV와 상태) context를 허용 | LOW | 수정: `n_state > 0`이면 거부하고 상태를 버림. `kv_only_entry_points_refuse_any_recurrent_state` |
| 따옴표 처리된 bare `Infinity`/`NaN`이 attention reader에 닿아 조용히 기본값이 됨 | LOW | 수정: Mamba2가 아닌 타입에서 거부. `attention_config_refuses_a_quoted_non_finite_number` |
| `decode_step.main`이 step 중간에 실패하면 `sscache`는 남은 채 global이 일부 진행될 수 있어, 같은 요청 안의 재시도 decode가 거기서 이어감 | LOW | 열림. 다음 요청은 다시 import하므로 요청 간 격리는 유지된다. dense KV도 같은 동작 |

변경 없이 확인한 사항: `xla_set_state_contract`는 `state_dims`를 읽기 전에 rank를 검증하고 `sum(ranks)`개만 읽으며 바이트 곱을 제한하므로 `xla_state_elements`는 overflow할 수 없다. prefill의 모든 실패 경로는 상태를 버리는 하나의 정리 코드를 지난다. 상태 결과는 shape가 검증되므로 짧거나 맞지 않는 결과 목록은 깔끔하게 실패한다. 테스트 hook은 `MLXCEL_XLA_TEST_SUPPORT`에서만 컴파일된다. `quote_non_finite_json`은 문자열 내용을 건드리지 않고 엄격 파싱이 실패한 뒤에만 돈다. 각 수정의 테스트는 수정을 되돌리면 실패한다.

**#3370에서 이월된 항목**의 처리:

1. `compile_vmfbs`는 순환 config에서 `prefill_embeddings`를 건너뛰고, shim은 `n_state > 0`일 때만 그 모듈을 선택 사항으로 본다.
2. `@state_import`는 `tests/xla_ssm_single_seq.rs`와 CLI 게이트에서 실제로 실행된다.
3. 순환 context의 `real_len = 0`은 shim이 거부하고, 호스트는 대신 상태를 리셋한다.
4. 정밀도: hip 기본값(f16)에서 토큰 단위로 정확하므로 f32 고정은 없다.
5. 전체 폭 norm인 `n_groups > 1`은 거부된다(`check_ssm_norm_grouping`).

## 5. 테스트와 검증

HF fp32 oracle(`spike/openxla/oracle_continuation.py`, Mamba2를 위해 바꿀 필요 없었음), "The capital of France is", greedy 40 토큰:

```
247 2846 273 253 5112 952 13 285 253 5347 273 253 5112 4687 15 733 310 253 5347 273 253 4687 273 6181 13 285 253 5347 273 253 4687 273 6181 15 187 187 510 5347 273 253
```

| 경로 | 장치 | 정밀도 | 결과 |
|------|------|--------|------|
| `mlxcel generate` (CLI, prefix prefill + decode) | hip | f16 (기본값) | 40/40 |
| `mlxcel generate` (CLI) | hip | f32 | 40/40 |
| `mlxcel generate` (CLI) | local-task | f32 (기본값) | 40/40 |
| `xla_oracle_check` (전체 prefill + decode) | hip | 기본값 | 40/40 |
| `xla_oracle_check` | local-task | 기본값 | 40/40 |
| `validate_arch.sh --single-only` | hip | 기본값 | PASS |

상태 리셋 경로를 타는 한 토큰 프롬프트("The", 10 토큰)도 hip에서 HF oracle과 같다: `1246 3688 7033 281 247 1332 323 253 3275 273`. 구현 리뷰는 hip 기본 정밀도에서 40 토큰 CLI 결과를 재현했다. finalize 커밋 이후 hip에서 40개 id를 다시 확인했다.

`tests/xla_ssm_single_seq.rs`는 합성 2층 Mamba2를 만들고, prefill과 decode를 지나는 세 경로가 같은 recurrence를 계산한다는 점을 이용해 상태 배관을 자기 자신과 대조한다.

- `prefill_to_decode_handoff_agrees_across_routes`: 전체 프롬프트 prefill, prefix prefill 후 decode 한 번, 한 토큰 prefill 후 나머지 프롬프트를 decode로 진행하는 경로가 토큰 단위로 같다.
- `consecutive_requests_start_from_a_fresh_state`: 한 세션의 두 번째 요청이 새 세션과 같다.
- `prefill_at_full_capacity_matches_the_decode_route`.
- `empty_prefix_and_refused_prefill_never_reuse_state`.
- `failed_state_reset_drops_the_previous_state`(리뷰 수정).
- `request_refused_in_rust_drops_the_previous_state`(보안 수정).
- `kv_only_entry_points_refuse_any_recurrent_state`(장치 불필요).

장치 테스트 6개는 일반 CI에서 `#[ignore]`이고 `cargo test --release --features xla-iree,test-utils --test xla_ssm_single_seq -- --ignored`로 실행한다. hip과 local-task에서 통과한다. 이전 요청의 상태를 prefill에 넣는 mutation은 처음 4개 테스트 중 3개를 실패시킨다.

추가된 단위 테스트: `state_ffi.rs` 4개(빈 레이아웃, 평탄화 순서, identity가 레이아웃을 구분, 상한 거부), `recurrent_config.rs` 2개(bare `Infinity` 파싱, 다중 그룹 거부와 통과하는 그룹별 경우), `recurrent_tests.rs` 1개(attention config에서 따옴표 처리된 비유한 수 거부).

hip의 dense 회귀: `xla_ragged_hardening` 6/6, `xla_greedy_timing` 3/3, `xla_decode_scalar_reuse` 4/4, `xla_prepared_prefill` 4/4. `llama-3.2-1b-instruct-4bit`의 greedy id(40 토큰)는 변경 전후가 같다.

정적 게이트: `cargo fmt --check`, mlxcel-xla lib와 tests(`iree` 유무 각각), root crate `--features xla-iree --all-targets`의 clippy `-D warnings`. `--features xla-backend --all-targets` clippy의 `src/multimodal/host_preprocessor_molmo_tests.rs`, `src/worker_failfast.rs` 실패는 origin/main(ea0b45d8)에서도 재현되므로 기존 문제다.

`cargo test --profile test-fast -p mlxcel-xla --lib`: PR 본문은 이전 리비전에서 512개 통과를 보고한다. 최종 head(25eeb40f, 이 리포트 작성 시 origin/main과 이미 최신 상태)에서는 513개 통과, 0개 실패, 21개 ignore다.

이 PR에서는 xla의 Mamba2 처리량이나 지연 시간을 측정하지 않았다.

## 6. 남은 작업과 한계

- **단일 시퀀스만.** `IreeRaggedLlama::load`는 순환 config를 거부하고, `xla_require_kv`는 상태가 있는 context의 ragged, slot, embeddings 진입점을 거부한다. #3340이 슬롯별 상태를 추가하고 이 가드를 풀며 `mlxcel-server`를 켠다. 그때까지 `validate_arch.sh --single-only`는 serve 게이트를 건너뛴다.
- **실제 체크포인트 하나, 긴 프롬프트 하나.** f16 토큰 정확성은 `mamba2-130m`(`n_groups = 1`)의 40 토큰과 10 토큰 continuation에 근거한다. #3370에서 측정한 f16 logits 최대 절대 오차 0.42는 그대로이므로, 더 긴 생성이나 다른 체크포인트에서는 근소한 동률이 뒤집힐 수 있다. `effective_precision`에서 순환 config를 f32로 고정하는 것이 대비책으로 남는다.
- **`n_groups > 1`은 지원이 아니라 거부.** Codestral Mamba는 gated norm 그룹화를 `mamba_ssm`과 대조해 결론 내야 한다. Nemotron-H(#3344)는 그룹별로 정규화하므로 검사를 통과한다.
- **step 중간의 decode 실패**는 같은 요청 안의 재시도에서 상태 global을 일부 진행된 채로 남길 수 있다(LOW, 열림. dense KV도 같다).
- **성능 측정 없음.** xla의 Mamba2 decode와 prefill 처리량, 요청마다 `state_zero`를 다시 0으로 채우는 비용은 측정하지 않았다.
- **CUDA 미실행.** `hip`(gfx1151)과 `local-task`만 돌렸다.
- **`test-utils`를 켠 clippy의 `private_interfaces` 실패**는 `IreeRaggedLlama`의 기존 문제이고 범위 밖이다.
- **배너 검사는 이름 기반.** `xla_model_has_no_kv_cache`는 `model_type == "mamba2"`를 비교한다. hybrid 계열에는 별도 규칙이 필요하다.

## 7. 학습 포인트

- 런타임에 두 번째 종류의 시퀀스별 상태가 생기면 첫 번째와 같은 수명 주기를 준다. `sscache`와 `state_import_pending`을 `kv_import_pending`에 맞춰 설계하니 import 규칙, 실패 처리, bucket 동작이 새 설계가 아니라 이미 아는 것이 되었다.
- 요청 간 격리는 실패 경로뿐 아니라 거부 경로에서도 성립해야 한다. native 계층에 닿기 전에 거부된 요청은 native 정리 코드를 실행하지 않는다. 모든 진입점 맨 앞에서 무조건 상태를 버리는 편이 어떤 검사가 먼저 실패하는지 따지는 것보다 단순하고 우회하기 어렵다.
- 낡은 상태도 shape로는 유효하므로 사용 시점의 검증으로는 재사용을 잡을 수 없다. 요청이 상태를 세우기 전까지는 상태가 없도록 하고, 없으면 decode가 거부하게 하는 것이 유일한 방어다.
- 가드는 오늘의 config가 아니라 불변식에 맞춰 쓴다. `has_kv == 0`을 검사하는 `xla_require_kv`는 Mamba2에는 맞았지만 epic이 향하는 hybrid에는 틀렸다. `n_state > 0`이 진입점에 실제로 없는 것을 나타낸다.
- 파서를 관대하게 만들면 모든 호출자가 받아들이는 범위가 넓어진다. 관용 처리가 쓰였는지 추적하고 그 결과를 해석할 수 없는 reader에서는 거부하게 했기 때문에, Mamba2를 위한 수정이 다른 계열의 값을 조용히 바꾸지 않았다.
- 정밀도 게이트는 사용자가 실제로 받는 정밀도에서 돌린다. #3370은 f16 logits 오차를 측정하고 greedy 토큰은 바뀌지 않을 것으로 예상했고, 이 PR은 f32 고정 여부를 정하기 전에 실제 CLI 경로에서 이를 확인했다.
