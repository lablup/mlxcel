# PR #2792: #2787, #2788, #2789가 남긴 LOW 잔여 항목 해소

**날짜**: 2026-10-31 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: main 1a257bcd 위 498bac80에서 구현 완료 (커밋 3개). PR 본문의 로컬 게이트는 GB10(spark-102)에서 통과했다. 작성 시점에 worktree에서 추가 로컬 게이트가 돌고 있었고, 호스팅 check(`cargo-clippy`, `OpenXLA feature link`, `OpenXLA feature compile`, `CPU-only link`, `2-host logical`, WebUI job들)는 pending 상태였다
**언어**: Rust, C, Markdown, YAML
**위험도**: 낮음에서 중간 (다섯 항목 중 넷은 보고, 검증, 테스트 gating만 바꾼다. IREE `local-task` device는 이제 IREE factory 대신 우리 C 코드가 만들므로 production device 경로에 새 native 코드가 들어가고, `PipelineUnavailable.stage`의 타입이 바뀐다)

## 요약

PR #2792 (issue #2790을 닫음)는 #2787, #2788, #2789의 리뷰와 report가 남긴 LOW 잔여 항목 다섯 개를 PR 하나와 검증 한 번으로 해결한다. layer별 KV storage 보고는 이제 요청한 mode를 저장하는 layer, mode와 무관하게 FP16을 저장하는 layer, KV를 전혀 갖지 않는 layer를 구분하고, 모든 model-owned family가 설정된 table로 fallback하지 않고 실제 storage를 보고한다. sliding window가 0이거나 `i32::MAX`보다 크면 (그리고 Gemma 3의 `sliding_window_pattern`이 0이면) 나중에 panic하지 않고 load가 실패한다. IREE `local-task` CPU device는 8 MiB worker stack으로 직접 만들어지므로 CUDA를 load한 process에서도 시작된다. `XlaPreparedPrefillLogits`는 production build에서 빠지고 Qwen2.5-VL diagnostic engine의 유일한 구현이 된다. pipeline availability gate는 stage가 원인인 실패에서만 열리고, forward timeout이 더 이상 entry stage를 지목하지 않는다.

## 1. 문제 정의

다섯 항목 모두 issue 작성 전에 main 863ddb72에서 다시 확인했다. 보안 항목은 없다. 같은 게이트(fmt, clippy, `models::`와 `distributed::` lib test, release `cuda,xla-iree` build 한 번)를 공유하므로 하나로 묶었다.

### 1.1 KV storage 보고가 요청 mode를 저장하지 않는 layer를 여전히 셌다

#2788은 server의 `kv_cache_mode_applied_layers`가 FP16 sliding-window layer를 quantized로 세지 않도록 `LanguageModel::kv_cache_storage_modes`를 추가했다. 세 경우가 여전히 틀렸다.

- **Llama 4**: `make_llama4_caches_with_modes`는 `(idx + 1) % 4 != 0`인 layer를 FP16 `ChunkedKVCache`로 만드는데, `Llama4`에 storage override가 없어 `--kv-cache-mode int8`에서 모든 layer가 applied로 보고됐다.
- **KV가 없는 layer**: Qwen3-Next, Qwen 3.5, Bailing linear의 linear-attention layer, LFM2 convolution layer, NemotronNAS `no_op` block이 요청 mode를 보고했다.
- **mode table이 없는 model-owned family** (DeepSeek V4, RecurrentGemma, Muse Glimmer, Mamba, Mamba2, RWKV7)는 `None`을 돌려줬고, server는 설정된 table로 fallback해 모든 layer가 applied라고 보고했다.

반환 타입 `Option<Vec<KVCacheMode>>`로는 "이 layer는 아무것도 저장하지 않는다"를 표현할 수 없었다.

### 1.2 sliding window나 pattern이 0이면 load 실패 대신 panic했다

Gemma 3 `ModelArgs`는 `sliding_window_pattern: 0`을 받아들였다. `layer_rope_params`가 `is_multiple_of`를 써서 load는 성공했지만, `is_sliding_layer`와 cache builder는 `% self.sliding_window_pattern`을 계산하므로 #2788 이후 scheduler build 시점에 `kv_cache_storage_modes` 안에서 panic했다. 별개로 `sliding_window: 0`은 `max_size == 0`인 `RotatingKVCache`를 만들었고, `speculative_ring_cursor`와 buffer-origin 경로의 `rem_euclid(self.max_size)`가 0으로 나눴다. 모든 family가 `usize` window를 `as i32`로 cast했으므로 `i32::MAX`를 넘는 값은 조용히 wrap됐다.

### 1.3 GB10에서 CUDA를 load한 상태로 IREE `local-task` device가 시작되지 않았다

`xla_llama_create_impl`은 모든 device를 `iree_runtime_instance_try_create_default_device`로 만들었다. `local-task`이면 IREE driver factory가 전역 flag registry에서 executor를 만들고(`iree_task_executors_create_from_flags`), 그 기본 worker stack은 128 KiB다. glibc는 static TLS를 각 thread stack 안에 두는데, test binary와 CUDA 의존성의 `PT_TLS` 합이 235,072 byte(cuBLASLt 94,240, OpenBLAS 61,440, cuSOLVER 53,264 등)라서 `pthread_create`가 EINVAL을 돌려줬다: `thread creation failed with 22; [runtime-init] failed to open IREE HAL device 'local-task'`. `diagnostics` feature만 `--task_worker_stack_size=8388608`을 IREE의 process 전역 flag registry에 parse해 이를 피했다. production과 `tests/xla_prepared_prefill.rs`의 `local-task` test 세 개는 spark-102에서 실행되지 않았다.

### 1.4 `XlaPreparedPrefillLogits`가 diagnostic 코드를 중복하고 모든 IREE build에 들어갔다

#2789는 integration test 하나를 위해 `XlaPreparedPrefillLogits`를 추가했지만, plain `iree`에서 public이어서 production bundle도 이를 compile했다. 본문은 `Qwen25VlLanguageDiagnosticEngine::load`와 `capture`를 slot reset과 M-RoPE check만 빼고 반복했으므로, `prefill_logits`를 반복 호출하면 빈 cache에서 시작하지 않았다.

### 1.5 pipeline availability gate가 엉뚱한 실패에 열리고 엉뚱한 stage를 지목했다

`RemotePipelineRuntime::exchange`는 `exchange_within`의 모든 `Err`에 gate를 열었다. send 실패뿐 아니라 coordinator 자체 오류(serialization, `local_addr`, 닫힌 receive channel)도 포함됐다. open 로그는 항상 "did not answer within {stage_timeout}"이라고 했고, `docs/distributed.md`는 full stage timeout만 gate를 연다고 적고 있었다. forward의 `RunEntry` exchange는 `stage_peers[0]`으로 가고 reply는 모든 stage가 실행된 뒤 마지막 stage에서 오므로, 뒤쪽 stage가 멈춰도 timeout은 항상 entry stage를 지목했다. 그 결과 `broadcast_best_effort`는 첫 probe가 `failed_peer`를 고칠 때까지 정상인 entry stage에 teardown 대기 0을, 실제로 멈춘 stage에 2초 전체를 줬다.

## 2. 기술적 선택과 그 이유

### 2.1 두 번째 default method를 추가하지 않고 trait 반환 타입을 바꿈

`kv_cache_storage_modes`는 이제 `Option<Vec<Option<KVCacheMode>>>`를 돌려주며, 안쪽 `None`은 KV cache가 없는 layer다. default는 `kv_cache_layer_modes()`의 각 항목을 `Some`으로 감싼다. #2788 report가 지적했듯 trait default는 forwarding 누락을 숨긴다. default가 있는 새 method는 forward를 잊은 wrapper에서도 조용히 compile된다. 기존 method의 signature를 바꾸자 compiler가 모든 forwarder(`LoadedModel`, `language_model_ref.rs`의 `&M` impl, `src/vision/mod.rs`, VLM wrapper들)를 나열했고, 전부 손봐야 했다. table 없는 text model을 감싸는 VLM wrapper 다섯 개(Granite 4 Vision, Inkling VL, Kimi K3 VL, Muse Glimmer VLM, Nemotron-H Nano Omni VL)는 이전에 이 method를 forward하지 않았고 이제 forward한다.

바깥 `None`은 "scheduler가 table로 cache를 만드는 dense model"이라는 한 가지 의미만 유지한다. `log_effective_kv_cache_application`은 이 경우에만 설정된 table로 fallback한다.

### 2.2 builder 자신의 predicate로 채우는 `LayerKv` 어휘 하나

`KvCacheLayerModes::storage_modes`는 `layer_kind(layer_idx) -> LayerKv` closure를 받는다. 종류는 넷이다: `Full`(주입된 mode의 `KVCache`), `Sliding(window)`(`RotatingKVCache::stored_mode`를 거침), `Fp16Only`(mode를 무시하는 cache), `None`. `LayerKv::stored_mode`가 종류에서 저장 mode로 가는 유일한 mapping이다. table 없는 family는 `table_less_storage_modes(n_layers, layer_kind)`를 쓰고, 이 함수는 같은 mapping을 FP16 요청으로 적용하므로 attention layer는 `Some(Fp16)`, recurrent나 SSM layer는 `None`을 보고한다.

각 family는 cache builder가 이미 쓰는 predicate를 넘긴다: `Qwen3NextConfig::is_linear_layer`, `Lfm2Config::is_attention_layer`, RecurrentGemma의 `block_types`, 그리고 새 `NASTransformerBlock::has_kv_cache`(`AttentionSubblock::Attention`만 KV를 저장하고, linear 대체 block은 쓰지 않는 slot만 가진다). Llama 4는 `uses_chunked_cache(idx)`를 추출해 cache builder, mask 선택 두 곳, `maybe_trim_front` loop 세 곳, storage override에서 함께 호출하므로 보고와 cache 배치가 어긋날 수 없다.

### 2.3 window를 load 시점에 검증하고, 검증된 `i32`를 돌려주고, constructor assert는 유지

`validate_sliding_window(family, window) -> Result<i32, String>`는 0과 `i32::MAX` 초과 값을 거부한다. `cache_window(family, window, has_sliding_layer)`는 model에 sliding layer가 있을 때만 이를 호출하므로, `layer_types`가 전부 full attention인 config는 어떤 window로도 load된다(`a_zero_window_without_sliding_layers_still_loads`). Gemma 3, GPT-OSS, Laguna, Mellum, ExaOne 4, Ministral 3, AFMoE는 검증된 `i32`를 저장하고 `as i32` cast를 없앴다. Gemma 4는 `Gemma4TextModel::from_weights`와 `Gemma4StageModel::from_filtered_weights` 둘 다에서 검증하지만(stage 경로는 리뷰 후 24674274에서 추가), 수십 곳에서 쓰이는 공유 `TextConfig`의 cast는 유지한다. 검증 후에는 그 cast가 손실 없이 동작한다. Gemma 3는 `Gemma3Backbone::from_weights`가 이미 쓰던 "sliding_window_pattern must be positive" 문구로 `sliding_window_pattern: 0`도 거부한다.

`RotatingKVCache` constructor assert는 직접 호출하는 쪽을 위해 남겼고, 조용한 clamp는 추가하지 않았다. 잘못된 config는 운영자 오류이므로 checkpoint가 기대하는 것과 다른 window의 cache를 만들기보다 load 시점에 family 이름과 함께 드러나야 한다.

### 2.4 IREE flag를 설정하지 않고 `local-task` device를 직접 만듦

worker stack을 키우는 방법은 둘이었다: IREE flag registry에 `--task_worker_stack_size`를 parse하기(`diagnostics`의 방식), 또는 executor를 직접 구성하기. registry는 process 전역이므로 production shim이 여기에 쓰면 같은 process의 다른 IREE 사용자 동작도 바뀌고, issue는 registry를 건드리지 말라고 요구했다. `csrc/xla_local_task.c`는 IREE의 `local_task/registration/driver_module.c`를 그대로 따른다.

1. `iree_task_executor_options_initialize` 후 `worker_stack_size = 8 MiB`.
2. `MLXCEL_XLA_LOCAL_TASK_WORKERS`가 설정되면 `iree_task_topology_initialize_from_group_count(n)`로, 아니면 현재 NUMA node에서 `iree_task_topology_initialize_from_physical_cores`로 topology를 만들며 상한은 64(IREE의 `--task_topology_max_group_count` 기본값)다.
3. `iree_hal_create_all_available_executable_loaders`, `iree_hal_allocator_create_heap`, `iree_hal_task_driver_create`, 그리고 `iree_runtime_instance_try_create_default_device`와 같은 방식으로 만든 proactor pool을 넘겨 `iree_hal_driver_create_default_device`.

중간 reference는 다음 객체가 retain한 뒤 release하고, 각 실패 경로는 자신이 만든 것을 release한다. `local-task`가 아닌 URI는 여전히 `iree_runtime_instance_try_create_default_device`를 거친다. device를 만드는 FFI entry point 두 개(`xla_llama_create`, `xla_aux_create`)가 worker 수를 받고, Rust가 호출 전에 `local_task_workers_for(device)`로 이를 결정한다.

worker 수는 Rust에서 검증한다. 미설정이면 `0`(물리 core 수), 아니면 `1..=256`(`IREE_TASK_TOPOLOGY_MAX_GROUP_COUNT`)의 정수여야 한다. IREE가 더 큰 값을 조용히 clamp했기 때문에 상한은 리뷰 후 498bac80에서 추가됐다. 사용한 IREE symbol은 모두 pinned CUDA tree와 `iree-dist-3.12.0rc20260721-linux-x86_64` 둘 다에서 export되는 것을 확인했다.

`diagnostics` flag helper는 그대로다. `local-task`에는 이제 불필요하지만 diagnostic CLI target 설정에는 여전히 쓰인다.

### 2.5 prefill parity 타입을 전용 `test-support` feature 뒤에 둠

`XlaPreparedPrefillLogits`를 `diagnostics` 뒤에 두는 안은 issue에서 기각됐다. 그러면 parity test가 flag registry로 큰 stack을 설정하는 diagnostics 경로로 옮겨가 1.3을 가렸을 것이다. 대신 mlxcel-xla에 `test-support = ["iree"]`를 추가하고, root `test-utils`가 `mlxcel-xla?/test-support`를 forward하며, `[[test]] xla_prepared_prefill`은 `xla-iree`와 `test-utils`를 요구한다. 타입은 `any(test-support, diagnostics)`에서만 compile된다. `Qwen25VlLanguageDiagnosticEngine`은 이제 이 타입을 감싸고, M-RoPE check를 closure로 `prefill_logits_checked`에 넘긴다. 이 함수는 slot을 reset하고, payload를 준비하고, check를 실행한 뒤 prefill한다. `IreeRaggedLlama::load`가 `RAGGED_B_VALUES = [4, 8]`만 받으므로 `b_max`는 4로 두고 그 이유를 주석으로 남겼다.

### 2.6 타입이 있는 오류로 gate 개방 여부를 결정

`exchange_within`은 stage가 원인인 두 실패를 `StageUnreachable { peer, cause }`와 `StageUnreachableCause::{Timeout(Duration), SendFailed(String)}`(새 module `stage_unreachable.rs`)로 감싼다. 닫힌 receive channel, serialization 오류, `local_addr`는 plain `anyhow` 오류로 남는다. `exchange`는 error chain을 `downcast_ref`로 훑어 `StageUnreachable`이 있을 때만 gate를 연다. 호출자와 test가 "timed out"에 match하므로 `Timeout`의 표시 문구("remote stage X timed out after Y")는 바꾸지 않았다.

지목 규칙: `RunEntry` timeout은 `mark_unavailable(None, ..)`를 호출한다. reply가 마지막 stage에서 오므로 timeout만으로는 어느 stage가 멈췄는지 알 수 없기 때문이다. `RunEntry` send 실패는 도달하지 못한 대상이 entry stage이므로 여전히 entry stage로 지목한다. 지목된 stage 없이 gate가 열려 있는 동안 `broadcast_best_effort`는 모든 stage에 정상 `teardown_ack_timeout`을 주고, 다음 probe가 멈춘 stage를 지목한다. `PipelineUnavailable.stage`는 `Option<String>`이 되며, `Display`는 지목된 stage에는 기존 문구를, 없으면 "an unidentified stage of the remote pipeline"을 출력한다.

open 로그는 이제 원인을 밝힌다. timeout이면 "did not answer within", send 실패면 "could not be reached ({reason})"(reason은 `loggable_peer_text`를 거침), 지목된 stage가 없으면 run 단위 문구다. 498bac80은 이를 probe 경로까지 넓혔다. `probe_failure_cause`가 실패한 probe의 error chain에서 `StageUnreachable` 원인을 읽으므로, send가 실패한 probe가 gate를 다시 열 때 더 이상 timeout으로 기록되지 않는다.

`StageAvailability`의 unit test는 `stage_availability_tests.rs`로 옮겼고, 새 `runtime_log_capture_tests.rs`의 thread별 `tracing` capture(`capture_logs`, `gate_open_lines`)로 runtime test가 open 로그 문구를 검증한다.

## 3. 변경 요약

파일 83개, +2205 / -448, 커밋 3개.

| 커밋 | 내용 |
|---|---|
| `cfa55cac` | 다섯 항목 전부 |
| `24674274` | Gemma 4 pipeline stage의 window 검증, 밀려난 `gpt_oss.rs` doc comment 복구 |
| `498bac80` | `MLXCEL_XLA_LOCAL_TASK_WORKERS` 상한 256, probe 재개방 로그가 실제 원인을 보고 |

| 영역 | 파일 | 변경 |
|---|---|---|
| Trait와 forwarder | `mlxcel-core/src/generate.rs`, `language_model_ref.rs`, `src/loaded_model.rs`, `src/vision/` 아래 16개 | 새 반환 타입, VLM wrapper 5개가 처음으로 forward |
| 공유 helper | `src/models/model_owned.rs` | `LayerKv`, `table_less_storage_modes`, `validate_sliding_window`, `cache_window` |
| Storage override | `llama4.rs`, `qwen3_next.rs`, `qwen3_5.rs`, `qwen4_exp.rs`, `bailing_moe_linear.rs`, `lfm2.rs`, `nemotron_nas.rs`, `deepseek_v4.rs`, `recurrent_gemma.rs`, `muse_glimmer.rs`, `mamba.rs`, `mamba2.rs`, `rwkv7.rs`, `nemotron_h.rs`, `jamba.rs`, `falcon_h1.rs`, `plamo2.rs`, `granitemoehybrid.rs`, `kimi_linear.rs`, `kimi_k3.rs`, `inkling/runtime.rs` | layer별 실제 storage 보고 |
| Window 검증 | `gemma3.rs`, `gemma4.rs`, `gpt_oss.rs`, `exaone4.rs`, `laguna.rs`, `step3p5.rs`, `ministral3.rs`, `mellum.rs`, `afmoe.rs`, `tensor_parallel/llama_runtime.rs` | load 시 검증, `i32` 저장 |
| Server와 example | `scheduler/paged_layout.rs`, `examples/logit_trace/kv_mode.rs` | `KvStorageSummary`, `kv_cache_mode_stateless_layers`, trace 줄의 stateless 수 |
| IREE shim | `csrc/xla_local_task.{c,h}`, `csrc/xla_iree.c`, `csrc/xla_aux.c`, `build.rs`, `src/local_task.rs`, `iree.rs`, `auxiliary.rs` | 직접 만드는 local-task device, worker 수 parsing, test-support probe |
| Test support | `mlxcel-xla/Cargo.toml`, root `Cargo.toml`, `batch.rs`, `lib.rs`, `.github/workflows/ci.yml`, `tests/xla_prepared_prefill.rs` | `test-support` feature, 공유 prefill 타입, CI link 단계에 `test-utils` 추가 |
| Pipeline gate | `pipeline/stage_unreachable.rs`(신규), `runtime.rs`, `stage_availability.rs`, `mod.rs`, test 파일 | 타입이 있는 원인, 선택적 지목 |
| 문서 | `docs/distributed.md`, `docs/environment-variables.md` | gate 규칙, `MLXCEL_XLA_LOCAL_TASK_WORKERS` |

운영자에게 보이는 동작:

- startup 로그 `resolved KV cache mode applied to model caches`에 `kv_cache_mode_stateless_layers`가 추가되고, `kv_cache_mode_total_layers`는 KV를 가진 layer만 센다. hybrid model(Qwen3-Next, LFM2, Jamba 등)에서는 total이 이전보다 작다.
- 새 변수 `MLXCEL_XLA_LOCAL_TASK_WORKERS`.
- sliding window가 0이거나 너무 큰 config는 `{family}: sliding_window must be between 1 and 2147483647, got N`으로 load가 실패한다.
- gate 로그가 timeout과 send 실패를 구분하고, stage를 아직 모른다고 말할 수 있다.

## 4. 트레이드오프와 남은 한계

- **predicate는 여전히 family마다 반복된다.** Llama 4는 이제 `uses_chunked_cache`를 공유하지만, 대부분의 family는 layout별 공유 함수 하나가 아니라 같은 config method를 호출해 builder의 predicate를 storage override에 넘긴다. layer 선택 방식을 바꾸는 family는 두 호출 지점을 모두 고쳐야 한다. family 그룹마다 storage test가 하나씩 있고(Llama 4, Qwen3-Next, RecurrentGemma, Mamba, DeepSeek V4), 나머지 override는 각 family test suite 통과에 기댄다.
- **DeepSeek V4는 모든 layer를 `Fp16Only`로 보고한다.** local cache는 `RotatingKVCache`와 압축 pooling 상태로 이뤄지는데, 보고는 layer를 FP16 attention cache 하나로 취급한다. applied 수에는 이것으로 충분하지만 pooling 상태를 설명하지는 않는다.
- **local-task probe test는 다시 읽은 값이 아니라 설정한 stack을 보고한다.** `xla_local_task_probe`는 IREE에 넘긴 상수를 돌려준다. stack이 실제로 적용됐다는 증거는 동작이다: CUDA를 link한 binary에서 worker 두 개가 시작되고, 128 KiB에서는 EINVAL로 실패한다(5절의 mutation check). flag registry assertion은 "registry가 compile되지 않음"과 "registry가 128 KiB 기본값"을 둘 다 허용한다.
- **기본 상한과 명시 상한이 다르다.** 미설정이면 물리 core 수에 상한 64, 명시하면 256까지 가능하다. IREE 자체 기본값을 따른 것이고 문서에 적혀 있다.
- **local-task test는 수동으로만 돈다.** 호스팅 CI는 `xla_prepared_prefill`을 link만 한다(`--no-run`). `#[ignore]` test는 pinned IREE runtime이 필요해 spark-102에서만 실행했다. aarch64 IREE dist가 없으므로 dist 경로는 aarch64 실행이 아니라 symbol export와 header compile로 확인했다.
- **CUDA 정밀도 미달, 변경 없음.** `MLXCEL_XLA_DEVICE=cuda`에서 `local_task_gemma3_vlm_prefill_matches_mlx_mask`는 기본 `f16` CUDA 정밀도로 1e-3 허용치를 넘는다(최대 5.4e-3). main에서도 같고, `MLXCEL_XLA_PRECISION=f32`에서는 통과한다.
- **지목되지 않은 teardown은 지목된 경우보다 느리다.** `RunEntry` timeout과 다음 probe 사이에는 어느 stage인지 모르므로 멈춘 stage가 teardown 대기 2초 전체를 받는다. 실제로 멈춘 stage에 대한 비용은 main과 같고, 차이는 정상인 entry stage가 더 이상 대기 0을 받지 않는다는 점이다.
- **`PipelineUnavailable.stage`는 타입이 바뀐 public field다** (`String`에서 `Option<String>`). workspace 안의 코드는 고쳤고, 이 field에 match하는 외부 crate는 `as_deref()`가 필요하다.
- **문서 사소한 점**: `docs/environment-variables.md`의 새 `MLXCEL_XLA_LOCAL_TASK_WORKERS` 행에서 semicolon 뒤 "The"가 대문자다.
- **범위 밖** (issue 기준): 설정 가능한 `stage_timeout`, "probe 전까지 미지목"을 넘는 stage별 `RunEntry` 지목, `RotatingKVCache`의 `Int8` 저장, quantized `ChunkedKVCache`, PaliGemma와 Gemma3n OpenXLA mask.

## 5. 검증

### 5.1 수정 전 실패 확인

새 test는 모두 main 1a257bcd에서, 또는 main에서 compile되지 않는 경우 main 동작을 되돌린 mutation에서 실패한다.

| Test | main 동작에서의 실패 |
|---|---|
| 기존 `local_task_*` test 세 개, main binary | `thread creation failed with 22; [runtime-init] failed to open IREE HAL device 'local-task'` |
| `local_task_executor_uses_8_mib_stacks_without_iree_flags`, stack을 128 KiB로 되돌림 | `xla_local_task_probe failed: ... thread creation failed with 22` |
| `llama4_chunked_layers_report_fp16_storage_under_int8` | `Some([Some(Int8) x8])` 대 layer 3과 7만 `Int8` |
| `qwen3_next_linear_layers_report_no_kv_storage` | `Some(Int8)` 여덟 개 대 `[None, None, None, Some(Int8), ...]` |
| `recurrent_gemma_reports_...`, `mamba_reports_every_layer_stateless`, `deepseek_v4_reports_fp16_storage_for_every_layer` (`None`을 돌려주는 override) | `left: None` |
| `gemma3_rejects_zero_sliding_window_pattern_at_load`, `sliding_window_zero_is_a_load_error` (검증 끔) | load 성공 |
| `run_entry_timeout_does_not_name_the_entry_stage` (main의 지목 규칙) | `failed_peer`가 entry stage |
| `send_failure_opens_the_gate_with_a_send_reason` | open 로그가 `did not answer within 1s` |

추가 unit test는 `validate_sliding_window`의 0, 1, `i32::MAX`, `i32::MAX + 1`, stateless layer가 있는 `KvStorageSummary`, `parse_local_task_workers`의 미설정, 0, 음수, 숫자 아님, 257, 1000, 실패한 probe가 지목하는 미지목 outage를 다루는 `StageAvailability`, 그리고 wall-clock bound 없는 순수 error-chain 검사인 `probe_failure_cause`를 다룬다.

### 5.2 로컬 게이트 (spark-102, GB10)

- `cargo fmt --all -- --check`, `make verify-env-var-docs`: clean.
- Clippy `-D warnings`: `mlxcel`, `mlxcel-core`, `mlxcel-xla` lib와 test에 `--profile test-fast --features cuda`, `mlxcel-xla`와 `mlxcel`에 `--release --features cuda,xla-iree,test-utils`와 `--test xla_prepared_prefill`: clean.
- Lib test (`--test-threads=1`, `gpu-lock` 아래): `models::` + `vision::` 3429 통과, `distributed::` + `server::batch::scheduler` 1845, `mlxcel-core` `cache::` + `generate::` 624, `mlxcel-xla --lib` 370. 리뷰 수정 후: `distributed::pipeline::` 416, `mlxcel-xla` `local_task` 6.
- Integration (`cuda,test-utils`): `distributed_integration` 20, `pipeline_e2e` 30, `pipeline_remote_real_models` 10, `pipeline_server_remote_real_models` 7, `pipeline_ci_multi_stage_real_models` 4(`--include-ignored --skip glm_moe_dsa`, `MLXCEL_REQUIRE_MODELS=1 MLXCEL_SDPA_DETERMINISTIC=1`), `disaggregated_serving_e2e` 2, `disaggregated_router_e2e` 8: 모두 통과.
- `cargo test --release --features cuda,xla-iree,test-utils --test xla_prepared_prefill -- --ignored`: 기본 `local-task` device에서 4/4. `MLXCEL_XLA_LOCAL_TASK_WORKERS=0`과 `abc`는 변수 이름과 함께 load 실패, `2`는 통과.
- Feature gating: root의 `use mlxcel_xla::XlaPreparedPrefillLogits`는 `cargo check --features cuda,xla-iree`에서 E0432로 실패하고 `test-utils`를 더하면 통과한다. `--features cuda,xla-reference-diagnostics --lib --tests` check는 clean(둘 다 `test-fast` profile).

### 5.3 실제 checkpoint smoke (`--kv-cache-mode int8`, release `mlxcel-server`)

| Model | 로그 |
|---|---|
| `llama-4-scout-17b-16e-instruct-4bit` | `applied_layers=12 total_layers=48 stateless_layers=0` (dense layer 12개) |
| `gemma-3-1b-it-4bit` | `applied_layers=4 total_layers=26`, #2788과 동일 |

hybrid나 table 없는 checkpoint는 server smoke로 돌리지 않았다. 이들의 storage 보고는 위 unit test가 다룬다.

## 6. 학습 포인트

- **forwarder를 모두 찾으려면 signature를 바꾼다.** default가 있는 새 trait method는 이를 잊은 wrapper에서 조용히 compile된다. 기존 method의 반환 타입을 바꾸면 "모든 wrapper 찾기"가 compiler 오류 목록이 되고, 이번에는 한 번도 forward하지 않던 VLM wrapper 다섯 개를 찾았다.
- **"없음"에 자기 값을 준다.** `Option<Vec<KVCacheMode>>`는 cache가 없는 layer를 표현하지 못해, 보고가 mode를 지어내거나 설정된 table로 fallback했다. 안쪽 `Option`(과 `LayerKv::None`)으로 stateless layer가 정식 답이 됐고, 바깥 `None`은 한 가지 의미를 유지했다.
- **static TLS는 thread stack 위에 있다.** glibc에서 process의 static TLS보다 작은 stack으로 `pthread_create`를 호출하면 메모리 부족이 아니라 EINVAL이 난다. 큰 CUDA library를 load하면(cuBLASLt만 TLS 94 KB) library의 기존 기본 stack 크기가 한계 아래로 내려갈 수 있다. binary와 `ldd` 출력 전체의 `PT_TLS`를 합하면 빠르게 확인할 수 있다.
- **전역 설정보다 직접 구성을 택한다.** IREE flag registry는 process 전역이므로 device 하나를 고치려고 flag를 설정하면 다른 모든 사용자에게 영향이 간다. IREE public API로 명시적 option을 주어 executor를 만들면 전역 부작용 없이 device를 고칠 수 있고, probe test가 registry가 기본값으로 남았는지 확인한다.
- **오류에 반응하기 전에 원인 주체로 분류한다.** availability gate를 연다는 것은 remote stage에 대한 판단이므로 stage가 일으킨 오류만 이를 촉발해야 한다. `anyhow` chain에서 `downcast_ref`로 찾는 타입 있는 오류는 기존 `anyhow` 흐름을 유지하면서 한 호출 지점이 그 판단을 내리게 한다.
- **프로토콜이 지목할 수 없는 것을 지목하지 않는다.** forward reply는 모든 stage가 실행된 뒤에만 온다. 그 timeout은 run이 멈췄다는 사실만 말하고 어디서 멈췄는지는 말하지 않는다. "미지목"으로 기록하고 다음 stage별 probe가 지목하게 하는 쪽이 요청을 보낸 stage를 지목하는 것보다 정확하다.
