# PR #2776: 모델 소유 cache에 KV mode, KV bound, release 적용

**날짜**: 2026-10-31 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Markdown
**위험도**: 낮음에서 중간 (operator가 요청하면 8개 family가 quantized KV를 저장하기 시작하므로 FP16이 아닌 mode에서 수치가 바뀐다. KV bound와 `--context-shift`를 함께 쓰는 모든 model-owned family는 이제 bound를 넘어 자라지 않고 bound에서 멈춘다)

## 요약

PR #2776 (issue #2766을 닫음)은 #2748이 KV cache를 모델 안에 두는 family에 남긴 잔여 항목 세 가지를 해결한다. 이제 `--kv-cache-mode`가 ExaOne 4, GPT-OSS, Ministral 3, Step 3.5 (그리고 Step-3.7 VLM), Mellum, Laguna, NemotronNAS의 cache에 도달한다. LongCat-Flash와 LongCat-Flash-Ngram은 MLA cache가 어떤 quantized mode도 지원하지 않는 latent pair를 담기 때문에 quantized table을 경고 한 번과 함께 거부하고 FP16을 보고한다. `--max-kv-size` (또는 `--ctx-size` window)와 `--context-shift`를 함께 쓰면 scheduler는 `ModelOwned` layout을 context shifting 없이 서비스한다. 그래서 trim이 조용히 아무 일도 하지 않는 대신 admission 거부와 `truncated: true` 정지로 bound가 지켜진다. single-slot family에 대해서는 공유 slot을 마지막으로 초기화한 sequence를 scheduler가 기록하고, 바로 그 sequence가 release될 때 slot을 비운다. 그래서 idle 상태의 server가 마지막 요청의 KV를 더 이상 상주시키지 않는다. ExaOne 4 Int8은 실제 checkpoint에서 `logit_trace`, `mlxcel generate`, `mlxcel-server`로 검증했고, NemotronNAS slot release는 49B 4-bit checkpoint에서 GPU 메모리로 검증했다. LongCat은 여전히 tiny config 테스트로만 검증된다.

## 1. 문제 정의

### KV mode, bound, release가 원래 작동하는 곳

server와 CLI는 layer마다 KV mode를 결정하고 두 곳에 적용한다. 하나는 scheduler의 pool cache이고, 다른 하나는 `LanguageModel::set_kv_cache_layer_modes` (`src/lib/mlxcel-core/src/generate.rs`의 기본 구현은 no-op)를 통한 모델이다. `--max-kv-size`와 `--context-shift` 조합은 `src/server/batch/scheduler/paged_layout.rs`의 `enforce_max_kv_size_for`가 pool cache를 순회하며 앞부분을 잘라 강제한다. sequence의 KV는 engine이 sequence를 close할 때 해제된다. 세 메커니즘 모두 scheduler pool이 cache를 가지고 있다고 가정한다. `ModelOwned` sequence-state layout인 모델은 pool entry가 비어 있고, cache는 `ModelOwnedSequenceState` (batching family) 또는 모델의 `RefCell` slot 하나 (#2748 이후의 NemotronNAS, LongCat)에 있다.

### `--kv-cache-mode`가 경고 없이 버려짐

#1330은 Gemma 3/4, Qwen 3.5/3.6/3.8, Qwen3-Next, LFM2, Llama 4, AFMoE에 모델 쪽 hook을 구현했다. 이 PR의 8개 family는 hook을 구현하지 않았고 모든 cache를 FP16인 `KVCache::new()` / `RotatingKVCache::new(..)`로 만들었다. pool cache는 mode를 받았지만 이 layout에서는 비어 있다. `--kv-cache-mode int8`을 준 사용자는 FP16 KV와 FP16 메모리 사용량을 그대로 얻었고, `kv_cache_layer_modes()`가 `None`을 반환했기 때문에 `MLXCEL_TRACE_KV_CACHE_MODE=int8`로 실행한 `examples/logit_trace`는 "cannot reach the caches" 오류로 실패했다.

### `--context-shift`에서 model-owned cache가 bound를 넘어 자람

context shifting이 켜져 있으면 KV bound는 trim으로만 강제되는데, trim은 자를 pool cache를 찾지 못했다. 그래서 모든 `ModelOwned` layout (8개 family만이 아니라 Gemma 3/4와 Qwen 3.5도)에서 생성이 `--max-kv-size`를 넘어 context window까지 계속됐다. shifting이 꺼져 있을 때(기본값)는 admission 거부(`admission.rs`)와 finish 단계(`finish.rs`)가 cache를 순회하지 않고 token 수를 세므로 bound가 이미 지켜졌다.

### single-slot family가 마지막 요청의 KV를 유지함

NemotronNAS와 LongCat은 batching을 지원하지 않고 공유 slot 하나를 가진다. scheduler는 다음 prefill(`prefill.rs`, `planned_prefill.rs`)에서만 이 slot을 초기화했으므로, 요청이 끝난 뒤에도 server가 idle인 동안 KV 전체가 할당된 채 남았다. 49B NemotronNAS checkpoint에서 973-token 요청이면 약 250 MiB다. sequence가 release될 때마다 초기화하는 방식은 틀리다. #2748 security review가 지적했듯이, 다른 요청이 decode하는 동안 대기 중인 요청을 cancel하면 진행 중인 sequence의 상태가 지워진다.

## 2. 기술적 선택과 그 이유

### family마다 #1330의 Gemma 3 패턴을 따름

6개 batching wrapper(ExaOne 4, GPT-OSS, Ministral 3, Step 3.5, Mellum, Laguna)에 `src/models/model_owned.rs`의 `KvCacheLayerModes` field, `mode_for_layer(layer_idx)`로 `KVCache::new_with_mode` / `RotatingKVCache::new_with_mode`를 호출하는 `make_caches_with_modes(Option<&KvCacheLayerModes>)` builder, 그리고 cache를 만드는 모든 경로가 쓰는 private `fresh_caches()`를 추가했다. 해당 경로는 `reset_caches`, `prepare_sequence_state`, `with_or_create_sequence_state` 안의 lazy builder다. `set_kv_cache_layer_modes`는 table을 저장하고 fallback slot을 다시 만든다. `kv_cache_layer_modes`는 table을 반환한다. public `make_caches()`는 FP16을 유지하므로 모델 밖에서 자기 cache를 만드는 호출자는 바뀌지 않는다.

issue의 판단 규칙은 attention이 `update_and_fetch`로 K/V를 읽는 family는 mode를 따른다는 것이다. `update_and_fetch`가 모든 mode의 저장과 dequantize를 처리하기 때문이다. mode를 따르는 7개 family 모두 여기에 해당한다. 모든 builder를 `fresh_caches()`로 통일한 것이 중요한 이유는 server가 쓰는 경로가 sequence별 경로이기 때문이다. fallback slot만 고쳤다면 CLI warmup은 quantized가 되고 server가 처리하는 모든 sequence는 FP16으로 남았을 것이다.

### NemotronNAS는 table을 slot이 아니라 block으로 indexing

NemotronNAS는 attention subblock이 있는 block마다 cache 하나를 가지므로, `no_op` block이 나타나면 slot `i`가 layer `i`가 아니다. server는 `num_layers()` (block마다 entry 하나) 기준으로 table을 만들므로, `fresh_caches`는 이제 block을 먼저 enumerate하고 attention이 있는 block만 고른 뒤 `mode_for_layer(block_index)`를 쓴다. `nemotron_nas_kv_mode_table_is_indexed_by_block_layer`가 이를 고정한다. block 1이 `no_op`이면 slot 1은 block 2에 속하므로 block 2의 mode를 가져야 한다(기대값 `[Int8, Int8, Fp16, Int8]`, main은 전부 FP16). `replace_with_linear` block은 쓰지 않는 slot을 table의 mode로 유지하며 비용은 없다. `supports_batching()` 주석에서 KV-mode table이 이 family를 설명하지 못한다는 문장은 삭제했다. 지금도 맞지 않는 것은 paged layout뿐이다.

### LongCat은 FP16을 유지하고 그렇게 보고함

LongCat의 MLA는 `update_and_fetch` 한 번으로 `(kv_latent, k_pe)`를 저장한다. "K" slot은 `kv_lora_rank` 폭이고 "V" slot은 RoPE key이므로 둘 다 head별 K/V row가 아니다. `mla::latent_layout_supports_mode`는 이미 이 layout을 FP16으로 제한한다. 한 slot에서 calibrate한 Turbo codebook이 다른 slot에 적용되고, 비대칭 mode는 latent는 그대로 두고 RoPE key만 quantize하기 때문이다. PR은 `iquestloopcoder` 선례를 따른다. `set_kv_cache_layer_modes`는 경고를 한 번 남기고(`DirectEngine`이 sequence마다 mode를 주입하므로 `Cell<bool>`로 막음), `kv_cache_layer_modes`는 모든 layer에 FP16을 보고한다. scheduler 시작 로그(`log_effective_kv_cache_application`)는 이제 설정된 table보다 모델 자신의 보고를 우선하므로, operator는 LongCat에 대해 요청한 mode가 아니라 FP16을 보게 된다.

issue의 파일 목록 밖에서 두 곳을 고쳤다. `src/lib/mlxcel-core/src/mla/cache.rs`의 `MLA_LATENT_CACHE_FAMILIES`에 `longcat_flash`를 추가했다. 두 `model_type` 모두 `LongcatFlashNgramModel`을 load하지만 n-gram 쪽만 목록에 있어서, CLI resolver는 plain variant를 FP16이 아니라 요청한 mode로 매핑했다. `src/vision/step3p7.rs`는 두 KV-mode hook을 Step 3.5 backbone으로 전달한다. 이것이 없으면 backbone이 mode를 따르게 된 뒤에도 VLM은 trait의 no-op을 유지해 mode를 무시했을 것이다.

### family별 trim을 추가하지 않고 context shift를 끔

`BatchScheduler::with_context_retention`은 이제 `effective_context_retention(policy)`를 저장한다. KV bound가 설정되어 있고, policy가 `context_shift`를 켜고, 모델의 `sequence_state_layout().backend`가 `SequenceStateBackend::ModelOwned`이면, `context_shift: false`인 policy를 반환하고 process당 한 번 경고한다(worker가 idle sleep에서 깨어날 때마다 scheduler를 다시 만들기 때문에 static `AtomicBool` 사용). `with_max_kv_size`도 같은 함수를 다시 적용하므로 bound와 policy 중 어느 쪽이 먼저 설치되든 결과가 같다(`context_shift_fallback_holds_when_the_bound_is_set_last`).

대안인 family별 front-trim hook은 issue에서 기각됐다. rotating, dense, latent layout이 각자 trim을 구현해야 하고, trim이 틀리면 attention이 조용히 망가진다. admission 거부와 `truncated: true`는 눈에 보이고 이미 테스트된 정지 방식이다. 대가는 Gemma 3/4, Qwen 3.5, 이 PR의 family에서 끝없이 굴러가는 생성을 원하던 operator가 bound에서 정지를 받는다는 점이다. 이 PR 이전에 그 operator가 받은 것은 자신이 설정한 bound를 무시하고 무한히 자라는 생성이었고, 그것도 rolling window는 아니었다.

### slot을 채운 sequence만 release

`BatchScheduler`에 `model_owned_slot_holder: Option<SequenceId>`를 추가했다. 두 prefill 초기화 지점은 이제 `reset_model_owned_slot_for(seq_id)`를 호출한다. 이 함수는 batching을 지원하지 않는 모델에 대해 기존 `make_caches()` 초기화를 그대로 하고 holder를 기록한다. `release_sequence_caches`는 `release_model_owned_slot_if_held_by(seq_id)`를 호출한다. id가 holder이고 모델이 batching을 지원하지 않으면 `reset_runtime_state()`를 호출하고 holder를 비운 뒤, MLX가 해제된 buffer를 cache에 붙잡고 있지 않도록 `mlxcel_core::clear_memory_cache()`를 호출한다. 다른 id는 아무것도 바꾸지 않는다. 이것이 #2748 review가 요구한 성질이다. prefill한 적 없는 대기 요청은 진행 중인 slot을 지울 수 없다.

이 검사는 NemotronNAS와 LongCat만이 아니라 scheduler가 서비스하는 batching 미지원 family 전부에 적용된다. Phi4MM, OCR wrapper, Nemotron VoiceChat의 `reset_runtime_state()`는 fallback 상태만 교체한다. 이 family들은 모두 active sequence 하나로 제한되므로 holder가 아직 decode 중인 sequence일 수는 없다.

### dense 대조군이 shift가 일어났음을 증명하게 함

`dense_layout_keeps_context_shift_under_a_kv_bound`의 첫 버전은 paged 기본 decode storage를 썼다. 그곳에서는 pool-backed FP16 cache의 front trim이 no-op이므로 shift 없이도 테스트가 통과했다. commit `28991823`은 이 테스트를 `Dense` decode storage에서, 모든 projection이 0인 one-layer Llama(`cycle_llama`, token `t`를 `t + 1`로 매핑하고 stop을 내지 않음)로 실행하고, pool cache의 live length가 bound 16에 도달하고 17을 넘지 않는지 확인한다. 이것이 없었다면 대조군은 "dense는 여전히 shift한다"와 "어디서도 shift하지 않는다"를 구분하지 못했을 것이다.

## 3. 변경 요약

26개 파일, +1267 / -96. commit 세 개(수정 `91fe46c0`, 문서 정정 `ef0b6c95`, dense 대조군 `28991823`)와 origin/main merge(`69650626`).

| 영역 | 파일 | 변경 |
|---|---|---|
| batching model-owned family | `exaone4.rs`, `gpt_oss.rs`, `ministral3.rs`, `step3p5.rs`, `mellum.rs`, `laguna.rs` | `KvCacheLayerModes` field, `make_caches_with_modes`, 모든 builder 경로의 `fresh_caches()`, trait hook 두 개, `#[cfg(test)] sequence_kv_modes` accessor |
| single-slot family | `nemotron_nas.rs`, `longcat_flash_ngram.rs` | NemotronNAS: block 기준 mode table, `slot_live_len` / `slot_kv_modes` 테스트 accessor. LongCat: 경고 한 번과 함께 FP16만 허용 |
| VLM wrapper | `src/vision/step3p7.rs` | 두 KV-mode hook을 Step 3.5 backbone으로 전달 |
| MLA resolver | `src/lib/mlxcel-core/src/mla/cache.rs` | `MLA_LATENT_CACHE_FAMILIES`와 테스트에 `longcat_flash` 추가 |
| Scheduler | `scheduler/config.rs`, `mod.rs`, `handoff.rs`, `prefill.rs`, `planned_prefill.rs`, `paged_layout.rs` | context-shift fallback, `model_owned_slot_holder`, holder 범위의 release, 시작 로그에 모델이 보고한 mode 사용 |
| Engine 주석 | `src/lib/mlxcel-core/src/engine/direct.rs` | 공유 `RefCell` slot 하나를 가지는 것은 NemotronNAS와 LongCat뿐이고 나머지 6개는 sequence별로 상태를 가짐 |
| 테스트 | `model_owned_kv_mode_tests.rs` (신규), `scheduler_model_owned_slot_tests.rs` (신규), `nemotron_nas_engine_tests.rs`, `longcat_flash_ngram_engine_tests.rs`, `gpt_oss_tests.rs`와 `mixed_cache_batched_decode_tests.rs`의 visibility 변경 | 신규 테스트 34개 |
| 문서 | `docs/supported-models.md`, `docs/llama-server-compat.md` | sliding layer 주의사항을 포함한 family별 KV-mode 표, context-shift fallback |

family별 결과:

| Family | `--kv-cache-mode` |
|---|---|
| ExaOne 4, GPT-OSS, Ministral 3, Mellum, Laguna | layer별로 적용 |
| Step 3.5와 Step-3.7 VLM | layer별로 적용 |
| NemotronNAS | 적용, block layer index 기준 |
| LongCat Flash / LongCat Flash N-gram | FP16만, 경고 한 번, FP16 보고 |

## 4. 트레이드오프와 잔여 항목

- **대부분의 mode에서 sliding layer는 FP16으로 남는다.** 이 family들은 sliding-window layer를 `RotatingKVCache`로 만드는데, 이 cache는 `fp16+turbo4`일 때만 quantized 데이터를 저장하고 `int8`, `turbo4`, `turbo4-delegated`, `fp16+turbo3`에서는 FP16을 유지한다. 그래서 `int8`에서 GPT-OSS, Mellum, Laguna, Step 3.5, hybrid ExaOne 4 checkpoint의 global layer는 quantize되지만 sliding layer는 그렇지 않다. Gemma 3/4에도 같은 제약이 이미 있다. 첫 commit의 문서는 반대로 적었고 `ef0b6c95`가 정정했다.
- **시작 보고가 sliding layer를 과장한다.** `kv_cache_layer_modes()`는 요청된 table을 반환하므로, 해당 layer에 대해 server의 `applied_layers` 수와 `logit_trace`는 rotating cache가 실제로 저장하지 않는 mode를 보고한다. 실제 저장 mode를 보고하려면 rotating cache의 fallback 규칙을 모델에 노출해야 한다.
- **Laguna의 DFlash round loop**는 자체 FP16 cache를 유지하며 table이 그 cache에는 닿지 않는다.
- **`--context-shift`를 쓰는 모든 model-owned family의 동작 변경.** bound가 설정되면 Gemma 3/4, Qwen 3.5, 이 PR의 family는 이제 `truncated: true`로 bound에서 멈춘다. 경고가 원인을 알려준다. layout별 실제 trim이 생기면 fallback을 없앨 수 있지만 이번에는 미뤘다.
- **builder chain 안에서 fallback은 한 방향이다.** `effective_context_retention`이 `context_shift`를 한 번 끄면, 이후 `with_max_kv_size(None)`이 다시 켜지 않는다. scheduler는 깨어날 때마다 한 번 만들어지므로 실제로는 발생하지 않는다.
- **LongCat은 실제 weight로 검증되지 않았다**(560B, 들어가는 호스트 없음). #2748과 같다. FP16 보고와 resolver 매핑은 tiny config 테스트가 다룬다.
- **NemotronNAS 메모리 감소는 main과 나란히 비교하지 않았다.** 아래 측정은 요청마다 메모리가 load 직후 수준 근처로 돌아옴을 보여주지만, main은 같은 부하에서 실행하지 않았다.
- **Metal과 ROCm은 로컬에서 실행하지 않았다.** 아래 gate는 GB10(CUDA)이다.

## 5. 검증

실패 우선 확인: test accessor만 남기고 main의 동작(hook 제거, fallback과 release 비활성화, MLA 목록에서 `longcat_flash` 제외)으로 신규 테스트를 실행했다. 의도한 대조군 두 개(`dense_layout_keeps_context_shift_under_a_kv_bound`, `context_shift_is_kept_without_a_kv_bound`)를 제외한 모든 신규 테스트가 실패했다. 대표적인 실패:

| 확인 항목 | main | PR |
|---|---|---|
| Int8 주입 후 `kv_cache_layer_modes()` (8개 family 전부) | `None` | `Some([Int8, ..])` (LongCat은 FP16) |
| block 1이 `no_op`이고 block 2가 FP16일 때 NemotronNAS slot mode | `[Fp16, Fp16, Fp16, Fp16]` | `[Int8, Int8, Fp16, Int8]` |
| `model_owned_layout_stops_at_the_kv_bound_under_context_shift` (bound 16) | sequence가 6 + 40까지 감 | bound에서 정지, `truncated: true` |
| `completing_the_slot_holder_frees_the_single_slot` | slot live length 11 | 0 |
| 완료 후 holder | `Some(SequenceId(0))` | `None` |

mode를 따르는 family마다 Int8에서 8-token greedy 테스트도 있다. logit이 유한하고 첫 token이 FP16과 같은지 확인한다.

GB10 로컬 gate(`MLX_ENABLE_TF32=0`): 신규 테스트 34개 통과, `--lib models::` 2491개 통과, `--lib server::batch::` 587개 통과, mlxcel-core `engine::`, `mla::`, `cache::turbo::allowlist` 130개 통과, KV advisor, registry, turbo-args 테스트 109개 통과, `cargo fmt --check`, mlxcel(lib, tests, `logit_trace`)과 mlxcel-core(lib, tests)에 clippy `-D warnings`.

GB10 실제 checkpoint, `gpu-lock` 하에서:

- `exaone-4.0-1.2b-4bit`에서 `MLXCEL_TRACE_KV_CACHE_MODE=int8`로 `logit_trace` 성공. main에서는 `install`이 cannot-reach 오류로 실패한다. FP16 trace 대비 top-1 불일치는 전체 15/256, 결정된 위치에서 0/87, perplexity -0.3%. Int8이 cache에 도달하고 FP16에 가깝게 유지됨을 보여준다.
- 같은 checkpoint에서 `mlxcel generate --kv-cache-mode int8`(1370-token prompt, greedy)은 완료되고, 몇 token 뒤 FP16과 다른 continuation이 나온다. mode가 버려졌다면 불가능한 결과다.
- ExaOne 4에서 `mlxcel-server --kv-cache-mode int8`은 요청을 완료하고 `kv_cache_mode_effective=int8 applied_layers=30/30`을 로그로 남긴다.
- ExaOne 4에서 `mlxcel-server --max-kv-size 128 --context-shift`는 fallback 경고를 남기고, `ignore_eos`를 준 10-token prompt는 생성 117 token에서 `truncated: true`, `stop_type: "limit"`으로 멈춘다.
- `Llama-3_3-Nemotron-Super-49B-v1-mlx-4bit` server에 동일한 greedy 973-token 요청 두 개: 출력이 byte 단위로 동일. GPU 메모리(nvidia-smi)는 load 후 27006 MiB, 각 요청 중 최대 28774 MiB, 첫 요청 후 27046 MiB, 두 번째 요청 후 27014 MiB. 이 길이에서 요청 하나의 KV가 약 250 MiB이므로 release 시 slot이 비워진다.

## 6. 학습 포인트

- **trait의 hook은 그 뒤의 builder만큼만 유효하다.** model-owned family에는 cache를 만드는 경로가 여러 개 있다. fallback slot, `prepare_sequence_state`, `with_or_create_sequence_state` 안의 lazy builder다. 모두 주입된 table을 써야 하며, 그렇지 않으면 mode가 일부 sequence에만 도달한다. wrapper마다 `fresh_caches()` 하나를 두는 것이 이들을 맞추는 저렴한 방법이고, fallback이 아니라 실제 sequence에서 mode를 읽어 확인하는 테스트가 누락을 잡는다.
- **layer별 table은 만든 쪽이 쓴 index로 indexing한다.** server는 `num_layers()` 기준으로 mode를 결정한다. cache 목록이 layer의 부분집합인 모델(NemotronNAS, `no_op`이나 linear 대체 block이 있는 family)은 layer index를 거쳐 매핑해야 하고, 두 indexing을 구분하려면 테스트의 layer 목록에 빈 자리가 있어야 한다.
- **no-op trim은 조용한 bound 위반이다.** scheduler pool을 순회하는 강제 로직은 `ModelOwned` layout에서 아무것도 하지 않는다. 어떤 메커니즘이 layout에 작용할 수 없을 때는 받아들이는 것보다 시작 시 경고와 함께 끄는 편이 안전하다.
- **대조군 테스트는 올바른 이유로 실패할 수 있어야 한다.** 첫 dense 대조군은 paged 기본값이 trim을 no-op으로 만들었기 때문에 통과했고, shifting에 대해서는 아무것도 증명하지 못했다. 오류가 없다는 사실이 아니라 관찰 가능한 효과(live length가 bound 근처에서 제한됨)를 확인해야 한다.
- **release는 소유자에게 한정한다.** 공유 single-slot 상태에서는 slot을 채운 주체를 기록하고 그 주체가 release될 때만 비운다. release마다 무조건 초기화하면 cancel된 대기 요청이 실행 중인 요청을 망가뜨린다.
