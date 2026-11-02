# PR #2788: #2776이 남긴 KV mode 보고 공백 해소

**날짜**: 2026-10-31 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: Rust, Markdown
**위험도**: 낮음 (보고와 metadata 변경, startup panic 하나를 FP16 fallback으로 바꿈, 경고가 모델마다 출력됨. aligned 경우나 Turbo가 아닌 mode에서 cache가 저장하는 내용은 바뀌지 않는다)

## 요약

PR #2788 (issue #2783을 닫음)은 #2776 리뷰와 그 report(4절)가 model-owned KV 영역에 남긴 LOW 잔여 항목 네 가지를 해결한다. architecture registry는 이제 #2776이 연결한 7개 family에 대해 `fp16`, `int8`, `turbo4`를 광고하고, LongCat은 FP16만 유지한다. 새 `LanguageModel::kv_cache_storage_modes`가 각 cache가 실제로 저장하는 mode를 보고하므로, server의 `kv_cache_mode_applied_layers` 집계와 `logit_trace`가 FP16 sliding-window layer를 quantized로 세지 않는다. `--kv-cache-mode int8`에서 Gemma 3 1B는 26/26 대신 4/26을, GPT-OSS 20B는 24/24 대신 12/24를 보고한다. 32의 배수가 아닌 sliding window에서 `fp16+turbo4`는 더 이상 startup에서 panic하지 않고, sliding layer가 경고 한 번과 함께 FP16으로 fallback한다. context-shift fallback 경고는 process 전역 static에서 worker별 flag로 옮겨져, multi-model server의 영향받는 모든 모델이 각자 한 번씩 경고하고 idle sleep에서 깨어날 때는 반복하지 않는다.

## 1. 문제 정의

### 오래된 registry metadata

`ModelType::capabilities`는 `src/models/registry.rs`의 `supports_turbo_kv`에 매치되는 type에만 `kv_modes`를 `KV_TURBO`로 설정한다. #2776 이후 ExaOne 4, GPT-OSS, Step 3.5, Step-3.7, Mellum, Laguna, NemotronNAS가 `int8`과 `turbo4`를 따르지만 목록에는 `Ministral3`만 있었으므로, `mlxcel arch --json`은 나머지 7개에 대해 여전히 `["fp16"]`을 보고했다.

### layer별 보고가 sliding layer를 과대 보고함

`RotatingKVCache::update_and_fetch` (`src/lib/mlxcel-core/src/cache.rs`)는 `Turbo4Asym`에서만 quantize한다. `Int8`, `Turbo4`, `Turbo4Delegated`, `Turbo3Asym`에서는 FP16으로 저장한다. 각 family의 `kv_cache_layer_modes()`는 주입된 table을 반환하고, 이 table의 두 소비자(`src/server/batch/scheduler/paged_layout.rs`의 `log_effective_kv_cache_application`, `examples/logit_trace/kv_mode.rs`)는 그것을 저장 상태로 취급했다. Gemma 3에 `int8`을 준 operator는 `applied_layers=26/26`을 봤지만 그중 22개 layer는 FP16을 담고 있었다. table로 `RotatingKVCache`를 만드는 모든 family가 같았다: Gemma 3, Gemma 4, GPT-OSS, ExaOne 4, Laguna, Step 3.5, Ministral 3, Mellum, AFMoE.

### 정렬되지 않은 window에서 `fp16+turbo4`가 panic함

`RotatingKVCache::new_with_mode_and_seed`는 `Turbo4Asym` cache의 `max_size`가 `turbo::BLOCK_SIZE`(32)의 배수인지 assert한다. 9개 builder 모두 `config.sliding_window`를 그대로 넘겼으므로, 예를 들어 window가 100인 checkpoint는 첫 cache 생성에서 abort된다. Gemma 3는 이미 이 위험이 있었고, #2776이 새로 연결한 family로 범위를 넓혔다. 로컬 checkpoint 중에는 이를 유발하는 것이 없다(window는 128, 512, 1024, 4096).

### context-shift 경고가 process당 한 번만 출력됨

#2776은 KV bound가 있을 때 model-owned layout의 `--context-shift`를 끄고, 함수 내부의 `static WARNED: AtomicBool` 뒤에서 경고했다. static을 쓴 이유는 worker가 idle sleep에서 깰 때마다 scheduler를 다시 만들기 때문이다. 부작용으로 multi-model server에서는 영향받는 첫 모델만 경고를 남겼다. 또 경고가 `effective_context_retention` 안에서 출력되었으므로, 출력 여부가 `with_context_retention`과 `with_max_kv_size`의 호출 순서에 의존했다.

## 2. 기술적 선택과 그 이유

### `kv_cache_layer_modes`를 바꾸지 않고 "주입된 table"과 "저장 상태"를 분리

가장 단순한 수정은 `kv_cache_layer_modes()`가 sliding layer에 FP16을 반환하게 하는 것이었다. 하지만 그러면 `set_kv_cache_layer_modes`의 round-trip이라는 계약이 깨진다. `logit_trace`는 주입 후 `applied == modes`를 확인하고, `model_owned_kv_mode_tests`는 주입한 table과 돌려받은 값을 비교한다. 이 PR은 두 번째 trait method `kv_cache_storage_modes()`를 추가하고, 기본 구현을 `kv_cache_layer_modes()`로 둔다. 저장 상태를 설명하는 소비자(startup log와 logit_trace의 추가 출력)만 새 method로 바꾼다. `kv_cache_layer_modes`를 forward하는 모든 wrapper(`LoadedModel`, `language_model_ref.rs`의 `&M` blanket impl, VLM wrapper 11개)도 새 method를 forward한다. 그렇지 않으면 VLM이 조용히 trait 기본 구현으로 돌아가 과대 집계가 다시 생긴다.

### 규칙은 한 번, 그것이 반영하는 코드 옆에

`RotatingKVCache::stored_mode(max_size, requested)`는 `const fn`이며, 32의 양의 배수 window에서 `Turbo4Asym`일 때만 `Turbo4Asym`을, 그 외에는 `Fp16`을 반환한다. `update_and_fetch` 바로 옆에 두었고, 두 코드가 함께 바뀌어야 한다는 doc comment를 달았다. `KvCacheLayerModes::storage_modes(window_of)`는 sliding layer를 이 함수로 매핑한다. 9개 family는 각각 `kv_cache_storage_modes`를 override하면서 자기 cache builder가 쓰는 것과 같은 sliding 판정(Gemma 3는 `sliding_window_pattern`, GPT-OSS는 `layer_types`, ExaOne 4는 `is_local`, AFMoE는 `uses_sliding` 등)과 같은 window 값(ExaOne 4의 `unwrap_or(512)` 기본값 포함)을 넘긴다.

cache의 `mode` field를 기준으로 쓰지 않은 이유는, `Int8`에서 sliding `RotatingKVCache`가 `mode == Int8`로 만들어지고 내부적으로만 FP16을 저장하기 때문이다. field를 읽으면 버그가 그대로 재현된다.

### panic하는 경우만 downgrade

`KvCacheLayerModes::rotating_mode_for_layer(layer_idx, window)`는 layer의 mode가 `Turbo4Asym`이고 window가 정렬되지 않았을 때 `Fp16`을, 그 외에는 table의 mode를 그대로 반환한다. 9개 builder 모두 `RotatingKVCache`에 이 함수를 쓰고, full-attention layer는 `mode_for_layer`를 유지한다. 다른 mode는 FP16으로 저장되더라도 의도적으로 `Fp16`으로 바꾸지 않는다. rotating cache의 `mode`는 `can_truncate_to`와 snapshot 재사용에도 쓰이므로, 바꾸면 지금 잘 동작하는 설정의 해당 경로가 달라진다. cache는 sequence마다 다시 만들어지므로 경고는 `KvCacheLayerModes` 인스턴스의 `Cell<bool>`로 한 번만 출력한다(LongCat 거부와 같은 패턴). `RotatingKVCache::new_with_mode`는 직접 호출자를 위해 assert를 유지하므로, model builder 밖의 오용은 여전히 크게 실패한다.

### 경고를 builder 밖으로 옮기고 worker 단위로 범위 지정

`effective_context_retention`은 이제 `(policy, forced_off)`를 반환하고 아무것도 출력하지 않는다. scheduler는 `context_shift_forced_off`와 `context_shift_fallback_warned: Arc<AtomicBool>`(기본값은 scheduler마다 새 flag)를 가진다. `warn_context_shift_fallback_once()`는 flag를 swap하고 기존 문구를 출력하며, builder chain이 확정된 뒤 `run`의 첫 단계에서 호출된다. `spawn_model_worker_with_batch_config`는 load/sleep/wake loop 바깥에서 worker thread마다 flag 하나를 만들고, 다시 만들 때마다 `with_context_shift_fallback_warned`로 넘긴다. worker 하나가 모델 하나를 서비스하므로, 몇 번 sleep하든 모델당 경고 한 번이 된다. `with_max_kv_size`가 `context_shift_forced_off`에 OR로 기록하는 이유는, 이미 강제로 꺼진 policy는 `context_shift: false`이므로 재평가하면 기록이 지워지기 때문이다.

## 3. 변경 요약

38개 파일, +648 / -78, commit 하나(`af0ca3b4`).

| 영역 | 파일 | 변경 |
|---|---|---|
| core cache 규칙 | `src/lib/mlxcel-core/src/cache.rs` | `RotatingKVCache::stored_mode` |
| trait | `src/lib/mlxcel-core/src/generate.rs`, `language_model_ref.rs` | 기본 구현이 있는 `kv_cache_storage_modes`, `&M`을 통해 forward |
| 공용 helper | `src/models/model_owned.rs` | `rotating_mode_for_layer`(경고 한 번), `storage_modes` |
| family | `gemma3.rs`, `gemma4.rs`, `gpt_oss.rs`, `exaone4.rs`, `laguna.rs`, `step3p5.rs`, `ministral3.rs`, `mellum.rs`, `afmoe.rs` | sliding layer builder가 `rotating_mode_for_layer` 사용, `kv_cache_storage_modes` override |
| forwarder | `src/loaded_model.rs`, `src/vision/{mod, gemma4_vl, gemma4_unified, qwen3_5_vl, qwen4_exp_vl, lfm2_vl, step3p7, got_ocr, mage_vl, llmjp_vl, llmjp_vl_text}.rs` | `kv_cache_storage_modes` forward |
| registry | `src/models/registry.rs` | 7개 type을 `supports_turbo_kv`에 추가 |
| scheduler | `scheduler/config.rs`, `mod.rs`, `run_loop.rs`, `paged_layout.rs`, `src/server/model_worker.rs` | forced-off 기록, worker별 flag, `run`에서 경고, startup 집계를 storage mode로 |
| example | `examples/logit_trace/kv_mode.rs` | 저장 상태가 다르면 `kv cache storage: {n}/{total} layers {mode}, {s} sliding layers fp16` 출력 |
| 테스트 | `rotating_turbo_window_tests.rs`, `model_owned_kv_mode_tests.rs`, `scheduler_model_owned_slot_tests.rs`, `registry.rs`, `gemma3_tests.rs`(fixture 공개 범위) | 5절의 새 테스트 |
| 문서 | `docs/supported-models.md`, `docs/llama-server-compat.md` | 저장 상태 집계와 misaligned window 설명, `kv_modes` 목록, 모델별 경고 문구 |

`recipes/registry/*.json`은 건드리지 않았다. 이 snapshot은 release 때 `make recipes-registry`로 다시 생성된다.

## 4. 트레이드오프와 잔여 항목

- **sliding 판정이 family마다 두 번 쓰인다.** 각 override는 자기 cache builder의 판정과 window를 반복한다. 나중에 sliding layer를 고르는 방식을 바꾸는 family는 두 곳을 모두 고쳐야 하며, 그렇지 않으면 보고가 저장 상태와 어긋난다. 테스트는 Gemma 3와 GPT-OSS만 고정한다. 나머지 7개 override는 storage 전용 assertion이 아니라 family 테스트 suite 통과로만 확인된다.
- **`stored_mode`와 `update_and_fetch`의 동기화는 comment에 의존한다.** rotating cache가 `Int8` quantized 저장을 지원하게 되면(이 PR 범위 밖) 같은 PR에서 `stored_mode`도 바꿔야 한다.
- **misaligned window 수정은 실제 checkpoint로 검증되지 않았다.** 로컬에는 window가 32의 배수가 아닌 checkpoint가 없다. window 100(Gemma 3)과 4(GPT-OSS)인 unit test가 유일한 근거다.
- **logit_trace의 storage 출력은 균일한 요청을 가정한다.** 저장 mode가 요청 `mode`와 같은 layer 수를 센다. 혼합 table(예: Boundary-V)에서는 숫자는 맞지만 mode 하나로만 표기된다.
- **Metal과 ROCm은 로컬에서 실행하지 않았다.** gate는 GB10(CUDA)이다.
- **범위 밖**(issue에 명시): `RotatingKVCache`에 `Int8` 등 다른 mode를 저장하는 것, Laguna의 DFlash round-loop cache, layout별 실제 context-shift trim, #2776 리뷰의 slot-release 범위 메모.

## 5. 검증

failing-first: 새 테스트는 main에 없는 API를 호출하므로, 각 수정을 main의 동작으로 되돌린 branch를 빌드했다(registry 항목 7개 제거, `rotating_mode_for_layer`가 table을 그대로 통과, `storage_modes`가 주입된 table 반환, 경고 flag를 process 전역 static으로 복원). 새 mlxcel 테스트 6개가 모두 실패했다.

| 테스트 | main 동작에서의 실패 |
|---|---|
| `model_owned_kv_families_advertise_quantized_modes` | `[Fp16]` vs `[Fp16, Int8, Turbo4]` |
| `gemma3_sliding_layers_report_fp16_storage_under_int8`, `gpt_oss_...` | `Some([Int8, Int8])` vs `Some([Fp16, Int8])` |
| `turbo4_asym_on_a_misaligned_window_falls_back_to_fp16` | `RotatingKVCache` constructor assert에서 panic |
| `gemma3_storage_modes_follow_the_turbo4_asym_window_alignment` | 같은 panic |
| `context_shift_fallback_warns_once_per_model` | 두 번째 모델이 경고하지 않음 |

mlxcel-core 테스트 `stored_mode_reports_fp16_except_for_aligned_turbo4_asym`은 6개 mode를 window 512, 100, 0에서 확인하고, aligned 결과가 constructor가 실제로 만드는 mode와 같은지도 확인한다. `context_shift_fallback_does_not_warn_when_nothing_was_forced_off`는 대조군이다.

GB10 로컬 gate(`--profile test-fast --features cuda`, `--test-threads=1`, `gpu-lock` 사용): mlxcel-core `cache::` 586개 통과. `models::registry`, `models::model_owned_kv_mode_tests`, `scheduler_model_owned_slot_tests`, `models::gemma3`, `models::gpt_oss`, `models::model_owned` 124개 통과(#2776의 `model_owned_layout_stops_at_the_kv_bound_under_context_shift`, `context_shift_fallback_holds_when_the_bound_is_set_last` 포함). 영향받는 family(`exaone4 laguna step3p5 ministral3 mellum afmoe gemma4 mixed_cache vision::step3p7 kv_cache_mode`) 245개 통과. `cargo fmt --check`와 clippy `-D warnings` 통과.

실제 checkpoint(`MLX_ENABLE_TF32=0`, "이전"은 `29c2f0c2`의 main build):

| 실행 | 이전 | 이후 |
|---|---|---|
| `mlxcel-server -m gemma-3-1b-it-4bit --kv-cache-mode int8` | applied 26/26 | applied 4/26 (global layer) |
| `mlxcel-server -m gpt-oss-20b-mxfp4 --kv-cache-mode int8` | applied 24/24 | applied 12/24 (full-attention layer) |
| `mlxcel-server -m gemma-3-1b-it-4bit --kv-cache-mode fp16+turbo4` (window 512) | | applied 22/26, fallback 경고 없음 |
| `exaone-4.0-1.2b-4bit`에서 int8 `logit_trace` | | 성공, storage 출력 없음(sliding layer 없음) |
| `gemma-3-1b-it-4bit`에서 int8 `logit_trace` | | `kv cache storage: 4/26 layers int8, 22 sliding layers fp16` 출력 |

`fp16+turbo4` 행은 aligned 대조군이다. 이 경우 sliding layer가 실제로 quantize되므로 집계가 줄지 않고 22/26으로 올라간다.

## 6. 학습 포인트

- **setter의 round-trip인 getter를 상태 보고용으로 겸용하지 않는다.** `kv_cache_layer_modes`는 "무엇이 주입되었나"에 답했는데, 두 소비자가 이를 "무엇이 저장되었나"로 읽었다. 첫 번째와 같은 기본 구현을 가진 두 번째 method를 추가하면 기존 호출자는 모두 그대로 맞고, 보고 경로만 바꿀 수 있다.
- **타입의 field가 동작을 설명하지 못하면 field가 아니라 규칙에서 동작을 도출한다.** `mode == Int8`로 만든 rotating cache는 FP16을 저장한다. 보고는 `update_and_fetch`가 적용하는 것과 같은 규칙에서 나와야 한다.
- **trait 기본 구현은 forwarding 누락을 숨긴다.** 기본 구현이 있는 새 trait method는 forward를 잊은 wrapper에서도 문제없이 컴파일된다. 형제 method를 forward하는 wrapper는 새 method도 forward해야 하며, 형제 method를 grep하는 것이 이를 찾는 방법이다.
- **fallback은 실패하는 경우로 좁힌다.** 지원하지 않는 sliding mode를 모두 FP16으로 바꾸면 더 깔끔해 보이지만, 지금 동작하는 설정의 truncation과 snapshot 재사용까지 바뀐다. 바꿀 필요가 있었던 것은 panic하는 조합뿐이었다.
- **"한 번만 경고"의 범위는 경고 대상의 범위와 맞춘다.** multi-model process에서 모델 단위 조건에 process 전역 static은 잘못된 범위다. 모델별 worker가 소유하고 rebuild 사이에 공유하는 flag가 수명과 정확히 맞는다. builder 단계가 아니라 `run`에서 경고하면 builder 호출 순서 의존도 사라진다.
