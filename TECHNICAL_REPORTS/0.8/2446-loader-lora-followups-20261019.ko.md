# PR #2446: #2438, #2439 이후 남은 로더 및 LoRA 후속 작업 정리

**작성일**: 2026-10-19
**상태**: CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료. PR은 열려 있음. "미해결 사항"의 항목은 계속 열려 있음

## 요약

#2445를 닫는다. #2438과 #2439가 남긴 후속 작업 5건을 이슈 요청대로 한 PR에서 처리하고, 항목 1을 실제 체크포인트로 검증하다 발견한 블로커 1건을 함께 고친다. 블로커는 `estimate_kv_arch`가 `config.json`을 일반 serde로 파싱해서 `Infinity` / `NaN` 값을 거부하던 문제이다.

## 1. 문제 정의

- **`kv_arch`가 Nemotron-H `layer_types`를 무시한다.** `classify`는 `LAYER_COUNT_KEYS` 항목을 요구하고, `count_attention_layers`는 `layers_block_type`, `block_types`, `hybrid_override_pattern`만 읽는다. #2439로 이제 로드되는 transformers 5 Nemotron-H 설정(`layer_types`만 있음)은 추정값을 얻지 못하고, `num_hidden_layers`가 있으면 모든 레이어를 attention으로 센다. `contains("attention")` 필터는 `M` 레이어인 `linear_attention`도 잘못 센다.
- **`kv_arch`가 실제 Nemotron-H 설정을 파싱하지 못한다.** `nvidia-nemotron-3-nano-30b-a3b-4bit` 같은 설정에는 `time_step_limit: [0.0, Infinity]`가 있다. 일반 serde가 이를 거부해서, 패턴 유무와 관계없이 KV 추정이 "unavailable"이 된다. `main`에서 `mlxcel inspect -m`은 이 체크포인트에 대해 `KV cache: 0 bytes (unavailable ...)`를 출력한다. 항목 1을 검증하는 중에 발견했다.
- **OptiQ 체크포인트에서 오해를 부르는 stale-index 경고.** OptiQ 인덱스는 441 / 809개 텐서를 `optiq/optiq_vision.safetensors`에 매핑한다. `validate_index_shards`가 하위 디렉터리 이름을 거부하므로 `collect_shard_paths`가 "references shards that don't match the on-disk files (likely a repackaged mlx-community quant)"를 출력한다.
- **텍스트 전용 Muse Glimmer가 이미지 마커를 받아들인다.** `reject_muse_glimmer_text_fallback`은 `LoadedModel::MuseGlimmerVLM`에서만 실행되므로, `LoadedModel::MuseGlimmer`는 이미지와 비디오 마커 id를 텍스트 모델에 그대로 넘긴다.
- **정수 LoRA 인자가 퓨전을 중단시킨다.** `compute_lora_delta`가 `lora_a` / `lora_b`를 캐스팅 없이 `mlxcel_core::matmul`에 넘기는데, 브리지의 이 함수는 실패를 반환하지 않으므로 정수 쌍이 들어오면 dense base를 포함해 프로세스가 종료된다. 런타임 경로는 claim 시점에 이미 f32로 캐스팅한다.
- **Nemotron-H가 LoRA 폭 오류를 "unclaimed"로 숨긴다.** 선택적 projection 4개가 `UnifiedLinear::from_weights(..).ok()`를 사용하므로 `claim_linear`의 폭 오류가 버려지고, 로드가 나중에 `ensure_runtime_terms_claimed`에서 실패한다.

## 2. 기술적 선택과 그 이유

- **Nemotron-H `layer_types`** (`src/execution/kv_arch.rs`): `hybrid_override_pattern`이 없는 `model_type` `nemotron_h`에 대해 `layers_block_type` / `layer_types` 배열을 `NEMOTRON_H_BLOCK_NAMES`로 매핑하고, `*` 항목을 세며, 레이어 수는 `NemotronHConfig::post_init`과 같이 배열 길이로 정한다. `NEMOTRON_H_BLOCK_NAMES`는 복사하지 않고 `src/models/nemotron_h.rs`에서 `pub(crate)`로 바꿨다. `linear_attention`은 `M` 레이어이며 더 이상 `contains("attention")` 필터로 세지 않는다.
- **`kv_arch`의 설정 정제 읽기**: 모델 로더와 같이 파일을 먼저 `sanitize_config_json`에 통과시킨다. 이것이 없으면 항목 1은 실제 체크포인트에서 효과가 없다. 또한 `Infinity` / `NaN`이 있는 다른 설정에서도 추정이 가능해진다. 이전에는 KV 0 바이트로 보고되었다.
- **OptiQ 사이드카**: `optiq_vision_sidecar_path`를 같은 검증(상대 경로, 일반 컴포넌트만, `.safetensors`)과 함께 `mlxcel_core::weights`로 옮기고 `loading::vlm`이 재사용한다. `collect_shard_paths`는 `validate_index_shards` 전에 선언된 사이드카만 인덱스의 shard 목록에서 제외하며, 하위 디렉터리 경로를 가리킬 때만 그렇게 한다(최상위 이름은 일반 shard이다). 설정은 모델 로더와 같은 `Infinity` / `NaN` 치환으로 읽는다. 선언되지 않은 하위 디렉터리 항목, 잘못된 사이드카 경로, 읽을 수 없는 설정은 목록을 그대로 두므로 이전처럼 검증에 실패하고 glob 폴백을 탄다.
- **Muse Glimmer 텍스트 전용 마커**: 검사를 토큰 id에 대한 헬퍼로 분리했다. `MuseGlimmerTextWrapper`가 체크포인트의 `image_token_id` / `video_token_id`(`i32` 필드 2개, `load_muse_glimmer_text`가 설정)를 보관하고, `reject_muse_glimmer_text_only_markers`가 `LoadedModel::MuseGlimmer`의 두 지점에서 실행된다. 오류 메시지는 체크포인트가 텍스트 전용으로 로드되었음을 알린다. 이후 커밋에서 비디오 마커 검사를 `contains`로 바꿨다.
- **정수 LoRA 인자**: `compute_lora_delta`가 matmul 전에 두 인자를 f32로 캐스팅한다. `runtime_lora`(claim 시점)와 `src/lora/embedding.rs`가 이미 그렇게 한다. scale이 f32 배열이므로 delta는 이미 f32였다. 따라서 부동소수점 쌍에서는 matmul 입력 정밀도만 달라지고, f32 어댑터(`mlx_lm.lora`가 쓰는 형식)에서는 변화가 없다.
- **Nemotron-H 선택적 projection**: shared-expert `up_proj` / `down_proj`와 latent projection 2개는 `{prefix}.weight`가 없을 때만 `None`을 반환하고, 그 외에는 생성자 오류를 `?`로 전파한다. 그래서 #2438의 `claim_linear` 폭 오류가 나중의 "unclaimed" 실패 대신 로드 오류로 나온다.

## 3. 변경 내용

- `src/execution/kv_arch.rs`: `layer_types` 매핑, 정제된 설정 읽기, 새 테스트.
- `src/models/nemotron_h.rs`: `NEMOTRON_H_BLOCK_NAMES`를 `pub(crate)`로 변경, 선택적 projection 4개의 오류 전파. `src/models/nemotron_h_tests.rs`: 새 테스트.
- `src/lib/mlxcel-core/src/weights.rs`, `src/loading/vlm.rs`, `src/loading/vlm_lfm2_vl_tests.rs`: `optiq_vision_sidecar_path` 이동 및 재사용, `collect_shard_paths`에서 사이드카 제외.
- `src/multimodal/muse_glimmer_runtime.rs`, `src/models/muse_glimmer.rs`, `src/loading/vlm_muse_glimmer.rs`, `src/server/model_worker.rs`, `src/multimodal/vlm_runtime.rs`, `src/loading/vlm_muse_glimmer_tests.rs`: 마커 헬퍼, 래퍼 필드, 호출 지점 2곳, 테스트.
- `src/lora/loader.rs`, `src/lora/loader_tests.rs`: `compute_lora_delta`의 f32 캐스팅, 테스트.

## 4. 검증

GB10, `--features cuda`, `test-fast` 프로파일, 모든 GPU 실행은 `gpu-lock` 아래에서, `MLXCEL_SDPA_DETERMINISTIC=1`. "Main"은 같은 worktree에서 74f9ae1d로 빌드한 바이너리이고, "fixed"는 이 브랜치이다.

- `cargo test --profile test-fast --features cuda --lib -- kv_arch lora:: muse_glimmer nemotron_h loading::vlm multimodal:: model_worker memory_estimate --test-threads=1`: 877개 통과, 0개 실패.
- `cargo test --profile test-fast -p mlxcel-core --features cuda --lib -- weights:: runtime_lora --test-threads=1`: 38개 통과.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings`, `cargo fmt --all -- --check`, 다섯 가지 contract 검사(`check_llama_compat_manifest`, `check_kernel_dtype_keys`, `check_kernel_port_dispatch`, `check_crate_versions`, `insert_apache_header --check`): 모두 통과.

새 테스트는 수정을 되돌리면 모두 실패한다(각 수정을 제자리에서 되돌리고, 테스트 바이너리를 다시 빌드한 뒤 복원).

| 테스트 | 되돌렸을 때 결과 |
|--------|------------------|
| `execution::kv_arch::tests::nemotron_h_layer_types_match_the_equivalent_pattern` | panic: `layer_types` 설정에 추정값 없음 |
| `execution::kv_arch::tests::a_config_file_with_infinity_still_estimates` | panic: 추정값 없음(serde가 `Infinity`를 거부) |
| `weights::tests::test_collect_shard_paths_skips_only_the_declared_optiq_sidecar` (mlxcel-core) | `left: ["model.safetensors", "stray.safetensors"]`, glob 폴백 발생 |
| `loading::vlm::muse_glimmer::tests::text_only_muse_glimmer_refuses_image_and_video_markers` | panic: 텍스트 전용 모델이 마커를 받아들임(호출 지점을 되돌림) |
| `lora::loader::tests::an_integer_lora_pair_fuses_like_the_same_values_in_f32` | 프로세스 중단, exit 134: `[matmul] Only inexact types are supported but int32 and int32 were provided` |
| `models::nemotron_h::tests::a_wrong_width_lora_term_on_an_optional_projection_fails_the_load` | panic: shared expert 없이 로드가 성공함 |

Nemotron-H 테스트는 `cfg(not(all(target_os = "macos", feature = "metal")))`이다. Metal은 레이어가 만들어지기 전에 이 모델의 런타임 LoRA를 거부하기 때문이다.

실제 체크포인트(`~/models_archive/mlx/...`):

| 항목 | 체크포인트와 명령 | Main | Fixed |
|------|-------------------|------|-------|
| 1 | `nvidia-nemotron-3-nano-30b-a3b-4bit`, `mlxcel inspect -m` | `KV cache: 0 bytes (unavailable ...)` | `48.0 MiB (hybrid: 6 attention layer(s) hold KV)` |
| 1 | 같은 체크포인트에 transformers 5 스타일 설정(`hybrid_override_pattern`을 `layer_types`로 바꾸고 `num_hidden_layers` 제거), `mlxcel inspect -m` | `0 bytes (unavailable ...)` | `48.0 MiB (hybrid: 6 attention layer(s) hold KV)`, 패턴 설정과 동일 |
| 2 | `lfm2.5-vl-3b-optiq-4bit`, `mlxcel generate`, 텍스트 프롬프트와 `--image tests/fixtures/test_image_shapes.png` | stale-index 경고 출력 | 경고 없음, 두 프롬프트 모두 생성 텍스트가 main과 동일 |
| 2 | `muse-glimmer-30b-optiq-4bit`, `mlxcel generate --image ... --show-reasoning` | stale-index 경고 출력 | 경고 없음, 생성 텍스트가 main과 동일 |
| 3 | `muse-glimmer-30b-optiq-4bit`의 텍스트 전용 뷰(최상위 shard, 사이드카 항목이 없는 인덱스, `optiq_vision` 없음, `MuseGlimmer`로 감지), 프롬프트 `"<\|image\|> What is in the picture?"` | 받아들이고 마커 id로 생성 | `Error: this Muse Glimmer checkpoint is loaded text-only (no vision tower), but the prompt contains image or video marker tokens ...`, exit 1 |
| 3 | 같은 뷰, 일반 프롬프트 | 생성 | 동일한 텍스트 |
| 4 | `llama-3.2-1b-instruct-bf16`(dense), `mlxcel generate --adapter`, `mlx_lm.lora`로 학습한 14, 15번 레이어의 q_proj 어댑터, 인자를 `round(x * 100)`으로 반올림해 int32로 저장, scale은 1e4로 나눔 | `terminate called after throwing an instance of 'std::invalid_argument' ... [matmul] Only inexact types are supported`, exit 134 | 로드 및 생성 성공, 같은 정수 값을 f32로 저장한 경우와 텍스트 동일 |
| 4 | 같은 조건, f32 쌍둥이 어댑터 | 생성 | main과 텍스트 동일 |
| 5 | `nvidia-nemotron-3-nano-30b-a3b-4bit`(`n_shared_experts: 1`이므로 모든 MoE 레이어가 새 경로로 shared expert를 만든다), `mlxcel generate --show-reasoning`, 24 토큰 | 생성 | 텍스트 동일 |

CLI의 미디어 준비는 서버 worker 지점인 `prepare_request_vlm_embeddings`를 거치므로 항목 3 실행이 그 호출 지점을 검증한다. 단위 테스트는 `prepare_vlm_embeddings_with_cache` 지점과 worker가 호출하는 헬퍼를 다룬다. `prepare_request_vlm_embeddings` 자체를 구동하는 단위 테스트는 없으므로 worker 쪽은 실제 실행으로만 검증된다.

## 5. 미해결 사항

리뷰에서 선언된 사이드카 항목이 이 이슈 범위 밖의 다른 두 곳에서도 잘못된 shard로 취급되는 것을 확인했다.

- `src/downloader/completeness.rs`: `missing_indexed_shards`가 `optiq/optiq_vision.safetensors`를 누락으로 세므로, store에서 해석된 OptiQ 스냅샷이 불완전으로 분류된다.
- `src/server/webui/catalog_fs.rs`: 이를 안전하지 않은 shard 경로로 보고한다.
- `weight_footprint_bytes`도 사이드카의 바이트를 메모리 사전 점검에서 뺀다.

앞의 두 곳은 `mlxcel_core::weights::optiq_vision_sidecar_path`를 재사용할 수 있다.

## 6. 변경 요약

| 항목 | 값 |
|------|----|
| 변경된 파일 | 14 |
| 추가된 줄 | 622 |
| 삭제된 줄 | 87 |

| 커밋 | 메시지 |
|------|--------|
| `c4e6442d` | fix: close loader and LoRA follow-ups from #2438 and #2439 (#2445) |
| `4df6e022` | fix(kv_arch): sanitize config.json before the KV estimate parses it (#2445) |
| `21a7207d` | fix(muse_glimmer): use contains for the video marker check (#2445) |
| `e630c5ca` | fix(weights): sanitize the OptiQ config read and keep top-level shards (#2445) |
