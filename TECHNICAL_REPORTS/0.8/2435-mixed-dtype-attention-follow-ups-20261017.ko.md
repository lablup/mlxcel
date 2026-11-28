# PR #2435: 혼합 bf16/fp16 경로에서 attention dtype 사용

**작성일**: 2026-10-17
**상태**: 머지 전에 CUDA(GB10)에서 실제 체크포인트로 구현 및 검증 완료
**위험도**: 낮음 (q, k, v의 dtype이 같으면 바뀐 판단이 모두 기존과 동일. 스냅샷 변경은 조용한 no-op을 명시적 거부로 바꿈. 테스트 7개 추가)

## 요약

이슈 #2422는 #2417 리뷰에서 발견된 혼합 bf16/fp16 경로의 결함 네 가지를 묶은 것이다: CUDA 쿼리 청크 게이트, Gemma 2 softcap composite, 모순된 오버레이 헤더, turbo4에서의 잘못된 스냅샷 skip 사유. fp16 TurboQuant KV 캐시 위의 bf16 모델은 `result_type(q, k, v)`, 즉 float32로 attention을 수행한다. 이 PR은 네 곳 모두 q의 dtype이나 잘못된 가정 대신 이 타입을 기준으로 삼게 한다. 스냅샷 테스트를 작성하던 중, 값이 채워진 대칭 `turbo4` `KVCache`가 스냅샷 경로에서 `Ok`를 반환하면서 아무것도 저장하지 않는 문제도 발견했으며, 같은 변경으로 고쳐진다.

## 문제 정의

네 항목 모두 하나의 사실에서 나온다. bf16 + fp16은 float32로 promote되지만, 네 곳은 여전히 q의 dtype이나 주석을 기준으로 판단했다.

1. **CUDA 청크 게이트 (MEDIUM).** `cuda_sdpa_materializes_scores`는 cuDNN flash 적합성을 `array_dtype(q)`로 판단했고, `materializing_sdpa_query_chunk`도 score 행 크기를 이 값으로 계산했다. fp16 K/V 위의 bf16 q에서는 MLX가 score당 4바이트로 materializing fallback을 실행한다. 게이트는 청크 분할을 건너뛰거나(head_dim <= 128) score당 2바이트로 예산을 잡아, 긴 프롬프트에서 메모리 예산을 넘겨 OOM이 날 수 있었다.
2. **Softcap composite (LOW).** `compiled_softcap_sdpa`와 `_gqa`는 마스크를 q의 dtype으로 캐스팅하고 v의 dtype을 반환했다. fast SDPA는 둘 다 `result_type(q, k, v)`를 사용한다.
3. **오버레이 헤더 (LOW).** `src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp`의 헤더는 bf16 + float16이 bf16이라고 했지만, 표와 뒤쪽 주석은 float32를 준다.
4. **스냅샷 skip 사유 (LOW).** `snapshot_standard`와 `snapshot_rotating`은 캐시 모드보다 먼저 keys와 values가 모두 있는지 확인했다. Turbo4Asym은 V를 `v_packed`에 보관하므로 Gemma 4는 모드 이름 대신 `rotating cache has only one of keys/values`를 로그로 남겼다. `docs/turbo-kv-cache.md`는 non-FP16 donation이 설계상 fail-closed라고 하므로, 잘못된 것은 동작이 아니라 사유였다.

### 진행 중 발견: 조용한 스냅샷 누락

값이 채워진 대칭 `turbo4` `KVCache`(keys와 values가 모두 `None`, offset 8)로 항목 4를 테스트하자, 스냅샷이 `Ok`를 반환하면서 아무것도 저장하지 않았다. main에서 probe 테스트로 확인했다. keys/values 쌍 검사가 "둘 다 없음"을 빈 캐시로 취급해 건너뛰었다. 그래서 값이 채워진 양자화 캐시가 빈 캐시와 구분되지 않았고, donation된 스냅샷에는 그 레이어의 상태가 조용히 빠져 있었다. 모드를 먼저 검사하면 이 문제가 닫힌다.

## 기술적 선택과 그 이유

- **dtype용 브리지 헬퍼 하나.** `sdpa_attention_dtype(q, k, v)`는 `sdpa_mask_for_inputs` 옆에 있으며 백엔드의 promotion 표를 읽는다. CUDA에서는 패치된 오버레이 표이므로, Rust 게이트가 커널의 실제 동작과 어긋날 수 없다. cuDNN 적합성과 score 행 바이트 계산 모두 이 헬퍼를 쓰고, `cuda_sdpa_materializes_scores`는 이제 `k`를 받는다.
- **게이트 판단을 순수 함수로 분리.** 예산은 `OnceLock`으로 한 번만 읽으므로 실제 `attention` 진입점을 다른 예산으로 테스트할 수 없다. 판단을 `sdpa_query_chunk_within_budget(.., budget)`으로 분리해 테스트가 예산을 직접 넘긴다. `lib.rs`의 마스크 없는 causal 경로도 같은 게이트를 쓴다.
- **Softcap도 fast SDPA와 같은 헬퍼 사용.** 마스크는 `sdpa_mask_for_inputs`를 거치고 출력은 attention dtype으로 캐스팅한다. grouped decode 분기도 포함한다.
- **쌍 검사보다 모드 검사를 먼저.** 값이 채워진 non-FP16 캐시는 모드로 먼저 거부한다. 빈 양자화 레이어(KV 공유)는 계속 건너뛰므로 Gemma 4 donation 경로는 그대로 동작한다.
- **오버레이: 주석만 고치고 표는 테스트로 고정.** 틀린 것은 헤더뿐이었다. CUDA 빌드에서 `result_type(bf16, fp16) == float32`를 테스트로 고정한다.

### CUDA 오버레이의 순서 의존적 3항 fold

새 promotion 표 테스트가 사실 하나를 더 드러냈고, `dtype.cpp`에 주석으로 남겼다. 패치된 fold는 CUDA에서 순서에 의존한다. `result_type(bf16, fp16)`은 float32이지만 `result_type(fp16, bf16, bf16)`은 bf16이다. 첫 번째 쌍 단계가 만든 타입을 다음 단계가 그대로 유지하기 때문이다. 게이트와 softcap 경로는 `(q, k, v)` 순서로 호출하고 k, v는 TurboQuant(항상 fp16)에서 오므로 순서는 bf16, fp16, fp16이며 결과는 float32이다. TurboQuant는 bf16이 나오는 fp16 우선 순서를 만들지 않는다. 이후 호출자가 인자 순서를 바꿔도 fold가 대칭이라고 가정하지 않도록 주석에 기록했다.

## 변경 내용

- `src/lib/mlxcel-core/cpp/mlx_cxx_internal.h`, `mlx_cxx_bridge.h`, `mlx_cxx_bridge.cpp`: `sdpa_attention_dtype`. softcap과 softcap GQA는 `sdpa_mask_for_inputs`를 쓰고 attention dtype을 반환한다.
- `src/lib/mlxcel-core/src/layers.rs`: `sdpa_query_chunk_within_budget`, attention dtype 기준 게이트와 행 크기. 테스트 `query_chunk_gate_sizes_mixed_half_kv_from_the_attention_dtype`, `sdpa_attention_dtype_follows_the_backend_promotion_table`.
- `src/lib/mlxcel-core/src/lib.rs`: 마스크 없는 causal 경로가 같은 게이트를 사용.
- `src/lib/mlxcel-core/src/ffi_tests.rs`: `test_compiled_softcap_sdpa_mixed_half_kv_attends_in_float32`.
- `src/lib/mlxcelverse/overlays/cuda/mlx/dtype.cpp`: 헤더 수정과 순서 의존성 주석.
- `src/models/kv_snapshot.rs`: `snapshot_standard`와 `snapshot_rotating`에서 쌍 검사 전에 모드를 검사.
- `src/models/kv_snapshot_tests.rs`: `populated_turbo_caches_are_refused_by_mode`, `empty_turbo_caches_are_skipped_not_refused`.
- `src/models/gemma4.rs`: `gemma4_unified_mask_tests::turbo4_snapshot_donation_names_the_mode` (`Cache::snapshot_into` 경유).
- `docs/turbo-kv-cache.md`, `CHANGELOG.md`.

합계: 파일 12개, +471 / -30.

## 동작 변경

q, k, v의 dtype이 같으면 게이트, softcap 경로, 오버레이는 기존과 같다. CUDA에서 fp16 위의 bf16이면 게이트가 head_dim 128에서 청크 분할을 하고, head_dim 256에서는 score당 4바이트로 크기를 계산한다. 이 조합에서 softcap composite는 v의 dtype 대신 float32를 반환한다. 모든 `kv_snapshot` 사용처는 값이 채워진 non-FP16 캐시를 모드 이름이 담긴 오류로 거부한다: Gemma 3, AFMoE, Gemma 4, Llama 4, Muse Glimmer. Qwen3-Next/3.5, LFM2, Bailing은 이미 모드를 먼저 검사했다. Jamba, Granite MoE Hybrid, Nemotron-H, Plamo2, Falcon-H1은 주입되는 KV 모드가 없다.

## 검증

- 청크 게이트: `layers::tests::query_chunk_gate_sizes_mixed_half_kv_from_the_attention_dtype`는 같은 dtype의 판단도 assert한다. `layers::tests::sdpa_attention_dtype_follows_the_backend_promotion_table`은 표를 고정한다.
- Softcap: `ffi_tests::test_compiled_softcap_sdpa_mixed_half_kv_attends_in_float32`는 출력 dtype을 확인하고 float32 기준값과 비교한다. 마스크가 유한한 bias를 가지므로 마스크 캐스트만 되돌려도 실패한다.
- 스냅샷: `models::kv_snapshot::tests::{populated_turbo_caches_are_refused_by_mode, empty_turbo_caches_are_skipped_not_refused}`와 `gemma4_unified_mask_tests::turbo4_snapshot_donation_names_the_mode`.
- Revert 확인: 수정을 되돌리면 청크 게이트, softcap, 값이 채워진 turbo, Gemma 4 테스트가 실패한다(`None` 대 `Some(16)`, 출력 dtype fp16, `has only one of keys/values`). promotion 표 테스트와 빈 캐시 테스트는 이미 올바른 동작을 보호하므로 main에서도 통과하도록 의도했다.
- GB10 실행(`--features cuda`, `--test-threads=1`, 전부 통과): `-p mlxcel-core`의 `layers::tests`(102), `softcap`(18), `ffi_tests`(112)와 루트의 `models::kv_snapshot`(25), `models::gemma4`(121), `gemma3`, `gemma2`, `afmoe`, `llama4`, `muse_glimmer`. Clippy `-D warnings`(core와 루트 `--lib --tests`), fmt, `scripts/ci` 검사도 깨끗하다.
- 실제 체크포인트. 모두 `MLXCEL_SDPA_DETERMINISTIC=1`, temperature 0. "변경 전"은 main `cadc6366`이며 각 변경 전 arm은 자기 자신과 일치함을 확인했다.
  - `mlxcel-server -m ~/models/mlx/gemma-4-26b-a4b-it-4bit --kv-cache-mode fp16+turbo4`, 1987 토큰 프롬프트, 300 토큰: 출력이 일관되고 응답은 변경 전과 바이트 단위로 동일하다. skip 사유가 `rotating cache has only one of keys/values`에서 `rotating cache mode Turbo4Asym is not supported by model-state snapshots`로 바뀐다.
  - `mlxcel generate`, 200 토큰: `qwen3-0.6b-4bit`, 그리고 기본 KV와 `fp16+turbo4`의 `gemma-2-2b-it-4bit` 모두 변경 전후 바이트 단위로 동일하다. 이 체크포인트는 fp16이므로 두 arm 모두 같은 dtype이다.
  - 혼합 softcap 경로에는 bf16 Gemma 2가 필요한데 mlx-community의 Gemma 2는 모두 fp16이다. `gemma-2-2b-it-4bit`를 로컬에서 bf16으로 변환한 것(커밋하지 않음)은 기본 KV에서 바이트 단위로 동일하다. `fp16+turbo4`에서는 645번째 문자에서 갈라지지만 출력은 일관되며, float32 masked score에서 예상되는 결과이다.

## 미검증 및 후속 조치

- 혼합 softcap 경로는 공개된 bf16 Gemma 2 체크포인트가 없어 로컬 bf16 변환본으로만 검증했다.
- 청크 게이트의 메모리 효과는 긴 프롬프트의 OOM 재현이 아니라 게이트 판단 테스트로 검증했다.
- 3항 promotion fold는 변경하지 않고 문서화만 했다. 다른 순서로 dtype을 넘기는 호출자는 `sdpa_attention_dtype`을 거쳐야 한다.
