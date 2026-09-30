# 기술 보고서: PR #2070 - ROCm SliceUpdate가 아직 참조 중인 source를 donate하지 않도록 수정

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: C++ 및 HIP (ROCm overlay), C++ (cxx bridge), Rust (테스트), Markdown

**위험도**: 중간 (모든 KV cache 쓰기가 지나가는 경로에서 overlay primitive 두 개의 버퍼 재사용 규칙이 바뀝니다. 모델에서 보이는 효과는 정확성 수정이며, decode 처리량은 변화 없음으로 측정되었습니다. Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않습니다)

## 요약

이슈 #2052(에픽 #1801의 일부)는 #2050을 구현하던 중 발견되었습니다. ROCm의 GPU `slice_update`는 source의 데이터 버퍼 소유자가 하나이기만 하면, 호출자가 source 배열을 아직 들고 있어도 결과를 source에 그대로 써 넣었습니다. 이슈는 mlxcel에서 `slice_update` 이후에 갱신 전 배열을 읽는 경로가 있는지, 즉 잠재 버그가 아니라 실제로 발생하는 정확성 버그인지 확인하라고 요구했습니다. 실제 버그였습니다. 이전 overlay로 gfx1151에서 두 모델 경로가 잘못된 출력을 냈습니다.

- DeepSeek-V4 `PoolingCache` chunked prefill: 첫 logit이 CPU stream 대비 상대 오차 약 9e-3만큼 벗어났습니다. 수정 후 차이는 1e-5입니다.
- `RotatingKVCache`의 prompt-cache snapshot (Gemma 3/4, AFMoE, Muse Glimmer): 다음 토큰의 ring 쓰기가 저장된 snapshot 안에 기록되었습니다. 기본값인 in-place KV warmup 쓰기가 이를 가리고 있었고, `MLXCEL_KV_INPLACE_WRITE=0`에서는 기존 테스트가 이전 빌드에서 실패했습니다.

`SliceUpdate::eval_gpu`와 `DynamicSliceUpdate::eval_gpu`는 이제 upstream CUDA 및 Metal과 같이 `copy_gpu`를 호출하며, `copy_gpu`는 `array::is_donatable()`이 성립할 때(배열과 버퍼가 각각 참조 하나)만 donate합니다. 평가 전에 버려진 source는 여전히 donate되며, KV cache가 쓰는 방식(`self.keys = slice_update(self.keys, ...)`)이 바로 이 경우입니다. 이 PR은 HIP graph capture 중 donation을 강제하던 `DynamicSliceUpdate`의 `graph_active()` override도 제거합니다. HIP graph가 꺼진 채로 컴파일되므로 이 override는 한 번도 동작할 수 없었습니다. 두 모델의 decode 처리량은 실행 간 편차를 넘는 변화가 없었습니다.

## 1. 문제 정의

### 배열이 아니라 버퍼를 센 것

MLX `array`는 `array_desc_`에 대한 handle이고, `array_desc_`가 데이터 버퍼의 shared pointer를 가집니다. `slice_update`가 만들어진 뒤에도 배열을 읽을 수 있게 만드는 것은 두 가지입니다. 같은 `array_desc_`에 대한 다른 handle(Rust 변수, cache 필드)과, 그 배열을 입력으로 받는 아직 평가되지 않은 노드(lazy `Slice`나 `Copy`)입니다. 둘 다 데이터 버퍼를 직접 잡고 있지 않습니다. 버퍼를 잡는 것은 배열뿐입니다.

overlay의 `SliceUpdate::eval_gpu`(`mlx/backend/rocm/indexing.hip`)와 `DynamicSliceUpdate::eval_gpu`(`mlx/backend/gpu/primitives.cpp`)는 `in.data_shared_ptr().use_count() == 1`과 contiguity 검사로 donation을 결정했습니다. 위의 두 경우 모두 이 값은 1이므로, 출력이 source의 버퍼를 가져가고 커널은 source가 여전히 가리키는 메모리에 update를 썼습니다. 이후 source를 읽거나 source 위의 lazy 노드를 평가하면 갱신된 값이 보였습니다.

고정된 upstream MLX(81ba1c6a)는 `array::is_donatable()`, 즉 `array_desc_.use_count() == 1 && array_desc_->data.use_count() == 1`을 씁니다. CUDA의 `SliceUpdate::eval_gpu`는 `copy_gpu`만 호출하고, 그 안의 `set_copy_output_data`는 `is_donatable(in, out)`이 성립할 때만 donate합니다. Metal도 같은 경로를 탑니다. 버퍼만 세는 규칙을 가진 backend는 ROCm overlay뿐이었습니다.

### 재현된 두 개의 잘못된 출력 경로

**DeepSeek-V4 `PoolingCache` chunked prefill.** prompt 모드에서 `PoolingCache::accumulate_windows`는 완성된 압축 window를 `buf_kv`의 lazy read로 반환하고, 같은 호출 안에서 새 remainder tail을 `slice_update`로 `buf_kv` 앞부분에 덮어씁니다. 한 chunk가 이전 remainder에 걸친 window를 완성하면서 새 remainder도 남기면, 반환된 window(`r_kv`)는 tail 쓰기가 덮어쓰는 바로 그 행에서 이전 remainder를 읽어야 합니다. 모델의 cache barrier(`eval_state`)는 window를 읽기 전에 버퍼를 먼저 평가합니다. 이전 overlay에서는 tail 쓰기가 `buf_kv`의 버퍼를 donate했기 때문에, `r_kv`가 평가될 때는 새 tail을 읽었습니다. `tiny_model_chunked_prefill_with_pool_remainder_matches_cpu`(압축 비율 4에서 prompt를 14 + 4로 분할)는 첫 logit에서 CPU stream과 상대 오차 약 9e-3만큼 달랐고, 수정 후에는 1e-5 이내로 일치합니다.

**`RotatingKVCache` prompt-cache snapshot.** `ModelStateSnapshot::push_tensor`는 `ModelStateTensor::new`로 cache 텐서를 담는데, 이는 live keys 배열에 대한 lazy `ffi::copy`입니다. copy 노드는 keys 배열을 잡을 뿐 버퍼는 잡지 않습니다. 한 토큰짜리 suffix가 ring을 한 바퀴 돌면 `RotatingKVCache`는 `slice_update`로 slot 하나를 덮어씁니다. 이전 overlay에서는 이 쓰기가 keys 버퍼를 donate했고, snapshot의 copy는 그 뒤에 평가되어 저장했던 토큰 대신 새 토큰을 담았습니다. prompt caching과 함께 `RotatingKVCache`를 쓰는 Gemma 3, Gemma 4, AFMoE, Muse Glimmer가 영향을 받습니다. 기존 테스트 `rotating_inplace_warmup_and_wrap_keep_shared_snapshots_intact`가 이전 빌드에서 통과한 이유는, 기본으로 켜진 in-place warmup 쓰기(`MLXCEL_KV_INPLACE_WRITE`)가 `SliceUpdate`가 아니라 `inplace_slice_write`를 거치기 때문이었습니다. 이 변수를 0으로 두면 테스트가 실패했습니다. PR은 steady-state wrap을 직접 재현하는 `rotating_steady_state_snapshot_survives_the_next_wrap_write`를 추가합니다.

### 확인했고 영향이 없는 경로

PR 작성자는 update 중에 source를 계속 들고 있을 수 있는 mlxcel의 모든 `slice_update` 호출자를 추적했습니다.

- 모든 모드의 `KVCache` append와 trim.
- `RotatingKVCache`의 speculative 및 MTP 경로.
- `RingSlidingKVCache`.
- Inkling과 Gemma 4의 MTP rollback.
- cache detach와 adopt.
- paged pool 쓰기와 snapshot restore.
- VLM embedding merge와 RoPE merge.

이 경로들에서 source는 평가 전에 버려지거나(재할당 패턴) 별도 버퍼로 materialize된 뒤에만 읽힙니다. 단일 토큰 FP16 decode 쓰기는 별도 primitive인 `inplace_slice_write`를 쓰며 바뀌지 않았습니다.

해가 없던 donation 하나는 사라집니다. `llama3.rs`의 paged per-sequence fallback(batched 경로가 거절할 때 사용)에서는 이전 sequence의 lazy gather가 공유 slab을 아직 잡고 있으므로, 첫 sequence 이후의 각 sequence는 이제 slab에 쓰는 대신 복사합니다. 기존 쓰기가 올바랐던 것은 gather가 필요한 행을 이미 읽었기 때문인데, 새 규칙은 그 사실을 알 수 없으므로 복사합니다.

## 2. 변경 요약

- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/indexing.hip`** (`SliceUpdate::eval_gpu`): 직접 작성한 `can_donate` 분기와 `out.copy_shared_buffer(in)`을 `copy_gpu(in, out, ctype, stream())` 한 줄로 바꿉니다. copy 종류(Scalar, Vector, General)는 이전과 같은 방식으로 고릅니다. 주석은 `LOCAL_FIXES.md` 항목 24를 가리킵니다.
- **`src/lib/mlx-cpp/patches-rocm/mlx/backend/gpu/primitives.cpp`** (`DynamicSliceUpdate::eval_gpu`): 같은 교체와 함께 `rocm::graph_active()` forward declaration과 `|| graph_active()` 조건을 제거합니다(3절).
- **`src/lib/mlxcel-core/cpp/mlx_cxx_bridge.{h,cpp}`**, **`src/lib/mlxcel-core/src/lib.rs`**: `DynamicSliceUpdate` primitive를 만드는 테스트 전용 bridge `slice_update_dynamic(src, update, start, axes)`. 이 primitive를 만드는 모델 경로가 없으므로, 테스트가 그 `eval_gpu`에 닿는 방법은 이것뿐입니다. bridge는 범위를 벗어난 axis나 길이가 `axes`와 다른 `start`를 거부하고, 각 start offset을 `[0, src_dim - update_dim]`으로 clamp합니다. GPU 커널은 이 offset에 bounds check 없이 쓰는데 이 진입점은 safe Rust이기 때문입니다.
- **`tests/rocm_slice_update_source.rs`** (신규, 272줄, `#![cfg(feature = "rocm")]`): 평가된 source를 들고 있는 상태에서 `slice_update`(None), `slice_update_reduce`(Sum, Max), `slice_update_dynamic`으로 GPU update를 한 번 실행하고, 출력과 source가 바뀌지 않았음을 확인합니다. 나머지 두 케이스는 dynamic start clamp와, 버려진 source로도 올바른 출력이 나오는지를 확인합니다. 각 테스트는 본문 동안 `lock_default_device`를 잡습니다.
- **`tests/rocm_slice_update_reduce.rs`**: #2050이 우회책으로 넣었던 private source copy를 제거합니다. 이제 GPU op가 먼저 `src`에서 실행되고 CPU reference가 같은 `src`를 읽으므로, 이 테스트도 이번 수정을 지키는 역할을 합니다. 가장 큰 케이스의 최대 메모리는 약 1.3 GB에서 1.1 GB로 줄었습니다.
- **`src/models/deepseek_v4_tests.rs`**: `pooling_cache_remainder_survives_an_overlapping_tail_write`(`PoolingCache` 순서만 따로 재현)와 `tiny_model_chunked_prefill_with_pool_remainder_matches_cpu`(모델 수준, CUDA의 기본 TF32 matmul을 고려한 3e-3 허용 오차).
- **`src/lib/mlxcel-core/src/cache.rs`**: `rotating_steady_state_snapshot_survives_the_next_wrap_write`.
- **`src/lib/mlxcel-core/src/generate.rs`**: `ModelStateTensor::new`의 doc comment가 이제 capture는 평가되면 버퍼를 공유하는 lazy copy이며, capture가 배열을 참조하는 동안 이후의 `slice_update`가 복사하기 때문에 올바르게 유지된다고 설명합니다.
- **`src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`**: 항목 24. 메커니즘, 재현된 두 버그, `graph_active()` 판단 근거, 테스트, 처리량 수치를 담고 #1813의 upstreaming 후보로 표시했습니다.

커밋: `338be1cc`는 수정과 테스트, `24e22975`는 dynamic start clamp 추가와 새 테스트 보강, `198f7a0f`는 항목 24에 처리량을 기록합니다.

## 3. 기술적 선택과 그 이유

### 조건을 고치는 대신 `copy_gpu` 호출

이슈는 두 곳에서 버퍼 카운트를 `in.is_donatable()`로 바꾸자고 제안했습니다. PR은 한 걸음 더 나아가 직접 작성한 분기를 지우고, upstream CUDA의 `SliceUpdate::eval_gpu`와 upstream `DynamicSliceUpdate::eval_gpu`처럼 `copy_gpu`를 호출합니다. `copy_gpu`는 이미 `is_donatable(in, out)`을 적용하며, 두 참조 카운트뿐 아니라 dtype과 크기 호환성도 검사합니다. 규칙의 사본을 overlay에 남기면 upstream과 맞춰야 할 곳이 두 군데가 됩니다. 사본을 지우면 이 부분에서 overlay의 slice update가 upstream과 같아지고, fork가 들고 다니는 diff도 줄어듭니다.

### `graph_active()` override 제거

`DynamicSliceUpdate`는 `rocm::graph_active()`가 참이면 항상 donate했습니다. 주석에 이유가 있었습니다. HIP graph capture 중에는 비동기 파이프라인이 버퍼의 use count를 올리므로 일반 규칙이면 새 버퍼로 복사하게 되고, capture된 graph를 replay할 때마다 고정된 capture 입력에서 cache를 다시 만들어 누적이 사라진다는 것입니다. PR은 두 가지 이유로 이를 제거합니다.

첫째, 동작할 수 없었습니다. `graph_active()`는 `use_hip_graphs()`가 참일 때 `CommandEncoder` 생성자에서만 설정되고, `mlx/backend/rocm/device.cpp`의 `use_hip_graphs()`는 상수 `false`를 반환하며, `MLX_GRAPH_PREFILL_REPLAY`는 `use_hip_graphs()` 뒤에서만 읽힙니다. override는 죽은 코드였습니다.

둘째, 동작했다면 본 버그와 같은 이유로, 더 심하게 틀렸을 것입니다. 다른 배열이 공유하는 버퍼까지 donate했기 때문입니다. HIP graph를 다시 켠다면 capture 중의 누적은 다른 코드가 아직 읽을 수 있는 source에 쓰는 방식이 아니라 capture 쪽에서 해결해야 합니다. 항목 24에 이를 기록해 두었으므로, 다음에 HIP graph를 켜는 사람이 override를 되살리지 않을 수 있습니다.

### 버려진 source의 donation은 유지

이 수정은 donation을 끄지 않습니다. KV cache는 `self.keys = slice_update(self.keys, ...)` 재할당 패턴으로 씁니다. 이전 handle이 교체되면 배열의 참조는 하나가 되고 `is_donatable`이 성립하므로, update는 여전히 버퍼를 재사용합니다. donation이 존재하는 이유가 이 경우이고, 처리량이 변하지 않은 이유도 이것입니다. `dropped_source_update_is_correct`가 이를 고정합니다.

### 테스트 전용 bridge에서 clamp

`slice_update_dynamic`은 start offset을 배열 데이터로 받습니다. MLX는 이를 bounds check하지 않고, GPU 커널은 받은 offset에 그대로 씁니다. 테스트 전용 함수라도 safe Rust에서 호출할 수 있으므로, bridge는 각 offset을 update가 들어가는 위치로 clamp하고, `dynamic_slice_update_clamps_an_out_of_range_start`는 범위를 벗어난 offset이 마지막 slot에 쓰는지 확인합니다. 예외 대신 clamp를 택해 계산된 offset으로도 bridge를 쓸 수 있게 하면서, 나중에 모델 경로가 이를 쓰기 시작해도 `src` 밖에 쓸 수 없게 했습니다.

### 비용이 없다고 말하기 전에 처리량 측정

이슈는 이 지름길이 KV append 속도 때문에 들어갔을 것으로 보았습니다. PR은 gfx1151에서 `scripts/bench_decode.sh`(pp512, tg128)로 decode 처리량을 측정했습니다. 수정 전과 후의 실행을 번갈아 돌렸고, 각 실행은 다른 GPU 프로세스나 컴파일러가 없는 상태에서 90초 대기 후 시작했으며, GPU 상태를 1초마다 기록했습니다.

| 모델 | 수정 전 (tok/s) | 수정 후 (tok/s) |
|---|---|---|
| Qwen3-0.6B-4bit | 279.0, 279.8 | 279.5, 279.9 |
| Meta-Llama-3.1-8B-Instruct-4bit | 35.68, 35.67, 35.82 | 35.90, 35.69 |

실행 간 편차(1% 미만)를 넘는 변화는 없습니다. 8B 모델의 수정 후 실행 하나(35.40)는 컴파일러와 겹쳐서 제외했습니다. 결과는 분석과 일치합니다. decode 쓰기는 `inplace_slice_write`를 거치고 prefill 쓰기는 재할당 패턴을 쓰므로, 두 경로 모두 donation을 잃지 않았습니다.

## 4. 검증

PR 작성자 (gfx1151):

- `tests/rocm_slice_update_source.rs`: source를 들고 있는 네 케이스가 이전 빌드에서 실패하고, 수정 후 여섯 개 모두 통과합니다.
- `pooling_cache_remainder_survives_an_overlapping_tail_write`, `tiny_model_chunked_prefill_with_pool_remainder_matches_cpu`, `rotating_steady_state_snapshot_survives_the_next_wrap_write`는 이전 빌드에서 실패하고 수정 후 통과합니다. `rotating_inplace_warmup_and_wrap_keep_shared_snapshots_intact`는 `MLXCEL_KV_INPLACE_WRITE=0`일 때 이전 빌드에서 실패합니다.
- private source copy를 뺀 `tests/rocm_slice_update_reduce.rs`: 8/8.
- mlxcel-core `cache` 테스트: 588개 통과, 2개 실패(dev profile의 `paged_detach` `debug_assert`. 이전 빌드에서도 같고 `test-fast` gate에서는 컴파일되지 않음). mlxcel의 `deepseek_v4`, `kv_snapshot`, `prompt_cache`, `gemma4` 테스트 통과.
- Qwen3-0.6B-4bit에서 `scripts/ci/rocm_smoke.sh`, 두 crate의 `-D warnings` clippy, fmt, fast script gate 통과.

오케스트레이터 검증 (gfx1151, origin/main `0d17303d` 위로 rebase한 브랜치):

- rebase에서 `LOCAL_FIXES.md` 충돌이 두 번 났고, 두 번 모두 항목 24(이 PR)를 항목 25(#2051의 것, PR #2073으로 머지됨) 앞에 두는 방식으로 해결했습니다.
- `make verify-rocm`은 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke 실행(32 토큰)이 통과했습니다.
- `verify-test-rocm`은 정확히 세 target에서 실패했고, 모두 이 PR과 무관한 알려진 baseline 실패입니다. `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, 그리고 #2037의 `gelu_approx_matches_mlx_nn_bit_for_bit`와 `tests::family_order_is_exhaustive`입니다.
- `tests/rocm_slice_update_source.rs` 6/6 통과. `tests/rocm_slice_update_reduce.rs`는 private source copy 없이 8/8 통과. mlxcel-core lib 테스트 1832개 통과.
- #2072가 추적하는 간헐적 `rocm_mxfp4_quant` 테스트는 이번 실행에서는 우연히 통과했습니다.

## 5. 학습 포인트

- **버퍼의 참조 카운트는 배열의 참조 카운트가 아닙니다.** MLX에서 배열을 읽는 lazy 노드는 배열을 잡지 데이터 버퍼를 잡지 않습니다. 버퍼만 세는 donation 검사는 "이 메모리를 잡은 곳이 없다"를 "이 배열을 다시 읽을 곳이 없다"로 취급하는데, graph에 아직 평가되지 않은 reader가 있으면 둘은 다릅니다. `array::is_donatable()`은 둘 다 검사하며, backend는 이를 재구현하지 말고 `copy_gpu`를 통해 써야 합니다.
- **lazy snapshot은 donation 규칙에 의존합니다.** `ModelStateTensor::new`는 즉시 복사하지 않고, MLX가 아직 참조되는 배열에 절대 쓰지 않는다는 점에 기댑니다. 이 규칙을 깨는 backend는 snapshot, prefix cache 등 "지금 저장하고 나중에 읽는" 모든 구조를 조용히 망가뜨립니다.
- **빠른 기본 경로가 느린 경로의 버그를 가릴 수 있습니다.** rotating cache snapshot 버그는 모든 ROCm 빌드에 있었지만, in-place 쓰기는 다른 primitive이므로 `MLXCEL_KV_INPLACE_WRITE` 기본값에서는 보이지 않았습니다. 빠른 경로를 끄고 기존 테스트를 돌린 것이 이전 빌드가 틀렸음을 보여 주었습니다. cache 의미론 테스트는 두 쓰기 경로를 모두 다뤄야 합니다.
- **테스트 안의 우회책은 버그 리포트입니다.** #2050의 테스트는 안정적인 reference를 얻으려고 GPU op에 source의 private copy를 주었습니다. 그 copy가 이 결함의 첫 증거였고, 이를 제거하면서 reduce 테스트가 이번 수정의 두 번째 guard가 되었습니다.
- **guard를 유지하기 전에 동작할 수 있는지 확인하세요.** `graph_active()` override에는 그럴듯한 주석이 있었지만, 플래그를 따라가 보니 상수 `false`였습니다. 정확성에 민감한 primitive 안의 죽은 코드는, 그 코드가 겨냥했던 기능이 다시 켜질 때 잘못된 규칙을 달고 되살아나기 쉽습니다.

## 6. 주의 사항과 검증하지 않은 부분

- **Metal과 CUDA**는 이 호스트에 없어 실행하지 않았습니다. overlay 변경은 ROCm 빌드에만 영향을 줍니다. 새 DeepSeek-V4 테스트와 rotating snapshot 테스트는 ROCm 전용이 아니어서 모든 GPU backend에서 실행되며, 해당 backend는 이미 `is_donatable`을 쓰므로 통과할 것으로 예상합니다.
- **gfx1151만** 실행했고, 처리량은 두 모델에서만 측정했습니다. prefill에서 재할당이 아닌 방식으로 `slice_update`를 쓰는 모델은 이전에 받던 donation을 잃을 수 있습니다. 점검에서 그런 경우는 찾지 못했지만 benchmark가 모든 모델 계열을 다루지는 않습니다.
- **paged per-sequence fallback은 이제 복사합니다.** `llama3.rs` fallback 경로에서 첫 sequence 이후의 각 sequence가 공유 slab을 복사합니다. 일반 경로는 batched 경로이므로 따로 benchmark하지 않았습니다.
- **점검은 코드 읽기입니다.** 영향 없는 경로 목록은 호출자 추적에서 나왔고, 기존 cache 및 모델 테스트 통과로 뒷받침되지만 경로마다 테스트가 있는 것은 아닙니다.
- **HIP graph는 여전히 꺼져 있습니다.** `use_hip_graphs()`를 켜면 capture 중 KV 누적에 대한 별도 설계가 필요하며, 제거한 override를 그대로 되살려서는 안 됩니다.

## 7. 남은 작업

- #1813: 다른 upstreaming 후보와 함께 항목 24를 fork에 제안.
- #2072: 이 PR과 무관한 간헐적 `rocm_mxfp4_quant` 실패. 이번 실행에서는 우연히 통과했습니다.
- `verify-test-rocm`의 baseline 실패 세 개(`prefill_dense_gemm_matches_qmm_bytes_where_eligible`와 #2037의 두 개)는 이 PR 밖에서 추적합니다.

참고: #2052 (이 PR로 닫힘), #1801, #2050, #1813, #2037, #2051, #2072, PR #2073.
