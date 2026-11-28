# 기술 보고서: PR #2103 - Paged attention (v1, v2 partial, merge)을 HIP으로 port

**날짜**: 2026-10-05

**상태**: gfx1151 호스트에서 구현 및 측정 완료. head `0bf1c97c`, origin/main `c05d5438` 위, 머지 대기 중.

**언어**: C++ (HIP kernel source, kernel holder, port table, launcher shape 검사), Rust (`PagedDecodeBackend::Rocm`, dispatch selector와 memo key, `gridDim.z` guard, autotune runner label, kernel bench 예제, unit test), CMake와 `build.rs` (header 등록), Markdown (환경 변수, 설치, benchmark 결과), Makefile (주석)

**위험도**: 중간 (ROCm에서 server의 batched decode, MLA split-KV, sparse paged decode의 기본 paged decode 경로가 바뀝니다. Metal과 CUDA에도 host 쪽 shape 거부와 holder refactor가 들어가며, 두 backend는 개발 호스트에서 실행할 수 없었습니다)

## 요약

Issue #2068 (#1814의 일부, epic #1801)은 세 paged-attention kernel, 즉 v1 split-K decode, v2 partial, v2 merge의 HIP port를 요청했습니다. 이 PR 이전에는 세 port table에 `.rocm` 항목이 없었으므로, ROCm에서 server의 batched paged decode, MLA split-KV, sparse paged decode가 모두 gather-then-SDPA로 실행되었고, ROCm gate의 test 36개가 `skipping ... lablup/mlxcel#1814` 줄을 남기고 skip되었습니다.

이 PR은 세 CUDA 본문을 hipRTC로 옮긴 `paged_attention_hip.h`를 추가하고, 세 `.rocm` slot을 채우며, `paged_decode_backend()`가 새 `PagedDecodeBackend::Rocm`을 반환하게 하여 v1 dispatch selector가 ROCm에서 native kernel을 고를 수 있게 합니다. HIP 본문은 CUDA 본문과 똑같이 `<input>_shape`에서 geometry를 읽습니다. 이는 #2100의 overlay 수정이 머지되면서 가능해졌습니다. PR의 첫 commit은 그 수정이 없는 상태를 우회하는 workaround를 담았고, 이후 commit이 이를 제거했으므로 Metal과 CUDA의 kernel 본문과 template argument는 main과 byte 단위로 같습니다. 세 launcher에는 모든 backend에서 동작하는 shape 거부도 추가되었습니다.

gfx1151에서 이전에 skip되던 test 36개가 test 수정 없이 실행되어 통과하며, HIP 본문을 의도적으로 변형하면 그중 29개가 실패합니다. Meta-Llama-3.1-8B-Instruct-4bit로 server paged decode를 측정한 결과, 요청당 decode는 batch 4, ~16K token에서 1.28x (arm당 3회), 단일 sequence ~16K와 ~32K에서 1.40x와 1.77x (arm당 1회) 빨라졌습니다. ~4K는 측정 가능한 변화가 없습니다.

## 1. 문제 정의

### 1.1 빈 slot 세 개

세 kernel은 각각 `KernelPorts` table (`paged_attention_ports()`, `paged_v2_partial_ports()`, `paged_merge_ports()`)을 가지고 있었고, Metal과 CUDA 항목만 있고 `.rocm = nullptr`였습니다. Support predicate가 `has_kernel_port`를 읽으므로 ROCm에서는 false를 답했고, kernel 앞의 production gate (`cache/paged.rs`, `mla/mod.rs`, `paged_v2/sparse.rs`)는 모두 gather fallback으로 내려갔습니다. PR #2059가 이미 `test_support/kernel_ports.rs`에 skip macro를 넣어 두었으므로 test는 실패 대신 눈에 보이게 skip했고, issue의 목표는 table을 채우면 test를 수정 없이 실행되게 하는 것이었습니다.

### 1.2 Predicate를 따르지 않는 gate

`paged_decode_backend()`는 `metal_is_available()`이나 `cuda_is_available()`이 true가 아니면 `Other`를 반환했습니다. ROCm에서 v1 table을 채워도 `Other`로 매핑되어 `select_pooled_paged_dispatch`가 native를 고를 수 없었습니다. Library entry point에서 v1 kernel에 도달하려면 table을 채우는 것만으로는 부족했습니다.

### 1.3 Boolean으로 구분하는 holder 두 개

`PagedV2PartialHolder`와 `PagedMergeHolder`는 `bool use_cuda`를 받았고, false 쪽은 Metal 본문을 compile했습니다. 세 번째 backend에서는 이 flag가 Metal로 읽히므로, ROCm 항목을 넣기 전에 holder가 자기 backend를 명시해야 했습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `paged_attention_hip.h` (신규, 430줄) | `PAGED_ATTENTION_DECODE_HIP_SOURCE`, `PAGED_ATTENTION_V2_PARTIAL_HIP_SOURCE`, `PAGED_ATTENTION_MERGE_HIP_SOURCE`: 철자만 바꾼 CUDA 본문 |
| `paged_attention.cpp` | `MLXCEL_BRIDGE_ROCM_BACKEND` 아래에서 `fast::hip_kernel`을 호출하는 `PagedAttentionKernelHolderHip`, `.rocm` getter, host shape 거부 |
| `paged_attention_v2.cpp` | `make_partial_kernel(GpuKernelBackend)`, `GpuKernelBackend`로 구분하는 `PagedV2PartialHolder`, template getter로 backend마다 holder 하나, `.rocm` getter, host shape 거부 |
| `paged_attention_v2_merge.cpp` | partial과 같은 refactor (`make_merge_kernel`, `PagedMergeHolder`), `.rocm` getter, host shape 거부 |
| `layers.rs` | `paged_decode_backend()`가 반환하는 `PagedDecodeBackend::Rocm`, `select_pooled_paged_dispatch`의 CUDA 규칙, memo tag 3, 신규 또는 확장 unit test 세 개 |
| `cache/paged.rs` | `gridDim.z <= 65535` guard를 `Cuda`처럼 `Rocm`에도 적용, 주석 |
| Autotune op | `paged_decode_splits`와 `paged_decode_v2_chunk`가 ROCm tactic을 `cuda` 대신 `rocm`으로 label |
| `examples/paged_attention_kernel_bench.rs` | Selector label이 production처럼 decode port predicate와 `gpu_backend_kind()`를 읽음 |
| Build | `turbo/CMakeLists.txt`와 `mlxcel-core/build.rs`에 header 등록 |
| 주석과 문서 | `kernel_ports.rs`, `lib.rs`, `mlx_cxx_bridge.h`, `mla/mod.rs`, Makefile 주석, `environment-variables.md`, `installation.md`, 신규 `rocm-paged-attention-gfx1151-2026-10-05.md` |

Branch의 commit은 다섯 개입니다: port (`16b6b25d`), bench label (`586e7e09`), #2100 이후 `_shape` 전환 (`f4ca4b9a`), review 수정 (`0d1afcb3`), 결과 문서 (`0bf1c97c`). origin/main 대비 diff는 22개 파일, 840줄 추가, 109줄 삭제입니다.

## 3. Port

### 3.1 같은 kernel, 다른 철자

각 HIP 본문은 CUDA 본문의 thread mapping, kernel 이름, input, output, grid, template argument를 그대로 유지합니다. Header 주석에 적힌 대로, ROCm `CustomKernel::eval_gpu`는 Metal 방식의 전체 thread grid를 threadgroup tuple로 CUDA와 같은 방식으로 올림 나눗셈하므로, 모든 `blockIdx`와 `threadIdx`는 CUDA 본문에서와 같은 의미입니다. 차이는 다음과 같습니다.

- 32-lane all-reduce는 width를 명시한 `__shfl_xor(v, o, 32)`입니다. HIP의 `__shfl_xor_sync`는 mask를 무시하는 호환 shim일 뿐이기 때문입니다.
- 음의 무한대는 `-__builtin_huge_valf()`입니다. hipRTC는 `fast::hip_kernel`이 앞에 붙이는 header만으로 본문을 compile하는데, `INFINITY`는 `<cmath>`에서 오기 때문입니다.
- KV 읽기는 명시적 `(float)`를 유지합니다. `hip_bfloat16`은 `explicit` operator로만 float로 변환됩니다.

Launch와 각 문자열을 compile하는 `fast::hip_kernel` 호출은 이미 CUDA launch가 있는 `.cpp` 파일에 남습니다. Header는 data만 담습니다. 그래서 `make verify-kernel-dtype-keys`의 pin 범위가 유지되고, `EXPECTED_IN_SCOPE`와 `9 in scope` 개수는 바뀌지 않습니다.

### 3.2 Wave32 guard, 그리고 merge kernel에 guard가 없는 이유

#1814 port 요구 사항은 lane 사이에서 reduction이나 shuffle을 하는 모든 port에 `__AMDGCN_WAVEFRONT_SIZE__`와 `__AMDGCN_WAVEFRONT_SIZE`에 대한 `#error` 검사 두 개를 넣으라고 합니다. Lane 16에서 시작하는 fold는 64-lane CDNA wave에서 lane 절반을 버리고도 유한하지만 틀린 결과를 내기 때문입니다.

- **v1 decode와 v2 partial**은 16에서 시작하는 XOR butterfly로 32 lane에 걸쳐 dot product를 fold합니다. 이 둘에는 guard가 있습니다. #2067에서 확인했듯이 ROCm 10의 AMD clang은 gfx1151, gfx942, gfx90a 어디에서도 두 macro를 정의하지 않으므로 guard는 동작하지 않고, 가정을 강제하기보다 기록하는 역할입니다. Header 주석은 두 번째 근거를 덧붙입니다. 같은 `threadIdx.y`를 공유하는 32 lane은 block의 선형 순서에서 연속이므로, wave64 target에서도 width 32를 명시한 shuffle이 각 butterfly를 자기 행 안에 가둘 것이라는 점입니다. Wave64 장치가 없었으므로 이는 실행하지 않은 추론으로 적혀 있습니다.
- **merge**에는 lane 수준 연산이 없습니다. Output element마다 thread 하나이고, shuffle도 barrier도 없습니다. 어떤 wavefront 크기에서도 맞으므로 guard를 두지 않습니다. 거기에 `#error`를 넣으면 wave64에서 올바른 kernel을 거부할 뿐입니다. Issue의 acceptance checkbox는 "세 table 모두 wave32 guard와 함께"라고 적었습니다. PR은 port 요구 사항 자체의 범위 ("lane 사이에서 reduction이나 shuffle을 하는 모든 port")를 따라 이 문구와 의도적으로 다르게 했고, 체크한 항목 옆에 이유를 적었습니다.

### 3.3 자기 backend를 명시하는 holder

`PagedV2PartialHolder`와 `PagedMergeHolder`는 이제 `mlxcel::GpuKernelBackend`를 받습니다. `make_partial_kernel` / `make_merge_kernel` switch가 Metal, CUDA, HIP 본문 중 하나를 compile하고, template getter `get_partial_kernel<Backend>()`가 backend마다 static holder 하나를 줍니다. Server가 요청별 worker에서 동시에 처음 사용에 도달하고, MLX device 조회가 throw하면 `call_once`가 initializer를 다시 실행하므로 `std::call_once`는 유지됩니다. `None` arm과 ROCm이 아닌 build의 throw는 switch를 exhaustive하게 만들 뿐이며, holder는 자기 backend를 명시한 table 항목을 통해서만 도달하므로 실행되지 않습니다.

## 4. #2100 workaround 제거

CUDA 본문은 head 수, block 크기, merge head 수를 `q_shape`, `k_pool_shape`, `v_in_shape`에서 읽습니다. ROCm에서는 #2100 전까지 이것이 fault를 냈습니다. Vendored `fast::hip_kernel`이 `<input>_shape`를 pointer로 선언했지만 launch는 값으로 넘겼기 때문입니다 (LOCAL_FIXES item 30).

첫 commit (`16b6b25d`)은 이를 우회했습니다. Geometry를 새 template argument `NumQHeads`, `NumKVHeads`, `PoolBlockSize`로 모든 backend에 넘기고, merge head 수를 `gridDim.y`에서 읽고, `static_assert`로 `_shape`, `_strides`, `_ndim`이 HIP 문자열에 들어가지 않게 했습니다. 대가는 Metal과 CUDA launch의 변경이었습니다. 두 본문이 읽지도 않는 template argument가 추가되어 JIT cache key까지 바뀌었고, 두 backend는 호스트에서 실행할 수 없었습니다.

#2100이 머지된 뒤 commit `f4ca4b9a`가 workaround를 제거했습니다 (세 파일, 12줄 추가, 68줄 삭제). HIP 본문은 이제 CUDA 본문과 똑같이 `_shape`를 읽고, Metal과 CUDA launch는 다시 main과 같습니다. 이 보고서를 위해 origin/main `c05d5438`과 비교한 결과, 세 launcher 파일의 Metal과 CUDA source 문자열 여섯 개가 모두 byte 단위로 같고, `.metal` 파일은 바뀌지 않았으며, diff는 어떤 `TemplateArg` 목록도 건드리지 않습니다. 제거로 세 backend가 geometry를 읽는 방법이 하나로 남고, 테스트할 수 없는 backend의 kernel은 그대로 유지됩니다.

## 5. 모든 backend의 host 쪽 shape 거부

세 launcher는 이제 launch를 만들기 전에, 모든 backend에서, 본문이 index할 수 없는 shape를 거부합니다. `std::invalid_argument`를 throw하며, bridge가 이 함수들을 `Result`로 선언하므로 throw는 Rust에 `Err`로 전달됩니다. #2067이 SSM update kernel에 한 방식을 따른 것입니다.

- **v1 decode와 v2 partial**: `q`, `k_pool`, `v_pool`은 rank 4여야 합니다. `q`는 `D >= 1`인 `[B, Hq, 1, D]`, `k_pool`의 block 크기와 head 수는 1 이상, D는 `q`와 같아야 하며, `v_pool`은 axis 1부터 3까지 `k_pool`과 같아야 합니다. Axis 0은 비교하지 않습니다. MiniMax-M3의 sparse launch가 두 pool을 `[rows, 1, 1, D]`로 reshape하고 K에는 index-key side head가 더 있어서 K의 행이 V보다 많기 때문입니다.
- **merge**: `v_in` rank 3, `v_in`의 N과 H를 가진 rank 2 `lse_in`, rank 1 `o_indptr`, 그리고 1부터 1024까지의 head dim입니다. Threadgroup이 `(D, 1, 1)`이므로 D는 launch 가능한 block 폭이어야 합니다.

처음 버전의 검사는 rank와 head dim만 비교했습니다. #2103 review에서 V-pool 검사가 너무 느슨하다는 점이 발견되었습니다. 본문은 K의 block 크기와 head stride로 V를 주소 계산하므로, axis 1부터 3이 다른 V는 범위 밖에서 읽힙니다. Commit `0d1afcb3`이 이를 조이고 `D = 0`과 merge 폭 거부를 추가했습니다.

이 거부는 Metal과 CUDA에서도 실행됩니다. 이전에 launch되어 범위 밖을 읽던 호출은 이제 error를 반환합니다. ROCm의 production caller는 조건에 맞는 shape를 넘기지만 (변경 후 paged, MLA, autotune, layers selector test 550개 통과), Metal과 CUDA 경로는 이 검사와 함께 실행되지 않았습니다.

## 6. `PagedDecodeBackend::Rocm`

- **검출.** `paged_decode_backend()`는 v1 predicate `paged_attention_decode_available()`이 true이고, Metal과 CUDA가 모두 없고, `crate::hardware::gpu_backend_kind()`가 `Rocm`일 때 `Rocm`을 반환합니다. Metal을 먼저, CUDA를 두 번째로 확인하는 순서는 그대로이므로 두 backend의 답은 바뀌지 않습니다. `Other` 문서는 이제 CPU 전용 build와 사용 가능한 GPU가 없는 기계만 다룹니다.
- **Selector.** `select_pooled_paged_dispatch`에서 `Rocm`은 CUDA 규칙 `slab_count <= NATIVE_MAX_SLABS`를 따릅니다. Metal이 gather로 보내는 batch 1과 긴 context 구간을 포함해 single-slab layer라면 모두 native입니다. HIP kernel은 CUDA kernel의 port로 pool을 같은 방식으로 읽고, 더 좁은 구간을 정당화할 ROCm 측정 batch나 context 상한이 없습니다.
- **Memo key.** `PagedDispatchCache::pack_key`는 Metal 0, CUDA 1, Other 2 다음으로 `Rocm`에 tag 3을 줍니다. 2-bit tag 필드는 이제 가득 찼습니다.
- **`gridDim.z` guard.** CUDA launch는 `batch * query_heads`를 `gridDim.z`에 넣고, CUDA는 이를 65535로 제한합니다. 이 한도는 bridge 호출이 반환될 때가 아니라 graph가 evaluate될 때 걸리므로 launcher의 `Result`로는 보고할 수 없고, `cache/paged.rs`가 미리 gather로 내려갑니다. HIP port는 같은 grid를 쓰므로 guard는 이제 `Cuda | Rocm`에 걸립니다. ROCm은 장치가 보고하는 한도에 기대지 않고 CUDA 한도를 그대로 씁니다.
- **Test.** `selector_rocm_backend_follows_the_cuda_rule`은 다섯 개의 (batch, context) 쌍과 네 개의 slab 수에 대해 `Rocm`을 `Cuda`와 shape마다 비교합니다. `paged_decode_backend_names_the_resolved_backend`는 v1 port가 있는 곳에서 답을 `gpu_backend_kind()`와 대조합니다. Memo test는 ROCm 질의가 Metal이나 Other의 cell을 잘못 공유해 답을 받지 않는지 확인하고, pack-key test는 포화된 ROCm key를 포함해 네 key가 서로 다르고 decision bit와 empty sentinel과 겹치지 않는지 확인합니다.

작은 소비자 두 곳도 같은 검출을 따릅니다. Kernel bench 예제는 ROCm에서 production이 더는 내리지 않는 gather 결정을 출력하는 대신 `select=` 열을 `Rocm`으로 표시합니다. 두 paged autotune op은 runner id로 `rocm`을 반환하므로, ROCm tactic이 다른 kernel의 tactic 옆에 `cuda`로 저장되지 않습니다.

## 7. 정확성 근거

### 7.1 Skip되던 test 36개

이 port가 없어서 ROCm에서 skip되던 mlxcel-core test 36개 (`57d8ed29`에서 `skipping ... lablup/mlxcel#1814` 줄 36개)가 이제 gfx1151에서 실행되어 통과하며, 그중 어느 것도 수정하지 않았습니다. Issue 본문은 34개로 셌지만 gate는 36개를 출력했습니다. 범위는 200 step에 걸친 gather 경로 대비 v1 decode와 GQA 및 batch matrix, host reference와 gather 경로 대비 v2 partial과 merge (f32와 f16 pool, 빈 요청, 잘린 window, GQA head mapping, 한 geometry에서 두 pool dtype), cascade, sparse, MLA split-KV입니다.

### 7.2 Mutation 검사

HIP lane fold를 16 대신 8에서 시작하고 HIP merge를 base 2 대신 base e로 계산하면 이 test 중 29개가 실패합니다. v2 launch, cascade, sparse, split-KV, batched decode, 그리고 v1에 대한 `test_fused_paged_decode_native_vs_fallback_matrix`입니다. Head dim 8인 v1 test 두 개는 fold 변경을 보지 못합니다. 8개 dim이 lane 0부터 7에 들어가고, 빠진 fold 단계는 0끼리만 합쳤을 것이기 때문입니다. 즉 test는 launch 실패만이 아니라 fold나 rescale이 틀린 port도 잡아냅니다.

### 7.3 Gate

PR 기준, gfx1151 (Radeon 8060S, ROCm 10)에서 `c05d5438` 위로 rebase한 `f4ca4b9a`: `make verify-rocm` OK, test binary 147개에서 11869 passed, 0 failed, 378 ignored, ROCm smoke OK, `#1814` skip 줄 없음. `0d1afcb3` 이후: paged, MLA, autotune, layers selector 통과 (test 550개), `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings`, `cargo fmt --check`, `dead_doc_pointers` 통과. `make verify-kernel-dtype-keys verify-kernel-port-dispatch`는 9 in scope로 통과하고 `EXPECTED_IN_SCOPE`는 바뀌지 않았습니다.

Orchestrator 검증, origin/main `c05d5438`과 최신 상태인 head `0bf1c97c`: `make verify-rocm`의 모든 단계 통과, 11,869 passed, 0 failed, 378 ignored, smoke OK, log에 `skipping ... #1814` 줄 0개.

## 8. 측정 결과

### 8.1 `bench_decode.sh`가 아니라 server를 측정한 이유

#1814 port 요구 사항은 `scripts/bench_decode.sh`를 명시합니다. 그러나 decode profile은 `bench_decode.sh`가 dense KV cache로 decode하기 때문에 paged attention을 decode의 0%로 잡았고, 이 script는 이 kernel에 도달하지 않습니다. Issue #2068의 acceptance criterion 자체가 server의 paged decode를 구동하는 `scripts/benchmark_paged_decode_production.sh` (issue #899)를 명시하며, 이것을 실행했습니다.

### 8.2 방법

Meta-Llama-3.1-8B-Instruct-4bit, `mlxcel-server --parallel 4 --ctx-size 131072`, 요청당 decode token 128개로 `scripts/bench_serving_concurrency.py`, `f4ca4b9a`에서 측정했습니다. 두 arm은 같은 binary를 씁니다. Before는 `MLXCEL_PAGED_ATTENTION_NATIVE=0` (이 변경 전 모든 ROCm build가 실행하던 gather-then-SDPA), after는 미설정 (fused v2)입니다. 각 case는 arm마다 server를 새로 띄웠고, 한 case의 before와 after arm은 `/sys/class/kfd/kfd/proc`이 비어 있고 compiler가 없는 55초짜리 `scripts/rocm_gpu_guard.sh` window 하나 안에서 연달아 실행했습니다. 외부 GPU process나 compiler를 본 시도는 버리고 다시 실행했습니다. 모든 after arm의 server log에는 `paged decode v2: fused v2 launch`가 찍히고, before arm log에는 없습니다. 이후 commit은 host shape 검사, 주석, autotune label만 바꾸고 kernel 본문이나 dispatch는 바꾸지 않으며, 다시 측정하지 않았습니다.

### 8.3 요청당 decode tok/s

| Case | Arm당 실행 횟수 | Before | After | After / before (중앙값) |
|---|---|---|---|---|
| batch 4, ~1K prompt | 3 | 5.7 / 5.5 / 5.6 | 6.1 / 6.3 / 6.0 | 1.09x |
| batch 4, ~4K prompt | 3 | 10.1 / 10.6 / 10.2 | 10.4 / 10.6 / 10.5 | 1.03x, 실행 간 편차 안 |
| batch 4, ~16K prompt | 3 | 10.9 / 10.9 / 10.9 | 13.9 / 13.9 / 13.9 | 1.28x |
| batch 1, ~16K prompt | 1 | 17.2 | 24.1 | 1.40x |
| batch 1, ~32K prompt | 1 | 11.2 | 19.8 | 1.77x |

Batch 4 case는 arm마다 세 번 실행했고 실행 간 편차는 최대 0.5 tok/s입니다. ~1K와 ~16K의 향상은 이 편차 밖이고, ~4K의 향상은 편차 안이므로 측정 가능한 변화 없음으로 보고합니다. 두 단일 sequence case는 공유 호스트에서 그만큼 긴 guard window를 얻기가 드물어 arm마다 한 번만 실행했으므로, 비율에 편차 추정이 없고 각각 한 쌍에 기댑니다. 방향은 batch 4의 경향 (context가 길수록 향상이 커짐)과 맞지만, 1.40x와 1.77x는 단일 관측으로 읽어야 합니다.

첫 token까지의 시간은 모든 case에서 몇 퍼센트 안으로 변하지 않았습니다. Prefill은 이 kernel을 쓰지 않으므로 예상대로입니다. 예를 들어 batch 4, ~16K에서 before 155.1초, after 151.1초 (중앙값)입니다.

### 8.4 Aggregate 열을 쓰지 않는 이유

Script는 aggregate tok/s, 즉 level의 wall-clock 구간 동안의 전체 completion token도 보고합니다. 긴 context에서 이 값은 반대로 움직입니다 (batch 4, ~16K: before 1.9 / 2.0 / 2.0, after 1.7 / 1.7 / 1.7). 그 지점에서 첫 token까지 약 150초가 걸리는 prefill이 지배하고, 각 요청이 멈추기 전에 생성한 token 수에 따라 달라지는데 그 수가 두 수치 경로 사이에서 다르기 때문입니다. Kernel을 측정하는 값이 아닙니다. 결과 페이지는 이 값을 싣고 이유를 설명하며, PR과 이 보고서는 어떤 주장에도 이 값을 쓰지 않습니다.

## 9. 기술적 선택과 그 이유

- **CUDA 본문을 줄 단위로 옮긴다.** CUDA와 HIP은 grid 의미, `template_args`, runtime compile을 공유합니다. 본문이 같으면 backend가 어긋날 때 볼 곳이 하나입니다.
- **Lane끼리 통신하는 곳에만 guard를 둔다.** v1과 v2 partial에는 guard가 있고 merge에는 없습니다. Checkbox 문구와는 다르지만 port 요구 사항의 범위를 따릅니다.
- **#2100이 들어오자마자 template argument workaround를 제거한다.** `_shape`를 읽으면 backend 사이에 geometry 계약이 하나로 유지되고 Metal과 CUDA가 main과 byte 단위로 같게 남습니다. 둘 다 여기서 실행할 수 없었으므로 중요합니다.
- **Launch를 CUDA launch 옆에 둔다.** dtype-key checker의 pin 범위와 개수가 바뀌지 않습니다.
- **잘못된 shape를 모든 backend에서 host에서 거부한다.** 어느 backend에서든 범위 밖 읽기보다 bridge의 `Err`가 낫습니다. Axis 0 예외로 MiniMax-M3의 sparse launch가 계속 동작합니다.
- **ROCm은 CUDA selector 규칙과 CUDA grid 한도를 따른다.** 같은 kernel, 같은 grid이고, ROCm 고유 상한은 측정된 적이 없으며, 알려진 한도 대신 장치가 보고하는 한도를 믿지 않았습니다.
- **Holder, tag, label에 backend 이름을 쓴다.** Boolean `use_cuda`, `cuda` runner id, `Other` fallback은 세 번째 backend가 생기면 모두 틀리게 읽힙니다.

## 10. 남은 위험과 검증하지 않은 것

- **Metal과 CUDA는 실행하지 않았습니다.** Kernel 본문, template argument, table 항목은 바뀌지 않았지만, holder refactor (같은 `metal_kernel`/`cuda_kernel` 호출, backend마다 holder 하나)와 새 shape 거부의 영향을 받습니다.
- **Wave64 장치가 없습니다.** ROCm 10 clang에서 guard가 동작하지 않으므로 wave64 build는 compile 시점에 막히지 않습니다. Wave64에서 v1과 v2 partial fold의 정확성은 명시한 shuffle 폭과 lane 배치로부터 추론한 것이며 실행하지 않았습니다.
- **긴 context 지점은 단일 실행입니다.** Batch 1의 1.40x와 1.77x는 각각 한 쌍입니다.
- **Guard window가 port 요구 사항보다 짧습니다.** 측정은 55초 idle window를 썼고, #1814 요구 사항은 `bench_decode.sh`에 대해 90초를 명시합니다.
- **Memo tag 필드가 가득 찼습니다.** Backend tag 네 개가 두 bit를 모두 쓰므로, 다섯 번째 backend는 key 배치를 넓혀야 합니다.

## 11. 후속 작업

- **Pool element가 2^32를 넘으면 32-bit index 계산이 넘칩니다.** 세 backend 모두 KV 주소를 `(row * block_size + slot) * stride_kv + kv_head * dim`으로 32-bit unsigned 산술 (Metal 본문은 `uint`, CUDA와 HIP 본문은 `uint32_t`)로 계산합니다. Element가 2^32개를 넘는 pool에서는 이 offset이 wrap되어 잘못된 block을 읽습니다. KV head 8개, 차원 128인 pool이라면 token slot 4,194,304개입니다. 이 PR 이전부터 있던 문제이고 모든 backend가 공유하며, rank와 axis별 일치만 보고 전체 크기는 보지 않는 새 shape 거부로는 잡히지 않습니다. 고치려면 세 본문 모두 64-bit offset으로 바꾸거나 (또는 한도를 넘으면 host에서 거부하거나) 해야 하며, Metal과 CUDA가 바뀌므로 그 호스트가 필요합니다.
- **느린 batched server decode는 scheduling 문제입니다.** 이 server에서 batch 4의 요청당 decode (5.6에서 13.9 tok/s)는 같은 모델의 단일 stream `bench_decode.sh` 속도 35 tok/s보다 훨씬 낮습니다. 요청당 속도에 나머지 세 요청이 prefill하는 동안 기다린 시간이 포함되기 때문입니다. 두 arm에서 같으며, 이 kernel로 해결되는 문제가 아닙니다.

## 12. 학습 포인트

- **Port table을 채우는 것만으로 충분하지 않을 수 있습니다.** `paged_decode_backend()`가 predicate의 backend를 읽지 않고 backend를 나열했기 때문에, 새 variant를 알기 전까지 v1 kernel에 도달할 수 없었습니다.
- **원인이 고쳐지면 workaround는 바로 걷어냅니다.** Template argument 우회는 동작했지만 테스트할 수 없는 두 backend를 바꿨습니다. #2100 이후 제거하면서 main과 byte 단위 동일성이 돌아왔습니다.
- **Guard는 lane끼리 통신하는 곳에만 둡니다.** Element마다 thread 하나인 merge kernel은 wave 크기와 무관하며, guard를 넣으면 올바른 kernel을 거부합니다.
- **Mutation 검사는 suite가 무엇을 볼 수 있는지 보여 줍니다.** Fold mutation은 head dim 8인 v1 test 두 개에서는 드러나지 않았고 matrix test가 잡았습니다. Mutation이 없었다면 작은 D에서의 사각지대를 알 수 없었습니다.
- **수치와 함께 실행 횟수를 적습니다.** 세 번의 중앙값과 단일 실행이 섞인 표는 어느 쪽인지 밝혀야 하고, kernel이 아닌 다른 것이 지배하는 열은 인용하지 말고 설명한 뒤 제쳐 두어야 합니다.

Refs: #2068, #1814, #1801, #2100, #2067, #2059, #2061, #1803, #899, #898, #634.
