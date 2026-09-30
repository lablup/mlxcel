# 기술 보고서: PR #2084 - ROCm에서 in-flight batch 메모리 제한과 cache limit 적용

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, `3c9edea0` 위로 rebase, 머지 대기 중.

**언어**: C++ (ROCm overlay: allocator, command encoder, eval), Rust (런타임 기본값, bench 바이너리, 테스트), Bash (bench 하네스), Markdown

**위험도**: 중간 (ROCm eager 경로가 이제 인코딩 중에 GPU event를 기다리며 host를 멈추므로 모든 ROCm 평가에 영향이 있습니다. Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않고, 공유 Rust 코드의 유일한 변경은 `rocm` 밖에서 기존 기본값을 유지합니다)

## 요약

이슈 #2062(#1814에서 분리, epic #1801)는 UMA 호스트에서 ROCm 백엔드의 메모리가 어디로 가는지 밝히고 제한된 기본값을 넣으라고 요구했습니다. gfx1151에서 pp512/tg128로 실행하면 `Meta-Llama-3.1-8B-Instruct-4bit`은 weight 4.75 GB에 대해 live MLX buffer peak이 20.60 GB였고, `Qwen3-30B-A3B-4bit`은 17.17 GB에 대해 23.56 GB였습니다. 이슈는 buffer cache를 원인으로 가정했지만 측정 결과는 달랐습니다. cache는 모든 phase 경계에서 0.4에서 1.7 GB였고, peak은 측정 pass의 prefill 안에서 살아 있던 임시 buffer였습니다. ROCm eager 경로는 command batch를 2000 op마다 한 번만 commit했고, 그 commit을 MLX scheduler에 등록하지 않았기 때문에 host가 GPU보다 훨씬 앞서 인코딩했고 모든 batch의 임시 buffer가 동시에 살아 있었습니다. 이와 별개로 allocator의 `set_cache_limit`은 인자를 저장만 하고 아무도 읽지 않았습니다.

PR은 메커니즘마다 하나씩, 두 가지 제한을 추가합니다.

- **In-flight 제한** (overlay, `LOCAL_FIXES.md` 항목 28): `gpu::eval`이 primitive마다 할당한 바이트를 세고, eager 경로는 `MLX_ROCM_MAX_INFLIGHT_MB`(기본값 1024 MiB)의 4분의 1에 도달하면 batch를 commit하며, in-flight 총량이 예산을 넘는 동안 host는 가장 오래된 committed batch를 기다립니다. `0`은 이전 동작을 복원합니다.
- **Cache 제한**: `malloc_async`가 cache miss 시 cache가 limit을 넘었으면 limit의 4분의 3까지 비우고, mlxcel은 `rocm` 빌드에서 `MLXCEL_CACHE_LIMIT` 기본값을 2 GiB로 둡니다. 사전 로드 추정이 읽는 `memory_limit()`은 바뀌지 않습니다.

`scripts/bench_decode.sh`로 측정하면(각 3회, 중앙값, GPU 유휴 상태) peak은 20.60에서 6.14 GB로, 23.56에서 18.58 GB로 내려갑니다. Decode는 +1.5%와 -0.2%, 8B의 prefill은 -1.8% 변합니다. 조사 과정에서 실제로는 아무것도 잡고 있지 않은 allocator 기능 네 가지(decode arena, graph deferral, async pool, GTT/managed fallback)를 찾았고, 제한의 첫 버전에서 프로세스 종료 시 heap corruption이 발생했는데 최종 버전은 raw HIP event를 써서 이를 피합니다.

## 1. 문제 정의

#1814의 ROCm spike는 weight가 약 4.5 GB인 8B 모델에서 약 20 GB peak을 측정했습니다. UMA 호스트(Radeon 8060S, 96 GiB VRAM carve-out, host에 보이는 메모리 31 GiB)에서 이 메모리는 OS와 다른 모든 GPU 사용자가 쓰는 같은 풀에서 나옵니다. #1814의 분리(2026-09-30)에서 이 항목이 #2062가 되었고, 나머지 여덟 하위 이슈와 의존성이 없습니다.

이슈의 계획은 먼저 peak을 설명하고, 모든 allocator knob을 비교한 뒤, 기존 `resolve_cache_limit()` 경로로 기본 cache limit을 넣는 것이었습니다. 첫 단계가 전제를 뒤집을 경우도 적어 두었습니다: "If step 1 shows the excess is not buffer cache (for example the arena or pool slack), fix it in the overlay instead and add a `LOCAL_FIXES.md` item, rather than capping a counter that does not hold the memory." 첫 단계가 실제로 전제를 뒤집었고, PR은 그 분기를 따르면서도 acceptance criteria가 요구하는 cache 기본값도 함께 넣습니다.

## 2. 진단

### 2.1 Peak은 cache가 아니라 live buffer였습니다

`mlxcel-bench-decode`는 이제 로드 후, warmup pass 후, 측정 pass 후에 `active`, `cache`, phase별 peak을 출력합니다. origin/main 기준:

| 모델 | Weight (warmup 후 `active`) | Warmup 후 cache | Warmup pass peak | 측정 pass peak | 측정 pass 후 cache | VRAM 증가량 |
|---|---:|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | 4.75 GB | 1.64 GB | 6.19 GB | 20.60 GB | 0.42 GB | 21.19 GB |
| Qwen3-30B-A3B-4bit | 17.17 GB | 0.62 GB | 17.56 GB | 23.56 GB | 0.50 GB | 24.09 GB |

MLX의 peak은 `active`만 세고 cache는 세지 않으며, 장치 전체의 `mem_info_vram_used` 증가량이 그 값과 0.6 GB 이내이므로 메모리를 잡고 있던 것은 live array입니다. 20 토큰 측정 pass와 128 토큰 측정 pass가 모두 20.60 GB에 도달하므로 peak은 decode가 아니라 측정 pass의 prefill에서 생깁니다. Warmup pass의 peak이 낮은 이유는 그 prefill이 lazy weight 로드와 겹치고, cross-stream wait 때문에 작은 batch 여러 개로 쪼개지기 때문입니다.

### 2.2 임시 buffer가 한꺼번에 살아 있던 이유

`CommandEncoder::add_temporary`는 op의 모든 입력과 scratch buffer를 그 batch의 completion handler가 실행될 때까지 살려 둡니다. Eager 경로(`use_hip_graphs()`가 `false`를 반환)는 `MLX_MAX_OPS_PER_BUFFER` op(2000)마다 batch를 commit했고, 그 commit을 MLX scheduler에 등록하지 않았습니다. 그 결과는 두 가지입니다.

- Scheduler의 outstanding task 10개 상한이 적용되지 않아 host가 GPU를 한 번도 기다리지 않았습니다.
- `set_memory_limit`도 같은 task 집계를 통해서만 작동하므로 `MLXCEL_MEMORY_LIMIT` 역시 host를 기다리게 하지 못했습니다.

Host는 GPU가 prefill을 실행하는 것보다 훨씬 빨리 인코딩하므로 여러 batch의 임시 buffer가 함께 살아 있었습니다. `MLX_MAX_OPS_PER_BUFFER=50`도 소용이 없었습니다. 기다림 없이 commit만 늘리면 host는 여전히 그만큼 앞서 있습니다.

### 2.3 임시 buffer의 정체

f16 체크포인트인 8B에서 초과분(weight 대비 15.85 GB)의 대부분은 `QuantizedMatmul`의 dequantize-and-GEMM 경로가 512행 activation에 대해 weight 행렬마다 할당하는 f16 복사본이며, MLP 행렬 하나당 117 MB입니다. 이 경로를 끄면(`MLX_ROCM_QMM_DEQUANT_GEMM=0`) peak은 6.78 GB였지만 prefill 속도는 20분의 1이었습니다(1069.34 대비 51.75 tok/s). MoE 모델은 bf16이고, bf16 affine 4-bit matmul은 weight 복사본을 할당하지 않는 fork의 fused WMMA 커널을 타므로, 같은 변수를 바꿔도 peak은 23.56 GB 그대로였습니다. 이 모델의 초과분은 같은 방식으로 붙잡혀 있던 다른 op의 출력이며, in-flight 제한이 이것도 대부분 없앱니다.

### 2.4 기존 knob으로는 해결되지 않았습니다

이슈가 나열한 모든 knob을 변경 전에 측정했습니다(각 1회, 전체 표는 `docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`). `MLXCEL_CACHE_LIMIT=1GB`, `MLXCEL_MEMORY_LIMIT`, `MLX_ROCM_NO_ASYNC_POOL`, 두 managed-fallback 스위치, `MLX_ROCM_FINEGRAINED=0`, `MLX_GRAPH_NODEFER=1`, `MLX_MAX_OPS_PER_BUFFER=50`, `MLXCEL_CACHE_CLEAR_INTERVAL=0`(8B만)은 모두 peak을 20.60 GB와 23.56 GB로 남겼습니다. `MLX_ROCM_QMM_DEQUANT_GEMM=0`만 위의 비용을 치르고 8B를 움직였고, `MLX_ROCM_USE_ASYNC_POOL=1`은 두 모델 모두를 죽였습니다. 주기적 cache clear는 이 실행에서 의미가 없습니다. 주기가 256 토큰인데 실행은 128 토큰만 decode합니다.

## 3. 발견된 죽은 기능

이슈는 초과분의 후보로 allocator 메커니즘 여러 개를 나열했습니다. 이번 실행에서 그중 어느 것도 메모리를 잡고 있지 않았습니다.

- **Decode arena** (`decode_arena_begin`): `decode_arena_begin`과 `decode_capture_begin`은 mlxcel에도 overlay의 나머지 부분에도 호출하는 곳이 없고, `use_hip_graphs()`는 `false`를 반환합니다. Arena는 할당되지 않습니다(용량과 high-water mark 모두 0).
- **Graph deferral** (`MLX_GRAPH_NODEFER`): 같은 이유로 graph deferral이 실행되지 않으므로 스위치가 아무것도 바꾸지 않습니다.
- **Async pool** (`MLX_ROCM_USE_ASYNC_POOL`, `MLX_ROCM_FORCE_ASYNC_POOL`; `MLX_ROCM_NO_ASYNC_POOL`은 강제로 끔): 기본값은 꺼짐입니다. 켜면 두 모델 모두 warmup pass 중 `gather_rows_kernel`에서 GPU memory fault가 나서 중단되었고, fork 자체의 주석이 경고하는 바로 그 실패입니다.
- **Managed/GTT fallback** (`MLX_ROCM_ALLOW_MANAGED_FALLBACK`, `MLX_ROCM_NO_MANAGED_FALLBACK`): 장치 할당이 실패할 때만 쓰이는데, 그런 일은 없었습니다. 어느 실행에서도 `mem_info_gtt_used`가 0.01 GB 넘게 오르지 않았습니다.
- **Cache limit 자체**: `RocmAllocator::set_cache_limit`은 `max_pool_size_`를 저장했지만 아무도 읽지 않았고, exact-size cache(`min_utilization` 1.0)는 `clear_cache()` 전까지 본 적 있는 모든 buffer 크기를 유지합니다. Limit의 기본값은 allocator의 memory limit, 이 호스트에서 76.8 GiB였습니다. `MLXCEL_CACHE_LIMIT`은 ROCm에서 한 번도 효과가 없었습니다.

이들은 제거하지 않고 기록만 했습니다. Fork의 코드이고, HIP graph를 다시 켜면 arena와 deferral은 다시 살아납니다.

## 4. 변경 요약

`update/issue-2062-rocm-cache-limit`의 커밋 두 개이며 `3c9edea0` 위로 rebase되어 있습니다.

- **`b72b03a6`** `fix(rocm): bound in-flight batch memory and enforce the cache limit`: overlay 변경, ROCm cache 기본값, bench 카운터, 하네스 로그 옵션, 결과 페이지와 데이터, 테스트, 문서.
- **`5f665765`** `fix(rocm): count each eval's last batch and trim the cache with slack`: 리뷰의 MEDIUM 지적 두 건. `gpu::finalize`가 이제 `commit_and_throttle`을 거치므로 모든 eval과 `async_eval`의 마지막 batch도 예산에 들어가고, miss 경로의 trim은 limit이 아니라 limit의 4분의 3까지 내려가며, synchronize가 실패하면 in-flight 추적을 버립니다. 모든 측정을 이 커밋에서 다시 했습니다.

영역별 파일:

- Overlay: `mlx/backend/rocm/device.cpp`와 `device.h`(예산 파싱, `commit_and_throttle`, `throttle_inflight`, `release_inflight`), `eval.cpp`(primitive별 바이트 집계, finalize), `allocator.cpp`와 `allocator.h`(`thread_allocated_bytes()`, miss 경로 trim), `LOCAL_FIXES.md` 항목 28.
- mlxcel: `src/execution/runtime.rs`(`DEFAULT_CACHE_LIMIT_BYTES`, 순수 함수 `cache_limit_bytes`), `src/execution/runtime_tests.rs`, `src/lib/mlxcel-core/src/memory.rs`(cache-limit 테스트), `tests/rocm_inflight_bound.rs`.
- 측정: `src/bin/bench_decode.rs`(phase 카운터, 적용된 cache limit), `scripts/bench_decode.sh`(`BENCH_RAW_DIR`로 runner 로그 보관).
- 문서: `docs/benchmark_results/rocm-memory-gfx1151-2026-09-30.md`와 `data/rocm-memory-gfx1151-2026-09-30/` 아래 원시 행, `docs/installation.md`(새 "Memory footprint" 소절), `docs/environment-variables.md`.

## 5. 수정 내용

### 5.1 In-flight 제한

Allocator는 `malloc`과 `malloc_async`가 내준 바이트를 cache hit까지 포함해 thread-local 단조 카운터로 셉니다(`rocm::thread_allocated_bytes()`). `gpu::eval`은 각 primitive의 `eval_gpu` 앞뒤로 이 값을 읽고 차이를 encoder의 열린 batch에 더합니다(`add_batch_bytes`). 카운터가 thread-local이고 증가만 하므로, 그 사이 worker thread가 완료된 batch를 해제해도 값이 흔들리지 않습니다.

Eager 경로의 `needs_commit()`은 이제 2000 op에 도달하거나 열린 batch가 `inflight_budget_ / 4`를 할당하면 참이 됩니다. `commit_and_throttle()`은 commit한 뒤 `throttle_inflight()`를 호출하고, 이 함수는 batch 뒤에 raw HIP event를 기록하고, event가 끝난 batch를 목록에서 빼고, committed 총량이 예산을 넘는 동안 가장 오래된 batch를 기다립니다. 기다림은 `hipEventSynchronize`이며(장치가 blocking-sync 모드이므로 스레드는 잠듭니다), `MLX_ROCM_GPU_WATCHDOG_SECS`가 설정된 경우에는 다른 host wait와 같이 50마이크로초 간격으로 polling하다가 기한에 포기합니다. Event 기록이나 기다림이 실패하면 stream의 오류로 저장되어 다음 synchronize가 던지고, 추적은 끝납니다. Decode-step capture나 stream capture 중에는 아무것도 추적하지 않습니다.

`MLX_ROCM_MAX_INFLIGHT_MB`는 바이트로 환산했을 때 `size_t`에 들어가는 음이 아닌 10진 정수 전체여야 하며, 그 외 값은 stderr 경고 한 줄과 함께 1024를 유지합니다.

제한은 구조상 근사치입니다. 임시 buffer만이 아니라 할당 전체를 세고(KV cache를 늘리는 batch도 포함), batch의 buffer는 event가 끝난 직후 worker thread가 해제합니다.

### 5.2 Hysteresis가 있는 cache limit 적용

`malloc_async`에서 cache miss가 났을 때 cache가 `max_pool_size_`를 넘었으면, allocator는 cache된 buffer를 limit의 4분의 3까지 해제합니다(`get_cache_memory() - max_pool_size_ / 4 * 3`). Trim은 miss에서만 합니다. Footprint가 커지는 유일한 곳이고, 이미 HIP 할당 비용을 치르는 곳이기 때문입니다. `free()`와 cache hit은 여전히 `hipFree`를 호출하지 않습니다. Fork가 이를 피하는 이유는 학습 단계 사이에 같은 크기의 activation 수십 GB를 해제하면 blocking drain이 되기 때문입니다. Trim이 있으면 active와 cache의 합은 live set과 limit과 현재 요청 하나의 합을 넘지 않습니다. 4분의 1 여유 덕분에 shape이 계속 바뀌는 작업에서 blocking `hipFree` 한 라운드가 여러 번의 miss를 감당하고, cache가 limit에 붙어 있을 때 miss마다 `hipFree`가 일어나는 일을 막습니다.

### 5.3 `MLXCEL_CACHE_LIMIT`의 ROCm 기본값

`DEFAULT_CACHE_LIMIT_BYTES`는 `cfg(feature = "rocm")`에서 `Some(2 GiB)`, 그 외에는 `None`입니다. `cache_limit_bytes(raw, default)`는 다음 규칙의 순수 함수입니다. 설정 안 됨 또는 빈 값은 기본값, 대소문자 무관 `0`이나 `none` 또는 0으로 파싱되는 크기는 제한 없음, 유효한 크기는 그 크기, 파싱할 수 없는 값은 기본값입니다. 마지막 규칙 덕분에 오타로 ROCm 제한이 조용히 사라지지 않습니다. `resolve_cache_limit()`은 결과를 `mlxcel_core::memory::set_cache_limit`으로 적용합니다. 하네스 로그에는 적용값이 `[Memory] cache limit: 2.15 GB`로 찍히며, 이는 2 GiB를 10진 GB로 나타낸 값입니다.

### 5.4 `memory_limit()`은 그대로

ROCm에서 `mlxcel inspect`와 `--estimate-memory`의 추정은 allocator의 `memory_limit()`을 사용 가능한 메모리로 읽습니다(#1805). Cache 기본값은 `set_cache_limit`을 거치고 in-flight 예산은 백엔드 내부 값이므로, `memory_limit()`은 76.80 GiB 그대로이고 모든 추정값도 이전과 같습니다. `runtime_tests::the_cache_default_leaves_the_memory_limit_the_estimator_reads`가 런타임 초기화가 이 값을 건드리지 않음을 확인합니다. 대신 memory limit을 낮췄다면 아무것도 제한하지 못했을 것이고(2.2절) 모든 추정값만 바뀌었을 것입니다.

## 6. 결과

### 6.1 하네스로 측정한 전후 비교

`scripts/bench_decode.sh`, pp512/tg128, 각 3회 번갈아 실행, 중앙값입니다. "변경 전"은 두 제한을 모두 끈 같은 바이너리(`MLXCEL_CACHE_LIMIT=none MLX_ROCM_MAX_INFLIGHT_MB=0`)이며, origin/main의 peak을 정확히 재현했습니다. 다른 개발 유닛이 같은 GPU에서 profiling 중이었기 때문에, 모든 실행은 0.25초마다 `/sys/class/kfd/kfd/proc`와 프로세스 목록을 샘플링하는 가드를 거쳤고, 다른 GPU 프로세스나 컴파일러와 겹친 실행은 버리고 다시 실행했습니다.

| 모델 | 설정 | MLX peak | VRAM 증가량 | 정상 상태 (측정 pass 후 active + cache) | Prefill tok/s | Decode tok/s |
|---|---|---:|---:|---:|---:|---:|
| Llama-3.1-8B-Instruct-4bit | 변경 전 | 20.60 GB | 21.19 GB | 5.16 GB | 1064.83 | 37.02 |
| Llama-3.1-8B-Instruct-4bit | 기본값 | 6.14 GB | 6.73 GB | 5.18 GB | 1045.87 | 37.57 |
| Qwen3-30B-A3B-4bit | 변경 전 | 23.56 GB | 24.11 GB | 17.67 GB | 300.07 | 61.79 |
| Qwen3-30B-A3B-4bit | 기본값 | 18.58 GB | 19.16 GB | 17.56 GB | 303.75 | 61.69 |

Weight 대비 peak 초과분은 8B에서 15.85 GB에서 1.39 GB로, MoE 모델에서 6.39 GB에서 1.41 GB로 줄어듭니다(표와 2.1절의 weight로 계산). 두 모델 모두 약 1.4 GB로, 1 GiB 예산과 같은 규모입니다. 각 행의 3회 실행에서 peak과 VRAM 증가량의 변동은 최대 0.14 GB였습니다. Decode는 +1.5%와 -0.2%로 이슈의 2% 기준 안입니다. 8B의 prefill은 1.8% 줄었는데, host가 훨씬 앞서 큐에 쌓는 대신 이제 GPU를 기다리기 때문입니다. MoE 모델의 +1.2%는 잡음입니다. 기본값 3회에서 prefill은 264.33에서 315.02 tok/s, 제한 없이 291.52에서 306.43 tok/s 사이였습니다. GTT는 최대 0.01 GB 올랐고 host `VmHWM`은 0.75에서 0.87 GB 사이였습니다. `generate_with_stats`가 이미 prefill 후 cache를 비우므로 정상 상태는 거의 그대로이고, 움직인 것은 peak입니다.

### 6.2 값 선택

두 sweep 모두 제한의 첫 버전(host wait가 `hipEventQuery`로 spin하고, eval의 마지막 batch를 세지 않고, cache를 limit까지만 trim하던 버전)에서 각 1회 실행했습니다. 최종 버전을 측정한 것은 6.1절입니다.

- **In-flight 예산**: 256에서 4096 MiB까지 8B peak은 5.28, 5.75, 6.36, 7.34, 9.36 GB, MoE 모델은 17.82에서 21.66 GB였고 decode는 변하지 않았습니다. Peak은 설계대로 예산을 따라갑니다. 기본값이 1024 MiB인 이유는 예산의 4분의 1보다 큰 임시 buffer는 자기 batch를 따로 commit하고, 더 큰 weight 행렬을 가진 모델(70B MLP 행렬은 f16으로 약 470 MB, 측정이 아니라 계산)은 예산이 그런 행렬 몇 개를 담을 수 있어야 GPU가 쉬지 않기 때문입니다. 더 작은 예산이 측정 가능한 이득을 주지도 않았습니다.
- **Cache limit**: 128 MiB까지 모든 값에서 decode는 제한 없음 대비 2% 안이었습니다. Prefill은 달랐습니다. 128 MiB에서 8B는 16% 느려졌고(892.73 tok/s), 117 MB f16 weight 복사본이 더 이상 cache에 남지 못해 행렬마다 다시 할당되기 때문으로 보입니다. 512 MiB 이상에서는 측정 가능한 차이가 없었습니다. 2 GiB는 비용 없는 최솟값의 4배로, 더 큰 모델의 더 큰 weight 복사본을 위한 여유입니다. 이번 실행에서 cache가 2 GiB에 도달한 적은 없어 기본값이 걸리지는 않았습니다. 이 값은 오래 실행되는 프로세스가 주기적 clear 사이에 쌓을 수 있는 양을 제한하며, 그러지 않으면 exact-size cache는 76.8 GiB까지 커질 수 있었습니다.

## 7. Pool 기반 HipEvent의 heap corruption

In-flight 제한의 첫 버전은 fork의 `HipEvent` wrapper로 batch를 추적했습니다. `HipEvent`는 소멸될 때 handle을 함수 내부 static pool에 돌려줍니다. 프로세스 종료 시 static 소멸 과정에서 `CommandEncoder`가 그 pool보다 늦게 소멸될 수 있고, 그러면 `HipEvent` 멤버가 이미 사라진 pool에 handle을 돌려줍니다. `tests/rocm_mxfp4_quant.rs`가 종료 시 `malloc_consolidate(): unaligned fastbin chunk detected`로 중단되었습니다.

최종 버전은 encoder가 소유하는 raw `hipEvent_t` handle을 저장합니다. `InflightBatch { done, bytes }`의 deque와 재사용용 spare event vector입니다. `release_inflight()`는 in-flight event를 spare 목록으로 옮기고, encoder 소멸자는 이를 호출한 뒤 모든 spare에 `hipEventDestroy`를 부르며, fault 난 장치는 모든 destroy에서 fault를 돌려주므로 오류는 무시합니다. `device.h`의 `InflightBatch` 주석과 `LOCAL_FIXES.md` 항목 28 모두 `HipEvent`를 쓰지 않는 이유를 기록해, 다음 변경에서 이것이 다시 들어오지 않게 합니다. 이후 `rocm_mxfp4_quant`는 3회 모두 통과했습니다.

## 8. 기술적 선택과 그 이유

- **이슈가 지목한 카운터가 아니라 메모리를 잡고 있던 메커니즘을 고칩니다.** 이슈는 fork 안에서만 `max_pool_size_`를 바꾸는 것을 거부하고 운영자가 볼 수 있는 cache 기본값을 선호했습니다. 측정이 문제를 command batch 수명으로 옮겼으므로, 이슈의 fallback 조항대로 주 수정은 overlay에 있습니다. Cache limit은 적용되지도 제한되지도 않았으므로 cache 기본값도 `MLXCEL_CACHE_LIMIT`을 통해 함께 넣습니다.
- **Op 수나 scheduler task가 아니라 바이트로 제한합니다.** Host wait 없이 commit만 늘린 것은 효과가 없었고, MLX scheduler의 상한은 바이트가 아니라 task를 셉니다. Primitive별 바이트를 세면 op 구성과 관계없이 실제로 커진 양을 제한합니다.
- **Stream 전체가 아니라 가장 오래된 batch를 기다립니다.** In-flight 총량이 예산을 넘을 때만, 가장 오래된 batch만 기다리므로 예산만큼의 작업이 큐에 남아 GPU가 굶지 않습니다. Commit마다 stream을 synchronize하면 매번 큐가 비게 됩니다.
- **Miss에서만, 4분의 3까지 trim합니다.** `free()`와 cache hit에서 blocking `hipFree`를 피하고(fork의 학습 경로 우려), shape이 계속 바뀔 때 trim 비용을 나눕니다.
- **파싱할 수 없는 `MLXCEL_CACHE_LIMIT`은 기본값을 유지합니다.** 제한을 끄려면 `0`이나 `none`을 명시해야 합니다. 변경 전에는 잘못된 값이 제한 없음을 뜻했는데, ROCm에서는 이제 문서가 약속한 기본값이 조용히 사라지는 일이 됩니다.
- **`memory_limit()`은 바꾸지 않습니다.** ROCm에서 사전 로드 추정의 입력이고, 두 제한 모두 필요로 하지 않습니다.
- **In-flight 제한에 `0`이라는 탈출구를 둡니다.** 이전 동작을 정확히 복원하며, 덕분에 "변경 전" 행을 같은 바이너리로 재현할 수 있었습니다.

## 9. 검증

PR 본문 기준, gfx1151에서:

- 전후 sweep(6.1절)과 결과 페이지의 knob, in-flight 예산, cache limit sweep.
- `cargo test -p mlxcel-core --lib memory::tests::`, 새 `cache_limit_bounds_the_free_buffer_cache` 포함(1 MiB limit에서 서로 다른 크기의 buffer 64개를 해제한 뒤 cache가 8 MiB 미만이어야 함. Trim을 제거하면 49.8 MB가 남아 실패).
- `cargo test -p mlxcel --lib execution::`(151개 통과, 새 cache 기본값 테스트 포함)와 `--test rocm_inflight_bound`(한 번의 평가에서 512행에 대한 f16 4-bit 8192x8192 matmul 40개, dequantize 복사본 5 GiB. 기본값에서 peak은 1.75 GiB 올랐고, `MLX_ROCM_MAX_INFLIGHT_MB=0`에서는 6.25 GiB 올라 3 GiB 기준을 넘어 실패).
- `rocm_gpu_faults`, `rocm_mxfp4_quant`(3회), `rocm_cpu_blas_finegrained`, `rocm_slice_update_source`, `rocm_strided_scan`, `rocm_fft_plan_cache`, `dead_doc_pointers`: 통과. `scripts/ci/rocm_smoke.sh` OK. 변경된 crate와 target에 대해 clippy `-D warnings`.
- 리뷰(pr-reviewer): CRITICAL, HIGH 없음. MEDIUM 두 건은 `5f665765`에서 고쳤고, 위의 모든 항목을 그 커밋에서 다시 실행했습니다.

`3c9edea0` 위로 rebase한 브랜치로 gfx1151에서 오케스트레이터가 검증한 내용:

- `make verify-rocm`이 모든 단계를 실행했습니다. Versions, kernel dtype keys, kernel port dispatch, llama-compat, `verify-rocm-overlay`(backend 파일 108개와 core 파일 15개, `LOCAL_FIXES` 항목 28개), fmt, `--features rocm` workspace clippy, ROCm smoke(32 토큰)가 통과했습니다.
- `verify-test-rocm`은 정확히 한 target, `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`(bf16)에서 실패했습니다. #2081에서 추적 중인 마지막 baseline 실패이며 이 PR이 만든 것이 아닙니다. mlxcel-core lib은 1838개 통과했습니다.
- `tests/rocm_inflight_bound.rs`는 전체 suite 안에서 통과했습니다.

이 보고서를 쓰며 커밋된 데이터로 다시 확인한 것: 6.1절의 중앙값은 `harness-bounds-off.csv`, `harness-defaults.csv`, `memory-lines.txt`와 일치합니다(기본값에서 peak 6.14, 6.14, 6.14 GB와 18.57, 18.58, 18.58 GB, 제한 없이 세 번 모두 20.60 GB와 23.56 GB).

## 10. 학습 포인트

- **제한하기 전에 어느 카운터가 메모리를 잡고 있는지 측정합니다.** 이슈의 계획은 cache limit이었습니다. Allocator의 peak은 정의상 cache를 제외하므로, weight보다 훨씬 큰 peak과 작은 cache는 첫 측정부터 live buffer를 가리키고 있었습니다.
- **저장된 limit이 적용된 limit은 아닙니다.** ROCm에서 `set_cache_limit`과 `set_memory_limit`은 모두 호출은 성공했지만 아무것도 바꾸지 않았습니다. 하나는 읽히지 않았고, 다른 하나는 eager 경로가 우회한 scheduler 집계에 의존했습니다. 효과 자체를 검사하는 테스트(cache 테스트, in-flight 테스트)만이 둘을 구분합니다.
- **Back-pressure 없는 batching은 batch 크기만 바꿉니다.** Host가 기다리게 만들기 전까지는 commit을 늘려도 아무것도 움직이지 않았습니다.
- **Static 소멸 순서는 GPU handle pool에도 적용됩니다.** 함수 내부 static pool을 쓰는 pooled RAII wrapper는 종료 시 그 pool보다 오래 살 수 있는 객체 안에서 안전하지 않습니다. 이 실패는 관련 없는 테스트의 종료 과정에서 heap corruption으로 드러났습니다.
- **이전 동작을 정확히 재현하는 스위치를 남깁니다.** `MLX_ROCM_MAX_INFLIGHT_MB=0`과 `MLXCEL_CACHE_LIMIT=none` 덕분에 한 바이너리로 전후 표의 양쪽을 만들 수 있었고, 빌드 차이가 비교에서 빠졌습니다.

## 11. 주의사항, 검증하지 않은 것, 남은 작업

- **사전 로드 추정은 여전히 8B의 peak보다 낮습니다.** `mlxcel inspect --max-tokens 640`은 8B를 5.58 GB(weight와 KV cache에 1.20을 곱하고 activation 항을 더함), MoE 모델을 20.72 GB로 추정합니다. 8B의 peak은 이제 6.14 GB로 추정보다 0.56 GB 높고(이전에는 15 GB 높았음), MoE 모델은 이제 추정보다 낮습니다. ROCm용 1.20 headroom factor 재보정은 이 PR에 포함하지 않았습니다.
- **예산과 cache sweep은 제한의 첫 버전에서 실행했습니다.** 최종 버전은 eval의 마지막 batch를 세고, spin 대신 block하고, 여유를 두고 trim한다는 점이 다르며, 최종 버전을 측정한 것은 하네스 표뿐입니다.
- **제한은 근사치입니다.** KV cache 증가 같은 영구 할당까지 세고, buffer는 event 완료 직후에 해제됩니다.
- **`MLX_ROCM_MAX_INFLIGHT_MB` 파싱에 대한 자동 테스트가 없습니다**(리뷰의 LOW 지적, 고치지 않음).
- **Metal과 CUDA는 실행하지 않았습니다.** 건드린 공유 코드는 `runtime.rs`(`rocm` 밖에서 기본값 `None` 유지, `the_default_cache_limit_is_rocm_only`로 고정), `memory.rs`(MLX 자체 allocator가 그쪽에서도 만족시키는 새 테스트), `bench_decode.rs`(출력만)입니다.
- **모델 두 개, 호스트 하나.** 1024 MiB와 2 GiB 기본값은 512 prompt 토큰에서 8B와 30B MoE로 골랐습니다. 더 큰 모델과 더 긴 prompt는 추론한 것이지 측정한 것이 아닙니다.
- **Upstream.** 항목 28은 fork에 올릴 후보입니다(#1813).

참고: #2062, #1814, #1801, #1805, #1813, #2081, #627.
