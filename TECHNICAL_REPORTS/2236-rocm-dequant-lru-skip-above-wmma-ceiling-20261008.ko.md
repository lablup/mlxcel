# 기술 보고서: PR #2236 - WMMA 상한 위로 라우팅된 GEMM은 Dequant LRU를 건너뜀

**날짜**: 2026-10-08

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 브랜치 커밋 `8964f610`(코드)와 `79a8c9bb`(문서), origin/main `6dbfe7d7` 위로 리베이스, 전체 `make verify-rocm` 게이트 통과 후 머지. #2151을 닫음. 후속: #2232.

**언어**: HIP/C++(ROCm 오버레이 `qmm.hip`, `rocm.h`, `no_rocm.cpp`; mlxcel-core 브리지 `mlx_cxx_bridge.cpp`, `mlx_cxx_bridge.h`), Rust(mlxcel-core `lib.rs`, 신규 `rocm_qmm_cache.rs`; 신규 `tests/rocm_qmm_dequant_cache.rs`), Markdown(`LOCAL_FIXES.md`, `docs/environment-variables.md`, `docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md`)

**위험도**: 낮음. 상한으로 라우팅된 GEMM은 이전과 같은 dequantize 커널과 같은 hipBLASLt GEMM을 실행합니다. 캐시 조회와 삽입만 건너뛰며, logit trace는 바이트 단위로 동일합니다. 캐시 경로를 쓰는 f16과 비 WMMA 호출자는 캐시를 그대로 씁니다. 새 카운터는 dequantize 경로의 GEMM마다 한 번 올리는 relaxed atomic이고, 캐시 리팩터링은 기존 static들을 구조체 하나로 옮길 뿐 축출 규칙은 바꾸지 않습니다.

## 요약

#2085 이후 RDNA 3.5에서 128행 이상의 bf16 affine GEMM은 fused WMMA 커널을 떠나 dequantize + hipBLASLt로 갑니다. 이 경로는 dequantize한 가중치를 모두 8개 행렬 또는 256 MB 크기의 LRU에 저장했습니다. forward pass 한 번이 실행하는 서로 다른 projection은 8개보다 훨씬 많으므로, prefill은 한 번도 hit 없이 캐시를 순환시켰고, 마지막 항목들(bf16 가중치 최대 256 MB와 양자화 원본에 대한 참조)은 decode 내내 살아 있었습니다.

PR #2236은 `select_qmm_route`의 상한 분기에서만 반환되는 `QmmRoute::DequantGemmAboveWmmaCeiling`을 추가합니다. `QuantizedMatmul::eval_gpu`는 이 경로를 `DequantGemm`처럼 실행하되, command encoder가 GEMM 뒤에 해제하는 임시 버퍼로 dequantize합니다. 캐시는 이제 hit, miss, insert, eviction, bypass 카운터를 가진 구조체 하나에 있고, `rocm::dequant_cache_stats()`, 브리지, `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`로 읽을 수 있습니다.

gfx1151에서 Gemma 3 4B 4-bit(bf16)를 pp2048/tg128로 돌렸을 때, 변경 전 캐시는 조회 340번에 hit 0번이었고 종료 시 항목 6개(230 MiB)를 들고 있었습니다. 변경 후 GEMM은 캐시를 340번 bypass하고, 캐시는 비어 있으며, prefill 후 active memory는 2.80 GB에서 2.56 GB로 줄었습니다. MLX peak memory는 4.52 GB 그대로이고 prefill과 decode는 실행 간 노이즈 범위 안입니다. Llama 3.1 8B 4-bit(f16 scales)는 변하지 않았고, 역시 조회 320번에 hit 0번이면서 224 MiB를 들고 있습니다. 그 기본값을 끄는 작업은 #2232입니다.

## 1. 문제 정의

### 1.1 상한 경로와 그 캐시

`src/lib/mlx-cpp/patches-rocm/mlx/backend/rocm/quantized/qmm.hip`의 `select_qmm_route`는 양자화 matmul의 실행 방식을 정합니다. WMMA 대상인 bf16 shape에 대해 `hipblaslt_is_faster = env != 1 && dequant && M >= wmma_qmm_max_m(d) && !fp8()`를 계산하며, `wmma_qmm_max_m`은 RDNA 3.5에서 128입니다. 이 PR 전에는 이 분기가 `QmmRoute::DequantGemm`을 반환했는데, 이는 f16 체크포인트와 비 WMMA shape가 `if (dequant)`를 통해 도달하는 것과 같은 열거자입니다.

`QuantizedMatmul::eval_gpu`에서 `DequantGemm`은 가중치, scales, bias 포인터로 키를 만든(`DequantCacheKey`) 프로세스 전역 LRU를 조회했습니다. miss가 나면 가중치를 dequantize하고, `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE`(기본 8)와 `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES`(기본 256 MB)가 모두 0이 아니면 삽입했습니다. 각 항목은 양자화 원본 배열에 대한 참조도 들고 있었습니다.

### 1.2 캐시가 hit하지 않는 이유

prefill은 forward pass마다 각 projection을 한 번씩 건드리고, 모델에는 캐시가 담을 수 있는 것보다 훨씬 많은 projection이 있습니다(Llama 계열 8B는 32개 레이어에 각 7개). 모든 조회는 miss이고, 모든 삽입은 오래된 항목을 축출하며, 축출되기 전에 다시 조회되는 항목은 없습니다. miss는 구조적이므로 키 순서나 해시를 바꿔도 해결되지 않습니다. 배치 1의 decode는 한 행을 실행해 qmv를 타고 캐시에 도달하지 않으므로, prefill 끝에 남은 항목은 쓰이지 않습니다.

`LOCAL_FIXES.md` 항목 29와 #2085 기술 보고서는 캐시가 prefill에서 hit하지 않으면서 이후에도 최대 256 MB를 살려 둔다고 이미 기록했습니다. 메모리 비용은 측정되지 않았습니다.

### 1.3 측정된 비용

카운터를 추가한 빌드(`87538835`와 똑같이 라우팅하고 캐시하는 카운터 전용 빌드)에서 Gemma 3 4B 4-bit는 pp2048/tg128에서 조회 340번을 했습니다. hit 0, miss 340, insert 340, eviction 334이고, 종료 시 항목 6개(241,172,480 바이트, 230 MiB)가 캐시에 남아 있었습니다. 이 230 MiB는 prefill 끝부터 decode 내내 할당된 채로 있었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `patches-rocm/.../quantized/qmm.hip` | 상한 분기에서만 반환되는 `QmmRoute::DequantGemmAboveWmmaCeiling` 신설. `eval_gpu`는 dequantize + GEMM 분기에서 이를 받음. `use_cache`는 `DequantGemm`일 때만 true이므로 새 경로는 임시 버퍼로 dequantize하고 `bypasses`를 올림. `quantized_matmul_runs_dequant_gemm`은 두 열거자를 모두 받음. 캐시 static들은 `evict_to()`를 가진 `struct DequantCache`로 이동. `DequantCacheCounters`가 relaxed atomic hit, miss, insert, eviction, bypass 카운터를 가짐. `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`이면 `print_dequant_cache_stats()`를 `std::atexit`에 등록 |
| `patches-rocm/.../rocm.h`, `no_rocm.cpp` | `struct DequantCacheStats`와 `MLX_API dequant_cache_stats()`. stub은 0을 반환 |
| `mlxcel-core/cpp/mlx_cxx_bridge.{cpp,h}` | `rocm_dequant_cache_stats()`가 일곱 값을 `rust::Vec<uint64_t>`로 반환. `MLXCEL_BRIDGE_ROCM_BACKEND`가 없으면 0 일곱 개 |
| `mlxcel-core/src/lib.rs`, `rocm_qmm_cache.rs`(신규) | FFI 선언과 `DequantCacheStats`, `dequant_cache_stats()`를 담은 `pub mod rocm_qmm_cache`. 단위 테스트 `stats_are_zero_off_rocm` |
| `tests/rocm_qmm_dequant_cache.rs`(신규) | ROCm에서 케이스 4개를 각각 자식 프로세스 하나로 실행 |
| `LOCAL_FIXES.md` 항목 29, `docs/environment-variables.md`, `docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md` | 새 동작, stats 변수, 날짜가 붙은 측정 섹션 |

커밋은 두 개입니다. 코드 변경(`8964f610`)과 측정 기록(`79a8c9bb`)입니다. 파일 11개, 812줄 추가, 53줄 삭제.

## 3. 설계

### 3.1 호출 플래그가 아니라 별도 열거자

GEMM이 WMMA shape였다가 행 수 때문에 다른 경로로 보내졌다는 사실을 아는 곳은 상한 분기뿐입니다. 새 열거자는 `eval_gpu`에 두 번째 조건 묶음을 두지 않고 그 사실을 전달합니다. 두 경로가 공유하는 것은 그대로 공유합니다. 같은 분기가 배치 shape 검사, dequantize 커널, hipBLASLt GEMM과 그 rocBLAS 폴백을 처리합니다. 다른 조건은 하나뿐입니다.

```cpp
const bool use_cache = route == QmmRoute::DequantGemm && cache_cap > 0 &&
    cache_max_bytes > 0;
```

`use_cache`가 false이면 가중치는 기존의 캐시 꺼짐 경로를 탑니다. `w_dequant`가 새로 할당되고, `enc.add_temporary(w_dequant)`가 GEMM이 끝날 때까지 이를 살려 둡니다. `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0`이 이미 쓰던 경로입니다.

경로의 다른 독자는 `quantized_matmul_runs_dequant_gemm`이며, mlxcel의 dense prefill 경로는 이 함수로 `quantized_matmul`과 `dequantize` + `matmul`이 같은 바이트를 내는지 판단합니다. 이제 두 열거자를 모두 받으므로, 상한으로 라우팅된 shape에 대한 답은 바뀌지 않습니다.

### 3.2 범위: 상한 경로만

이슈는 다른 `DequantGemm` 호출자는 그대로 두고 먼저 측정하라고 했습니다. f16 체크포인트와 비 WMMA shape는 이 PR에서도 캐시를 씁니다. 아래 측정에서 f16 캐시도 hit하지 않는 것으로 나왔지만, 그 기본값을 바꾸는 것은 따로 확인할 경우가 있는 별개의 결정이므로(#2156, `DequantGemm`을 타는 f16 배치 4 decode로, 같은 projection을 매 스텝 재사용하는 decode 루프는 hit할 수 있음) #2232로 넘겼습니다.

경계 사례는 라우팅 로직에서 바로 나옵니다. `MLX_ROCM_WMMA_QMM=1`은 `env == 1`이 되어 상한 분기를 타지 않으므로, fused 커널이 실행되고 캐시도 bypass 카운터도 움직이지 않습니다. `MLX_ROCM_WMMA_QMM_MAX_M` 오버라이드는 상한을 바꾸고, 여전히 새 열거자를 만듭니다. fp8 경로는 영향이 없습니다.

### 3.3 카운터와 stats 훅

이슈의 완료 조건은 bf16 prefill 후 캐시가 비어 있음을 단언할 방법이 필요했습니다. `eval_gpu` 안의 캐시 `static` 지역 변수는 밖에서 읽을 수 없었으므로, `dequant_cache()` 뒤의 `struct DequantCache`(뮤텍스, LRU 리스트, 바이트 수, 항목 맵)로 옮겼고, 두 벌이던 축출 루프는 `evict_to()` 메서드 하나가 되었습니다. 축출 동작은 같으며, `evict_to()`가 eviction 카운터도 올립니다.

카운터는 relaxed ordering의 `std::atomic<uint64_t>`입니다. dequantize 경로의 GEMM마다 한 번 올리고, hot path에서는 읽지 않습니다. `read_dequant_cache_stats()`가 카운터를 읽고, 캐시 뮤텍스 아래에서 항목 수와 바이트 수를 읽습니다. 그 위에 독자가 셋 있습니다.

- `rocm.h`의 `rocm::dequant_cache_stats()`. `no_rocm.cpp`에 stub이 있습니다.
- 브리지를 거치는 `mlxcel_core::rocm_qmm_cache::dequant_cache_stats()`. ROCm이 아니면 0을 반환합니다.
- `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`. stderr에 `[ROCm] qmm dequant cache: ...` 한 줄을 출력하는 `std::atexit` 핸들러를 등록합니다. 핸들러는 static 캐시가 생성된 뒤 `dequant_cache()` 안에서 등록되므로 캐시가 소멸되기 전에 실행됩니다. 상한 경로는 bypass를 세기 전에 `dequant_cache()`를 호출하므로, 캐시 자체를 한 번도 쓰지 않아도 보고가 등록됩니다.

### 3.4 테스트

캐시, 카운터, 환경 변수 노브가 프로세스 전역이고 한 번만 읽히므로, `tests/rocm_qmm_dequant_cache.rs`는 각 케이스를 새 자식 프로세스에서 실행합니다. 부모 테스트는 자식을 하나씩 차례로 실행해 GPU를 공유하지 않게 합니다. 각 자식은 `MLX_ROCM_QMM_DEQUANT_M_THRESHOLD=1`로 어떤 행 수에서든 dequantize 경로를 대상으로 만들고, 양자화 GEMM 하나를 두 번 실행해 각 결과를 CPU 스트림의 f32 `dequantize` + `matmul`과 비교하며(상대 허용 오차 2e-2), 각 패스 뒤 카운터를 확인합니다.

- `bf16_above_default_ceiling`(256행, 4096x4096): bypass 1번, 이어서 2번, 빈 캐시, GPU의 `dequantize` + `matmul`과 같은 출력 바이트.
- `bf16_above_env_ceiling`(64행, `MLX_ROCM_WMMA_QMM_MAX_M=32`): 환경 변수 상한으로 같은 확인.
- `bf16_forced_wmma`(`MLX_ROCM_WMMA_QMM=1`): 어떤 카운터도 움직이지 않음.
- `f16_cached`(128행): 첫 패스에 miss와 insert, 두 번째 패스에 hit, `n * k * 2` 바이트 항목 하나.

상한 분기가 다시 `DequantGemm`을 반환하게 하면 두 bf16 상한 케이스가 첫 패스에서 실패하고(`misses=1 inserts=1 bypasses=0 entries=1`), forced-WMMA와 f16 케이스는 여전히 통과합니다. 4개 중 2개가 실패합니다.

## 4. 프로덕션 영향

gfx1151(Radeon 8060S)에서 `scripts/bench_decode.sh <model> --prompt-tokens 2048 --max-tokens 128`로 측정했습니다. 변경 전은 main `87538835`, 변경 후는 그 커밋에 변경을 더한 것이며, 3회 중앙값(범위)이고, 각 실행은 `scripts/rocm_gpu_guard.sh --idle-secs 60` 아래에서 두 쪽을 번갈아 돌렸습니다.

| 모델 | Scales | 구분 | Prefill tok/s | Decode tok/s | MLX peak | Prefill 후 active | 종료 시 active |
|---|---|---|---:|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | bf16 | 변경 전 | 2872 (2819-2910) | 62.75 (62.68-62.76) | 4.52 GB | 2.80 GB | 2.97 GB |
| gemma-3-4b-it-4bit | bf16 | 변경 후 | 2868 (2839-2945) | 62.71 (62.55-62.73) | 4.52 GB | 2.56 GB | 2.73 GB |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | 변경 전 | 1136 (1106-1136) | 36.63 (32.36-36.84) | 6.92 GB | 4.75 GB | 4.75 GB |
| Meta-Llama-3.1-8B-Instruct-4bit | f16 | 변경 후 | 1132 (1048-1143) | 36.67 (36.58-36.92) | 6.92 GB | 4.75 GB | 4.75 GB |

캐시 카운터(warmup과 측정 prefill 합계, 변경 전은 카운터 전용 빌드):

| 모델 | 구분 | Hits | Misses | Inserts | Evictions | Bypassed | 종료 시 항목 | 종료 시 바이트 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| gemma-3-4b-it-4bit | 변경 전 | 0 | 340 | 340 | 334 | 0 | 6 | 241,172,480 |
| gemma-3-4b-it-4bit | 변경 후 | 0 | 0 | 0 | 0 | 340 | 0 | 0 |
| Meta-Llama-3.1-8B-Instruct-4bit | 변경 전 | 0 | 320 | 320 | 318 | 0 | 2 | 234,881,024 |
| Meta-Llama-3.1-8B-Instruct-4bit | 변경 후 | 0 | 320 | 320 | 318 | 0 | 2 | 234,881,024 |

RDNA 3.5의 bf16 체크포인트에서는 prefill 후와 실행 종료 시 active memory가 캐시에 남던 항목 크기만큼인 0.24 GB 줄어듭니다. MLX peak memory는 변하지 않는데, peak는 prefill 안에서 호출마다 만드는 dequantize 사본에서 오고 캐시는 그것을 없앤 적이 없기 때문입니다(그 사본은 항목 28이 제한합니다). prefill과 decode는 실행 간 범위 안이며, 속도 향상은 주장하지 않습니다. Llama 3.1 8B는 f16 대조군입니다. 이 모델의 GEMM은 상한 경로를 타지 않으므로 두 쪽이 같게 나옵니다.

main `ad844354` 위로 리베이스한 변경에서 모델당 한 번씩 돌린 확인 실행은 변경 후 행과 일치했습니다. Gemma 3 4B는 prefill 2885 tok/s, decode 62.14 tok/s, peak 4.52 GB, prefill 후 active 2.56 GB, bypass 340번, 빈 캐시였고, Llama 3.1 8B는 prefill 1119 tok/s, decode 36.66 tok/s, peak 6.92 GB, active 4.75 GB, miss 320번, hit 0번, 항목 2개였습니다.

Metal과 CUDA는 영향이 없습니다. 변경은 ROCm 오버레이에 있고, 브리지 함수는 ROCm이 아니면 0을 반환합니다.

## 5. 문서

- **`LOCAL_FIXES.md` 항목 29.** LRU가 prefill에서 hit하지 않으면서 마지막 항목을 살려 둔다고 적은 문장이 이제 `QmmRoute::DequantGemmAboveWmmaCeiling`, Gemma 3 4B 수치(변경 전 조회 340번에 hit 0번, 230 MiB 캐시됨; 변경 후 bypass 340번, 빈 캐시, prefill 후 active 2.80에서 2.56 GB, peak 변화 없음), 카운터, `dequant_cache_stats()`, `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`, 새 테스트를 설명합니다.
- **`docs/environment-variables.md`.** `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE` 설명에 `MLX_ROCM_WMMA_QMM_MAX_M` 위로 보내진 bf16 GEMM은 캐시를 쓰지 않는다는 내용이 추가되었고, `MLX_ROCM_QMM_DEQUANT_CACHE_STATS=1`이 문서화되었습니다.
- **`docs/benchmark_results/rocm-bf16-qmm-route-gfx1151-2026-09-30.md`.** "2026-10-07: the dequantized-weight cache above the WMMA ceiling (issue #2151)" 섹션이 추가되어 방법, 두 표, logit 확인, 확인 실행을 담습니다. 세 번째 Gemma 변경 전 실행은 첫 실행이 하네스에서 실패해 2026-10-08에 했다는 점, 그리고 리베이스된 헤드에서 Gemma의 종료 시 active memory(2.56 GB, `87538835`에서는 2.73 GB)가 main의 변화와 함께 움직였고 그 차이를 분리하지 않았다는 점도 적혀 있습니다.

## 6. 검증

gfx1151(Radeon 8060S), ROCm 10.0.0 / HIP 7.15에서 실행했습니다.

- **새 테스트.** `cargo test --release --features rocm --test rocm_qmm_dequant_cache -- --test-threads=1`: 통과. 상한 분기가 다시 `DequantGemm`을 반환하게 하면 4개 케이스 중 2개(두 bf16 상한 케이스, 첫 패스에서)가 실패하고, forced-WMMA와 f16 케이스는 여전히 통과합니다.
- **라우팅 환경 변수 테스트.** `cargo test --release --features rocm --test rocm_qmm_env -- --test-threads=1`: 통과.
- **Logits.** gemma-3-4b-it-4bit에서 `examples/logit_trace`(README.md를 코퍼스로, 256토큰 청크 4개이므로 모든 projection이 256행으로 실행되어 상한 경로를 탐)를 변경 전후로 실행: trace가 바이트 단위로 동일. `scripts/compare_logit_traces.py --decided 2.0`은 1024개 중 0개 불일치와 동일한 perplexity를 보고합니다.
- **대상 게이트.** `make verify-rocm-overlay verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `cargo test --test dead_doc_pointers`, `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings`, `cargo clippy -p mlxcel --features rocm --test rocm_qmm_dequant_cache -- -D warnings`: 통과.
- **전체 게이트.** `6dbfe7d7` 위의 head `79a8c9bb`에서 `MLXCEL_ROCM_SMOKE_MODEL=models/mlx/Qwen3-0.6B-4bit`로 실행한 `make verify-rocm`: `[verify-rocm] OK`, cargo 테스트 스위트 162개에서 12191개 통과, 0개 실패, 399개 무시(mlxcel-core lib 스위트에서 8906개 통과, 165개 무시), ROCm 스모크는 GPU에서 32개 토큰을 생성했습니다. `ae343d84` 기준의 이전 실행도 통과했습니다(12190개 통과, 0개 실패). 이후 main에 bridge 파일을 건드리는 #2218이 들어와 브랜치를 리베이스하고 게이트를 다시 실행했습니다.

검증하지 않은 것: 이 호스트에 없는 Metal과 CUDA. 브리지 stub은 단위 테스트 `stats_are_zero_off_rocm`이 다루지만, 그 백엔드에서 빌드하거나 실행하지 않았습니다.

## 7. 기술적 선택과 그 이유

- **상한 경로에서만 캐시를 건너뜀.** 이슈가 측정한 경로이고 캐시의 miss 패턴이 알려진 경로입니다. f16 경로도 hit 0번이었지만 #2156의 decode 유사 경우는 hit할 수 있으므로, 그 기본값 변경은 여기에 합치지 않고 #2232에서 다룹니다.
- **전역 캐시 크기 변경 대신 새 열거자.** `MLX_ROCM_QMM_DEQUANT_CACHE_MAX_BYTES`를 전역으로 줄이면 쓸모없는 삽입은 그대로이고 메모리도 여전히 묶이므로, 이슈에서 기각했습니다. miss가 구조적이므로 키를 바꾸는 방안도 기각했습니다.
- **캐시 꺼짐 경로 재사용.** 임시 할당과 `add_temporary`는 `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE=0`용으로 이미 있었으므로, 새 경로는 메모리 관리 코드를 새로 추가하지 않습니다.
- **카운터는 항상 켜 둠.** GEMM마다 relaxed atomic 증가 한 번은 dequantize와 GEMM에 비하면 무시할 수 있고, 항상 있으므로 벤치마크가 재빌드 없이 환경 변수 하나로 읽을 수 있습니다.
- **테스트 케이스마다 자식 프로세스 하나.** 캐시, 카운터, 노브가 프로세스 전역이고 한 번만 읽히므로, 각 케이스가 0에서 시작하려면 새 프로세스가 필요합니다.

## 8. 남은 위험과 후속 작업

- **#2232: f16 캐시.** Llama 3.1 8B(f16 scales)는 조회 320번에 hit 0번, insert 320번, eviction 318번이었고, 실행 후 항목 2개(224 MiB)를 살려 두었으며, 세 번 실행 모두 같았습니다. #2232는 #2156의 decode 유사 경우를 카운터로 확인한 뒤 ROCm에서 `MLX_ROCM_QMM_DEQUANT_CACHE_SIZE`의 기본값을 끄자고 제안합니다.
- **상한은 gfx1151에서만 측정됨.** 변경은 상한 경로를 타는 모든 곳에 적용되지만, 그 경로와 이 수치는 gfx1151에서 측정했습니다.
- **베이스 간 종료 시 메모리 차이.** Gemma의 실행 종료 시 active memory는 리베이스된 헤드에서 2.56 GB, `87538835`에 변경을 더한 것에서 2.73 GB였습니다. 두 베이스 사이에 main이 바뀌었고 그 차이는 분리하지 않았습니다.
- **ROCm 외 stub은 실행되지 않음.** `stats_are_zero_off_rocm`이 stub을 다루지만 Metal이나 CUDA에서 실행된 적은 없습니다.

## 9. 학습 포인트

- **캐시는 hit율이 있어야 메모리 값을 합니다.** 이 LRU는 이 행 수의 forward pass가 결코 주지 않는 재사용을 위해 추가되었습니다. 카운터가 실행 한 번으로 이를 드러냈습니다. 조회 340번에 hit 0번입니다.
- **peak와 상주 메모리는 다른 수치입니다.** 캐시를 건너뛰자 prefill 후 active memory는 0.24 GB 줄었지만 MLX peak는 변하지 않았습니다. peak는 캐시가 없앤 적 없는 호출당 임시 버퍼에서 오기 때문입니다. peak만 측정했다면 효과가 없는 것으로 보였을 것입니다.
- **라우팅에 관한 사실은 경로에 담습니다.** GEMM이 왜 다른 경로로 갔는지 아는 지점에 열거자를 추가하니, `eval_gpu`의 차이는 한 줄로 끝났고 경로의 다른 독자인 `quantized_matmul_runs_dequant_gemm`도 한 줄 변경으로 올바르게 유지되었습니다.
- **변경 전 상태를 이전 라우팅에서 측정 가능하게 만듭니다.** 변경 전 카운트는 `87538835`처럼 라우팅하는 카운터 전용 빌드에서 나왔고, 이로써 카운터의 효과와 라우팅 변경의 효과를 분리했습니다.
