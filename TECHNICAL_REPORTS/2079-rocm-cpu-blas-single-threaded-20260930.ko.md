# 기술 보고서: PR #2079 - fine-grained 메모리 위에서 CPU 스트림 BLAS를 단일 스레드로 실행

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: C++ (ROCm overlay allocator), Rust (테스트), Markdown

**위험도**: 정확성 측면에서는 낮음, CPU 스트림 속도 측면에서는 중간 (allocator hook 하나만 추가되며 GPU 커널은 바뀌지 않습니다. 측정한 gemv에서 CPU 스트림 BLAS가 약 7배 느려집니다. Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않습니다)

## 요약

이슈 #2072(에픽 #1801의 일부)는 `tests/rocm_mxfp4_quant.rs`의 `quantized_matmul_matches_dequantized_reference`가 gfx1151에서 간헐적으로 실패하며, 항상 gpt-oss-20b의 decode shape인 `qmm 2880x2880 M=1`에서 실패한다고 보고했습니다. 이슈는 `qmv_warp_shared_kernel`을 의심했고, 커널이 원인이었다면 ROCm의 gpt-oss decode가 조용히 잘못된 logit을 내고 있었을 수 있습니다.

커널은 옳았습니다. 틀린 것은 테스트의 CPU 기준값이었습니다. APU에서 ROCm allocator는 CPU 스트림 배열을 포함한 모든 배열에 fine-grained device 메모리를 줍니다. 멀티스레드 OpenBLAS(Debian 0.3.29, 32 스레드)가 `cblas_sgemm` 결과를 그 메모리에 쓰면 호출마다 다른 출력 열이 틀립니다. 틀린 열은 각각 한 OpenBLAS 스레드가 맡은 출력 구간의 마지막 열이고, 그 열의 64바이트 캐시 라인을 다음 스레드도 씁니다. MLX 없이 독립 C 프로그램으로도 재현되었습니다.

수정은 overlay allocator에 있습니다. `unified_malloc`이 첫 fine-grained 할당 시 `openblas_set_num_threads(1)`을 한 번 호출합니다. weak symbol로 선언해서 다른 BLAS로 빌드해도 링크되고 아무것도 바뀌지 않습니다. tolerance는 바뀌지 않았습니다. 비용은 CPU 스트림 BLAS 속도로, 재현에 쓴 gemv가 호출당 약 0.2초에서 약 1.4초가 되었습니다. 변경은 `LOCAL_FIXES.md` 항목 27이며 #1813의 upstream 후보입니다.

## 1. 문제 정의

실패하는 케이스는 무작위 `[2880, 2880]` weight를 GPU에서 mxfp4로 양자화하고, GPU에서 `quantized_matmul`을 실행한 뒤, CPU 스트림에서 만든 기준값과 비교합니다. 기준값은 같은 packed 바이트와 scale을 `dequantize_cpu`로 풀고 `want = astype(x, f32) @ dense.T`를 계산한 것입니다. 모든 dtype이 실패했고(`f32: 0.17341012 exceeds 0.0001`, `f16: 0.11339103`, `bf16: 0.19927761`), main `9484ffc2`에서 10회 중 7회, #2057 머지 시점에서 10회 중 6회 실패했으며, 같은 오차 값이 실행 사이에 반복해서 나왔습니다. 2880x2880의 M=8, M=64 케이스와 더 작은 shape는 한 번도 실패하지 않았습니다.

커널이 의심받은 이유는 두 가지였습니다. K=2880은 커널의 2048개 원소 shared memory chunk보다 큰 유일한 테스트 K이므로 두 chunk와 tail을 거치는 유일한 shape입니다. 또 테스트는 seed를 한 번 정하고 고정된 순서로 입력을 뽑으므로, 명목상 고정된 입력이 매번 다른 출력을 내는 것처럼 보였습니다. 수정 전까지 `make verify-rocm`은 비결정적이었고, hosted CI가 멈춘 동안 에픽의 모든 PR은 이 게이트를 통과해 머지됩니다.

## 2. 조사

이슈의 계획은 입력이 고정인지 증명하고, 프로세스 안 재현을 만들고, 커널과 기준값 중 어느 쪽이 틀렸는지 가리는 것이었습니다. 조사는 그 계획을 따랐고, 이슈가 가능성이 낮다고 본 쪽에서 끝났습니다.

### 2.1 배제한 가설

| 가설 | 확인 방법 | 결과 |
|---|---|---|
| 입력 비결정성 | 실행마다 `w`, `packed`, `scales`, `x`의 hash 비교 | 모든 실행에서 동일 |
| `shared_x`의 chunk 경계 race | chunk 방식 qmv 커널(`qmv_warp_shared_kernel`과 batched, gather 변형) 읽기 | 각 chunk 로드 후와 다음 로드 전에 모두 `__syncthreads()` 있음 |
| tail chunk (2880 - 2048 = 832개 원소) | 같은 코드 읽기 | tail은 `k < chunk_end`만 로드하고 읽음 |
| wave32 불일치 | 같은 코드 읽기 | `THREADS_PER_COL=16`은 wave32에 들어감 |
| GPU 양자화기가 CPU 경로와 다름 | GPU와 CPU `dequantize` 비교, GPU packed/scales와 CPU 양자화기 비교 | 비트 단위로 동일 |

### 2.2 틀린 쪽은 기준값

실패하는 프로세스 안에서는 매 반복마다 같은 열이 틀렸습니다. `want`를 한 번 계산해서 재사용하기 때문입니다. 실행 사이에 달라지는 값이 GPU 출력이 아니라 기준값이라는 첫 신호였습니다. 실패한 실행에서 dump한 텐서로 보면, GPU 출력은 packed 바이트를 numpy로 독립적으로 dequantize하고 내적한 값과 정확히 일치했습니다. CPU `matmul`은 4개 열에서 틀렸습니다.

증상의 모양도 이것으로 설명됩니다. 모든 dtype의 기준값은 CPU 스트림의 f32 matmul이므로 테스트 dtype과 무관하게 `cblas_sgemm`을 거칩니다. 그래서 f32, f16, bf16이 모두 실패했습니다. 같은 오차 값이 반복된 것도 결정적인 열 위치에 떨어지는 결함과 맞습니다.

### 2.3 CPU 스트림에서 분리

GPU 커널 없이 CPU 스트림에서 `matmul(x, w.T)`만 반복한 결과:

- 호스트가 달리 idle일 때 60회 중 35회 틀림
- 다른 GPU 프로세스가 도는 동안 200회 중 12회 틀림. 동시 GPU 부하는 원인이 아니라 오히려 빈도를 낮춤
- `OPENBLAS_NUM_THREADS=1`에서 300회 중 0회 틀림

### 2.4 독립 C 재현

`[1, 2880] x [2880, 2880]^T`에 대해 `cblas_sgemm`을 직접 호출하는 C 프로그램으로, 버퍼 위치와 OpenBLAS 스레드 수를 바꿔 가며 측정했습니다(gfx1151 호스트, Ryzen AI MAX+ 395, Debian OpenBLAS 0.3.29 pthread).

| 입력 | 출력 | OpenBLAS 스레드 | 틀린 호출 |
|---|---|---|---|
| fine-grained (`hipExtMallocWithFlags(hipDeviceMallocFinegrained)`) | fine-grained | 32 (기본값) | 50회 중 37회 |
| malloc | fine-grained | 32 | 300회 중 10회 |
| malloc | malloc | 32 | 0 |
| `hipHostMalloc` | `hipHostMalloc` | 32 | 0 |
| fine-grained | fine-grained | 1, 2, 4 | 0 |

이것으로 MLX, GPU 커널, 테스트가 모두 빠집니다. 결함에는 멀티스레드 OpenBLAS와 fine-grained 메모리에 있는 출력, 두 가지가 필요합니다. 입력이 fine-grained 메모리에 있으면 빈도가 올라가지만 필수 조건은 아닙니다.

### 2.5 틀린 열의 위치

MLX를 통해 보면 틀린 열은 각각 한 OpenBLAS 스레드 구간의 마지막 열이었습니다. 구간은 93열이었고 틀린 열은 650, 1022, 1859 등이었습니다(650 = 7 x 93 - 1). f32 출력 열은 4바이트이므로 64바이트 라인 하나에 16열이 들어가고, 650열의 라인(640열부터 655열)은 651열에서 시작하는 다음 스레드도 씁니다. 발견된 틀린 열은 모두 이런 공유 라인에 있었습니다.

## 3. 변경 요약

- **`patches-rocm/mlx/backend/rocm/allocator.cpp`**: `extern "C" void openblas_set_num_threads(int) __attribute__((weak))`를 선언하고, symbol이 해석되면 `std::once_flag` 아래에서 한 번 호출하는 `single_thread_cpu_blas_for_finegrained()`를 추가합니다. `unified_malloc`은 `hipExtMallocWithFlags(..., hipDeviceMallocFinegrained)`가 성공한 직후 이를 호출합니다. 주석에 측정값과 근거를 남겼습니다.
- **`tests/rocm_cpu_blas_finegrained.rs`** (신규): GPU에서 `w`와 `x`를 만들고 f64 호스트 기준값을 계산한 뒤, CPU 스트림에서 `[1, 2880] x [2880, 2880]^T` matmul을 16회 실행하고, 출력 scale 대비 1e-4를 넘게 틀린 열이 하나라도 있으면 실패합니다. 본문 전체에서 `lock_default_device`를 잡고, ROCm이 아닌 backend에서는 건너뜁니다.
- **`patches-rocm/LOCAL_FIXES.md`**: 런타임 목록의 항목 25 뒤에 항목 27. 동작 방식, 측정값, 비용, 택하지 않은 대안, #1813 upstream 메모를 담았습니다.
- **`docs/installation.md`** (ROCm 지원 표의 CPU device 행): CPU device의 BLAS 작업이 OpenBLAS 스레드 하나로 돈다는 점과 그 이유.

`tests/rocm_mxfp4_quant.rs`와 `qmm.hip`은 바뀌지 않았습니다. 버그를 드러낸 테스트는 기준값이 옳아졌기 때문에 이제 통과합니다.

## 4. 기술적 선택과 그 이유

### tolerance가 아니라 기준값 경로를 고침

이슈의 완료 조건은 tolerance 완화를 금지했습니다. 오차는 상대값 0.06에서 0.2로 반올림 오차 범위를 훨씬 벗어났으므로, bound를 넓혔다면 잡음이 아니라 CPU backend의 실제 잘못된 결과 버그를 숨겼을 것입니다.

### cacheable scratch 출력 대신 BLAS 스레드 하나

C 표를 보면 cacheable 출력이면 멀티스레드 OpenBLAS도 옳습니다. 따라서 대안은 BLAS에 malloc한 출력을 주고 결과를 fine-grained 버퍼로 복사하는 것이었습니다. MLX의 CPU backend는 여러 호출 지점(`cblas.cpp`, `conv.cpp`, `masked_mm.cpp`, LAPACK primitive)에서 BLAS와 LAPACK 결과를 배열 버퍼에 직접 쓰므로, 그 방법은 fork의 모든 호출 지점을 고쳐야 합니다. 스레드 하나는 한 곳에서 전부 고칩니다. C 재현에서 2와 4 스레드도 정확했지만 PR은 이를 MLX를 통해서나 다른 shape에서 측정하지 않았으므로, 근거가 있는 설정은 스레드 하나입니다.

### allocator에서, 첫 fine-grained 할당 때 설정

hook은 fine-grained 메모리가 만들어지는 곳에 있습니다. APU에서는 모든 배열이 이 경로에서 나오므로, 첫 fine-grained 할당은 그런 버퍼에 쓸 수 있는 어떤 CPU 스트림 BLAS 호출보다 먼저 일어납니다. fine-grained 경로를 타지 않는 discrete GPU 구성은 이를 호출하지 않고 OpenBLAS 스레드 수를 그대로 둡니다.

### weak symbol

`openblas_set_num_threads`는 OpenBLAS 전용입니다. weak로 선언하면 다른 BLAS로 링크한 빌드도 링크되고, 그때는 포인터가 null이라 hook이 아무것도 하지 않습니다.

## 5. 비용

- 재현에 쓴 gemv는 CPU 스트림에서 호출당 약 0.2초에서 약 1.4초가 되었습니다. CPU는 fine-grained 메모리를 느리게 읽고, 스레드가 하나면 그 느림이 가려지지 않습니다.
- `rocm_mxfp4_quant` target 전체가 8초에서 19초로 늘었습니다.
- 영향 없음: GPU 작업, 양자화 CPU matmul, bf16/f16 CPU matmul. 뒤의 둘은 BLAS가 아니라 MLX 자체 SIMD 커널을 씁니다. 그래서 `MLXCEL_DEVICE=cpu`의 모델 추론은 BLAS를 거의 쓰지 않습니다.
- 설정은 프로세스 전역입니다. 첫 fine-grained 할당 이후에는 같은 프로세스의 다른 OpenBLAS 사용자도 단일 스레드로 돕니다.

## 6. 검증

PR 작성자 (gfx1151):

- 측정 전에 HIP를 강제로 clean 재빌드했습니다(`hip_objs/`를 trash하고 reconfigure). main 기준선: 8회 중 3회 실패(오차 0.104, 0.173, 0.063).
- `tests/rocm_cpu_blas_finegrained.rs`는 allocator 변경을 되돌린 상태에서 5회 중 5회 실패했고(`allocator.cpp.o` timestamp로 확인, 모두 처음 8회 호출 안에서), 변경을 적용하면 6회 중 6회 통과했습니다.
- `cargo test --profile test-fast --features rocm --test rocm_mxfp4_quant -- --test-threads=1 quantized_matmul`이 50회 연속 통과했습니다. 49회는 시작 시점에 다른 KFD 프로세스가 보였고(매번 새 PID, 직전 실행의 프로세스가 종료 중이었을 가능성이 큼), 따라서 실행이 엄밀히 격리되지는 않았습니다.
- `rocm_mxfp4_quant` 전체(테스트 4개)가 통과했습니다. 새 테스트에 대한 `-D warnings` clippy, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`, `dead_doc_pointers`가 통과했습니다.
- gpt-oss-20b logit trace는 없습니다. 이슈는 커널이 원인일 때만 이를 요구했고, 커널 출력은 바뀌지 않았습니다.

orchestrator 검증 (gfx1151, `bf5bf525` 위로 rebase한 브랜치):

- rebase에서 `LOCAL_FIXES.md` 충돌이 났습니다. orchestrator는 #2076이 고친 항목 25, 그다음 런타임 목록에 이 PR의 항목 27, 그다음 #2076의 빌드 절과 항목 26 순서로 정리했습니다.
- `make verify-rocm`은 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke 실행(32 토큰)이 통과했습니다.
- `verify-test-rocm`은 정확히 알려진 기준선 target 세 개에서만 실패했습니다: `layers::tests::prefill_dense_gemm_matches_qmm_bytes_where_eligible`, 그리고 #2037의 `gelu_approx_matches_mlx_nn_bit_for_bit`와 `tests::family_order_is_exhaustive`. #2076의 게이트에서는 같은 단계가 `rocm_mxfp4_quant`에서도 실패했지만 여기서는 실패하지 않았습니다.
- 전체 suite 안에서 `tests/rocm_cpu_blas_finegrained.rs`가 통과했고 `tests/rocm_mxfp4_quant.rs`가 4개 중 4개 통과했습니다. mlxcel-core 라이브러리 테스트는 1835개 통과했습니다.

브랜치는 그 뒤 `929c80ab`(#2077) 위로 다시 rebase되었습니다. #2077은 paged KV cache와 서버 scheduler만 건드리며, 위 게이트는 그 base에서 다시 돌리지 않았습니다.

## 7. 학습 포인트

- **커널보다 기준값을 먼저 확인할 것.** 한 번 계산해서 재사용하는 기준값은, 불안정한 기준값을 불안정한 커널처럼 보이게 만듭니다. 한 실행 안에서는 매 반복 같은 틀린 값이고, 실행 사이에는 다른 틀린 값이기 때문입니다. 결정적인 검사는 dump한 packed 바이트에 대한 numpy 계산이라는 세 번째 독립 계산이었고, 그 결과는 GPU와 일치하고 CPU와는 달랐습니다.
- **fine-grained 메모리는 CPU에게 일반 메모리가 아니다.** 이 APU에서는 CPU 스트림이 쓰는 배열을 포함한 모든 배열이 allocator의 fine-grained 메모리를 받습니다. malloc한 메모리에서는 옳은 코드, 여기서는 인접 스레드가 같은 64바이트 라인에 쓰는 멀티스레드 OpenBLAS가, 그 메모리에서는 틀린 결과를 냈습니다. 이런 버퍼에 병렬로 쓰는 CPU 코드는 모두 같은 이유로 의심해야 합니다.
- **시스템을 독립 재현까지 줄일 것.** C `cblas_sgemm` 표는 질문을 "MLX의 어느 계층인가"에서 "어떤 버퍼 위치와 스레드 수인가"로 바꿨고, 그 행들이 수정(스레드 하나)과 택하지 않은 대안(cacheable 출력)을 그대로 뒷받침합니다.
- **orchestrator 자신의 잘못된 가설.** orchestrator는 처음에 실패를 입력 의존적이라고 보았고, 다음에는 커널의 chunk 방식 shared memory 루프의 race로 보았습니다. 또 커밋 사이 warm 트리 bisect를 시도했는데, 이는 무효였습니다. #2076이 #2075를 고치기 전에는 헤더만 바뀐 경우 warm ROCm 트리가 HIP object를 다시 컴파일하지 않았으므로, bisect 단계가 stale 커널을 돌릴 수 있었습니다. 두 가설은 입력을 hash하고 GPU 출력을 독립 decode와 비교하자 무너졌고, 이 PR의 재현 작업은 모두 HIP를 강제로 clean 재빌드한 상태에서 했습니다. 에픽에 주는 교훈은, GPU 증상을 bisect하기 전에 빌드가 각 단계의 내용을 실제로 컴파일하는지와 비교의 어느 쪽이 틀렸는지를 먼저 확인하라는 것입니다.
- **동시 부하는 원인이 아니라 빈도를 바꿨다.** CPU 루프는 다른 GPU 프로세스가 돌 때 덜 자주 실패했습니다(200회 중 12회 대 60회 중 35회). 배경 부하에 따라 움직이는 빈도는 경합 버그로 오해하기 쉽지만, 여기서는 idle 상태의 성질이었습니다.

## 8. 주의 사항과 검증하지 않은 것

- **하드웨어 메커니즘**은 밝혀지지 않았습니다. 근거는 위치(틀린 열이 64바이트 라인을 공유하는 스레드 구간 경계에 있음)와 조건(fine-grained 출력이 필요하고, 32 스레드에서 나타났으며 1, 2, 4 스레드에서는 나타나지 않음)입니다. 이 APU에서 fine-grained 메모리의 한 라인에 대한 동시 쓰기가 왜 데이터를 잃는지, 그리고 그것이 OpenBLAS 버전, 커널 드라이버, CPU 중 무엇에 달렸는지는 조사하지 않았습니다.
- **`rocm_mxfp4_quant`에서 왜 M=1만 실패했는지**는 살펴보지 않았습니다. 같은 K, N의 M=8과 M=64는 한 번도 실패하지 않았습니다.
- **2 또는 4 스레드**는 C 재현에서 정확했지만 MLX를 통해서는 시도하지 않았으므로, 비용이 덜한 스레드 수는 검증되지 않았습니다.
- **다른 BLAS와 LAPACK 경로**(convolution, 선형대수)는 같은 프로세스 전역 설정으로 덮이지만 개별로 테스트하지 않았습니다.
- **Metal과 CUDA**는 실행하지 않았습니다(이 호스트에서 사용할 수 없음). 변경은 `patches-rocm`에 한정되며 그 빌드들은 이를 컴파일하지 않습니다.
- **50회 완료 조건 루프**는 대부분의 실행 시작 시점에 다른 KFD 프로세스가 있는 상태에서 돌았습니다.

## 9. 남은 작업

- #1813: 다른 upstream 후보와 함께 항목 27을 fork에 제안.
- APU에서 CPU 스트림 f32 BLAS 속도가 중요해지면, cacheable scratch 출력(또는 측정을 거친 더 높은 스레드 수)이 기록된 대안입니다.
- 기준선 `verify-test-rocm` 실패 세 개(`prefill_dense_gemm_matches_qmm_bytes_where_eligible`와 #2037의 두 개)는 이 PR 밖에서 추적합니다.

참고: #2072 (이 PR로 close), #1801, #2075, PR #2076, #1808, PR #2057, PR #2071, #1813, #2037.
