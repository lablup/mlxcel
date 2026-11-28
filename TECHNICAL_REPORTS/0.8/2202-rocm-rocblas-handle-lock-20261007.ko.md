# 기술 보고서: PR #2202 - ROCm Device의 rocBLAS 핸들을 잠금 아래에서 한 번만 초기화

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 코드 헤드 `e337e6bf`(origin/main `33f60aa9` 위), PR 열림, 머지 대기. #2198을 닫음(#1801의 일부).

**언어**: C++ 및 HIP(ROCm 오버레이 `device.h`, `device.cpp`, `matmul.cpp`, `gemms/rocblas_gemm.cpp`, `quantized/qmm.hip`), Rust(신규 `tests/rocm_rocblas_handle_concurrency.rs`), Markdown(`patches-rocm/LOCAL_FIXES.md`)

**위험도**: 낮음에서 중간. 이 변경은 커널, GEMM 인자, 결과를 바꾸지 않습니다. 모든 rocBLAS 호출이 스트림 바인딩부터 GEMM enqueue까지 잡는 디바이스별 뮤텍스를 추가하므로, rocBLAS GEMM의 호스트 측 enqueue가 스레드 간에 직렬화됩니다. 이 비용은 측정하지 않았습니다. Metal과 CUDA 코드 경로는 건드리지 않았습니다.

## 요약

이슈 #2198은 ROCm 오버레이 `Device`의 잠금 없는 구간 두 곳을 보고했습니다. 첫째, `get_rocblas_handle()`은 `rocblas_create_handle`을 호출하기 전에 초기화 플래그를 세우므로, 그 구간에 도착한 두 번째 스레드가 null 핸들을 받았습니다. 둘째, 모든 GEMM이 공유 핸들 하나에 잠금 없이 `set_rocblas_stream()`을 호출하므로, 스레드 A의 재바인딩과 A의 GEMM 사이에 스레드 B가 재바인딩하면 A의 GEMM이 B의 스트림에 올라갑니다. 서버는 하나의 `Device`를 공유하는 여러 워커 스레드에서 각자 스트림으로 forward를 실행하므로, 두 구간 모두 GPU 하나에서도 도달 가능합니다.

PR은 rocBLAS 핸들을 이동 전용 `RocblasLease`로만 접근하게 만듭니다. `Device::acquire_rocblas(stream)`은 `rocblas_mtx_`를 잡고, 첫 사용 시 핸들을 만들고, 마지막으로 바인딩한 스트림과 다를 때만 스트림을 바인딩하며(실패하면 예외), 소멸할 때까지 뮤텍스를 쥔 리스를 반환합니다. 원자적 `rocblas_ready_` 플래그는 create가 성공한 뒤에만 release로 저장합니다. create가 실패하면 한 번 경고하고 예외를 던지며, 다음 호출이 재시도합니다. rocBLAS 호출 지점 8곳 모두 리스를 잡고, `get_rocblas_handle()`은 private이며 `set_rocblas_stream()`은 사라졌으므로, 보호되지 않은 호출은 더 이상 컴파일되지 않습니다.

새 테스트는 실패를 결정적으로 재현합니다. 수정을 제거하면 20회 중 20회 실패했습니다(자식 프로세스 200개 중 187개). 수정이 있으면 20회 중 0회, 자식 프로세스 200개 중 0개가 실패했습니다. 크래시 가드에 그쳤던 #2196의 JIT 캐시 테스트와 다른 점입니다. 오케스트레이터의 `e337e6bf` `make verify-rocm`은 통과했습니다. 157 스위트, 12,050 통과, 0 실패, 385 무시입니다.

## 1. 문제 정의

### 1.1 핸들 초기화 경쟁

`33f60aa9` 시점 main에서 `Device::get_rocblas_handle()`은 먼저 `rocblas_initialized_ = true`를 세우고, 그다음 아키텍처 검사와 `rocblas_create_handle`을 실행했습니다. 그 구간에 도착한 두 번째 스레드는 플래그를 보고, `rocblas_available_`이 아직 기본값 `true`인 것을 보고, 아직 `nullptr`인 `rocblas_`를 반환했습니다. 두 스레드가 어느 쪽도 쓰기 전에 플래그를 읽으면 둘 다 핸들을 만들고 하나는 누수됩니다. 여러 호출 지점이 GEMM 상태를 버리므로(예를 들어 f32 `rocblas_sgemm` 지점), null 핸들은 오류 없이 출력을 쓰지 않은 채로 남겼습니다.

`is_rocblas_available()`은 hipBLASLt가 처리하지 않는 첫 f32 GEMM과 모든 affine 양자화 matmul 경로 결정에서 실행되므로, 프로세스의 첫 양자화 forward가 핸들을 만듭니다. `-m <chat> --embedding-model <emb>`에서 두 모델에 동시에 도착한 첫 요청이 초기화에서 경쟁하며, 핸들을 먼저 만들어 주는 시작 시 워밍업 forward는 없습니다.

### 1.2 스트림 재바인딩 경쟁

모든 rocBLAS 호출 지점은 `set_rocblas_stream(stream)`을 호출한 다음 GEMM을 실행했습니다. rocBLAS는 호출마다가 아니라 핸들에 스트림을 바인딩하고, 오버레이는 디바이스마다 핸들 하나를 둡니다. 스레드 A의 재바인딩과 A의 GEMM 사이에 스레드 B가 재바인딩하면 A의 GEMM이 B의 스트림에 올라가고, A의 스트림은 그것을 기다리지 않으므로 A가 GEMM 실행 전에 출력을 읽을 수 있습니다. 첫 GEMM만이 아니라, 두 워커 스레드가 rocBLAS GEMM을 실행하는 동안의 모든 rocBLAS GEMM에서 열려 있습니다.

### 1.3 probed 플래그

`has_native_wmma()`와 `supports_cdna_mfma_gemm()`은 결과를 계산하기 전에 probed 플래그를 세우므로, 동시 호출자가 한 번 `false`를 읽고 non-WMMA 경로를 탔습니다. `is_rocblas_bf16_available()`은 호출자가 없었고, 복구 경로가 `hipDeviceReset()`을 호출하고 다른 스레드가 쓰는 중에 공유 핸들을 다시 만들었습니다. 둘 다 이 수정에 포함됩니다(3.3절).

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `patches-rocm/mlx/backend/rocm/device.h` | 이동 전용 `RocblasLease` 신설(`std::unique_lock<std::mutex>`와 핸들, `handle()` 접근자). `Device::acquire_rocblas(hipStream_t)` 추가. `get_rocblas_handle()`은 "호출자가 `rocblas_mtx_`를 잡는다"는 계약과 함께 private으로. 새 멤버 `std::mutex rocblas_mtx_`, `std::atomic<bool> rocblas_ready_`, `rocblas_warned_`, `rocblas_arch_supported_`, `arch_name_`. `has_native_wmma()`와 `supports_cdna_mfma_gemm()`은 `const` 인라인 getter. `set_rocblas_stream()`, `is_rocblas_bf16_available()`, probed와 bf16 멤버 제거. |
| `patches-rocm/mlx/backend/rocm/device.cpp` | 아키텍처 사실을 생성자가 이미 읽는 `hipDeviceProp_t`에서 `Device::Device`에서 한 번 계산(아키텍처 목록은 같고 파일 로컬 헬퍼로). 핸들은 뮤텍스 아래에서 생성하고, create 성공 뒤 `rocblas_ready_`를 release로 저장. `acquire_rocblas`는 바뀐 경우에만 스트림 바인딩. `is_rocblas_available()`은 ready이면 잠금 없이 응답. |
| `matmul.cpp`(람다 3개) | 각 람다가 `auto lease = device.acquire_rocblas(stream); rocblas_handle handle = lease.handle();`로 시작. 바깥의 핸들 획득 두 곳 삭제. |
| `gemms/rocblas_gemm.cpp`(람다 3개), `quantized/qmm.hip`(람다 2개) | 같은 두 줄. `dequant_rocblas_gemm`에서는 hipBLASLt 시도 뒤에 리스를 잡으므로 hipBLASLt GEMM은 리스를 쥐지 않음. |
| `tests/rocm_rocblas_handle_concurrency.rs`(신규) | 새 프로세스 동시성 테스트(4절). |
| `patches-rocm/LOCAL_FIXES.md` | 항목 37. |

커밋 하나, 파일 7개, 506줄 추가, 269줄 삭제입니다. 각 람다의 `handle` 지역 별칭 덕분에 rocBLAS 호출의 인자 목록은 그대로이고, 리스는 람다 안의 모든 호출보다 오래 삽니다.

## 3. 설계

### 3.1 리스

`Device`마다 핸들 하나를 `rocblas_mtx_` 아래에서 한 번 만들고, 모든 rocBLAS 호출은 이미 스트림을 바인딩한 리스를 통해 그 뮤텍스를 쥔 채로 합니다.

- **바인딩과 enqueue가 하나의 임계 구역.** 리스는 `acquire_rocblas`부터 스코프 종료까지 뮤텍스를 쥐므로, 이 스레드의 바인딩과 GEMM 사이에 다른 스레드가 핸들을 재바인딩할 수 없습니다. 재바인딩 경쟁이 닫힙니다.
- **바뀔 때만 재바인딩.** `acquire_rocblas`는 `rocblas_stream_`과 비교해 스트림이 다를 때만 `rocblas_set_stream`을 호출합니다. 실패하면 GEMM을 엉뚱한 스트림에서 실행하지 않고 상태를 담은 `std::runtime_error`를 던집니다.
- **스택 해제 시 해제.** 잠금은 리스 안의 `std::unique_lock`이므로, 람다에서 예외가 나면 뮤텍스가 풀립니다.
- **컴파일 시점 강제.** `get_rocblas_handle()`은 private이고 `set_rocblas_stream()`은 제거되었으므로, 새 호출 지점은 `acquire_rocblas` 없이 핸들에 닿을 수 없습니다. `patches-rocm` 아래에서 `get_rocblas_handle`, `rocblas_set_stream`, `set_rocblas_stream`을 grep하면 `device.cpp`와 `device.h`만 나옵니다.
- **호스트 측 enqueue만 직렬화.** GEMM 자체는 GPU에서 각자 스트림으로 계속 겹쳐 실행됩니다.

### 3.2 초기화 프로토콜

`get_rocblas_handle()`은 뮤텍스 아래에서 실행됩니다. `rocblas_ready_`가 이미 세워져 있으면 핸들을 반환합니다. 지원하지 않는 아키텍처에서는 예외를 던집니다. 그렇지 않으면 `rocblas_create_handle`을 호출하고, 실패하면 경고를 한 번 출력하고(`rocblas_warned_`로 보호), 아무것도 캐시하지 않고 예외를 던지며, 다음 호출이 재시도합니다. 성공하면 핸들을 저장하고, `rocblas_stream_`을 null로 되돌린 뒤(새 핸들은 null 스트림에 바인딩되어 있으므로 첫 리스가 재바인딩), `rocblas_ready_`를 release로 저장합니다.

이전에는 create 실패가 프로세스 전체에서 rocBLAS를 비활성화했습니다. 이제 일시적 실패는 재시도되고, 실패는 호출자에게 null 핸들을 돌려주는 대신 예외가 됩니다.

`is_rocblas_available()`은 `rocblas_ready_`를 acquire로 읽어, 핸들이 있으면 뮤텍스 없이 true를 반환하므로, 리스를 쥔 스레드가 교착 없이 호출할 수 있습니다. 그렇지 않으면 잠그고 create를 시도해 성공 여부를 보고합니다.

### 3.3 생성 시점의 아키텍처 사실

`rocblas_arch_supported_`, `has_native_wmma_`, `cdna_mfma_ok_`는 생성자가 이미 읽는 `hipDeviceProp_t`에서 `Device::Device`가 한 번 계산하며, 아키텍처 목록은 이전과 같습니다. `has_native_wmma()`와 `supports_cdna_mfma_gemm()`은 `const` getter가 되어, probed 플래그와 그에 대한 경쟁이 사라집니다.

이로 인해 지원하지 않는 아키텍처 경고가 생성 시점으로 옮겨 갑니다. 이슈 본문은 경고가 첫 사용 시 출력되는 것으로 적혀 있었지만, 지원 여부를 생성 시점에 알게 되면 `is_rocblas_available()`이 지원하지 않는 디바이스에서 잠글 필요가 없으므로 경고는 `Device`가 만들어질 때 한 번 출력됩니다. 이슈 본문과 의도적으로 다른 점이며, 바뀐 것은 메시지가 나타나는 시점이지 나타나는 여부가 아닙니다.

`is_rocblas_bf16_available()`은 멤버와 함께 제거했습니다. `device.cpp`와 `device.h` 밖에서는 호출자가 없었고, 실패 경로가 `hipDeviceReset()`을 호출하고 공유 핸들을 다시 만들었는데 다른 스레드가 도는 중에는 안전하지 않습니다.

### 3.4 기각: 스레드별 핸들

핀 시점 upstream CUDA(MLX `81ba1c6a`, `mlx/backend/cuda/cublas_utils.cpp`)는 cuBLASLt 핸들을 `static thread_local std::vector<CublasHandles> cache(gpu::device_count())`에 두고, 각각 `CHECK_CUBLAS_ERROR`로 만들며, 스트림은 `cublasLtMatmul`에 호출마다 넘깁니다(`encoder.stream()`). 공유되는 것이 없습니다. PR은 이를 따르지 않았습니다.

- rocBLAS는 호출마다가 아니라 핸들에 스트림을 바인딩하므로 스레드별 핸들도 바인딩이 필요하고, 포크는 디바이스마다 핸들 하나를 전제로 만들어져 있습니다.
- 스레드별 핸들은 각 워커의 첫 GEMM에서 생성되는데, 그 시점이 디코드 스텝 스트림 캡처 안일 수 있습니다.

리스는 직렬화를 대가로 공유 핸들 하나를 택한 것입니다. 프로파일에서 `rocblas_mtx_` 경합이 보이면, 생성 시점을 앞당긴 스레드별 핸들이 대안입니다.

## 4. 검증

### 4.1 테스트

`tests/rocm_rocblas_handle_concurrency.rs`(`#![cfg(feature = "rocm")]`)는 `first_gemm_from_many_threads_in_fresh_processes`를 실행하며, 이 테스트는 `#[ignore]` 자식 테스트를 `MLX_NO_HIPBLASLT=1`로 새 프로세스 10개에서 실행합니다(GEMM이 rocBLAS에 머물게 하기 위함이며, #2200이 이를 없앱니다). 각 자식은 barrier 뒤에서 각자 스트림을 쓰는 스레드 8개를 시작합니다. 각 스레드는 입력이 `-4..=4` 정수인 f32 `[64, 128] x [128, 64]` matmul을 33번 실행하고, 모든 출력을 호스트 `i32` 기준값과 정확히 비교합니다. 초기화 구간이 프로세스당 한 번이므로 새 프로세스가 필요합니다.

### 4.2 20회 실행 표

gfx1151에서 빌드마다 테스트를 20회, 회마다 자식 프로세스 10개로 실행했습니다.

| 빌드 | 실패한 실행 | 실패한 자식 프로세스 |
|---|---:|---:|
| main의 오버레이(수정 제거), 새 테스트만 | 20회 중 20회 | 200개 중 187개 |
| 이 PR | 20회 중 0회 | 200개 중 0개 |

수정이 없을 때 실패한 스레드들 가운데 354개는 출력이 0으로 남았고(null 핸들에서 GEMM 실행, 상태 버림), 86개는 부분 GEMM 하나가 모자란 합을 보고했습니다(다른 스레드의 스트림에 올라간 GEMM이 실행되기 전에 읽힘). 이 두 징후는 1.1절과 1.2절의 두 경쟁과 일치합니다.

크래시 가드에 그쳤던 #2196의 JIT 캐시 테스트와 달리, 이 테스트는 수정이 없으면 매 실행 실패를 재현하므로 리스의 회귀는 이 호스트에서 잡힐 것입니다.

### 4.3 게이트

- `cargo clippy --features rocm --test rocm_rocblas_handle_concurrency -- -D warnings`: 깨끗함.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: 통과.
- 오케스트레이터의 `e337e6bf`(main `33f60aa9` 위) `make verify-rocm`: 테스트 스위트 157개, 12,050 통과, 0 실패, 385 무시. clippy, smoke, 오버레이, 스크립트 게이트 OK.

검증하지 않은 것: Metal과 CUDA(이 호스트에 없음, 변경은 `patches-rocm/` 아래 파일과 `rocm` 게이트 테스트에만 닿음), 그리고 리스의 성능 비용.

## 5. LOCAL_FIXES 항목 37

항목은 `src/lib/mlx-cpp/patches-rocm/LOCAL_FIXES.md`의 "Fixes to the fork's kernels" 아래에 들어갑니다. 작업 전에 플래그를 세우는 초기화, 재바인딩 경쟁과 그 결과(null 핸들, 엉뚱한 스트림), probed 플래그, 생성 시점 아키텍처 사실과 경고 이동, 초기화 프로토콜, 리스와 직렬화, private 접근자, 제거된 bf16 probe, upstream 비교와 스레드별 핸들을 기각한 이유, 그리고 20회 중 20회 실패 수치를 가진 테스트를 기록합니다. 포크 정책 문구로 끝납니다. "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2198)." `docs/mlxcelverse/upstream/` 아래에는 아무것도 추가하지 않았습니다.

## 6. 기술적 선택과 그 이유

- **`set_rocblas_stream`만 잠그지 않고 리스 하나, 뮤텍스 하나.** 경쟁은 바인딩과 GEMM 사이에 있으므로 잠금이 둘 다 덮어야 합니다.
- **보호되지 않은 경로를 불가능하게.** private getter와 공개 재바인딩 없음은 다음 보호되지 않은 호출을 리뷰 사항이 아니라 컴파일 오류로 만듭니다.
- **실패하면 예외, 다음 호출에서 재시도.** 상태를 버리는 호출 지점에 null 핸들을 돌려주면 조용히 틀린 출력이 나오지만, 예외는 그렇지 않습니다. 재시도는 일시적 실패 한 번으로 프로세스 전체의 rocBLAS가 꺼지는 것을 막습니다.
- **아키텍처 사실은 생성 시점에 계산.** 디바이스의 불변 속성이므로, 지연 플래그를 없애면 잠금 없이 그 경쟁이 사라집니다.
- **`dequant_rocblas_gemm`에서는 hipBLASLt 시도 뒤에 리스.** hipBLASLt GEMM은 rocBLAS 핸들을 건드리지 않으므로 그 뒤에 줄 설 이유가 없습니다.
- **스레드별 핸들 대신 리스.** 3.4절.
- **호출되지 않는 bf16 probe 제거.** 복구 경로가 안전하지 않은 죽은 코드입니다.

## 7. 후속 이슈

- **#2200, hipBLASLt 공유 상태.** 시리즈의 다음 단위입니다. 디바이스마다 32 MB 워크스페이스 하나를 모든 스트림이 공유하고, `ensure_workspace`는 잠금 없이 해제하고 재할당하며, 초기화가 핸들과 워크스페이스가 준비되기 전에 `initialized`를 공개하고, 캐시된 파이프 matmul이 실패하면 다른 스레드가 아직 쓰고 있을 수 있는 `GemmPipe`를 파괴합니다. 이 테스트의 `MLX_NO_HIPBLASLT=1`도 없애므로, 그 뒤에는 테스트가 기본 bf16과 fp16 경로도 다룹니다.
- **#2197, `rocm::device()` 맵.** 프로세스 전역 맵에 대한 잠금 없는 `find`/`try_emplace`이며, 잠재적이고 멀티 GPU에서만 발생합니다.
- **낡은 LOCAL_FIXES 도입부 문구.** `LOCAL_FIXES.md`의 도입부 문구가 오래되었습니다. #2144가 이를 정리하며, 이 PR은 건드리지 않습니다.

## 8. 남은 위험

- **잠금 비용 미측정.** 이제 모든 rocBLAS GEMM이 호스트 측 enqueue 동안 `rocblas_mtx_`를 잡으므로, 여러 워커의 동시 rocBLAS enqueue가 직렬화됩니다. GPU에서 계속 겹쳐 실행되는 GEMM 실행 시간에 비하면 작을 것으로 예상하지만, 예상일 뿐 측정이 아닙니다.
- **리스의 범위는 람다.** 핸들을 리스 밖으로 저장하거나 리스가 파괴된 뒤 `lease.handle()`을 호출하는 미래의 호출 지점은 경쟁을 되살립니다. 헤더 주석이 이 계약을 적어 둡니다.
- **hipBLASLt와 디바이스 맵은 아직 열려 있습니다.** #2200과 #2197이 들어가기 전까지 멀티 모델 ROCm 서빙은 여전히 hipBLASLt 상태에서 경쟁할 수 있습니다.
- **호스트 하나.** 모든 실행이 gfx1151 한 대에서 이뤄졌습니다.

## 9. 학습 포인트

- **스트림 바인딩만 잠갔다면 고쳐지지 않았을 것입니다.** 실패는 바인딩과 사용 사이의 틈이므로 보호할 단위는 그 쌍이고, 리스가 이를 표현합니다.
- **프로세스당 한 번의 초기화에는 새 프로세스 테스트가 필요합니다.** 오래 사는 테스트 바이너리에서 스레드를 돌리면 초기화 구간을 기껏해야 한 번 맞힙니다. 회마다 자식 10개를 쓰면 드문 구간이 횟수로 바뀝니다.
- **실패 집계에서 징후를 나누십시오.** 0으로 남은 출력과 부분 GEMM 하나가 모자란 합은 각각 null 핸들과 엉뚱한 스트림에 대응하며, 이것이 두 경쟁이 모두 실재했고 모두 닫혔음을 테스트가 보이는 방식입니다.
- **버려진 GEMM 상태가 잘못된 핸들을 숨깁니다.** 예전 null 핸들은 여러 지점에서 오류를 내지 않았습니다. `acquire_rocblas`에서 예외를 던지면 실패가 보이는 곳으로 옮겨집니다.
- **안전한 경로를 유일한 경로로 만드십시오.** `get_rocblas_handle()`을 숨기고 `set_rocblas_stream()`을 지우면, 다음 rocBLAS 호출 지점이 잠금을 건너뛸 수 없습니다.
