# 기술 보고서: PR #2208 - hipBLASLt GEMM 상태를 스레드와 스트림 간에 안전하게 만들기

**날짜**: 2026-10-07

**상태**: gfx1151 호스트(ROCm/HIP)에서 구현 및 검증 완료. 코드 헤드 `df8bc628`(origin/main `5976af8d` 위), PR 열림, 머지 대기. #2200을 닫음(#1801의 일부).

**언어**: C++ 및 HIP(ROCm 오버레이 `gemms/hipblaslt_gemm.cpp`, `gemms/hipblaslt_gemm.h`, `device.cpp`), Rust(신규 `tests/rocm_hipblaslt_concurrency.rs`, 수정 `tests/rocm_rocblas_handle_concurrency.rs`), Markdown(`patches-rocm/LOCAL_FIXES.md`)

**위험도**: 중간. GEMM 인자와 결과는 바뀌지 않지만, 파이프 캐시가 프로세스 전역 맵 하나에서 스레드별 맵으로 바뀌고, 워크스페이스 버퍼가 디바이스당 하나에서 (디바이스, 스트림)당 하나로 바뀌며 `~CommandEncoder`가 해제합니다. 테스트 호스트에서는 휴리스틱이 워크스페이스를 요청하지 않았으므로, 스트림별 워크스페이스 경로는 실행 시점에 검증되지 않았고 구조상으로만 보장됩니다. Metal과 CUDA 코드 경로는 건드리지 않았습니다.

## 요약

이슈 #2200은 bf16과 fp16 GEMM의 기본 경로인 hipBLASLt GEMM 계층에서 #2198과 같은 부류의 경쟁을 찾았습니다. `gemms/hipblaslt_gemm.cpp`는 디바이스별 핸들을 `hipblasLtCreate` 실행 전에 세우는 일반 `bool`로 공개했고, 디바이스의 모든 스트림에 32 MB 워크스페이스 하나를 공유시켰으며 `ensure_workspace`가 이를 잠금 없이 해제하고 재할당했습니다. GEMM 파이프(레이아웃, matmul 디스크립터, 알고리즘)는 프로세스 전역 맵 하나에 저장되어, 스레드들이 잠금을 푼 뒤에 사용하고 서로의 항목을 지웠습니다. 서버는 디바이스 하나를 공유하는 여러 스레드에서 각자 스트림으로 hipBLASLt GEMM을 enqueue합니다.

PR은 세 가지를 합니다. 핸들은 대상 디바이스를 current로 둔 채 상태 뮤텍스 아래에서 만들고, 성공 후 release로 원자적 초기화 상태를 통해 공개합니다. 워크스페이스는 (디바이스, 스트림)당 버퍼 하나이며, 첫 사용 시 32 MB 상한 크기로 할당하고 해당 스트림의 `CommandEncoder`가 소멸할 때 해제합니다. 파이프 캐시는 스레드 로컬이며 matmul이 성공한 뒤에만 삽입합니다. 마지막 항목은 `std::shared_ptr<GemmPipe>`를 제안한 이슈와 다릅니다. gfx1151에서 다섯 가지 실험을 한 결과 `hipblasLtMatmul`이 받은 디스크립터 객체에 쓰기를 하므로, 소유 방식과 무관하게 파이프 하나를 두 스레드가 동시에 쓸 수 없었습니다.

새 테스트는 실패를 재현합니다. `hipblaslt_gemm.cpp`를 main으로 되돌리면 20회 중 20회 실패했습니다(자식 프로세스 200개 중 50개 사망, 12회는 자식이 멈춤). 수정이 있으면 20회 중 0회, 자식 프로세스 200개 중 0개가 실패했습니다. 테스트한 형상에서 휴리스틱이 워크스페이스 0을 요청했으므로, 이 호스트에서는 스트림별 워크스페이스 경로가 실행 시점에 검증되지 않았습니다. Llama-3.1-8B-4bit 디코드 처리량은 잡음 범위 안에서 변하지 않았습니다. 오케스트레이터의 `df8bc628` `make verify-rocm`은 이 PR과 무관한 main의 테스트 하나(4.3절)를 제외하고 모두 통과했습니다.

## 1. 문제 정의

### 1.1 공유 상태 목록

`84d3a7bc` 시점 main 기준, 모두 `gemms/hipblaslt_gemm.cpp`에 있습니다.

| 상태 | 기존 보호 | 문제 |
|---|---|---|
| `g_state[d].initialized`, `.available`, `.handle` | `state.mutex` 아래에서 쓰고 잠금 없이 읽음 | `initialized`를 `hipblasLtCreate` 실행 전에 세움. 일반 `bool`이므로 데이터 경쟁. 이를 본 읽기 스레드가 `available == false`나 null 핸들을 볼 수 있음. |
| `g_state[d].workspace`, `.workspace_size` | 초기화는 뮤텍스 아래, `ensure_workspace`는 잠금 없음 | 디바이스의 모든 스트림이 버퍼 하나를 공유. 잠금 없는 `hipFree`와 `hipMalloc`. |
| `g_caps[d]` | 프로브 전체를 `g_caps_mutex`로 보호 | `probed`를 `get_handle` 전에 세움. 초기화 구간에서 이것이 예외를 던지면 프로세스가 끝날 때까지 모든 capability가 `false`로 남음. |
| `g_pipe_cache`와 각 `GemmPipe`이 소유한 디스크립터 | 맵 조회는 `g_pipe_mutex` 아래, 파이프는 잠금 해제 후 사용 | 캐시된 matmul이 실패하면, 다른 스레드가 `hipblasLtMatmul`에 넘기고 있을 수 있는 항목을 파괴하고 지움. |
| `algo_cache`(와 fp8용) | 자체 뮤텍스, 항목을 값으로 복사 | 안전. |
| `hipblasLtHandle_t` 자체 | 없음 | 안전. `hipblaslt.h`는 핸들을 바꾸는 헬퍼만 동기화하라고 하며, 이 파일은 create 이후 핸들을 바꾸지 않음. |

### 1.2 실패 양상

- **공유 워크스페이스(출력 손상).** `hipblasLtMatmul`은 enqueue만 하므로, 알고리즘이 스크래치 메모리를 요청하는 두 스트림의 GEMM 두 개가 같은 버퍼 위에서 GPU에서 동시에 실행됩니다. 스트림 순서는 한 스트림 안의 재사용만 보호합니다.
- **워크스페이스 증가(GPU use-after-free).** `ensure_workspace`가 다른 스트림의 대기 중인 GEMM이 아직 쓰는 버퍼를 해제할 수 있었습니다.
- **파이프 삭제(호스트 use-after-free).** 캐시된 matmul이 실패하면(오래된 버킷 알고리즘) 스레드가 항목의 레이아웃과 디스크립터를 파괴하고 지우는데, 다른 스레드가 그것을 들고 있을 수 있었습니다. 재시도 경로는 `algo_cache`만 고치고 나쁜 알고리즘을 파이프에 남겼으므로, 같은 키의 다음 호출이 다시 실패하고 지웠습니다.
- **초기화 구간(잘못된 경로).** 플래그와 create 사이에 도착한 스레드는 `available == false`를 보고 그 GEMM을 rocBLAS로 넘겼습니다. `gemm_caps` 안에서는 예외 때문에 capability 표가 프로세스 내내 전부 `false`로 남아 fp8 prefill 경로가 꺼졌습니다.
- **디바이스가 current가 아님.** `hipblasLtCreate`가 `device_id`를 current로 두지 않고 실행됐습니다.

gfx1151에서는 휴리스틱이 워크스페이스를 요청하지 않으므로 워크스페이스 경쟁은 실행되지 않습니다(4.3절). 그곳에서 도달 가능한 것은 파이프 디스크립터와 초기화 구간입니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `gemms/hipblaslt_gemm.cpp` | `HipblasltState`가 핸들과 `std::atomic<int> init_state`를 가짐, `ensure_handle`. `stream_workspace`가 `ensure_workspace`를 대체. `std::unique_ptr<GemmPipe>`의 스레드 로컬 `GemmPipeCache`, `GemmPipe`는 디스크립터를 스스로 파괴하고 복사 불가. `gemm_caps`는 프로브 후에 probed 표시. 휴리스틱 miss 로깅. 파일 상단에 스레드 안전성 주석. |
| `gemms/hipblaslt_gemm.h` | 신규 `hipblaslt_release_stream_workspace(hipStream_t)`. |
| `device.cpp` | `~CommandEncoder`가 스트림 파괴 전에 이를 호출. |
| `tests/rocm_hipblaslt_concurrency.rs`(신규) | 새 프로세스 동시성 테스트(4.1절). |
| `tests/rocm_rocblas_handle_concurrency.rs` | `MLX_NO_HIPBLASLT=1` 제거. |
| `patches-rocm/LOCAL_FIXES.md` | 항목 38. |

커밋 하나, 파일 6개, 665줄 추가, 147줄 삭제입니다.

## 3. 설계

### 3.1 핸들 초기화

`HipblasltState`는 핸들, 뮤텍스, `init_state`(미시도, 준비됨, 사용 불가)를 가집니다. `ensure_handle`은 acquire 로드 값이 0이 아니면 바로 반환합니다. 그렇지 않으면 `state.mutex`를 잡고 다시 확인한 뒤, 현재 디바이스를 저장하고 `device_id`를 current로 설정하고, 로컬 변수에 `hipblasLtCreate`를 호출하고, 디바이스를 복원합니다. 성공했을 때만 `state.handle`을 저장하고 그다음 `init_state`를 release로 저장합니다. `get_handle()`과 `is_hipblaslt_available()`은 acquire 로드가 준비됨을 알려준 뒤에만 핸들을 읽습니다. create 실패는 한 번 경고하고 이전처럼 프로세스 내내 사용 불가로 남습니다. `is_hipblaslt_available()`이 모든 GEMM 경로 결정에서 실행되므로 매번 create를 재시도해서는 안 되기 때문입니다. 초기화가 하던 워크스페이스 할당(디바이스당 32 MB)은 없어졌습니다.

`gemm_caps`는 이제 프로브가 끝난 뒤에만 `probed`를 표시합니다. 일찍 `probed = true`를 두는 곳은 `get_handle`이 예외를 던진 경로 하나뿐이며, 그것은 create가 실패한 뒤에만 일어나고 그 실패는 영구적입니다. 잠금 순서는 `g_caps_mutex`, 그다음 `state.mutex`로 그대로입니다.

### 3.2 스트림별 워크스페이스

`stream_workspace(device_id, stream, required)`는 `g_ws_mutex` 아래에서 스트림의 버퍼와 크기를 반환합니다.

- `required == 0`이면 잠금 없이 `{nullptr, 0}`을 반환하고, 32 MB 상한을 넘는 요청은 예외를 던집니다.
- 버퍼가 없는 스트림은 `device_id`를 current로 두고 정확히 `kMaxWorkspaceBytes`를 할당받습니다. 파일의 모든 휴리스틱 preference가 이 크기로 상한을 두므로, 스트림이 살아 있는 동안 버퍼는 커지지도 이동하지도 않으며, 증가 경쟁이 사라집니다.
- 캡처 중인(`hipStreamGetCaptureInfo`가 `None`이 아닌) 스트림에 버퍼가 없으면 캡처 안에서 `hipMalloc`을 부르는 대신 예외를 던집니다.
- `hipMalloc`이 실패하면 `{nullptr, 0}`을 반환합니다. 튜닝 루프는 그 알고리즘을 건너뛰고, 다른 모든 지점은 null 버퍼를 `hipblasLtMatmul`에 넘기는 대신 `hipBLASLt: failed to allocate workspace of N bytes`를 던집니다.

`hipblaslt_release_stream_workspace(stream)`은 스트림의 버퍼를 찾아 항목을 지우고, 스트림을 동기화한 뒤 버퍼를 해제합니다. `~CommandEncoder`가 스트림이 파괴되기 전에 이를 호출하므로, 같은 핸들 값을 재사용하는 이후 스트림이 대기 중인 GEMM이 아직 읽고 있는 버퍼를 물려받지 않습니다. 테이블은 JIT 모듈 캐시(항목 35)처럼 의도적으로 누수시켜, 정적 소멸 시점이 인코더 소멸자와 경쟁하지 않게 합니다.

비용은 워크스페이스가 필요한 GEMM을 실행하는 스트림당 32 MB입니다. 휴리스틱이 워크스페이스를 요청하지 않는 디바이스는 아무것도 할당하지 않고, 초기화가 하던 디바이스당 32 MB는 없어졌습니다.

### 3.3 파이프 캐시

`thread_pipe_cache()`는 OS 스레드마다 자체 `unordered_map<GemmPipeKey, unique_ptr<GemmPipe>>`를 줍니다(비동기 MoE 경로의 HIP 호스트 콜백 스레드도 하나를 가집니다). 따라서 빠른 경로에 잠금이 없습니다. 스레드의 맵은 누수시켜, 종료 중 스레드가 끝나면서 이미 언로드된 hipBLASLt를 호출하는 일이 없게 합니다.

- **성공 후 삽입.** miss는 파이프를 로컬에서 만들고 matmul과 재시도를 실행한 뒤, matmul이 성공했을 때만 성공한 알고리즘과 함께 삽입합니다. 그래서 재시도에서 새로 얻은 알고리즘이 오래된 것을 대체합니다. 이전에는 오래된 것이 캐시에 남았습니다.
- **실패 시 삭제.** 캐시된 matmul이 실패하면 스레드 자신의 맵에서 항목을 지워 디스크립터를 파괴합니다. 다른 스레드는 그것을 들고 있지 않습니다.
- **휴리스틱은 계속 공유.** `algo_cache`가 휴리스틱 결과를 값으로 프로세스 전역에 유지하므로, 스레드가 형상을 처음 쓸 때의 비용은 `AlgoGetHeuristic`이 아니라 디스크립터 생성뿐입니다.

### 3.4 디버그 로깅

`MLX_ROCM_GEMM_DEBUG=1`은 이제 휴리스틱 miss마다 `[hipBLASLt algo] MNK=... batch=... algos=... workspace=...`를 출력하므로, 호스트가 자신의 형상에서 스트림별 워크스페이스 경로가 실행되는지 알 수 있습니다. 4.3절의 워크스페이스 0 결과는 이렇게 읽었습니다.

### 3.5 기각된 대안

- **스레드별 핸들**(고정 버전의 upstream CUDA, `mlx/backend/cuda/cublas_utils.cpp`): `algo_cache`와 파이프 캐시가 한 핸들에서 계산한 휴리스틱 결과를 스레드 간에 공유하고, 비동기 MoE 경로가 HIP 호스트 콜백 안에서 핸들을 만들게 됩니다.
- **`hipblasLtMatmul`에 걸친 리스**: 호출은 커널이 실행되기 전에 반환하므로, 리스로는 두 스트림의 커널이 한 버퍼를 쓰지 못하게 할 수 없습니다. 안전하게 만들려면 스트림 동기화(캡처 중 불가)나 무관한 스트림 사이의 이벤트 인계가 필요하며, 어느 쪽이든 GEMM을 직렬화합니다.
- **스레드 로컬 워크스페이스**: HIP 콜백 스레드가 모든 스트림의 비동기 MoE GEMM을 실행합니다.
- **upstream처럼 호출마다 할당자 임시 버퍼**: `hipblaslt_gemm_raw`, `hipblaslt_gemm_fp8_raw`, `hipblaslt_gemm_rowmajor_on_stream`은 `CommandEncoder`가 아니라 `hipStream_t`만 받고, 호출마다 `hipMalloc`하는 것은 캡처할 수 없습니다.

## 4. 검증

### 4.1 테스트

`tests/rocm_hipblaslt_concurrency.rs`(`#![cfg(feature = "rocm")]`)의 `hipblaslt_gemms_from_many_threads_in_fresh_processes`는 `MLX_NO_HIPBLASLT`와 `MLX_HIPBLASLT_NO_PIPE_CACHE`를 환경에서 제거한 새 프로세스에서 `#[ignore]` 자식 테스트를 10번 띄웁니다. 각 자식은 프로세스에서 GPU matmul이 한 번도 실행되기 전에, 자신의 스트림을 가진 스레드 8개를 배리어 뒤에서 시작합니다. 각 스레드는 bf16 TN `[64, 64] x transpose([64, 64])`, bf16 배치 `[4, 64, 64] x [4, 64, 64]`, f32 rocBLAS `[64, 128] x [128, 64]`를 33라운드 실행하며, 모든 스레드가 같은 형상을 써서 같은 파이프 키를 두고 경쟁합니다. 모든 출력을 호스트 `i32` 참조값과 정확히 비교하고, 부모는 자식마다 `[hipBLASLt caps] device` 줄이 정확히 한 번 나오는지 요구합니다. 이는 bf16 GEMM이 hipBLASLt에 도달했음을 증명합니다.

### 4.2 이슈와 다른 점

**이탈 1: `shared_ptr`가 아닌 스레드 로컬 파이프.** 이슈는 `g_pipe_cache`에 `std::shared_ptr<GemmPipe>`를 제안했습니다. 먼저 이를 구현했고 실패했습니다. gfx1151(hipBLASLt 1.4)에서 스레드 8개가 같은 형상을 enqueue했고, 변형마다 자식 프로세스 5개를 돌렸습니다.

| 변형 | 통과한 자식 |
|---|---:|
| 이슈의 설계: `shared_ptr`로 파이프 공유 | 5개 중 0개 |
| 디스크립터 공유, 알고리즘은 복사 | 5개 중 0개 |
| 파이프 캐시 우회(`MLX_HIPBLASLT_NO_PIPE_CACHE=1`, 같은 핸들에 호출마다 디스크립터) | 5개 중 5개 |
| rocBLAS만(`MLX_NO_HIPBLASLT=1`) | 5개 중 5개 |
| 스레드 로컬 파이프 | 5개 중 5개, 이어서 200개 중 200개 |

실패한 변형에서는 힙 손상(`malloc(): unaligned tcache chunk detected`), double free, `SIGSEGV`, `HSA_STATUS_ERROR_MEMORY_FAULT`가 나타났습니다. 통과한 변형은 핸들과 `algo_cache`는 공유하지만 디스크립터 객체는 하나도 공유하지 않습니다. 결론은 `hipblasLtMatmul`이 받은 디스크립터 객체를 변경하므로, 소유권을 공유하든 아니든 파이프 하나를 두 스레드가 동시에 쓸 수 없다는 것입니다. hipBLASLt 소스는 확인하지 않았으므로, PR은 어떤 필드에 쓰는지 특정하지 않습니다. `shared_ptr`에 대한 이슈의 인수 조건 항목은 이 측정에 의해 대체된 것으로 표시했습니다.

**이탈 2: bf16 테스트 입력을 `-1..=1`로.** 이슈와 테스트 초안은 `-2..=2`였습니다. 정확 비교를 만들다가 #2206이 드러났습니다. gfx1151에서 bf16 입력이 0과 크기 2를 섞으면 bf16 GEMM이 정확히 0이어야 할 출력에 약 2^-22의 잔차를 돌려줍니다. 단일 스레드에서 발생하고 hipBLASLt와 rocBLAS 모두에서 나타나며, `{-1, 0, 1}`과 `{-2, -1, 1, 2}`는 정확합니다. bf16 출력 반올림이 0이 아닌 출력에서는 이를 가리므로 정확 비교만 이를 잡습니다. 테스트는 #2206이 해결될 때까지 정확성을 유지하도록 `-1..=1`을 씁니다.

### 4.3 결과

gfx1151(ROCm 7.15)에서 빌드마다 테스트를 20회, 회당 자식 프로세스 10개로 실행했습니다.

| 빌드 | 실패한 실행 | 실패한 자식 프로세스 |
|---|---:|---:|
| `hipblaslt_gemm.cpp`를 main으로 복원(수정 제거) | 20회 중 20회 | 200개 중 50개 |
| 이 PR | 20회 중 0회 | 200개 중 0개 |

수정이 없을 때 사망한 자식 50개는 `SIGABRT` 28개(tcache와 double free 중단), `SIGSEGV` 19개, 출력이 쓰이지 않았거나 `HSA_STATUS_ERROR_MEMORY_FAULT`인 경우 3개였습니다. 12회에서는 자식이 120초 예산을 넘겨 멈추기도 했습니다(한 스레드가 오지 않는 GPU 완료를 계속 기다림). `tests/rocm_rocblas_handle_concurrency.rs`는 이제 `MLX_NO_HIPBLASLT`를 설정하지 않고 통과합니다.

**워크스페이스 경로는 실행되지 않음.** 휴리스틱은 두 bf16 테스트 형상(`MNK=64,64,64`, TN 배치 1과 NN 배치 4, 각각 알고리즘 8개)에 `workspace=0`을 돌려줬고, 이슈의 자체 프로브도 15개 형상의 모든 후보에서 0이었습니다. 따라서 gfx1151에서는 스트림별 워크스페이스 경로가 실행되지 않으며 테스트는 이를 구조상으로만 보호합니다. 그곳에서 테스트가 닿는 것은 공유 파이프 디스크립터와 초기화 구간입니다. 라이브러리가 전역 스크래치를 쓰는 split-K 알고리즘을 고르는 아키텍처(CDNA, gfx1201)는 측정하지 않았습니다.

**디코드, 변화 주장 없음.** Meta-Llama-3.1-8B-Instruct-4bit, 512토큰 프롬프트, 128토큰 생성, 두 릴리스 바이너리를 `scripts/rocm_gpu_guard.sh` 세션 하나에서 연달아 실행했습니다.

| 빌드 | tok/s | 중앙값 |
|---|---|---:|
| 이전(main의 `hipblaslt_gemm.cpp`) | 37.96, 37.97, 38.02 | 37.97 |
| 이후 | 37.89, 37.91, 37.92 | 37.91 |

중앙값은 0.2% 차이이고, 별도의 가드 세션에서 같은 이후 바이너리가 35.63 tok/s를 보였으므로 세션 간 편차는 약 6%입니다. 4비트 모델의 디코드는 M = 1에서 hipBLASLt에 도달하지 않으므로 변화를 기대하지 않았습니다.

### 4.4 게이트

- `cargo clippy --features rocm --test rocm_hipblaslt_concurrency --test rocm_rocblas_handle_concurrency -- -D warnings`: 경고 없음. `grep -n "ensure_workspace\|GemmPipe\*\|initialized" hipblaslt_gemm.cpp`: 결과 없음.
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay`: 통과.
- 오케스트레이터의 `df8bc628`(main `5976af8d` 위로 리베이스) `make verify-rocm`: 158 스위트, 12,104 통과, 1 실패, 386 무시. 실패는 `server::routes::props::props_tests::geometry_block_reports_the_resolved_batch_size_alias`입니다. #2205(기본 prefill 청크를 2048로 변경) 이후 main에서 결정적으로 실패하고, 단독 실행해도 똑같이 실패하며, 이 PR과 무관하고 별도로 수정 중입니다.
- 리베이스 전 `ec067c6e`에서 이 작업 단위 자체 게이트: 12,051 통과, 0 실패.

검증하지 않은 것: Metal과 CUDA(이 호스트에서 사용할 수 없음. 변경은 `patches-rocm/` 아래 파일과 `rocm` 게이트 테스트만 건드립니다).

## 5. LOCAL_FIXES 항목 38

항목은 "Fixes to the fork's kernels" 아래에 들어갑니다. 공유된 상태와 이제 각각이 어떻게 소유되는지, 측정한 파이프 변형 세 가지, 기각된 대안, 디버그 로깅, 20회 중 20회 수치를 가진 테스트를 기록하고, 포크 정책 문구로 끝납니다. "Applies to the fork; kept in mlxcelverse under the 2026-10-06 fork policy, not proposed there (lablup/mlxcel#2200)." `docs/mlxcelverse/upstream/` 아래에는 아무것도 추가하지 않습니다.

## 6. 기술적 선택과 그 이유

- **파이프를 공유하기 전에 측정.** 이슈의 설계는 종이 위에서는 타당했지만 5개 중 0개가 통과했습니다. 수정은 측정을 따릅니다.
- **잠금보다 스레드 로컬.** `hipblasLtMatmul` 전체에 `g_pipe_mutex`를 쥐면 MoE 세그먼트 루프를 포함해 프로세스의 모든 hipBLASLt GEMM의 호스트 enqueue가 직렬화됩니다. 스레드별 맵은 빠른 경로에 잠금이 없고, 공유 `algo_cache`가 첫 사용 비용을 디스크립터 생성으로 한정합니다.
- **성공한 뒤에만 파이프 삽입.** 부수 효과로 오래된 알고리즘 재시도 문제를 고치고, 반쯤 만들어진 파이프가 관찰되지 않게 합니다.
- **상한 크기의 스트림당 버퍼 하나.** 커지지 않으므로 경쟁할 재할당이 없고, 스트림 순서로 재사용되며 스크래치 메모리에 필요한 순서는 그것뿐입니다.
- **버퍼 없는 캡처 스트림에서는 예외.** 캡처 안의 `hipMalloc`은 캡처를 무효화합니다. 캡처 가능한 유일한 hipBLASLt 호출은 fp8 경로이며 M >= 64가 필요하고 디코드 캡처에서는 도달하지 않습니다.
- **create 실패는 영구 유지.** 경로 결정이 매 GEMM마다 호출합니다.

## 7. 후속 이슈

- **#2206, 정확한 0에서의 bf16 잔차.** Tensile 커널을 특정하고 MLX 밖에서 재현한 뒤 고치거나 한계를 문서화합니다. 그때까지 정확한 bf16 테스트는 입력을 `-1..=1`로 유지합니다.
- **#2197, `rocm::device()` 맵.** 시리즈의 다음 작업입니다. 프로세스 전역 맵에 대한 잠금 없는 `find`/`try_emplace`이며, 잠재적이고 멀티 GPU에서만 해당합니다.
- **누수되는 alpha/beta 쌍.** GEMM 경로의 `new float[2]` 쌍은 해제되지 않습니다. 이 PR의 범위 밖입니다.
- **GPU 가드 대기.** 다른 작업 단위가 컴파일 중이면 가드된 디코드 세션이 호스트 전역 잠금 뒤에서 몇 시간 대기할 수 있습니다.

## 8. 남은 위험

- **스트림별 워크스페이스는 실행 시점에 검증되지 않음.** gfx1151에서는 어떤 알고리즘도 워크스페이스를 요청하지 않았습니다. 이 경로의 버그는 CDNA나 gfx1201에서 먼저 드러날 것입니다.
- **디스크립터 변경의 원인 미특정.** 스레드 로컬 설계는 라이브러리 소스가 아니라 측정에 기반합니다. 이 동작을 바꾸는 hipBLASLt 버전이 나와도 설계가 깨지지는 않지만, 어떤 필드에 쓰는지는 확인하지 않았습니다.
- **메모리.** 스레드마다 파이프 맵을 가지고, 워크스페이스가 필요한 스트림마다 32 MB를 잡습니다. 둘 다 워커 스레드와 스트림 수로 제한됩니다.
- **정확한 bf16 검증 범위가 의도보다 좁음.** #2206이 해결될 때까지입니다.
- **호스트 하나.** 모든 실행이 gfx1151 한 대에서 이루어졌습니다.

## 9. 학습 포인트

- **수정의 전제를 시험하라.** 이슈의 공유 소유 모델은 옳아 보였지만 틀렸고, 다섯 가지 실험만이 이유를 보여줬습니다. 우회, rocBLAS 전용, 스레드별 변형이 "공유 상태"와 "공유 디스크립터 객체"를 구분해 줍니다.
- **소유권이 곧 안전은 아니다.** `shared_ptr`는 디스크립터를 살려 뒀지만, 호출된 쪽이 그것을 변경하므로 힙은 여전히 손상됐습니다.
- **워크스페이스를 논하기 전에 휴리스틱부터 읽어라.** 이 호스트의 워크스페이스 0은 무서운 경쟁을 테스트가 닿을 수 없는 것으로 바꿨고, 보고서는 그 사실을 밝혀야 합니다.
- **새 프로세스 테스트는 프로세스당 한 번의 구간을 횟수로 바꾼다.** 실행마다 자식 10개, 20회 실행입니다.
- **정확한 비교는 허용 오차가 가리는 것을 찾는다.** #2206이 그렇게 발견됐습니다.
