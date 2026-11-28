# 기술 보고서: PR #2049 - 첫 HIP 큐 이전에 ROCm blocking-sync 설정, FFT plan 캐시를 128로 복원

**날짜**: 2026-09-30

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: C++ (ROCm overlay의 HIP 런타임 호출), Rust (통합 테스트), Markdown

**위험도**: 중간 (모든 ROCm 프로세스가 장치 대기 모드를 설정하는 시점을 allocator의 hot path에서 바꿉니다. ROCm 전용 overlay와 `#![cfg(feature = "rocm")]` 테스트만 바뀌며, Metal과 CUDA 빌드는 `patches-rocm/`을 복사하지 않습니다)

## 요약

이슈 #1876 (에픽 #1801의 일부)은 CUDA 파일이 128을 쓰는데 왜 ROCm hipFFT plan 캐시는 8로 제한해야 하는지 물었습니다. 8을 넘기면 FFT를 많이 쓰는 작업이 GPU가 쉬는 상태로 영원히 멈췄고, 원인은 "rocFFT 내부 어딘가"로만 기록되어 있었습니다. 이 PR의 조사 결과, 멈춤은 rocFFT 안에 있지 않았고 살아 있는 plan 수와도 무관했습니다. ROCm 백엔드의 순서 버그였습니다. `hipSetDeviceFlags(hipDeviceScheduleBlockingSync)`는 `rocm::device()`에서 실행되었는데, 그 전에 allocator가 `hipStreamQuery(nullptr)`로 null stream의 HIP 큐를 이미 만들어 두었습니다. CLR은 그 큐에 non-interrupt 완료 signal을 주었고, 이 signal에는 이후 `hipFree`가 의존하는 async handler를 붙일 수 없습니다. 그래서 null stream을 동기화해야 하는 첫 `hipFree`가 돌아오지 않았습니다. 첫 plan eviction 때의 rocFFT `hipfftDestroy`는 mlxcel에서 null stream 작업 뒤에 `hipFree`를 부른 첫 코드였을 뿐입니다.

MLX도 hipFFT도 없는 30줄짜리 독립 HIP 프로그램이 이 멈춤을 재현합니다. null stream을 건드리고, 플래그를 설정한 다음, `hipMalloc`, `hipMemcpy`, null stream 커널, `hipFree`를 반복합니다. 첫 `hipFree`에서 멈추고, 앞의 null stream 접근만 빼면 끝까지 실행됩니다.

수정은 `rocm::ensure_device_flags(int)`와 `ensure_current_device_flags()`를 추가해, 장치에 큐가 하나도 없을 때 장치마다 한 번 플래그를 설정하고, `rocm::device()`보다 먼저 HIP에 닿을 수 있는 모든 경로에서 이를 호출합니다. plan 캐시는 128로 돌아가고, 새 통합 테스트 `tests/rocm_fft_plan_cache.rs`가 자식 프로세스에서 캐시 크기 16으로 eviction이 일어나는 round trip 48회를 돌려 멈춤이 돌아오면 실패합니다. 이 결함은 처음부터 FFT에 한정된 것이 아니었습니다. null stream에 대기 중인 작업이 있는 상태의 모든 장치 전체 동기화가 노출되어 있었으므로, 수정 범위는 이를 발견한 이슈보다 넓습니다.

## 1. 문제 정의

### 이슈에 기록된 내용

#1825의 FFT 이식(PR #1856, #1861)은 `fft.hip`의 hipFFT `LRUBytesKeyCache`에 `default_capacity` 8을 주고, rocFFT 내부의 근본 원인은 모른다는 주석을 달았습니다. 측정은 일관되었지만 오해를 부르는 형태였습니다. 26개 변환 형태를 돌리는 probe는 용량 8에서 끝까지 실행되고 16, 24, 32, 128에서는 항상 순서상 같은 위치에서 멈췄습니다. 케이스 순서를 바꾸면 그 위치에서 실행되는 것이 멈췄고, 한 형태를 60번 반복하면 멈추지 않았습니다. 이 패턴은 "살아 있는 plan이 너무 많다"를 가리켰고, 이슈가 제시한 가설들도 여기서 나왔습니다. rocFFT 런타임 컴파일, `hipfftSetAutoAllocation(handle, 0)`로 설정한 수동 work area, 그리고 `shared_ptr` 캡처 때문에 plan이 캐시 항목보다 오래 사는 문제입니다. 독립 프로그램은 hipFFT plan 64개를 문제없이 만들었으므로, 한계가 MLX의 사용 방식에 달려 있다는 것까지는 알았지만 어떻게인지는 몰랐습니다.

제한의 비용은 실제로 있었습니다. 장치 FFT를 쓰는 mlxcel 경로 두 개인 Kokoro 음성 합성과 Phi-4-multimodal 오디오 front end는 CUDA보다 훨씬 자주 plan을 다시 만들었습니다.

### 조사에서 찾은 것

이 부분이 PR의 핵심이며, 증상이 매 단계 원인과 다른 쪽을 가리켰기 때문에 추론 과정을 남길 가치가 있습니다.

**1단계: 멈춤의 시점 찾기.** probe를 `fft_numeric_probe --plans N`(원래 PR #1878)으로 다시 만들었습니다. 캐시 16에서는 `plan 7` 다음에, 32에서는 `plan 15` 다음에 멈췄습니다. probe 한 반복은 plan 두 개(R2C, C2R)를 만들므로, plan 7 다음 반복은 17번째 plan을, plan 15 다음 반복은 33번째 plan을 만듭니다. 두 경우 모두 eviction을 처음 일으키는 plan입니다. 멈춤은 살아 있는 plan 수의 문턱이 아니라 첫 `hipfftDestroy`에서 일어났습니다. LRU 캐시는 새 항목을 넣는 도중에 eviction을 하므로, 멈춤이 "다음 plan 생성이 막힌다"처럼 보였고 이슈도 그렇게 기술했습니다.

**2단계: 멈춤의 스택 찾기.** 멈춘 스레드에 rocgdb를 붙이면 `hipfftDestroy -> rocfft_plan_destroy -> ~ExecPlan -> ~TreeNode -> gpubuf_t::free -> hipFree -> ... -> std::condition_variable::wait`가 나왔습니다. GPU는 쉬고 있었고 rocFFT RTC 헬퍼 프로세스도 없었으므로 런타임 컴파일 가설은 배제되었습니다. rocFFT는 장치 버퍼를 해제하고 있었을 뿐이고, 대기는 HIP의 `hipFree` 안에 있었습니다. `hipFree`는 해제 전에 장치의 모든 stream을 동기화합니다(`Device::SyncAllStreams`).

**3단계: 대기가 깨어나지 않는 이유 찾기.** `AMD_LOG_LEVEL=4`에서 멈추기 직전 마지막 줄은 `rocvirtual.cpp:844 hsa_amd_signal_async_handler() failed to set the handler!`였습니다. ROCr는 event mailbox가 없는 signal, 즉 non-interrupt signal에는 async handler를 거부합니다. `SyncAllStreams`는 null stream에 marker를 넣고 그 marker의 완료 handler가 호출되기를 기다립니다. handler가 설치되지 않았으니 condition variable은 끝내 notify되지 않았습니다.

**4단계: non-interrupt signal의 출처 찾기.** CLR은 큐를 만들 때 적용 중인 장치 대기 모드에 따라 큐의 완료 signal ring 종류를 정합니다. 기본값인 active wait 모드에서는 호스트가 signal을 spin으로 기다리므로 CLR은 가벼운 `HSA_AMD_SIGNAL_AMD_GPU_ONLY`(non-interrupt) signal을 할당합니다. active wait가 켜져 있는 동안에는 `HwQueueTracker::ActiveSignal`이 handler가 필요한 명령마다 interrupt signal로 바꿔 줍니다. `hipDeviceScheduleBlockingSync`가 active wait를 끄면, 큐가 애초에 interrupt signal로 만들어졌다고 가정하고 이 교체를 멈춥니다. HIP 로그가 이 가정을 어긴 큐를 보여 주었습니다. null stream의 큐는 unified 버퍼를 처음 호스트에서 읽을 때 `allocator::Buffer::raw_ptr()`의 `hipStreamQuery(nullptr)`가 만들었고, 그 뒤에야 `rocm::device()`가 `hipSetDeviceFlags`를 호출했습니다. 그래서 null stream 큐는 프로세스가 끝날 때까지 non-interrupt signal을 유지했고, 그중 하나에 떨어지면서 handler가 필요한 첫 marker가 대기자를 멈추게 했습니다.

**5단계: MLX 밖에서 증명하기.** PR 본문의 독립 재현 프로그램은 나머지를 모두 걷어 냅니다(`hipcc --offload-arch=gfx1151 -O2`로 빌드).

```cpp
if (early_touch) (void)hipStreamQuery(nullptr);        // creates the null-stream queue now
(void)hipSetDeviceFlags(hipDeviceScheduleBlockingSync); // flips active wait off
// loop: hipMalloc, hipMemcpy H2D, kernel on the null stream,
//       keep 8 buffers live, hipFree the oldest
```

`timeout 30 ./repro 1 200`은 `iter 7`을 출력하고 exit 124로 끝나며, 첫 `hipFree`에서 멈춥니다. 앞의 null stream 접근만 뺀 `./repro 0 200`은 `done`을 출력합니다. hipFFT도, MLX도, plan 캐시도 없습니다. 결함 전체가 HIP 호출 두 개의 순서입니다.

### 이슈의 가설이 틀린 이유와 이로써 배제되는 것

- **살아 있는 plan 수.** 멈춤은 살아 있는 plan 수가 아니라 첫 eviction에 달려 있습니다. 독립 재현 프로그램에는 plan이 아예 없습니다.
- **rocFFT 런타임 컴파일.** RTC 헬퍼는 실행 중이 아니었고, 스택은 컴파일이나 캐시 lock이 아니라 `hipFree`에 있었습니다.
- **plan이 캐시 항목보다 오래 사는 문제.** `execute_fft` 주석은 캡처된 `shared_ptr`가 비동기 실행 동안 plan을 살려 둔다고 했습니다. 그렇지 않습니다. `CommandEncoder::launch_kernel`은 lambda를 동기적으로 호출하고 반환할 때 버립니다. 캐시가 "여유를 남겨야 한다"는 근거가 이 전제였고, 전제는 틀렸습니다.
- **rocFFT 자체.** rocFFT의 역할은 null stream 작업 뒤에 `hipFree`를 부른 첫 호출자였다는 것뿐입니다. rocFFT에 보고할 것은 없습니다. 대신 재현 프로그램은 CLR 보고 후보입니다(아직 제출하지 않았습니다).

설명되지 않은 부분이 하나 있습니다. 26개 형태에서는 용량 8에서도 eviction이 일어나는데, 왜 용량 12 이하는 끝까지 실행되고 16 이상은 첫 eviction에서 멈췄는지입니다. PR의 작업 가설은 첫 `hipFree` 시점의 null stream signal ring 위치입니다(재현 프로그램도 커널 8개 뒤에 처음 멈춥니다). 하지만 이 부분은 규명하지 않았습니다.

## 2. 변경 요약

- **`device.cpp` / `device.h`: `ensure_device_flags(int)`와 `ensure_current_device_flags()`.** 장치 인덱스마다 한 번 `hipDeviceScheduleBlockingSync`를 설정합니다. 모든 unified 할당과 unified 버퍼의 모든 호스트 읽기에서 실행되므로, 64 미만 인덱스는 lock 없는 atomic 비트마스크로 처리하고, 그 이상은 mutex와 vector로 처리합니다. 호출자가 다른 장치에 있을 때만 대상 장치로 전환하고 끝나면 원래 장치로 되돌립니다. `hipSetDevice`가 실패하면 장치를 기록하지 않아 다음 호출이 다시 시도합니다. `hipSetDeviceFlags`가 실패하면 오류를 지우고, 장치와 HIP 오류를 적은 한 줄을 stderr에 출력하며, 할당마다 재시도하지 않도록 장치를 기록합니다. `rocm::device()`는 이제 플래그를 직접 설정하지 않고 이 헬퍼를 호출합니다. 플래그 값 자체는 바뀌지 않았습니다.
- **`rocm::device()`보다 먼저 큐를 만들 수 있는 호출 지점.** `allocator.cpp`의 `unified_malloc`(프로세스의 첫 HIP 작업인 경우가 많습니다), `Buffer::raw_ptr()`의 두 분기(unified 메모리의 null stream query와 discrete 메모리의 `hipDeviceSynchronize`), `RocmAllocator` 생성자에서 pool 장치마다 만드는 opt-in async pool 해제 stream, `eval.cpp`의 `gpu::init()`에서 `hipFree(nullptr)` 전, 그리고 `host_stage.cpp`의 `staged_write`에서 장치 동기화 전입니다. 체크포인트 저장이 스레드 장치의 첫 동기화일 수 있기 때문입니다.
- **`fft.hip`.** `default_capacity`는 `mlx/backend/cuda/fft.cu`와 같은 128로 돌아갑니다. 캐시 주석은 실제 원인을 적고 `LOCAL_FIXES.md` 항목 18과 20을 가리킵니다. `execute_fft` 주석도 바로잡았습니다. 캡처된 plan은 enqueue 동안만 살고, `hipFree`가 먼저 장치를 동기화하므로 eager launch에서는 eviction이 여전히 안전하며, opt-in `MLX_GRAPH_PREFILL_REPLAY=1` 그래프 경로는 검토하지 않았다고 적습니다.
- **`tests/rocm_fft_plan_cache.rs` (신규, 285줄).** 부모 테스트는 `MLX_ROCM_FFT_CACHE_SIZE=16`과 120초 제한으로 같은 바이너리의 ignored 자식 테스트를 다시 실행하고, 시간 안에 끝나지 않으면 자식을 종료합니다. 자식은 서로 다른 길이로 rfft/irfft round trip 48회를 실행합니다(16칸 캐시에 plan 96개이므로 round trip 8회 뒤부터 eviction이 계속 일어납니다). 그다음 길이 2, 8, 400, 512, 1024, 1200, 2048에서 rfft, irfft, round trip, complex fft와 512 batch rfft를 CPU stream과 1e-5 이내로 비교합니다. 자식 본문은 `MLXCEL_ROCM_FFT_CHILD`가 설정되었을 때만 실행되므로 `--include-ignored` 실행이 공유 프로세스에서 이를 돌리지 않습니다.
- **`LOCAL_FIXES.md`.** 항목 18(FFT 이식)은 용량 128과 그 경위를 기술합니다. 새 항목 20은 순서 결함, CLR 메커니즘, 모든 호출 지점, 실패 처리, 근거와 재현 프로그램을 fork 쪽 upstream 반영 후보로 기록합니다.
- **`docs/environment-variables.md`, `docs/installation.md`.** `MLX_ROCM_FFT_CACHE_SIZE`의 기본값을 128로 적고, 멈춤이 해결되었다고 기술합니다(#1876).

커밋 이력: `84e8ad19`가 수정 본체입니다. `6e4ac7d6`은 `fft.hip` 주석에 `LOCAL_FIXES.md` 항목 번호를 적고, `9a9c1b06`은 헬퍼의 실패 경로를 드러내고 테스트가 자식 polling 실패 시 자식을 종료하도록 합니다. `78a53b15`와 `3ccab4e5`는 origin/main을 머지합니다(첫 번째는 PR #2046의 `LOCAL_FIXES.md` 항목 19를 가져오기 위한 것입니다).

## 3. 기술적 선택과 그 이유

### 캐시 제한이 아니라 순서를 고칩니다

캐시를 8로 두면 null stream 작업 뒤에 오는 다른 모든 `hipFree`와 장치 전체 동기화에 결함이 그대로 남습니다. FFT는 처음 걸린 호출자였을 뿐이고, 런타임에 장치 메모리를 해제하는 이후의 연산이 FFT 없이도 같은 식으로 멈출 수 있었습니다. 순서를 고치면 이런 멈춤 전체가 사라지고, 캐시 용량은 안전 한계가 아니라 메모리와 plan 재생성 사이의 절충으로 돌아갑니다.

### 프로세스 시작 시 한 번이 아니라 모든 진입점에서 일찍 설정합니다

HIP에는 어떤 코드든 첫 HIP 호출 이전에 실행되는 hook이 없고, MLX가 장치를 고르기 전에는 장치 인덱스를 알 수 없습니다. 플래그는 해당 장치의 첫 큐 이전에 그 장치에 설정되어야 하므로, `rocm::device()`보다 먼저 큐를 만들 수 있는 각 지점에서 헬퍼를 실행합니다. 첫 호출 이후의 비용은 atomic load 한 번과 비트 검사이며, 할당 경로에 있기 때문에 fast path를 lock 없이 만들었습니다.

### blocking-sync를 없애지 않고 유지합니다

`hipSetDeviceFlags`를 아예 빼도 모드는 일관됩니다(전부 active wait). 하지만 모든 GPU 대기 동안의 호스트 CPU 사용량이 바뀌고, 이는 fork가 처음에 blocking-sync를 설정한 이유이기도 합니다. PR은 플래그 값을 유지하고 적용 시점만 바꿉니다.

### 장치별로, 현재 장치만

헬퍼는 인덱스별로 따로 기록하므로 장치 0을 먼저 설정해도 장치 1이 누락되지 않고, 모든 장치를 순회하지도 않습니다. 기존 `rocm::device()` 주석이 그 이유를 기록하고 있습니다. 다중 GPU 호스트에서 다른 GPU에 context나 큐를 만드는 것이 TB5 링크 너머 discrete GPU의 큐를 멈추게 한 원인이었습니다. 헬퍼는 이 제약을 지키고 호출자의 현재 장치를 복원하므로 allocator에서 부작용 없이 부를 수 있습니다.

### 실패를 드러내고, 성공할 수 있는 것만 재시도합니다

`hipSetDevice` 실패는 플래그 설정을 시도조차 못 했다는 뜻이므로 장치를 기록하지 않고 다음 호출이 다시 시도합니다. `hipSetDeviceFlags` 실패는 기록하고 stderr에 한 번 알립니다. 이 한 줄이 없으면 active wait 모드로 남은 장치가 아무 흔적 없이 #1876을 재현하게 되고, 이는 이 PR이 없애려는 조용한 완전 멈춤 그대로입니다.

### 제한 시간이 있는 자식 프로세스에서 테스트합니다

멈춤 자체에는 제한 시간이 없고, plan 캐시 용량은 첫 FFT 때 환경 변수에서 읽으므로 같은 프로세스 안에서는 테스트할 수 없습니다. 자식 프로세스 방식은 새 환경과 외부 종료를 모두 제공합니다. 앞쪽 호출들을 주석 처리하면 테스트가 실패함(`the child did not finish within 120s`, round trip 48회 중 8회 뒤)을 확인했으므로, 이 테스트는 FFT 정확도뿐 아니라 순서 자체를 지킵니다.

## 4. 검증

작성자 실행 (gfx1151, Radeon 8060S, HIP 7.15.26333, ROCm 10.0 core, rocFFT 1.0.39):

- 캐시 128에서 probe(`--plans 40`) 연속 3회: 모두 exit 0, round trip 40/40 정상. 이 실행은 eviction이 없으므로 의미 있는 것은 아래 eviction 실행입니다.
- 수정 전 `plan 7`, `plan 15` 뒤에 멈췄던 캐시 16과 32에서 probe: 둘 다 exit 0, 40/40 정상. 캐시 16에 `--plans 100`도 끝까지 실행됩니다.
- probe 정확도 모드: 25개 비교 모두 CPU stream과 1e-5 이내, 최악은 4.2e-7(n=2048과 n=1200 round trip).
- `AMD_LOG_LEVEL=3`에서 순서 증명: `hipSetDeviceFlags ( 4 )`가 26번째 줄로, `hipExtMallocWithFlags`(32번째 줄), `hipStreamQuery ( <null> )`(50번째 줄), 첫 `Created SWq`(52번째 줄)보다 앞섭니다. 수정 전에는 플래그가 null stream의 `Created SWq` 뒤에 나왔습니다.
- `cargo test --features rocm --test rocm_fft_plan_cache`: 통과(9.97초, 6.91초). 앞쪽 호출을 주석 처리하면 실패합니다.
- 독립 재현 프로그램: 앞의 null stream 접근이 있으면 `iter 7`에서 exit 124, 없으면 `done`.
- 새 테스트에 대한 `-D warnings` clippy, `dead_doc_pointers`, 빠른 `make verify-*` 게이트: 통과.

오케스트레이터 검증 (gfx1151, origin/main `ac02b2dc`를 머지한 브랜치 head `3ccab4e5`):

- 회귀 테스트와 `dead_doc_pointers` 통과(자식은 약 7초에 끝남). 앞쪽 호출을 주석 처리하면 회귀 테스트가 실패합니다. 캐시 128에서 probe 3회, 캐시 16과 32, 정확도 모드 모두 끝까지 실행됩니다. Qwen3-0.6B-4bit으로 `scripts/ci/rocm_smoke.sh`가 32 토큰을 생성합니다.
- `make verify-rocm`의 모든 단계를 실행했습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` workspace clippy, ROCm smoke(32 토큰)가 통과했습니다.
- `verify-test-rocm`은 세 타깃에서 실패했고, 알려진 기준 실패 37개와 정확히 같았으며 그 외는 없었습니다.
  - `-p mlxcel-core --lib`: 35개 실패. 34개는 ROCm 이식이 없는 fused paged-attention 테스트(#1814)이고, 나머지 하나는 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`입니다.
  - `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit`와 `tests::family_order_is_exhaustive`, 둘 다 #2037에서 온 것입니다.
- 새 `tests/rocm_fft_plan_cache.rs`는 전체 스위트 안에서 통과했습니다.

## 5. 학습 포인트

- **설정에 따라 움직이는 문턱은 한계가 아니라 사건인 경우가 많습니다.** 두 캐시 크기에서 probe가 plan 번호를 출력하자 "살아 있는 plan이 N개를 넘으면 멈춘다"가 "첫 eviction에서 멈춘다"로 바뀌었습니다. 멈춤 위치를 여러 설정에서 잰 것이 plan 수 이론을 무너뜨렸습니다.
- **가설을 세우기 전에 스택부터 확보합니다.** 이슈의 가설 네 개는 모두 그럴듯했고 모두 틀렸습니다. rocgdb backtrace 하나가 문제를 rocFFT에서 `hipFree`로 옮겼고, `AMD_LOG_LEVEL=4` 한 줄이 `hipFree`에서 signal 종류로 옮겼습니다.
- **일부 HIP 장치 플래그는 첫 stream이 아니라 첫 큐 이전에 설정해야 합니다.** CLR은 큐를 만들 때의 대기 모드로 완료 signal 종류를 정하고 다시 바꾸지 않습니다. null stream을 건드리는 모든 HIP 호출(query, sync, copy, launch)이 큐를 만들 수 있습니다. "장치를 처음 쓸 때" 플래그를 설정하는 코드는 allocator가 하는 호출까지 포함해 어떤 종류든 첫 HIP 호출을 첫 사용으로 정의해야 합니다.
- **라이브러리를 탓하기 전에 런타임 수준까지 줄입니다.** 독립 재현 프로그램은 잘못된 순서의 HIP 호출 두 개로 결함을 보여 줍니다. 이것이 없었다면 자연스러운 다음 단계는 rocFFT가 하지도 않는 일에 대한 rocFFT 버그 보고였을 것입니다.
- **한계를 정당화하는 주석은 다시 확인합니다.** "캡처된 plan이 호출보다 오래 산다"는 주석이 캐시 여유의 근거였는데, `launch_kernel`에는 그런 동작이 없었습니다.

## 6. 검증되지 않은 부분

- **다른 GPU와 ROCm 버전.** gfx1151, HIP 7.15.26333, rocFFT 1.0.39만 측정했습니다. CLR 동작은 일반적이지만 다른 HIP 버전은 signal 종류를 다르게 정할 수 있습니다.
- **작은 용량에서의 문턱.** 용량 12 이하가 왜 끝까지 실행되었는지는 규명하지 않았고, signal ring 위치는 추정입니다.
- **FFT의 `MLX_GRAPH_PREFILL_REPLAY=1` 경로.** 이 경로는 hipFFT 호출을 나중에 실행되는 그래프에 기록하며, `execute_fft`의 eviction 안전성 논증은 eager launch만 다룹니다.
- **`hipSetDeviceFlags` 실패 경로.** stderr 메시지는 호출이 실패할 때만 나오며, 이를 강제하는 테스트는 없습니다.
- **Metal과 CUDA.** 이 호스트에서는 사용할 수 없습니다. 변경은 `patches-rocm/`과 `#![cfg(feature = "rocm")]` 테스트에만 닿으므로 두 빌드는 구조적으로 영향을 받지 않습니다.
- **다중 GPU 호스트.** 헬퍼는 설계상 요청된 장치만 건드리지만, 다중 GPU ROCm 호스트에서는 테스트하지 않았습니다.

## 7. 남은 작업

- 독립 재현 프로그램으로 CLR 보고를 제출합니다. 아직 제출하지 않았습니다.
- `LOCAL_FIXES.md` 항목 20(과 갱신된 항목 18)을 #1813의 일부로 ROCm fork에 제안합니다.
- 이 diff 밖의 기존 문제: `MLX_ROCM_FFT_CACHE_SIZE=0`일 때 `lru_cache.h`가 빈 리스트에서 eviction을 하고, 숫자가 아닌 값에서는 `std::stoul`이 예외를 던집니다.
- `MLX_GRAPH_PREFILL_REPLAY=1`에서의 FFT를 검토합니다.
- 기준 테스트 실패: #1814(fused paged-attention 이식, 실패 34개), bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` 실패, #2037의 실패 두 개.

참고: #1876 (이 PR로 닫힘), #1801, #1825, #1878, #2033, #2046.
