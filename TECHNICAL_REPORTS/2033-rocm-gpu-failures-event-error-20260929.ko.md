# 기술 보고서: PR #2033 - ROCm GPU 실패를 Event::error로 전달

**날짜**: 2026-09-29

**상태**: gfx1151 호스트에서 구현 및 검증 완료, 머지 대기 중.

**언어**: C++ (HIP), Rust

**위험도**: 중간 (ROCm의 모든 대기 경로를 건드리고 모든 프리미티브 실행 뒤에 검사를 추가합니다. overlay 안에서만 바뀌므로 Metal과 CUDA는 구조상 영향이 없지만, 해당 백엔드에서 실행해 보지는 않았습니다)

## 요약

이슈 #1804(에픽 #1801의 1단계)는 ROCm GPU 실패가 MLX의 `Event::error`(ml-explore/mlx#3742)에 도달해 mlxcel이 이를 오류로 보고하도록 만들라고 요구했습니다. 이 PR 전에는 ROCm overlay에 오류 저장 공간만 있었고 아무도 거기에 쓰지 않았습니다. HIP가 거부한 커널 실행은 출력을 쓰지 않은 채 남기고 eval은 `Ok`를 돌려줬으며, 비동기 디바이스 폴트가 나면 기다리던 eval은 영원히 돌았습니다. 브리지의 `stashed_launch_state`는 두 경우 모두 `Landed`로 읽었습니다.

이 PR은 Metal의 방식을 ROCm overlay로 옮깁니다. 각 `CommandEncoder`가 `Error`를 하나 소유합니다. 호스트 쪽의 모든 대기와 조회는 성공을 가정하지 않고 HIP 상태를 돌려주며, 스트림이 실패한 것을 발견한 대기나 조회는 그 상태를 인코더에 기록하고 그 `Error`를 이벤트에 붙입니다. `Event::wait()`는 이를 `Event::check_error()`로 던지고, `Event::is_signaled()`는 실패한 이벤트를 오류가 붙은 signaled 상태로 보고하므로 `array::is_available()`가 배열을 영원히 대기 상태로 두지 않고 예외를 던집니다. 동기 실행 실패는 `gpu::eval`에서 프리미티브마다 스레드의 pending HIP 오류를 한 번 읽어서 잡습니다. gfx1151에서 범위 밖 쓰기는 이제 기다리던 eval을 195 ms 만에 실패시키고, 이후 eval은 멈추지 않고 0 ms에 실패합니다.

## 1. 문제 정의

ml-explore/mlx#3742의 계약은 실패한 GPU 단계가 이벤트에 실린 예외로 드러난다는 것입니다. mlxcel 브리지는 여기에 의존합니다. `drain_pending_verification()`은 `stashed_launch_state()`를 읽고, 이 함수는 signaled 이벤트에 오류 포인터가 있을 때만 `Failed`를 보고합니다. ROCm에서는 아무도 그 포인터를 설정하지 않았으므로, 실패한 실행이 `Landed`로 읽히고 그 출력이 제대로 쓰인 것처럼 소비됐습니다.

gfx1151, ROCm 7.15에서 독립 프로브 커널로 측정한 결과(`patches-rocm/LOCAL_FIXES.md` 7번 항목에 기록)는 백엔드의 세 대기 경로가 모두 큐 폴트 이후에는 성립하지 않는 가정 위에 서 있음을 보여줬습니다.

- **프로세스가 중단되지 않습니다.** `HIP_SKIP_ABORT_ON_GPU_ERROR` 설정 여부와 관계없이 런타임은 프로세스를 살려두므로, 크래시가 무한 루프를 끊어주지 않습니다.
- **이후 모든 HIP 호출이 실패합니다.** 모든 스레드의 모든 호출이 `hipDeviceReset`까지 포함해 `hipErrorIllegalAddress`를 돌려줍니다. `HipEvent::wait`는 "`hipEventQuery`가 `hipSuccess`를 돌려줄 때까지" 돌았는데, 그런 일은 다시 일어나지 않습니다.
- **큐에 들어간 호스트 콜백은 실행되지 않습니다.** 백엔드는 `AtomicEvent` 신호와 완료 핸들러를 `hipLaunchHostFunc`로 처리합니다. 카운터를 기다리는 쪽이나, promise를 기다리는 `CommandEncoder::synchronize()`는 영원히 막힙니다.

동기 실행 실패에는 별도의 구멍이 있었습니다. 약 60개의 실행 지점이 아무것도 돌려주지 않는 `hipLaunchKernelGGL`이나 `<<<>>>`를 씁니다. 거부된 실행(너무 큰 블록, 해당 gfx용 코드 오브젝트 없음)은 스레드의 pending HIP 오류만 설정합니다. `launch_module_kernel`은 `hipModuleLaunchKernel`의 상태를 `(void)`로 버렸습니다. 그 결과 쓰이지 않은 버퍼를 두고 eval이 성공했고, 이슈에 적힌 NaN 출력이 여기서 나왔습니다.

이슈에 적힌 자연 재현 사례(mxfp4 hang, mxfp8 NaN, `arg_reduce` grid 초과)는 모두 PR #1818에서 원인이 고쳐졌으므로, 이 수정에는 의도적으로 폴트를 일으키는 테스트가 필요했습니다.

## 2. 변경 요약

- `patches-rocm/mlx/backend/rocm/device.{h,cpp}`: `CommandEncoder`에 `Error error_`, `set_device_error`(가장 먼저 난 오류를 유지), `check_launch(primitive)`가 추가됐습니다. 자유 함수로 `describe_device_error`(HIP 상태를 적고, `hipGetDevice`도 실패하면 컨텍스트가 사라져 프로세스를 재시작해야 한다고 알림), `record_stream_error`(새로 추가된, 디바이스를 바인딩하지 않는 `Device::find_encoder`로 스트림의 인코더를 찾고, CPU 스트림에는 의도적으로 누수시킨 프로세스 전역 fallback `Error`를 사용), `gpu_watchdog_seconds`가 생겼습니다. `synchronize()`는 모든 HIP 상태를 검사하고, 핸들러 promise를 기다리는 동안 약 1 ms마다 `hipStreamQuery`로 스트림을 확인하며, 스트림의 오류를 던집니다. `commit()`과 decode capture 경로는 실패한 `Worker::commit`을 기록합니다. 소멸자 경로의 graph destroy는 명시적으로 `(void)` 처리됩니다. `MLX_GRAPH_PREFILL_REPLAY=1` 스트림 캡처 중에 실행이 예외를 던지면 캡처를 끝낸 뒤 eager 실행으로 되돌아가고, 거기서 캡처 밖의 오류로 다시 던집니다.
- `patches-rocm/mlx/backend/rocm/eval.cpp`: 각 `eval_gpu` 뒤에 `encoder.check_launch(name)`이 스레드의 pending HIP 오류를 읽어 해당 프리미티브의 실패로 던집니다. `eval_gpu` 자체가 예외를 던지면 먼저 pending 오류를 지워서 다음 프리미티브가 누명을 쓰지 않게 합니다.
- `patches-rocm/mlx/backend/rocm/event.{h,hip}`: `HipEvent::wait`와 `AtomicEvent::wait`가 `[[nodiscard]] hipError_t`를 돌려주고, `HipEvent::completed()`는 `query()`가 됐습니다. `HipEvent` 루프는 `hipErrorNotReady`가 아닌 상태를 받으면 멈춥니다. `AtomicEvent`는 카운터를 공유 `State`로 옮기고, 여기에 마지막으로 신호를 보낸 GPU 스트림과 콜백 등록을 막은 상태를 함께 기록합니다. `status()`는 그 상태를 보고하거나 스트림을 조회합니다. `Event::wait`, `Event::wait(Stream)`, `Event::signal(Stream)`, `Event::is_signaled`는 `poison_event`/`fail_event`로 스트림의 `Error`를 붙입니다. CPU 스트림의 대기와 신호는 Metal, CUDA와 마찬가지로 `scheduler::wait_event`/`scheduler::signal_event`를 거칩니다.
- `patches-rocm/mlx/backend/rocm/worker.{h,cpp}`: `Worker::commit`이 `hipLaunchHostFunc` 상태를 돌려줍니다.
- `patches-rocm/mlx/backend/rocm/fence.cpp`: 이벤트 대기가 실패를 보고하면 `Fence::wait`가 예외를 던집니다.
- `patches-rocm/mlx/backend/rocm/jit_module.h`: `hipModuleLaunchKernel`을 `CHECK_HIP_ERROR`로 감쌌습니다.
- `patches-rocm/mlx/backend/rocm/utils.h`: `HipHandle::reset`은 소멸자에서 예외를 던지는 대신, 실패한 destroy를 핸들 타입마다 한 번 stderr에 알립니다.
- `mlxcel-core`: 테스트 전용 브리지 fixture `rocm_fault_probe_array(kind)`(ROCm이 아니면 예외를 던지는 stub)와 이를 감싼 타입 있는 래퍼 `mlxcel_core::rocm_faults`를 추가하고 `#[doc(hidden)]`으로 표시했습니다. `stashed_launch_state` 위 주석은 이제 ROCm도 오류를 붙인다고 적습니다.
- `tests/rocm_gpu_faults.rs`: 실패 유형마다 하나씩 두 개의 테스트. 큐 폴트 테스트는 자식 프로세스에서 실행됩니다.
- 문서: `LOCAL_FIXES.md` 7번 항목, `docs/installation.md`의 "GPU faults" 행, `docs/environment-variables.md`의 `MLX_ROCM_GPU_WATCHDOG_SECS`.

## 3. 기술적 선택과 그 이유

### Metal 방식을 따릅니다: 이벤트마다가 아니라 스트림마다 Error 하나

Metal의 command buffer 완료 핸들러는 인코더의 오류를 그 버퍼가 신호하는 모든 이벤트에 저장합니다. 고정된 upstream의 CUDA는 오류를 전혀 붙이지 않으므로 참고할 모델이 되지 못합니다. ROCm 포트는 Metal을 따릅니다. `Error`는 `CommandEncoder`에 있고, 실패한 스트림에서 신호되는 이벤트는 모두 그것을 가리킵니다. `Event::set_error`는 raw pointer를 저장하므로 가리키는 대상이 모든 이벤트보다 오래 살아야 합니다. 인코더는 `Device`가 쥐고 있어 프로세스 수명 동안 살고, CPU 스트림용 fallback은 의도적으로 누수시킨 static입니다. 이벤트마다 오류 객체를 두면 별도의 수명 관리가 필요한데 얻는 것이 없습니다. 큐 폴트 뒤에는 디바이스의 모든 이벤트가 같은 이유로 실패하기 때문입니다.

### 대기는 상태를 돌려주고, 예외는 Event 계층이 던집니다

저수준 대기(`HipEvent::wait`, `AtomicEvent::wait`)는 `[[nodiscard]]`가 붙은 `hipError_t`를 돌려주고, 실패를 예외로 바꾸는 것은 `Event` 계층뿐입니다. `poison_event` 뒤에 `Event::check_error()`를 부르는 방식입니다. `check_error`로 던지는 것은 다른 백엔드와 같은 방식이고 메시지도 같은 방식으로 소비되므로, 같은 이벤트를 나중에 다시 기다려도 오류가 두 번 보고되지 않습니다. `[[nodiscard]]`는 상태를 무시하는 호출자를 컴파일러가 잡게 만듭니다. 이전 코드가 상태를 잃어버린 경로가 바로 그것이었습니다. `record_stream_error`와 `poison_event`는 실패할 수 있는 HIP 호출을 하지 않고 디바이스도 바인딩하지 않으므로, 죽은 디바이스에서나 `const` 조회에서도 안전합니다.

### 실패한 이벤트는 signaled로 보고합니다

신호를 보내는 스트림이 실패했으면 `is_signaled()`는 오류를 붙인 채 `true`를 돌려줍니다. 기다리지 않는 두 독자에게 닿는 방법은 이것뿐입니다. 하나는 이벤트가 signaled되면 오류를 떼어내 던지는 `array::is_available()`이고, 다른 하나는 signaled 이벤트의 오류 포인터를 확인하는 브리지의 `stashed_launch_state()`입니다. `false`를 돌려주면 둘 다 결코 오지 않을 신호를 기다리게 됩니다. `AtomicEvent`는 실패 상태를 읽은 뒤 카운터를 다시 확인합니다. 첫 확인과 상태 조회 사이에 콜백이 값을 기록했을 수 있기 때문이며, 값이 기록됐으면 스트림이 이후에 무엇을 보고하든 대기는 완료된 것으로 봅니다.

### 동기 실행 실패는 프리미티브당 한 번 잡습니다

검사하지 않는 `hipLaunchKernelGGL`/`<<<>>>` 실행 지점 60개를 모두 고치면 diff가 커지고 회귀하기도 쉽습니다. HIP는 이미 거부된 실행을 스레드의 pending 오류로 기록하므로, `gpu::eval`에서 각 `eval_gpu` 뒤에 실행 스레드에서 한 번 읽으면 앞으로 생길 지점까지 모두 덮입니다. 측정된 비용은 thread-local `hipGetLastError` 한 번, 프리미티브당 약 17.5 ns입니다. `hipEventQuery`/`hipStreamQuery`의 `hipErrorNotReady`는 pending 오류가 되지 않는다는 것을 프로브로 확인했으므로, 파이프라인된 decode에서 이 검사가 잘못 걸리지 않습니다.

리뷰에서 두 가지를 다듬었습니다. `hipErrorOutOfMemory`는 건너뜁니다. 할당자의 캐시 해제 후 재시도, `unified_malloc`의 managed memory fallback, hipBLASLt의 workspace 없는 경로가 모두 실패한 `hipMalloc`에서 회복하면서 pending 오류를 지우지 않고, 회복하지 못한 할당은 이미 예외를 던지기 때문입니다. 이 예외 처리가 없으면 메모리 압박 속에서 성공한 eval이 실패로 처리됐을 것입니다. 또 `check_launch`는 `hipGetDevice` 호출로 실행 거부와 죽은 디바이스를 구분합니다. 실행 거부는 일반 오류를 던지고 스트림은 계속 쓸 수 있게 두며, 컨텍스트가 죽었으면 스트림의 `Error`를 오염시켜 이후 대기가 바로 실패하게 합니다.

### 스트림은 조회하되 드물게

폴트가 난 스트림은 `AtomicEvent` 카운터를 기록하는 콜백을 실행하지 않으므로, 기다리는 쪽은 신호가 아직 올 수 있는지 스트림에 물어봐야 합니다. 이 조회는 빈도를 제한합니다. 카운터는 매 반복마다, 시계는 64번마다, `hipStreamQuery`(약 54 ns)는 최대 1 ms에 한 번 읽습니다. 1 ms 안에 끝나는 대기, 즉 평범한 decode 경우에는 HIP 호출이 추가되지 않습니다. `synchronize()`도 같은 방식으로, 핸들러 future는 100 us마다, 스트림은 열 번째마다 확인합니다. 즉시 감지를 포기하는 대신 성공 경로 비용을 없앤 설계이며, 실제로는 약 1초 안에 폴트가 보고됩니다(측정값 195 ms).

### 이벤트 상태는 콜백 payload가 소유합니다

이전에는 `AtomicEvent` 카운터를 `add_completed_handler([buf = mem_]{})`가 살려뒀는데, 실패한 스트림은 이 핸들러를 실행하지 않습니다. 이제 카운터, 신호를 보낸 스트림, 큐 등록 실패 상태는 공유 `State`에 있고, `hipLaunchHostFunc` payload가 이를 `shared_ptr`로 쥡니다. 수명이 더 이상 완료 핸들러에 의존하지 않고, 다른 스레드의 대기자는 `State`에서 조회할 스트림과 콜백 등록 중에 난 실패를 읽을 수 있습니다.

### CPU 스트림은 스케줄러를 거칩니다

CPU 스트림에서의 `Event::wait(Stream)`과 `Event::signal(Stream)`은 이제 `scheduler::wait_event`와 `scheduler::signal_event`를 씁니다. 그래서 Metal, CUDA와 마찬가지로 CPU 스트림이 GPU 이벤트의 오류를 물려받고, 이미 실패한 CPU 스트림은 자신이 신호하는 이벤트를 오염시킵니다. 첫 초안은 CPU 스트림의 `Fence::update`를 GPU 인코더를 가정하는 경로로 보냈고, `rocm_smoke.sh`가 이를 잡았습니다. 두 번째 커밋이 CPU 스트림 fence 신호를 스케줄러 경로에 남겨둡니다.

### 디바이스는 복구하지 않고, watchdog은 opt-in입니다

큐 폴트 뒤에는 HIP가 `hipDeviceReset`을 포함한 모든 호출을 거부하므로 프로세스 안에서의 복구는 불가능합니다. PR은 리셋을 시도하지 않고 폴트를 보고하며, 오류 메시지에 이 사실을 적습니다. `mlxcel-server`의 배치 스케줄러는 이미 `try_eval` 오류를 요청 단위 실패로 매핑하므로(`decode_tick.rs`와 `prefill.rs`의 `abort_sequence_with_error`) 서버 변경은 필요 없었습니다. 폴트를 낸 요청은 실패하고, 이후 GPU 요청은 프로세스를 재시작할 때까지 같은 오류로 실패합니다. HIP의 종료 처리가 실행되지 않는 콜백을 기다리므로 종료에 SIGKILL이 필요할 수 있습니다.

영원히 도는 커널은 느린 커널과 구분할 수 없으므로, 이슈의 선택 사항이던 watchdog은 기본으로 꺼져 있습니다. 이슈가 스케치한 브리지 drain이 아니라 백엔드 대기 안에 두었는데, 모든 대기 경로가 거기를 지나기 때문입니다. 이름도 overlay의 다른 변수들처럼 `MLX_ROCM_GPU_WATCHDOG_SECS`입니다. 파싱은 `strtol`로 완전히 검증하며, `10s`, 음수, 오버플로 값은 추측하지 않고 stderr 경고 한 줄과 함께 무시합니다. 폴링 루프가 없는 `MLX_EVENT_BLOCKING`에서는 적용되지 않습니다.

### 폴트 테스트는 자식 프로세스에서 실행합니다

프로브 fixture는 `fast::hip_kernel`을 쓰므로 실행, 대기, 오류 부착은 실제 경로이고, 커널 본문만 인위적입니다(HIP가 거부하는 2048 스레드 블록, 그리고 버퍼 끝에서 2^40 바이트 너머에 쓰기). 큐 폴트는 프로세스 전체의 디바이스를 죽이므로, 범위 밖 쓰기 테스트는 테스트 바이너리를 자식으로 다시 실행하고 그 타이밍 보고를 파싱합니다. 자식은 `_exit`로 끝나 HIP의 종료 처리를 건너뜁니다. 그렇지 않으면 종료 처리가 실행되지 않을 콜백을 기다립니다. fixture는 브리지에 `#[doc(hidden)]`으로 있으며 운영 경로에서는 호출하지 않습니다.

## 4. 검증

작성자 실행 결과(PR 본문, gfx1151, ROCm 7.15, `--features rocm`):

- `cargo test --features rocm --test rocm_gpu_faults`: 2개 통과. 자식은 195 ms에 폴트를 보고했고 후속 eval은 0 ms에 실패했습니다.
- 음성 대조: overlay 파일 10개를 `origin/main`에서 되돌리고 fixture만 남기면 두 테스트 모두 실패합니다. 자식은 120 s 예산을 모두 쓰고 강제 종료됐고, 너무 큰 블록 실행은 쓰이지 않은 버퍼로 `Ok`를 돌려줬습니다.
- 리뷰 커밋 이후: `rocm_gpu_faults` 2개 통과(`MLX_ROCM_GPU_WATCHDOG_SECS=10s`에서도 통과, 잘못된 값 경고 출력), `rocm_smoke.sh` 통과(32 토큰, 258.8 tok/s), `cargo clippy -p mlxcel-core --features rocm --lib --tests -- -D warnings` 깨끗함, `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt`와 `cargo test --features rocm --test dead_doc_pointers` 통과.

오케스트레이터 검증(gfx1151, 브랜치를 #1806의 PR #2030이 포함된 origin/main `cdd531d5` 위로 rebase):

- `make verify-rocm`의 모든 단계가 실행됐습니다. versions, kernel dtype keys, kernel port dispatch, llama-compat, fmt, `--features rocm` 워크스페이스 clippy, ROCm smoke(32 토큰)가 통과했습니다.
- `verify-test-rocm`은 정확히 세 타깃에서 실패했고, 어느 것도 이 PR 때문이 아닙니다.
  - `-p mlxcel-core --lib`: 1796 통과, 35 실패, 1 무시. 35개는 #2030 실행 때와 같습니다. ROCm 포트가 없는 fused paged-attention 테스트 34개(#1814에서 추적)와 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible`입니다.
  - `-p mlxcel --lib`: 8711 통과, 1 실패. `gemma3_backbone_tests::gelu_approx_matches_mlx_nn_bit_for_bit`가 ROCm에서 1 ULP 차이로 실패하며, #2037이 추가한 테스트입니다.
  - `-p mlxcel --bin mlxcel`: 248 통과, 1 실패. `family_order_is_exhaustive`가 #2037이 추가한 `Speech` 패밀리를 빠뜨렸습니다.
- `tests/rocm_gpu_faults.rs`를 포함한 나머지 테스트 바이너리는 모두 통과했습니다.

#1806이 머지되면서 ROCm에서 `mlxcel-core` lib 테스트가 끝까지 실행되므로, 이번이 `Event::error` 변경에 대한 첫 전체 스위트 실행입니다. 이 점이 중요한 이유는 `check_launch`가 모든 프리미티브 뒤에 실행되고 모든 대기가 이제 상태를 검사하기 때문입니다. 성공 경로 어디에서든 오탐이 있었다면 스위트 전반에 새 실패로 나타났을 텐데, 실패 목록은 #2030과 같습니다.

## 5. 학습 포인트

- **복구를 설계하기 전에 실패 양상을 측정하십시오.** 이슈의 계획은 폴트 난 큐를 실패로 표시하고 복구할 수 있다고 가정했습니다. 프로브는 디바이스 컨텍스트가 프로세스 동안 사라지고, 모든 HIP 호출이 실패하며, 어떤 콜백도 실행되지 않음을 보여줬습니다. 그래서 목표가 복구에서 빠르고 정확한 보고로 바뀌었고, 세 대기 경로가 모두 멈춘 이유도 설명됐습니다.
- **성공을 기다리는 루프는 고착된 오류에서 영원히 돕니다.** `while (query() != hipSuccess)`는 "아직 안 됨"과 "실패"를 똑같이 다룹니다. 구체적인 "아직 안 됨" 상태에서만 루프를 돌고 나머지는 돌려주십시오.
- **완료 핸들러에 묶인 수명은 핸들러가 실행되지 않을 때 정확히 깨집니다.** keep-alive 핸들러는 성공 경로에서는 편하지만 실패 경로에서는 누수나 use-after-free 위험이 됩니다. 콜백 payload가 자신이 건드리는 것을 소유하게 하십시오.
- **호출 지점 60곳보다 병목 지점 한 곳의 검사가 낫습니다.** HIP의 스레드별 pending 오류는 실행 실패를 전달하는 기존 채널입니다. `gpu::eval`에서 프리미티브마다 한 번 읽으면 지금과 앞으로의 모든 `hipLaunchKernelGGL` 지점을 거의 비용 없이 덮을 수 있습니다. 단, 회복 경로가 남기는 상태(`hipErrorOutOfMemory`)는 제외해야 합니다.
- **프로세스를 죽이는 폴트는 자식 프로세스에서 테스트하십시오.** 프로세스 전체의 디바이스를 오염시키는 폴트는 다른 GPU 테스트와 같은 바이너리 안에 둘 수 없습니다. 필터를 건 바이너리 재실행과 `_exit`로 부모의 디바이스를 살려두고 타이밍도 잴 수 있습니다.

## 6. 검증하지 않은 것

- **Metal과 CUDA는 실행하지 않았습니다.** 그 경로에 있는 파일은 `mlx_cxx_bridge.cpp` 하나뿐이며, `MLXCEL_BRIDGE_ROCM_BACKEND` 없이는 fixture가 예외를 던지는 stub으로 컴파일되고 주석 하나가 바뀌었습니다. 동작 변화는 예상되지 않지만 측정하지는 않았습니다.
- **메모리 부족 예외 처리는 런타임에서 실행되지 않았습니다.** 브리지를 통해 메모리 부족 후 회복 상황을 강제로 만들 방법이 없습니다. 이 처리는 pending 오류를 보여준 프로브와 할당자 및 hipBLASLt 코드 읽기에 근거합니다.
- **watchdog 만료는 실행되지 않았습니다.** 테스트는 잘못된 값의 파싱만 다룹니다. 끝나지 않는 커널로 설정된 한도를 넘기는 테스트는 없으므로 `kWatchdogExpired` 메시지와 그 뒤 멈춘 스트림의 동작은 테스트되지 않았습니다.
- **`MLX_EVENT_BLOCKING` 경로는 테스트가 다루지 않습니다.** `hipEventSynchronize`가 폴트를 바로 돌려준다는 것에 의존합니다.
- **서버 경로는 실행이 아니라 코드 읽기로 확인했습니다.** 폴트를 주입한 상태에서 `mlxcel-server`에 실제 요청을 보내지 않았습니다. 요청 단위 실패 매핑은 기존 `abort_sequence_with_error` 코드에 근거합니다.
- **처리량 기준선이 없습니다.** 워크트리에 변경 전 바이너리가 없었습니다. 추가된 성공 경로 비용은 마이크로 측정값(프리미티브당 17.5 ns, GPU가 신호하는 `AtomicEvent`의 미도착 `is_signaled` 조회당 54 ns)으로 추정했으며, smoke 실행의 243에서 259 tok/s는 통제된 비교가 아닙니다.

## 7. 남은 작업

- #1813의 일부로 overlay 변경을 ROCm fork에 upstream합니다. `LOCAL_FIXES.md` 7번 항목에 기록돼 있습니다.
- #1814: fused paged-attention 커널의 ROCm 포트. `mlxcel-core` 실패 35개 중 34개가 여기에 해당합니다.
- ROCm에서의 bf16 `prefill_dense_gemm_matches_qmm_bytes_where_eligible` 실패를 분류합니다.
- #2037이 ROCm에서 만든 두 실패를 고칩니다. 1 ULP 차이의 `gelu_approx_matches_mlx_nn_bit_for_bit`와 `Speech` 패밀리를 나열하지 않는 `family_order_is_exhaustive`입니다.
- 선택 사항: 도는 커널로 watchdog 만료를 실행하는 테스트, 그리고 폴트를 주입한 `mlxcel-server` 실행.
