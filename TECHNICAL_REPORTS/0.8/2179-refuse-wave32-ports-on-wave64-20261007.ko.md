# 기술 보고서: PR #2179 - Wave64 장치에서 Wave32 전용 커널 포트 거부

**날짜**: 2026-10-07

**상태**: gfx1151 호스트에서 구현 및 검증 완료. 헤드 `1dc419c6`(origin/main `08e22698` 기준 최신), PR 열림, 머지 대기. #2147을 닫음(#1801의 일부).

**언어**: C++(`turbo/kernel_port.*`, `turbo/gpu_backend.*`, ROCm 오버레이, `mlx_cxx_kernels.cpp`, cxx 브리지), Rust(mlxcel-core FFI와 테스트, Mamba, Jamba, BitNet, 통합 테스트 2개), Python과 셸(`scripts/ci/check_kernel_port_dispatch.py`와 그 부정 테스트), Markdown(`LOCAL_FIXES.md`, `docs/installation.md`, CONTRIBUTING)

**위험도**: RDNA에서는 낮음(gfx1151은 같은 포트를 선택하고 같은 출력을 냄). CDNA에서는 중간: 퓨즈드 HIP 커널 10개가 더 이상 선택되지 않고 BitNet이 로드되지 않음. 이 동작은 wave64 장치에서 실행된 적이 없음.

## 요약

mlxcel 퓨즈드 커널의 셔플 기반 HIP 포트는 모두 32레인 웨이브프런트(RDNA, gfx1151)에서 작성되고 실행되었으며, 64레인 빌드를 막는 수단으로 `__AMDGCN_WAVEFRONT_SIZE`에 대한 `#error` 가드에 의존했습니다. ROCm 10의 AMD clang 23(HIP 7.15)은 gfx1151, gfx942, gfx90a 어느 타깃에서도 이 매크로를 어느 철자로도 정의하지 않으므로 가드는 한 번도 발동하지 않습니다. 설치 가이드가 컴파일된다고 적은 CDNA 장치(MI200, MI300)에서는 이 커널들이 그 폭에서 아무도 검증하지 않은 채로 실행되었을 것입니다.

PR #2179는 보호를 호스트 쪽으로 옮깁니다. 커널의 `KernelPorts` 테이블을 포트로 매핑하는 유일한 함수인 `port_for`는 이제 테이블이 `rocm_any_wave_size`로 표시되어 있거나 장치의 하드웨어 웨이브프런트가 32레인일 때만 ROCm 항목을 돌려주고, 그 외에는 "포트 없음"으로 답합니다. `has_kernel_port`, 모든 `*_available()` 술어, `select_kernel_port`가 모두 `port_for`를 거치므로 Rust 게이트와 C++ 런처의 답이 일치하고, 호출자는 이미 가진 그래프 폴백으로 갑니다. 테이블 5개(RoPE append, paged merge, Gumbel, rejection, 그리고 이슈 목록에 없던 xIELU)가 any-wave로 표시되고 10개는 wave32 전용으로 남습니다. 그래프 폴백이 없는 BitLinear 연산을 쓰는 BitNet은 wave64에서 로드 시점에 거부됩니다. CI 검사기에는 any-wave 표시의 정직성과 테스트 seam의 프로덕션 코드 유입 금지를 지키는 규칙 2개가 추가되었고, 각각 부정 테스트로 입증됩니다.

gfx1151에서는 바뀌는 것이 없습니다. 포트 술어 16개는 모두 참이고, 프로파일러에는 여전히 퓨즈드 커널이 보이며, 퓨즈드 RMSNorm을 켜고 끈 그리디 출력이 바이트 단위로 같습니다. wave64 거부 자체는 wave64 장치가 없으므로 테스트 전용 seam으로만 입증되었습니다.

## 1. 문제 정의

### 1.1 가드가 작동하지 않았던 이유

#1814 포트 규칙은 "wave64 타깃은 그럴듯한 오답을 내는 대신 컴파일에 실패해야 한다"고 요구했고, 각 포트는 이를 전처리기로 구현했습니다.

```c
#if defined(__AMDGCN_WAVEFRONT_SIZE__) && __AMDGCN_WAVEFRONT_SIZE__ != 32
#error "... assumes a 32-lane wavefront"
#endif
```

HIP에서는 `static_assert(warpSize == 32)`를 쓸 수 없습니다(`warpSize`는 `operator int()`를 가진 객체이지 상수식이 아님). 그래서 매크로가 유일한 컴파일 시점 수단이었습니다. AMD clang 23은 `hipcc -E -dM`으로 확인한 세 타깃 모두에서 두 철자 중 어느 것도 정의하지 않습니다(#2065, #2067, 이 PR에서 재확인). 따라서 `defined(...)` 검사는 항상 거짓이고 `#error`는 컴파일에 들어가지 않습니다. 포트의 주석도 이미 그렇게 적고 있었습니다. 소스에는 검사가 있어 보이지만 아무것도 하지 않는 실패 유형이라, 작동을 멈췄다는 신호도 없었습니다.

### 1.2 위험에 놓였던 것

명시적 셔플 폭 32는 각 버터플라이 리덕션을 32레인 안에 묶어 두므로 wave64에서도 성립해야 하지만, 커널의 다른 부분이 32폭 웨이브프런트를 가정하지 않을 때에만 그렇습니다. 예를 들어 `threadIdx.x % 32`나 `/ 32`로 구한 레인과 워프 인덱스, 워프별 공유 메모리 슬롯, 배리어 없는 워프 동기 단계가 있습니다. 이 중 어느 것도 wave64 장치에서 실행된 적이 없었고(#2107: "Wave64 (CDNA) untested"), 절반만 접힌 리덕션은 유한하고 그럴듯하지만 틀린 값을 조용히 돌려줍니다.

장치의 실제 폭은 MLX가 이미 알고 있었지만(`hipDeviceProp_t::warpSize`, 바인드 시 로그), mlxcel 쪽에서는 아무도 읽지 않았습니다. `Device::warp_size()`는 실행 폭 실험용인 `MLX_ROCM_FORCE_WARP_SIZE`가 덮어쓰므로 이 판단에 쓸 수 없었습니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| ROCm 오버레이 `rocm.h`/`rocm.cpp`/`no_rocm.cpp` | 새 `mlx::core::rocm::device_warp_size()`: `hipGetDevice`와 `hipDeviceGetAttribute(hipDeviceAttributeWarpSize)`, 오버라이드는 절대 따르지 않고 실패 시 0, ROCm 밖에서는 0을 돌려주는 스텁. 새 "Additions to the fork's API" 절에 `LOCAL_FIXES.md` 항목 32 |
| `turbo/gpu_backend.*` | `constexpr rocm_port_allowed(any_wave_size, warp_size)`; 하드웨어 폭을 프로세스당 한 번 캐시하고 32가 아니면 stderr에 한 번 알리는 `rocm_port_warp_size()`; 테스트 seam `set_rocm_port_warp_size_for_tests` |
| `turbo/kernel_port.*` | `KernelPorts::rocm_any_wave_size`(기본 false); `port_for`가 ROCm 항목을 보류; 그 이유일 때 `select_kernel_port`가 폭을 메시지에 명시 |
| 포트 테이블 5개 | `fused_rope_ports`, `paged_merge_ports`, `gumbel_ports`, `rejection_ports`, `xielu_ports`에 `.rocm_any_wave_size = true`; xIELU의 작동하지 않던 가드 제거 |
| `mlx_cxx_kernels.cpp`, 브리지 | `bitlinear_kernel_available`이 `bitlinear_ports()`를 읽음; BitLinear 거부 메시지가 BitNet 로드 시점 거부를 언급; 가드 주석이 호스트 쪽 보류를 보호 수단으로 명시 |
| cxx 브리지, `lib.rs` | `mamba1_selective_scan -> Result<()>`; 새 `rocm_port_allowed`, `rocm_device_warp_size`, `set_rocm_port_warp_size_for_tests` |
| Mamba, Jamba, BitNet | Mamba와 Jamba가 게이트를 통과한 스캔 실행 결과를 `expect`; BitNet 로드 거부 메시지가 wave64를 포함 |
| 패리티 테스트, `test_support` | 퓨즈드 MoE, relu2, SSM, fused norm, Mamba1, add3에 `skip_for_wave32_only_rocm_port`; paged 건너뛰기 메시지에 보류 언급 |
| 검사기 | `check_kernel_port_dispatch.py`에 규칙 5와 6(`--root` 추가); 사례 13개의 `check_kernel_port_dispatch_test.sh`, make 타깃과 CI에서 실행 |
| `build.rs` | `kernel_port.h/.cpp`와 `gpu_backend.h/.cpp`에 `rerun-if-changed` |
| 테스트 | `kernel_port_tests.rs`(진리표), `tests/rocm_wave_size.rs`, `tests/rocm_wave64_port_refusal.rs` |
| 문서 | `docs/installation.md`에 "Wave64 devices" 행; CONTRIBUTING에 보류 언급 |

커밋 4개: 구현(`9c875c6`), 리뷰 보강(`967933e`), origin/main 머지(`f7d9ec4`), #2164가 31번을 가져간 뒤 LOCAL_FIXES 항목을 32로 재번호(`1dc419c`). 파일 38개, 추가 1,356줄, 삭제 71줄.

## 3. 설계

### 3.1 모든 경로가 읽는 한 곳에 하나의 보류

`port_for`는 이미 해석된 `GpuKernelBackend`를 테이블 항목으로 매핑했고, `has_kernel_port`(모든 술어, 따라서 모든 Rust 게이트 뒤에 있음)와 `select_kernel_port`(모든 런처 뒤에 있음)가 이를 읽습니다. 이제 ROCm 분기는 `rocm_port_allowed(ports.rocm_any_wave_size, rocm_port_warp_size())`가 참일 때만 `ports.rocm`을 돌려줍니다. 런처나 술어마다 검사를 두지 않고 여기에 두었기 때문에, wave64 장치에서 술어와 그 런처가 다르게 답할 수 없습니다. 그런 불일치가 바로 `select_kernel_port`를 도입해 없앤 버그 유형입니다(#1801). 검사는 `ports.rocm`이 null이 아닐 때만 실행되므로 ROCm 항목이 없는 테이블은 장치를 조회하지 않습니다.

`rocm_port_allowed`는 순수 `constexpr`(`any_wave_size || warp_size == 32`)라 모든 백엔드에서 테스트할 수 있고, 조회 실패 값인 `0`은 "32 아님"으로 취급합니다. ROCm에서 `ports.rocm`이 있는데도 포트가 없으면 `select_kernel_port`는 폭을 명시한 메시지("validated only on a 32-lane wavefront and this device reports 64 lanes")를 던지므로, CDNA 사용자가 이 거부를 포트 누락으로 오해하지 않습니다.

### 3.2 `device_warp_size()`와 무시하는 오버라이드

폭은 LOCAL_FIXES 항목 32로 기록된 새 오버레이 호출 `mlx::core::rocm::device_warp_size()`에서 옵니다. 현재 장치의 `hipDeviceAttributeWarpSize`를 읽고 `MLX_ROCM_FORCE_WARP_SIZE`는 의도적으로 무시합니다. 이 변수는 MLX 내부에서 실행 폭을 실험하기 위한 것이고, 이것이 64레인 장치에서 32레인용 커널을 풀어 주게 두면 실험용 손잡이가 정확성 스위치가 됩니다. ROCm을 쓸 수 없거나 HIP 호출 중 하나가 실패하면 0을 돌려줍니다. 지우는 HIP 오류는 자신이 일으킨 것뿐입니다. 스레드에 이미 대기 중인 오류(`hipPeekAtLastError`)가 있으면 그 주인을 위해 남겨 두므로 어느 오류도 잘못 보고되지 않습니다.

`gpu_backend.cpp`의 `rocm_port_warp_size()`는 이 값을 함수 지역 static으로 프로세스당 한 번 캐시하고, 32가 아니면 퓨즈드 커널이 그래프 폴백을 쓴다(또는 BitNet이 거부된다)는 stderr 한 줄을 출력합니다. CDNA 운영자가 디코드가 느린 이유를 추측하지 않고 바로 알 수 있습니다.

### 3.3 any-wave 테이블

`KernelPorts::rocm_any_wave_size`의 기본값은 false이므로, 새 포트는 작성자가 따로 밝히기 전까지 wave32에 묶입니다. 표시된 테이블은 5개입니다.

| 테이블 | HIP 소스 |
|---|---|
| `fused_rope_ports` | `FUSED_ROPE_APPEND_HIP_SOURCE` |
| `paged_merge_ports` | `PAGED_ATTENTION_MERGE_HIP_SOURCE` |
| `gumbel_ports` | `GUMBEL_MAX_SAMPLE_HIP_SOURCE` |
| `rejection_ports` | `REJECTION_SAMPLE_HIP_SOURCE` |
| `xielu_ports` | `XIELU_HIP_SOURCE` |

이슈는 앞의 4개를 나열했습니다. 현재 코드를 읽어 보니 `XIELU_HIP_SOURCE`는 순수 원소별 연산(원소당 스레드 하나, 레인 간 단계 없음)이어서 함께 표시했고, 샘플러와 RoPE 본문에 처음부터 가드가 없던 것과 같은 이유로 `#error` 가드를 제거했습니다. 매크로를 정의하는 컴파일러에서는 그 가드가 wave64에서 올바른 커널의 hipRTC 컴파일을 실패시키기 때문입니다. 나머지 ROCm 테이블 10개(bitlinear, ssm, mamba1_scan, moe_gateup, moe_down, moe_fc1_relu2, add3_layer_norm, fused_norm, paged_attention, paged_v2_partial)는 wave32 전용으로 남습니다. 이들의 가드는 그대로 두고(해가 없고, 매크로를 정의하는 컴파일러에서는 작동함) 주석만 `port_for`를 보호 수단으로 명시하도록 고쳤습니다.

### 3.4 BitNet은 로드 시점에 거부

이 PR 이전에는 `bitlinear_kernel_available`이 포트 테이블과 무관하게 "GPU 백엔드가 있음"을 돌려주었습니다. wave64에서는 참으로 답해 BitNet이 로드되고, 첫 BitLinear matmul이 요청 도중 `select_kernel_port`의 거부에 걸렸을 것이며, 갈 그래프 폴백도 없습니다. 이제 술어는 `has_kernel_port(bitlinear_ports())`를 돌려주므로 런처가 디스패치하는 같은 테이블을 읽고 wave64에서 거짓으로 답하며, `reject_without_bitlinear_kernel`이 웨이브프런트를 명시한 메시지로 로드 시점에 체크포인트를 거부합니다. 런처의 거부 문구도 더 이상 "graph fallback"을 약속하지 않고 BitNet 로드 시점 거부를 가리킵니다.

### 3.5 `mamba1_selective_scan`이 `Result`를 돌려줌

`select_kernel_port`는 거부할 때 예외를 던집니다. `-> Result<...>`로 선언되지 않은 cxx extern을 통과하는 예외는 `noexcept` 경계를 넘어 프로세스를 끝냅니다. Mamba1 스캔이 그런 extern이었고 이제 wave64에서 거부할 수 있으므로 선언을 `-> Result<()>`로 바꿨습니다. Mamba와 Jamba는 같은 테이블을 읽는 `mamba1_scan_kernel_accepts()`로 실행을 게이트하므로, 게이트가 이미 포트를 확인했다는 메시지와 함께 결과를 `expect`합니다. 그곳의 실패는 게이트와 런처가 어긋났다는 뜻이고, 처리할 조건이 아니라 드러내야 할 버그입니다.

### 3.6 검사기 규칙 5와 6

테이블을 any-wave로 표시하면 소스에 대한 주장 하나만으로 아무도 테스트하지 않은 하드웨어에서 실행됩니다. `verify-kernel-port-dispatch`는 이제 그 주장을 검사합니다.

- **규칙 5.** 표시된 테이블은 각각이 컴파일하는 HIP 소스와 함께 `EXPECTED_ANY_WAVE`에 고정됩니다. 검사기는 표시된 테이블마다 `.rocm` 람다를 따라 `get_*()` 접근자, 그것이 돌려주는 `static` holder, holder 구조체, 한 단계의 헬퍼까지 추적해 모든 `fast::hip_kernel` 호출을 모읍니다. 컴파일되는 소스는 고정된 소스와 정확히 같아야 하고, 추가 인자 없이 전달되어야 하며(추가 인자는 스캔이 보지 못하는 헤더일 수 있음), 단일 raw 리터럴로 정의되어야 합니다(인접 리터럴은 이어 붙여지는데 첫 번째만 스캔됨). 주석을 제거한 본문은 `LANE_OPS`의 어떤 것과도 일치하면 안 됩니다: 셔플, ballot, vote, 마스크 인트린식, `warpSize`, AMDGCN 레인 빌트인(`ds_swizzle`, DPP, `readlane`, `permlane`, `ballot` 등), 웨이브프런트 매크로, `threadIdx.x`에 대한 32레인 산술. 고정 목록에 없는 표시 테이블도 실패하고, 표시를 잃은 고정 테이블도 실패하므로 집합은 어느 방향으로도 리뷰 없이 바뀔 수 없습니다.
- **규칙 6.** `set_rocm_port_warp_size_for_tests`는 `tests/`, `*_tests.rs`, `test_support/` 아래의 Rust와 정의부에만 나타날 수 있고, 정의 파일마다 정확한 사용 횟수(`SEAM_DEFINITIONS`)로 묶여 있어 큰 브리지 파일 안에 추가된 프로덕션 호출도 실패합니다.

`scripts/ci/check_kernel_port_dispatch_test.sh`는 런처 소스를 임시 트리로 복사하고, 복사본을 변형한 뒤 `--root`로 검사기를 실행합니다. 사례는 13개입니다. 실패해야 하는 변형 10개(셔플 기반 테이블 표시, 표시된 본문에 셔플, ballot, AMDGCN ballot 빌트인 추가, 표시된 항목을 다른 holder로 연결, 헤더를 넘기는 표시된 holder, 표시가 빠진 고정 테이블, 프로덕션 Rust와 C++에서의 seam 호출, 정의하는 브리지 파일 안의 두 번째 호출)와 통과해야 하는 대조군 3개(변형하지 않은 트리, 주석에만 등장하는 레인 인트린식, 테스트 모듈의 seam 호출)입니다. 각 변형은 치환 문자열이 더 이상 일치하지 않으면 큰 소리로 실패하므로, 변형이 조용히 적용되지 않아 사례가 통과하는 일은 없습니다. make 타깃과 CI 단계가 모두 실행합니다.

### 3.7 `build.rs`의 추적 누락

`mlxcel-core/build.rs`는 `kernel_port.*`와 `gpu_backend.*`를 `rerun-if-changed`로 등록하지 않았기 때문에, 이 파일을 고쳐도 추적되는 다른 파일이 바뀔 때까지 브리지 라이브러리가 낡은 상태로 남았습니다. 개발 중에 부정 검사(`port_for`의 보류를 끄고 wave64 테스트가 실패하기를 기대)가 통과하면서 드러났습니다. 테스트가 이전 빌드로 실행되고 있었던 것입니다. 이제 네 파일이 추적됩니다. 이 누락은 이 PR보다 오래되었고(헬퍼는 #1803부터 있었음), 웜 트리에서 포트 선택에 대한 이전 수정도 가렸을 수 있습니다.

## 4. 검증

### 4.1 테스트

- **진리표**(`kernel_port_tests.rs`, 모든 백엔드): cxx 브리지를 통해 `rocm_port_allowed`의 `(false, 32) = true`, `(false, 64) = false`, `(true, 64) = true`, `(false, 0) = false`. 2개 통과.
- **하드웨어 폭**(`tests/rocm_wave_size.rs`): gfx1151에서 `device_warp_size() == 32`; `MLX_ROCM_FORCE_WARP_SIZE=64`를 준 자식 프로세스(장치 로그로 MLX에 오버라이드가 적용됨을 확인)에서도 32; 포트 술어 16개 모두 참.
- **seam을 통한 wave64 거부**(`tests/rocm_wave64_port_refusal.rs`): 어떤 술어도 묻기 전에 보고 폭을 64로 설정하면, wave32 전용 술어 11개(테이블 10개를 덮음)는 모두 거짓이고 any-wave 5개는 참입니다. `bitlinear_matmul`, `fused_add_rms_norm`, `fused_moe_expert_kernel`(moe_gateup)은 폭을 명시한 메시지로 거부하고, 퓨전을 켠 `layers::fused_add_rms_norm`과 `residual_add3_layer_norm`은 그래프 경로와 같은 바이트를 돌려줍니다. `port_for`에서 보류를 제거하면 이 테스트는 11개 항목을 모두 나열하며 실패하므로, 지키려는 회귀를 실제로 감지합니다.

Rust 게이트가 `true` 답을 캐시하므로 seam은 첫 술어 호출 전에 설정해야 합니다. 그래서 거부 테스트는 seam을 먼저 설정하고 모든 검사를 순서대로 실행하는 테스트 하나만 가진 별도 바이너리입니다.

### 4.2 gfx1151은 그대로

`Meta-Llama-3.1-8B-Instruct-4bit`, `scripts/rocm_gpu_guard.sh` 아래의 릴리스 바이너리: `MLXCEL_FUSED_ADD_RMSNORM=0`과 `=1`의 그리디 64토큰 출력이 동일합니다(33.88, 33.90 tok/s, 각 1회, 성능 주장 아님). 샘플링 실행의 `rocprofv3 --kernel-trace`에는 `custom_kernel_mlxcel_fused_add_rms_norm`(wave32 전용, 544회 실행)과 `custom_kernel_mlxcel_gumbel_max_sample`(any-wave)이 여전히 선택되어 있고, stderr에 웨이브프런트 알림이 없습니다. 술어 16개가 모두 참이고 프로파일의 커널 이름이 같으므로 RDNA에서의 포트 선택은 이전과 같습니다.

### 4.3 게이트

PR 기준: 디버그 프로필의 대상 포트 테스트(fused norm, RoPE, MoE, relu2, SSM, Mamba1, Gumbel, rejection, fixed-key, paged v2, MLA, residual add3) 357개 통과. `cache::paged_detach` 테스트 2개는 게이트의 `test-fast` 프로필에는 컴파일되지 않는 `debug_assert`에서 실패하며 포트와 무관합니다. Apertus, BitNet, Cohere2 테스트 26개 통과. `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-fmt verify-rocm-overlay` 통과. 리뷰 수정 후 대상 테스트(mlxcel-core 29개, Mamba/Jamba 30개, 통합 파일 3개)를 다시 실행해 통과했습니다.

오케스트레이터 기준, 헤드 `1dc419c6`: `make verify-rocm`이 모든 단계를 통과했습니다. 테스트 11,969개 통과, 0개 실패, 381개 무시, smoke 단계 OK.

검증하지 않은 것: wave64 장치가 없어 실제 CDNA 하드웨어에서의 거부는 테스트되지 않았습니다. Metal과 CUDA는 호스트에 없었습니다. 이들의 `port_for` 분기는 바뀌지 않았고, 두 테이블 모두 포트가 있으므로 `bitlinear_kernel_available`의 답도 같습니다.

## 5. 기술적 선택과 그 이유

- **컴파일 시점 검사 대신 호스트 쪽 보류.** HIP에서 `static_assert(warpSize == 32)`는 컴파일되지 않고 매크로는 정의되지 않으므로, 현재 툴체인에는 컴파일 시점 수단이 없습니다. 하드웨어 속성은 신뢰할 수 있고 한 번 읽는 비용도 작습니다.
- **런처별이 아니라 포트 선택 시점.** 런처별 검사는 모든 런처에 판단을 복제하고 술어와 런처가 어긋날 여지를 만듭니다. `select_kernel_port`는 그것을 막으려고 존재합니다.
- **`Device::warp_size()`가 아닌 하드웨어 폭.** 오버라이드는 MLX 자체 커널의 실행 폭 실험용이며, mlxcel이 wave64에서 wave32 커널을 실행하게 만들면 안 됩니다.
- **기본값 false.** wave64를 고려하지 않고 작성한 포트는 기본적으로 보류됩니다. 그 대가로 레인 연산이 없는 올바른 포트도 누군가 표시할 때까지 보류되며, 표시는 검사기가 확인합니다.
- **표시를 기계적으로 검증.** any-wave 주장은 주석이 아니라 고정 목록과 부정 테스트를 갖춘 소스 스캔으로 강제됩니다. 주석은 작동하지 않던 `#error` 가드처럼 어긋날 수 있습니다.
- **BitNet은 로드 시점 거부.** BitLinear에는 그래프 폴백이 없으므로 wave64에서의 선택지는 검증되지 않은 커널, 요청 도중 실패, 로드 시점 거부뿐이고, 운영자가 대응할 수 있는 것은 마지막입니다.
- **seam의 정확한 사용 횟수.** 브리지 파일에 파일 단위 예외를 주면 수천 줄짜리 파일에 프로덕션 호출이 끼어들 수 있습니다.

## 6. 잔여 위험과 후속 작업

- **seam이 공개되어 있음.** `set_rocm_port_warp_size_for_tests`는 일반 cxx 브리지 함수이므로 `mlxcel-core`를 링크하는 어떤 크레이트도 호출할 수 있습니다. 규칙 6은 이 저장소만 스캔합니다. 기능 플래그나 테스트 전용 빌드 뒤로 숨기면 이 틈이 닫힙니다.
- **프로세스당 폭 캐시.** 폭은 첫 호출 시점의 현재 장치에 대해 한 번만 읽습니다. RDNA와 CDNA 카드를 섞은 호스트에서는 첫 답이 유지되므로, 다른 종류의 GPU로 옮겨 간 프로세스는 잘못된 보류를 적용하고, CDNA 장치에서는 wave32 전용 커널이 실행될 수 있습니다. 헤더 주석은 이 경우를 처리하지 않는다고 명시합니다.
- **실패한 조회가 0으로 캐시됨.** `device_warp_size()`가 한 번 실패하면(예: 첫 사용 시 일시적인 HIP 오류) 프로세스는 수명 내내 wave32 장치에서도 모든 wave32 전용 포트를 보류합니다. 안전한 쪽으로 실패하지만 느리고, stderr 알림에는 "query failed"가 나옵니다. 0 결과를 재시도하거나 캐시하지 않으면 끈질긴 성능 저하를 피할 수 있습니다.
- **wave64 실행 없음.** wave32 전용 포트 10개는 검증된 것이 아니라 거부된 것입니다. CDNA에서 포트를 검증해 any-wave로 표시하는 일은 커널별이고 CDNA 호스트가 필요하며 범위 밖이었습니다. any-wave 포트 5개도 wave64에서 실행된 적이 없고, 표시의 근거는 소스 스캔입니다.
- **레인 연산 스캔은 어휘적.** `LANE_OPS`는 현재 쓰이는 인트린식 계열, AMDGCN 빌트인, 흔한 `threadIdx.x` 인덱스 형태를 다룹니다. 다른 방식으로 레인 인덱스를 만드는 본문(예: 헬퍼 변수에 담은 뒤 `% 32`)은 잡히지 않습니다.

## 7. 학습 포인트

- **발동하지 않는 가드는 지켜지는 가드와 똑같아 보입니다.** `#if defined(MACRO) && MACRO != 32`는 매크로가 없으면 조용합니다. 툴체인이 제공하는 기호에 의존하는 검사는 지원하는 각 툴체인에서 그 기호가 있는지 확인하거나, 값이 확실히 존재하는 곳(여기서는 런타임의 장치 속성)에 검사를 두어야 합니다.
- **정책은 모든 경로가 이미 공유하는 함수에 둡니다.** 술어와 런처가 모두 `port_for`를 읽기 때문에 그곳의 조건 하나가 모든 호출자를 일관되게 바꿨고, 기존 그래프 폴백이 나머지를 처리했습니다.
- **테스트 없이 코드를 실행하게 하는 주장은 기계로 검사합니다.** any-wave 표시는 소스 텍스트에 대한 주장이며, 고정 목록, 접근자를 따라가는 스캔, 부정 변형이 그 주장이 어긋나지 않게 합니다.
- **통과한 부정 검사는 설명이 필요합니다.** 보류를 제거하면 거부 테스트가 깨져야 했는데 깨지지 않았고, 원인은 약한 테스트가 아니라 `rerun-if-changed` 누락으로 인한 낡은 빌드였습니다. 빌드 의존성의 틈은 어떤 검증이든 오래된 코드로 실행되게 만들 수 있습니다.
- **없는 하드웨어에서의 거부는 seam으로 입증하고, 그 테스트가 실패할 수 있음도 입증합니다.** 테스트 전용 폭 오버라이드는 폭을 읽는 지점 아래의 프로덕션 경로를 실행하고, 보류를 끄면 테스트가 실패하므로 테스트가 seam이 아니라 보류를 측정한다는 것이 드러납니다.
