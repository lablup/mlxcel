# PR #2604: Windows ROCm 빌드를 PATH와 ROCM_PATH 없이 실행

**날짜**: 2026-10-26
**상태**: Windows에서 완료. Linux ROCm은 실행하지 않음(CI ROCm 잡 중단 상태)
**언어**: Rust, C++, CMake, Markdown
**위험도**: 중간

## 요약

PR #2604(이슈 #2602)는 #2601 이후 Windows ROCm 빌드에 남아 있던 두 가지 실행 조건, 즉 ROCm `bin` 디렉터리가 `PATH`에 있어야 하고 `ROCM_PATH`/`HIP_PATH`가 설정되어 있어야 한다는 조건을 없앴다. ROCm DLL은 빌드 시점에 기록한 SDK에서 전체 경로로 delay-load하고, hiprtc JIT는 빌드 시점 include 디렉터리로 헤더를 찾으며, Windows 통합 GPU 메모리 한도는 OS RAM이 아니라 디바이스 풀을 기준으로 한다. `mlxcel generate`와 `mlxcel-server`를 일반 `cmd.exe`에서 `PATH=C:\Windows\System32;C:\Windows`, ROCm 변수 없음, 비어 있는 JIT 캐시, 작업 디렉터리의 가짜 ROCm DLL 조건으로 검증했다.

작업 중에 #2601에서 아무도 몰랐던 정확성 버그도 드러났다. AMD 디스플레이 드라이버가 System32에 더 오래된 `amdhip64_7.dll`과 `amd_comgr.dll`을 설치하는데, #2601 바이너리는 SDK가 아니라 드라이버의 HIP 런타임으로 실행되고 있었다.

## 1. 문제 정의

#2601 이후 Windows `--features rocm` 바이너리는 ROCm DLL 다섯 개(`amdhip64_7`, `rocblas`, `libhipblaslt`, `hipfft`, `hiprtc0716`)를 이름으로 import했고, hiprtc 헤더는 `ROCM_HOME`/`ROCM_PATH`/`HIP_PATH`로만 찾았다. 그래서 바이너리가 설정된 셸에 의존했고, DLL 검색 순서 문제도 남았다. SafeDllSearchMode에서는 작업 디렉터리를 `PATH`보다 먼저 검색하므로, 모델 파일 옆이나 다운로드 폴더에 심어 둔 DLL이 SDK 대신 로드될 수 있었다. #2599는 CUDA에서 같은 두 문제를 delay-load와 내장 include 디렉터리로 풀었다. 이 PR은 그 설계를 따르고, CUDA 설계에는 없는 ROCm 고유의 빈틈을 막는다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 링크 | `windows_rocm::select_delayload_dlls`와 `emit_delayload_link_args`가 import되는 DLL 다섯 개에 `/DELAYLOAD`를 내보내고(prefix는 맞지만 import되지 않는 `hiprtc-builtins*`, `hipfftw`는 제외), `delayimp`를 링크한다. `mlxcel-core/build.rs`와 루트 `build.rs` 양쪽에서 호출한다. |
| delay-load 훅 | 새 overlay `delayload.cpp`: ROCm 이름에 대한 `dliNotePreLoadLibrary`에서 `MLXCEL_ROCM_BIN_DIR`의 전체 경로로 로드한다. 프로세스당 한 번 `AddDllDirectory(<bin>)`를 호출하고 `amd_comgr*.dll`, `hiprtc-builtins*.dll`을 전체 경로로 미리 로드한다. |
| hiprtc 헤더 | `jit_module.cpp`가 순수 함수 `choose_rocm_include_dir()`로 include 디렉터리를 정한다. 순서는 `ROCM_HOME`, `ROCM_PATH`, `HIP_PATH`(빈 값은 미설정), 빌드 시점 `MLXCEL_ROCM_INCLUDE_DIR`(존재할 때), `/opt/rocm`. |
| 메모리 | `_WIN32`에서 통합 GPU allocator 한도는 시스템 RAM 항 없이 `total * 0.8`. Rust에 `GlobalMemoryStatusEx` 프로브를 추가해 Windows CPU 전용 추정이 0이 되지 않게 했다. |
| 빌드 가드 | 상대 경로 `ROCM_PATH`/`HIP_PATH`, 그리고 `amdhip64_*.dll`이 없는 SDK `bin`은 이제 빌드를 실패시킨다. 작업 디렉터리를 검색하는 바이너리가 조용히 만들어지지 않는다. |
| 문서 | 실행 제약 제거, DLL 해석 방식 설명, SDK 디렉터리는 관리자만 쓸 수 있어야 한다는 요구, Windows에서 `ROCM_PATH`/`HIP_PATH`는 빌드 전용. `LOCAL_FIXES.md` 57번. |

## 3. 기술적 선택과 그 이유

### 3.1 프로세스 검색 경로를 제한하지 않고 런타임 로드 DLL을 미리 로드

import되는 DLL 다섯 개는 delay-load 훅이 처리하지만, hiprtc와 HIP 런타임은 `amd_comgr`와 hiprtc builtins를 스스로 `LoadLibrary`로 불러오고 훅은 이 호출을 보지 못한다. 쉬운 해법인 `SetDefaultDllDirectories(LOAD_LIBRARY_SEARCH_DEFAULT_DIRS)`는 디스플레이 드라이버의 자체 로드를 포함해 프로세스 전체의 검색 순서를 바꾼다. 이 프로젝트가 통제할 수 없고 드라이버 버전별로 시험할 수도 없는 영역이다. 대신 이 PR은 해당 DLL을 기록된 `bin`에서 전체 경로로 먼저 로드한다. Windows는 디렉터리보다 이미 로드된 모듈 목록을 먼저 확인하므로, 이후 이름만으로 하는 로드는 SDK 사본에 묶인다. 대가는 훅이 모르는 DLL을 이름만으로 로드하면 여전히 일반 검색 순서를 따른다는 점이다. 보안 리뷰에서 SDK DLL에 들어 있는 이름을 조사했다. 선택적 DLL 세 개(`hsa-amd-aqlprofile64`, `hsa-ext-finalize64`, `dtif64a`)는 SDK에도 System32에도 없고, 생성과 서빙에서는 로드되지 않으며, 남은 노출로 문서화했다.

### 3.2 null 반환 대신 helper의 예외를 발생

이슈는 기록된 `bin`에 DLL이 없으면 훅이 null을 반환해 로더가 일반 오류를 내도록 하라고 했다. 개발 과정에서 `__delayLoadHelper2`가 null을 "의견 없음"으로 보고 일반 `LoadLibraryExA(name, NULL, 0)`로 재시도한다는 것을 확인했다. 이 재시도는 작업 디렉터리와 `PATH`를 검색하는데, 이 PR이 막으려는 바로 그 경로다. 그래서 훅은 기대 경로를 출력하고, 재시도가 실패했을 때 helper가 던지는 것과 같은 `ERROR_MOD_NOT_FOUND` 예외를 helper 자신의 `DelayLoadInfo`와 함께 발생시킨다. 구현 리뷰에서 helper 동작과 대조해 확인했다.

### 3.3 Windows APU 한도를 디바이스 풀로 제한

Linux 공식 `max(total * 0.8, sys_ram * 0.8)`은 APU 풀이 시스템 RAM에서 나온다고 가정한다. Windows WDDM에서는 큰 풀이 디바이스 메모리로 보고되는 BIOS carve-out이다. 1 GiB `hipMalloc`를 실패할 때까지 할당하는 프로브는 보고된 107.87 GiB, OS RAM 31.65 GiB 조건에서 105 GiB에서 멈췄다. 할당이 보고된 총량을 넘지 않으므로 Windows 한도는 `total * 0.8`(이 호스트에서 86.29 GiB)이고, `--estimate-memory`도 이 값을 보고한다.

### 3.4 빌드에서 닫힌 쪽으로 실패

리뷰 지적 중 두 건은 설정 오류가 빌드 실패가 아니라 덜 안전한 바이너리로 조용히 이어지는 경우였다. 상대 경로 `ROCM_PATH`는 `MLXCEL_ROCM_BIN_DIR`에 그대로 박혀 실행 시점 작업 디렉터리 기준으로 해석되고, 읽을 수 없거나 잘못된 `bin`은 `/DELAYLOAD` 인자를 하나도 내지 않아 이름 기반 import로 돌아간다. 이제 둘 다 문제 값을 명시하고 panic한다. CUDA helper는 기존 형태를 유지한다.

## 4. 발견한 잠재 결함: System32의 HIP 런타임

Adrenalin 드라이버는 ROCm 10.1보다 오래된 `amdhip64_7.dll`과 `amd_comgr.dll`을 System32에 둔다. 검색 순서에서 System32가 `PATH`보다 앞서므로, SDK를 `PATH`에 두고 실행한 #2601 바이너리는 드라이버의 HIP 런타임을 로드하고 수학 라이브러리만 `PATH`의 SDK에서 가져왔다. 드라이버의 런타임과 comgr가 서로 맞았기 때문에 동작했을 뿐이다. SDK의 `amdhip64_7.dll`을 드라이버의 `amd_comgr.dll`과 함께 로드한 프로브는 디바이스를 하나도 찾지 못했다. 전체 경로 delay-load와 comgr 사전 로드로 둘 다 SDK에서 오게 되어 조합이 결정적이 되었다.

## 5. 검증

Windows 11, Radeon 8060S(`gfx1151`), ROCm 10.1.0, VS 18:

- 두 바이너리에 `llvm-readobj --coff-imports`: ROCm DLL 다섯 개 모두 `DelayImport`. 시스템 외 eager import는 `libopenblas.dll`뿐.
- 깨끗한 `cmd.exe`(`PATH`는 System32만, ROCm 변수 없음, 새 `MLX_HSACO_CACHE_DIR`): Qwen3-0.6B-4bit 약 117~121 tok/s, Qwen3-1.7B-bf16 약 28.6~30.4 tok/s, 올바른 텍스트. #2601 바이너리가 `ROCM_PATH` 없이 실패하던 첫 JIT 컴파일 포함.
- 작업 디렉터리의 가짜 DLL(`version.dll`을 `amdhip64_7`, `amd_comgr`, `hiprtc0716`, `hiprtc-builtins0716`, `rocblas`로 이름만 바꾼 것, 두 번째 실행에서는 3.1의 선택적 이름 세 개와 `amd_comgr_3`, `atidxx64` 추가): 하나도 로드되지 않았고 모든 ROCm 모듈이 `C:\rocm\10.1.0\bin`에서 왔다. #2601 바이너리는 `rocblas.dll` 가짜만으로 시작 시 종료한다.
- 같은 환경의 `mlxcel-server`: `/v1/chat/completions`(스트리밍, 비스트리밍)와 `/v1/messages`가 "The capital of France is **Paris**."를 예상한 종료 사유와 함께 반환.
- `cargo test -p mlxcel-mlx-pin --lib`(49), `memory_estimate` 테스트(64), fmt, `--features rocm` clippy, overlay `drift`와 `records`.

## 6. 알려진 공백

- **Linux ROCm**은 빌드도 실행도 하지 않았다(호스트 없음, CI 잡 중단, #1811). 모든 변경은 Windows 조건 안에 있고, 리뷰에서 Linux `-I` 문자열과 allocator 공식이 그대로임을 확인했다.
- **Windows CUDA**는 다시 빌드하지 않았다(NVIDIA GPU 없음). 루트 `build.rs`의 호출은 그대로다.
- **DLL 누락 오류 경로**는 실행해 보지 않았다(SDK를 옮겨야 함).
- **SDK 디렉터리 권한**: 드라이브 루트 아래 디렉터리는 인증된 모든 사용자의 쓰기 권한을 상속한다. 문서에 관리자 전용 디렉터리를 요구하고 `icacls` 명령을 적었지만 빌드가 이를 검사하지는 않는다.
- **실행 시점 환경 변수**는 Linux, CUDA와 마찬가지로 빌드 시점 include 디렉터리보다 우선한다. SDK 디렉터리와 같은 보호가 필요하다.
- **이름만으로 로드되는 선택적 DLL**(3.1)은 프로파일링 경로가 이를 로드하게 되면 일반 검색 순서를 따른다.
- **기존 문제**: JIT 커널 캐시 기본 위치가 공유 임시 디렉터리(Linux에서는 `/tmp`)라서 다른 사용자가 먼저 만들 수 있다.

## 7. 학습 포인트

- Windows에서는 GPU 드라이버가 벤더 런타임의 오래된 사본을 System32에 둘 수 있고, System32는 `PATH`보다 우선한다. "SDK를 PATH에 두면 동작한다"가 SDK로 동작한다는 뜻은 아니다.
- delay-load 훅은 바이너리가 import하는 DLL만 다룬다. 라이브러리가 직접 `LoadLibrary`하는 DLL은 따로 다뤄야 하고, 전체 경로 사전 로드는 프로세스 기본 검색 순서를 바꾸는 것보다 범위가 좁은 해법이다.
- "null을 반환하면 된다"에 기대기 전에 delay-load helper의 fallback 동작을 확인해야 한다. fallback은 일반 검색이다.
