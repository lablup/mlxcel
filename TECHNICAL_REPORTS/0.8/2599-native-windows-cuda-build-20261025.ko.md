# PR #2599: Windows에서 CUDA로 mlxcel 네이티브 빌드와 실행

**Date**: 2026-10-25
**Status**: 로컬 검증 완료 (이 시점 기준 Linux/macOS CI 대기 중)
**Languages**: Rust, C++, CMake, Markdown
**Risk Level**: Medium

## 요약

PR #2599는 issue #58 스파이크를 실제로 동작하는 Windows 네이티브 빌드로 마무리한다. MSVC 환경에서 `cargo build --release --features cuda`를 실행하면 `mlxcel.exe`와 `mlxcel-server.exe`가 나오고, 두 바이너리 모두 CUDA 환경변수 없이, 툴킷 디렉터리를 `PATH`에 넣지 않은 상태로 단일 GPU 추론을 수행한다. 사용할 수 있는 NVIDIA Windows 장비가 ARM64(NVIDIA RTX Spark N1X, `sm_121`) 한 대뿐이어서 이슈 담당자와 합의해 `aarch64-pc-windows-msvc`에서 진행했다. 빌드 코드는 `x64`와 `arm64` 툴킷 배치를 똑같이 처리하지만 x86_64 GPU 장비에서는 아직 돌려 보지 않았다.

Phase 1 게이트는 MLX를 고치지 않고 통과했다. 고정된 업스트림 트리에 mlxcel의 CPU·CUDA 오버레이를 얹은 상태로, arm64 라이브러리 위치만 CMake에 알려 주자 MSVC 14.51과 nvcc 13.4로 끝까지 컴파일됐다. 남은 작업은 세 갈래였다. 하나는 CMake가 MLX 사용자에게 원래 해 주던 일(지연 로드 링크 옵션, 지연 로드 훅 오브젝트, OpenBLAS, 정적 라이브러리 정의)을 cargo 쪽에서 다시 해 주는 것이고, 둘째는 런타임 DLL 탐색, 셋째는 MSVC 링커만 잡아내는 잠복 결함들이다.

## 1. 문제 정의

이 PR 전까지 Windows는 문서에도 없었고 크레이트도 컴파일되지 않았다. 이슈는 막힌 지점을 세 층으로 나눴다.

- **업스트림 MLX**: Windows x86_64 CUDA는 업스트림 CI에서 빌드만 하고 실행은 하지 않는다. Windows on Arm CUDA는 업스트림 검증이 아예 없다.
- **mlxcel 빌드 코드**: `build.rs`가 Linux를 전제로 했다(`lib64`, `/usr/local/cuda/bin/nvcc`, GCC 플래그, 시스템 OpenBLAS). MLX가 CMake 타깃에 PUBLIC 옵션으로 붙여 둔 정적 링크용 지연 로드 설정은 cargo 링크까지 전달되지 않는다.
- **Rust 코드**: Unix API(`tokio::net::UnixListener`, `libc` errno 상수, 디렉터리 fd 기반 파일 연산)와, 이슈가 Windows 범위에서 뺀 분산 추론 스택이다.

완료 조건은 NVIDIA Windows 장비에서 네이티브 `--features cuda` 빌드가 Phase 5 스모크 테스트(GPU 생성, chat completions 서빙)를 단일 노드로 통과하는 것이었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 빌드 코드 | `mlxcel-core/build.rs`와 루트 `build.rs`가 함께 쓰는 `build_support/windows_cuda.rs` 추가. 툴킷·cuDNN 배치(`lib\<arch>`, `bin\<arch>`), Ninja 생성기, 슬래시로 쓴 CMake 경로, CUDA·cuDNN DLL용 `/DELAYLOAD`, `/INCLUDE:__pfnDliNotifyHook2`를 담당한다. 단위 테스트는 `mlxcel-mlx-pin`에서 돈다. |
| C++ 브리지 | MSVC 분기 추가: `/std:c++20`, `MLX_STATIC`, `/bigobj`, `/Zc:preprocessor`, `/fp:fast`. `MLXCEL_CXX_MARCH`는 `/arch:<값>`으로 넘긴다. |
| CPU 백엔드 | MLX의 CMake가 내려받은 OpenBLAS import 라이브러리를 링크하고 `openblas.dll`을 빌드된 실행 파일 옆에 복사한다. |
| CUDA 오버레이 | `delayload.cpp` 오버레이 신규, `jit_module.cpp` 헤더 탐색 수정, `matmul.cpp`에 `BlockMaskedMM::eval_gpu` 추가. |
| 의존성 | `tokenizers`의 기본 기능 `esaxx_fast` 제외, 브리지의 죽은 C++ `split()` 삭제. |
| 단일 노드 | `src/distributed/platform.rs`와 `ServerDistributedFlags`로 `generate`, `serve`, `mlxcel-server`가 모델을 해석하기 전에 분산 플래그를 거부한다. |
| Unix 게이팅 | `UnixListener` 경로와 Unix 전용 헬퍼·테스트를 `cfg`로 막았다. Windows 리스너는 `SO_REUSEADDR`를 더 이상 설정하지 않고, 다운로드 임시 파일은 배타적으로 생성한다. |
| 문서 | `docs/installation.md`에 Windows 절 추가, README, `docs/distributed.md`, `docs/environment-variables.md` 갱신. |

## 3. 기술적 선택과 그 이유

### 3.1 `mlx.dll`을 만드는 대신 cargo에서 지연 로드 링크 단계를 재현

MLX CUDA 백엔드는 Windows 바이너리를 링크할 때 CUDA·cuDNN DLL마다 `/DELAYLOAD`를 걸고 `delayload.cpp`를 사용자 쪽에 컴파일해 넣는다. `mlx`가 정적 라이브러리면 CMake는 두 가지를 PUBLIC 사용 요건으로 붙인다. cargo는 `mlx.lib`를 직접 링크하므로 둘 다 최종 링크에 닿지 않고, 그러면 실행 파일이 DLL을 즉시 import해서 툴킷이 `PATH`에 없으면 시작조차 하지 못한다.

이 PR은 정적 라이브러리를 유지하면서 링크 단계를 재현했다. 의존 크레이트의 `rustc-link-arg`는 그 크레이트를 링크하는 바이너리까지 전달되지 않으므로, `/DELAYLOAD` 목록은 공용 모듈 하나에서 두 빌드 스크립트가 각각 내보낸다. ROCm rpath를 처리하는 `rocm_rpath.rs`가 이미 쓰는 방식과 같다. 목록은 실제로 링크하는 import 라이브러리 뒤의 DLL로 한정했다. `cudart64_`와 `cublas64_`는 현재 pin에서 아무도 import하지 않지만(nvcc가 정적 cudart를 링크하고 MLX는 cuBLASLt만 호출한다) 목록에 남겼다. 나중 pin이 이들을 import하더라도 계속 지연 로드되게 하려는 것이다.

훅 오브젝트는 `mlx.lib` 안에 있지만 `__pfnDliNotifyHook2`를 참조하는 곳이 없어서, 그대로 두면 링커가 `delayimp.lib`의 null 기본값을 가져간다. `/INCLUDE`로 이 심볼을 강제하고, `mlx`를 `delayimp`보다 먼저 링크한다. 첫 버전은 `delayload.cpp`를 브리지에도 컴파일해 넣었는데, 구현 리뷰가 `mlx.lib`에 같은 정의가 이미 있다는 것을 찾아내서 제거했다.

이슈가 제시한 대안은 MLX를 `mlx.dll`로 빌드하는 것이었다. 배포할 때마다 DLL이 하나 늘고, 다른 플랫폼과 공유하는 링크 모델이 바뀌며, 아래의 런타임 DLL 탐색 문제도 그대로 남기 때문에 택하지 않았다.

### 3.2 런타임 DLL 탐색은 `PATH`가 아니라 훅에서 해결

`dumpbin`으로 보면 최종 실행 파일은 `cublasLt64_13`, `cufft64_12`, `cusolver64_12`, `cudnn64_9`, `nvrtc64_130_0`을 지연 로드하고, `nvcuda.dll`이나 `cudart`는 직접 import하지 않는다. 런타임 실패는 원인이 다른 세 가지가 차례로 나타났다.

1. `0xC06D007E`(지연 로드 모듈 없음): bin 디렉터리 경로가 백슬래시째 CMake에 넘어가 C++ 문자열 리터럴이 됐고, `\t`, `\b`, `\a`가 제어 문자로 바뀌었다. 이제 CMake로 넘기는 경로는 모두 슬래시로 쓴다(`windows_cuda::cmake_path`, 바로 그 경로로 테스트한다).
2. `failed to open nvrtc-builtins64_134.dll`: NVRTC는 builtins DLL을 이름만으로 연다. `AddDllDirectory`는 사용자 디렉터리를 요청한 `LoadLibraryEx` 호출에만 적용된다.
3. `Could not locate cudnn_engines_runtime_compiled64_9.dll`, 이어서 `cublasLt64_13.dll`과 fail-fast `0xC0000409`: cuDNN도 하위 라이브러리와 cuBLAS를 같은 방식으로 연다.

그래서 `delayload.cpp` 오버레이는 NVRTC와 cuDNN이 이름으로 여는 DLL을 전체 경로로 미리 올린다(cuBLAS와 NVRTC를 cuDNN 하위 라이브러리보다 먼저). 나중에 이름으로 다시 열 때는 이미 프로세스에 올라와 있는 모듈을 찾게 된다. 각 DLL은 `LOAD_LIBRARY_SEARCH_DLL_LOAD_DIR | LOAD_LIBRARY_SEARCH_DEFAULT_DIRS`로 로드하는데, 이 플래그는 DLL의 import를 그 DLL이 있는 디렉터리에서 찾고 현재 디렉터리나 `PATH`는 보지 않는다. 첫 버전은 `LOAD_WITH_ALTERED_SEARCH_PATH`를 썼고, 보안 리뷰가 그 방식으로는 두 곳 모두 여전히 탐색된다고 지적했다.

### 3.3 MLX의 CMake는 손대지 않음

`FindCUDAToolkit`은 호스트 프로세서가 `AMD64`일 때만 Windows 라이브러리 탐색 접미사를 설정하고, MLX의 `FindCUDNN`은 `lib/x64`를 찾으며, MLX는 `MLX_CUDA_BIN_DIR`의 기본값을 `bin/x64`로 둔다. 셋 모두 밖에서 설정할 수 있어서(`CMAKE_LIBRARY_PATH`, `CUDNN_LIBRARY_PATH`, `CUDNN_INCLUDE_PATH`, `MLX_CUDA_BIN_DIR`) `build.rs`가 값을 넘기고 CMake 오버레이는 추가하지 않았다. 코드 수정이 필요했던 곳은 NVRTC 헤더 탐색 하나다. 오버레이의 `default_cuda_toolkit_path()`는 설치 프로그램 디렉터리가 없으면 `directory_iterator`에서 예외를 던졌다. 이제 Windows에서는 `CUDA_HOME` / `CUDA_PATH` 다음 순서로 빌드에 쓴 툴킷의 `include\`(`MLXCEL_CUDA_INCLUDE_DIR`)를 쓴다. Linux가 `/usr/local/cuda`를 그대로 쓰도록 이 대체 경로는 Windows에만 적용했다.

### 3.4 분산 모드는 컴파일에서 빼지 않고 시작 시점에 거부

`src/distributed/`는 Windows에서도 컴파일된다(Unix API는 `target_os`로 막힌 RDMA 탐지 하나뿐이다). `cfg`로 막으면 없애야 할 컴파일 에러도 없이 복잡도만 늘어난다. 실제로 빠진 것은 Windows에서의 멀티 호스트 전송 검증이다. 그래서 모든 분산 플래그(1보다 큰 `--tp-size`, PP, 클러스터·분리 서빙 플래그)를 해당 플래그 이름을 담은 에러로 거부한다. 두 서버 CLI는 모델 참조를 해석하기 전에 검사하므로, 결국 거부될 repo id를 내려받는 일이 생기지 않는다. 프로그램에서 직접 호출하는 경우를 위해 `start_server`도 한 번 더 검사한다.

## 4. MSVC가 드러낸 잠복 결함

GNU ld는 `--gc-sections`를 쓰면 참조되지 않는 섹션을 먼저 버린 뒤에 그 안의 미정의 심볼을 검사하지만, MSVC는 검사를 먼저 한다. 이 PR 이전부터 있던 결함 세 개가 그렇게 드러났다.

- CUDA `matmul.cpp` 오버레이는 업스트림 `9a795735` 기준이라, 그 뒤에 업스트림이 추가한 `BlockMaskedMM::eval_gpu`가 없었다. `ops.cpp`의 vtable은 모든 CUDA 빌드에서 이 함수를 참조했고, Linux에서는 그 vtable을 쓰지 않아서 링크가 통과했을 뿐이다. 함수는 pin 버전 그대로 가져왔고, 파일의 나머지는 리베이스하지 않았으므로 오버레이의 기준 커밋 표기는 `9a795735`로 유지했다.
- `rust::Vec<std::unique_ptr<MlxArray>>`를 반환하는 C++ `split()`은 첫 공개 릴리스 때부터 cxx 선언이 없었다. 그래서 해당 `rust::Vec` 메서드가 한 번도 인스턴스화되지 않았다.
- `tokenizers`의 기본 기능 `esaxx_fast`는 esaxx C++를 정적 CRT(`/MT`)로 컴파일하는데, 이것은 sentencepiece와 MLX의 `/MD` 오브젝트와 함께 링크할 수 없다. 이 기능은 Unigram 학습만 빠르게 하고, mlxcel은 학습을 하지 않는다.

## 5. 검증

RTX Spark N1X(드라이버 616.62, CUDA 13.4.3, cuDNN 9.27, VS 2026)에서 `PATH`에는 `System32`만 두고, `CUDA_PATH` 없이, 빈 JIT 캐시로 실행했다.

| 항목 | 결과 |
|---|---|
| gemma-4-e4b-it-qat-4bit | 정답 출력, 상주 메모리 6.33 GB, 캐시가 찬 상태에서 약 16 tok/s |
| gemma-4-26b-a4b-it-4bit (MoE) | 정답 출력, 14.29 GB, 첫 실행 JIT 포함 10.07 tok/s |
| qwen3_5 27B 4-bit (cuDNN 경로 사용) | 정답 출력, 14.10 GB, 캐시가 찬 상태에서 4.3~4.5 tok/s |
| `mlxcel-server` | `/health`, `/v1/models`, 비스트리밍(`finish_reason: stop`)과 스트리밍(`[DONE]`) chat |
| 분산 거부 | `generate`, `serve`, `mlxcel-server`가 존재하지 않는 repo id로 0.2초 만에 거부 |
| 정적 검사 | Windows clippy `--lib --tests --bins -D warnings`와 fmt 통과, 대상 lib 테스트 17개와 `windows_cuda` 단위 테스트 7개 통과 |

Linux, ROCm, macOS 빌드는 리뷰에서 코드로 확인했다. 공유하는 Linux 링크 경로는 라이브러리 목록과 순서가 같고, 새 CMake·오버레이 분기는 모두 `WIN32` / `_WIN32` 안에 있다. 그 밖의 검증은 CI에 맡겼다.

## 6. 남은 공백

- x86_64 Windows GPU에서 실행하지 않았다. `x64` 배치는 CUDA 13 설치 프로그램(`bin\x64`)을 따르지만 검증되지 않았다.
- CUDA 12.x는 처리하지 않는다. DLL이 `bin\`에 있기 때문이다. cuDNN 설치 프로그램의 `lib\13.x\x64` 배치도 찾지 않으므로 재배포용 아카이브 배치가 필요하다.
- 실행 파일은 빌드 장비의 툴킷 디렉터리에 의존하고, `openblas.dll`은 cargo 프로필 디렉터리에만 복사된다. 패키징과 Windows CI 작업은 Phase 6 범위다.
- Qwen 3.5 MTP 추측 디코딩은 기존 설계상 CUDA에서 거부된다(정확성 커널이 Metal 전용). Linux에서도 같다.
- 디코드 처리량은 조사하지 않았다. 27B dense 모델이 약 4.4 tok/s이면 초당 약 63 GB의 가중치를 읽는 셈이고, 이 장비의 메모리 대역폭보다 한참 낮다.
- Windows CPU 전용 생성은 MLX의 스칼라 양자화 루프를 돈다. 24토큰 생성이 30분 안에 끝나지 않았다.

## 7. 학습 포인트

- cargo가 소비하는 정적 C++ 라이브러리는 CMake의 PUBLIC 사용 요건(링크 옵션, 인터페이스 소스, 여기서는 `MLX_STATIC` 같은 컴파일 정의)을 하나도 물려받지 못한다. 각각을 명시적으로 재현해야 하고, 링크 옵션은 최종 바이너리를 소유한 크레이트의 빌드 스크립트에서 내보내야 한다.
- Windows에서 `AddDllDirectory`는 서드파티 DLL 내부의 일반 `LoadLibrary` 호출에는 영향을 주지 않는다. 의존 DLL을 이름으로 여는 라이브러리를 만족시키려면 전체 경로로 미리 올리는 방법이 확실하다.
- Rust에서 CMake를 거쳐 C++ 리터럴까지 가는 문자열에는 모든 단계를 무사히 통과하는 구분자가 필요하다. 슬래시가 그렇다.
- MSYS는 일부 Windows 크래시 코드를 종료 코드 127로 보여 준다. `cmd`와 `%ERRORLEVEL%`로 보면 실제 NTSTATUS(`0xC06D007E`, `0xC0000409`)가 나온다.
