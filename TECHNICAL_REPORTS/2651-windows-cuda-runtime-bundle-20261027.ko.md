# PR #2651: Windows CUDA 빌드를 옮길 수 있는 런타임 번들에서 실행

**날짜**: 2026-10-27
**상태**: NVIDIA GPU 없는 Windows x64에서 빌드로 검증. NVIDIA 하드웨어 실행은 남아 있음
**언어**: C++, Python, Makefile, Markdown
**위험도**: 중간

## 요약

PR #2651(이슈 #2640)은 #2606/#2607의 Windows ROCm 런타임 설계를 Windows CUDA 빌드에 적용한다.
- **이전:** `--features cuda` 바이너리는 빌드 시점에 기록한 경로에 CUDA 툴킷과 cuDNN이 있어야 실행되었다.
- **이후:** 모든 CUDA, cuDNN, NVRTC DLL을 런타임 디렉터리 하나에서 로드한다. 디렉터리는 다음 순서로 고른다.
  - `MLXCEL_CUDA_RUNTIME_DIR`
  - 없으면 실행 파일 폴더(번들 표지 DLL이 있을 때)
  - 그것도 아니면 빌드 시점 구조
- **패키징:** 새 스크립트 `scripts/package_windows_cuda_runtime.py`가 번들을 만든다.

개발 호스트에 NVIDIA GPU가 없어서, 사용자는 이 이슈를 빌드로만 검증하기로 결정했다. x64 CUDA 빌드, 호스트 테스트와 패키징 테스트, NVRTC 헤더 확인, GPU 없이 도는 로더 하네스는 모두 통과했다. 디바이스 초기화 이후는 여전히 NVIDIA 호스트가 필요하다.

## 1. 문제 정의

Windows CUDA 빌드(#2599)는 빌드 시점에 기록한 툴킷의 `bin\<arch>`에서 CUDA 라이브러리를 delay-load하고, NVRTC는 기록해 둔 `include\`에서 헤더를 읽었다. 다른 머신에서 실행하려면 같은 툴킷을 같은 경로에 설치해야 했다. 문서에도 `openblas.dll`과 CUDA, cuDNN, NVRTC DLL 묶음은 아직 만들지 않았다고 적혀 있었다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 디렉터리 선택 | 헤더 전용 `overlays/cuda/.../runtime_dir.h`에 순수 함수 `choose_cuda_runtime_dir()`를 새로 두었다. `MLXCEL_CUDA_RUNTIME_DIR`는 절대 경로이고 런타임을 담고 있어야 하며, 아니면 대체 경로 없이 오류다. 값은 먼저 정규화한다(`.`, `..` 해석, 끝 구분자 제거). 실행 파일 디렉터리는 바이너리 자신의 delay-import 표에 적힌 표지 DLL이 있을 때 후보가 된다. 빌드 시점 구조는 확인 없이 쓰므로 개발 빌드는 이전과 같다. |
| 표지 DLL | nvcc가 cudart를 정적으로 링크하므로 찾을 cudart DLL이 없다. 그래서 표지는 `cublasLt64_13.dll`이다. |
| 로더 훅 | 번들에서는 라이브러리가 이름으로 여는 DLL을 CUDA DLL을 로드하기 전에 확인하고 전체 경로로 미리 로드한다. 대상은 `nvrtc64_*`, `nvrtc-builtins*`, `cublas64_*`, `cublasLt64_*`, `nvJitLink_*`, 필수 `cudnn_*` 하위 라이브러리 7개이고, `cusparse64_*`는 확인만 한다. 로드에 실패하면 기대한 경로를 밝히며 delay-load 예외를 일으킨다. 빌드 시점 경로는 upstream의 라이브러리별 조회와 `../nvidia/...` wheel 대체 경로를 그대로 쓴다. |
| NVRTC 헤더 | 헤더 순서는 `CUDA_HOME`, 번들 `include\`(런타임 디렉터리 안이나 옆), `CUDA_PATH`, 내장 툴킷 include다. `CUDA_HOME`이 설정되면 번들 include는 맨 뒤로 가서 CUTLASS/CuTe만 공급한다. |
| 패키징 | 스크립트가 `bin\`(DLL과 OpenBLAS), `include\`(JIT 헤더, 빌드 `_deps`의 CCCL, CUTLASS), `licenses\`(CUDA EULA, cuDNN, CCCL, CUTLASS)를 폴더나 zip으로 만든다. DLL glob마다 파일이 정확히 하나 맞아야 하고, 라이선스는 내용으로 구분한다. DLL, 헤더, 라이선스가 하나라도 빠지면 이름을 밝히며 실패하고, zip은 `.partial` 파일을 거쳐 쓰며, 이미 있는 출력은 거부한다. |
| 테스트 | `make verify-cuda-runtime-dir`(`make verify`에 포함)가 선택, 표지, 사전 로드 확인, include 순서를 다루는 호스트 C++ 테스트를 실행한다. `tests/test_package_windows_cuda_runtime.py`에 패키징 테스트가 있다. |
| 문서 | `docs/installation.md`에 런타임 디렉터리와 번들, 번들 밖 이름 로드, 재배포 주의사항, 빌드로만 검증한 범위를 밝히는 제한 표 행을 추가했다. `docs/environment-variables.md`에는 `MLXCEL_CUDA_RUNTIME_DIR`, `CUDA_HOME`, `CUDA_PATH`를 적었다. |

## 3. 기술적 선택과 그 이유

### 3.1 ROCm과 같은 규칙

CUDA 훅은 #2607에서 ROCm용으로 정한 규칙을 따른다.
- 프로세스당 디렉터리 하나
- DLL별로 다른 설치로 넘어가지 않음
- 훅이 다루는 DLL은 현재 디렉터리와 `PATH`에서 찾지 않음
- null을 돌려주면 헬퍼가 일반 검색으로 재시도하므로, null 대신 delay-load 헬퍼의 예외를 일으킴
- 경로는 UTF-8로 출력

예외는 빌드 시점 구조 하나다. 여기서는 upstream의 조회와 대체 경로를 그대로 두었기 때문에, 번들도 변수도 없는 개발 빌드는 이전과 똑같이 로드한다.

### 3.2 번들은 CUDA_PATH보다 우선하고 CUDA_HOME보다는 아래

CUDA 설치 프로그램은 `CUDA_PATH`를 머신 전체에 설정한다. #2648의 `HIP_PATH`와 같은 경우이므로 번들 헤더보다 아래에 둔다. `CUDA_HOME`은 의도적인 지정이라 맨 앞에 둔다.

구현 리뷰에서 첫 버전이 이렇게 동작하지 않는다는 것이 드러났다. NVRTC include 목록에서 번들 include가 선택된 CUDA 헤더 디렉터리보다 앞에 있었는데, NVRTC는 처음 맞는 것을 쓴다. 그래서 `CUDA_HOME`이 설정되면 번들 include를 맨 뒤에 두도록 고쳤다.

`CUDA_HOME/include`를 맨 앞으로 옮기지 않은 데는 이유가 있다. CUDA 13 툴킷의 최상위 `include/nv/`가 빌드의 CCCL `nv/`를 가리기 때문이다.

### 3.3 GPU 없이 정한 DLL 목록과 헤더 집합

GPU 없이는 generate 실행의 로드 모듈 목록을 얻을 수 없다. 그래서 DLL 목록은 x64 CUDA 13.4.3과 cuDNN 9.27 DLL의 import 표와 내장 DLL 이름에서 뽑았다. 예를 들어 cuBLASLt는 NVRTC를, cuSPARSE는 nvJitLink를 이름으로 연다. 목록은 코드에 고정해 두었고 하드웨어에서 확인이 필요하다고 표시했다.

JIT 헤더 집합은 MLX가 내장하는 헤더 17개를 NVRTC 13.4로 번들 `include\`만 써서 하나씩 컴파일해 찾았다. NVRTC 컴파일에는 디바이스가 필요 없다. 이 과정에서 CUTLASS가 include하는 `cuda_runtime.h`, `mma.h`, `crt/mma.h(pp)`가 추가되었다.

### 3.4 재배포 조건

JIT에 필요한 헤더 중 일부는 CUDA EULA Attachment A의 재배포 가능 파일 목록에 없다. `cooperative_groups*`, `math_constants.h`, `cuda_runtime.h`, `mma.h`, `driver_types.h` 등이다. 문서에 이를 밝히고, 번들을 공개 배포하기 전에 NVIDIA에 조건을 확인하라고 안내했다. 미리 빌드한 아카이브는 범위 밖이다.

## 4. 검증

호스트는 NVIDIA GPU가 없는 Windows 11 x64(Radeon 8060S)다. 툴킷은 sha256을 확인한 NVIDIA windows-x86_64 재배포 아카이브(CUDA 13.4.3, cuDNN 9.27.0.42)를 썼다.

- **빌드:** 처음으로 x64 Windows CUDA 빌드를 했고, 빌드 연결 코드는 고칠 필요가 없었다. `MLX_CUDA_ARCHITECTURES=120`으로 cold 빌드 32분, 리뷰 수정 후 재빌드 17분 37초가 걸렸다. `mlxcel.exe`는 133,756,928바이트다. `llvm-readobj`로 보면 cuBLASLt, cuFFT, cuSOLVER, cuDNN, NVRTC는 delay import이고, 일반 import에는 CUDA DLL이 없다.
- **검사:** clippy `--features cuda -D warnings`, fmt, `cargo test -p mlxcel-mlx-pin --lib`(49개)가 통과했다. 호스트 테스트는 통과하고, 선택 로직을 일부러 바꾸면 실패한다. 패키징 테스트 25개와 환경 변수 문서 검사도 통과했다.
- **실제 아카이브로 패키징:** 번들은 1830개 파일, 1968 MiB다. 파일 하나의 이름을 바꾸면 그 파일을 밝히며 실패하고, cuDNN의 `LICENSE`를 CUDA EULA로 넘기면 거부한다.
- **로더 하네스:** overlay `delayload.cpp`를 링크하고 delay-load 훅을 거쳐 cuBLASLt, NVRTC, cuFFT, cuDNN의 버전 함수를 부르는 프로그램이다. 툴킷 이름을 바꿔 치우고 `CUDA_PATH`를 지운 상태에서 결과는 다음과 같았다.
  - 실행 파일 디렉터리, override, 끝 구분자가 붙은 override 모두에서 CUDA, cuDNN, NVRTC 모듈이 전부 번들에서 로드되었다.
  - `nvcuda.dll`은 System32에서 왔다.
  - 상대 경로 override, 잘못된 폴더, cuFFT, cuSPARSE, `cudnn_ops`, `cudnn64` 각각의 제거, 표지 제거는 모두 기대한 경로를 밝히며 실패했다.
  - 툴킷이 있으면 개발 구조는 이전처럼 툴킷에서 로드했다.
- **mlxcel 자체:** 번들의 `mlxcel generate`와 `mlxcel-server`, 그리고 개발 빌드는 첫 디바이스 질의에서 0xC0000409로 끝난다. delay-load되는 CUDA DLL을 요청하기 전이다. 이 호스트에는 오래된 `nvcuda.dll`만 있고 NVIDIA GPU는 없다.

## 5. 남은 공백

- **NVIDIA 하드웨어 실행:** 다음은 NVIDIA Windows 호스트에서 실행해 보지 않았다.
  - 툴킷을 치운 새 번들 폴더에서 첫 NVRTC JIT를 포함한 generate와 서버
  - 실제 실행의 모듈 목록
  - mlxcel 자체에서 DLL을 지웠을 때의 실패
  - 개발 빌드가 그대로 동작하는지

  이 때문에 수락 기준 6개 중 4개가 열려 있다.
- **확인되지 않은 목록:** DLL 목록과 헤더 집합은 정적 분석과 NVRTC 단독 확인에서 나왔다.
- **번들 밖 이름 로드:** CUDA 라이브러리는 `nvcuda.dll`, `nvcudart_hybrid64.dll`, `zlibwapi.dll`, `cask_profile_interface.dll`, `nvtileiras64.dll`과 선택 cuDNN 하위 라이브러리 2개도 이름으로 연다. 번들은 이들을 담지도 확인하지도 않으므로, 개발 빌드와 마찬가지로 현재 디렉터리와 `PATH`를 포함한 표준 검색 순서를 따른다. 막으려면 강제 manifest나 프로세스 전체 검색 변경 같은 결정이 필요하며, DLL planting 후속 이슈(#2641)에서 다룬다.
- **패키징 공백:**
  - `--stage-dir`는 실패하면 일부만 만든 트리를 남긴다. 그 번들은 빠진 파일 경로를 밝히며 멈추므로 안전 쪽 실패다.
  - 스크립트는 훅이 선택으로 보는 cuDNN 하위 라이브러리 2개를 필수로 요구하므로, 이전 cuDNN 9는 패키징에 실패한다.
  - target 디렉터리에 CUDA 빌드 출력이 둘 있으면 `--deps-dir`가 필요하다.
- **Linux:** Linux에서 바뀐 동작은 빈 `CUDA_HOME`을 미설정으로 보는 것 하나다.

## 6. 학습 포인트

- **include 순서:** 헤더 검색 경로 목록은 우선순위 그 자체다. NVRTC는 처음 맞는 것을 쓰므로, 선택 함수에 적은 우선순위는 include 목록도 같은 순서로 만들 때만 지켜진다.
- **정적 런타임:** 런타임이 정적으로 링크되어 있으면 번들을 알아볼 런타임 DLL이 없다. 표지는 바이너리가 실제로 delay-load하는 DLL이어야 한다.
- **섞이는 라이선스 파일:** 재배포 아카이브가 모두 `LICENSE`라는 같은 이름의 파일을 담고 있으면, 한 곳에 풀 때 서로 덮어쓴다. 라이선스를 내용으로 구분해야 맞는 이름으로 틀린 조건이 배포되는 일을 막는다.
