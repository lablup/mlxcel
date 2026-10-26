# PR #2607: Windows ROCm 런타임을 옮길 수 있는 디렉터리에서 로드

**날짜**: 2026-10-26
**상태**: Windows에서 완료(gfx1151, ROCm 10.1.0). 다른 GPU와 드라이버는 실행하지 않음
**언어**: C++, Python, Markdown
**위험도**: 중간

## 요약

PR #2607(이슈 #2606)은 ROCm SDK가 없는 Windows 머신에서도 mlxcel의 ROCm 빌드가 실행되게 한다. #2604의 delay-load 훅은 모든 ROCm DLL을 빌드 시점에 기록한 SDK 경로에서 로드했다. 이제는 프로세스마다 런타임 디렉터리 하나를 고른다. 순서는 `MLXCEL_ROCM_RUNTIME_DIR`, 없으면 실행 파일 자신의 디렉터리, 없으면 빌드 시점 SDK의 `bin`이다. 새 스크립트 `scripts/package_windows_rocm_runtime.py`는 그 디렉터리에 들어갈 런타임 번들을 만든다. 번들에는 DLL 의존성 전체, HIP JIT 헤더, 아키텍처별로 거른 커널 데이터가 들어가며, gfx1151 기준 압축 후 76 MiB다. 런타임을 따로 배포하는 llama.cpp 방식을 따르되 번들은 mlxcel이 직접 만든다. llama.cpp의 ROCm zip에는 rocBLAS, hipBLASLt, hipRTC가 없기 때문이다.

SDK 폴더 이름을 바꿔 치운 상태에서, 번들과 `mlxcel.exe`만 둔 새 폴더로 `generate`(cold JIT)와 서버 엔드포인트 세 개를 모두 GPU에서 실행했다. ROCm 모듈은 전부 번들에서 로드되었고, System32, 작업 디렉터리, 사라진 SDK에서 온 것은 없었다.

## 1. 문제 정의

#2604 이후 Windows ROCm 바이너리는 SDK가 `PATH`에 없어도 되었지만, 빌드 시점에 기록한 바로 그 경로에 SDK가 있어야 했다. 그래서 바이너리가 빌드 머신의 디렉터리 구조에 묶였다. 일반 Windows 사용자에게 mlxcel을 배포하려면 바이너리가 함께 배포된 런타임을 찾아야 한다. 단 두 가지를 지켜야 한다. 하나는 #2604가 막은 DLL 검색 순서 구멍(작업 디렉터리와 `PATH`)을 다시 열지 않는 것이다. 다른 하나는 두 ROCm 설치의 DLL을 섞지 않는 것이다. #2604에서 확인했듯이 SDK의 HIP 런타임을 디스플레이 드라이버의 System32 `amd_comgr.dll`과 섞으면 디바이스가 0개로 잡힌다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 런타임 디렉터리 | 헤더 전용 `runtime_dir.h`에 순수 함수 `choose_rocm_runtime_dir()`를 새로 두었다. `delayload.cpp`가 이를 한 번 호출하고, delay-load되는 ROCm DLL 전부, `amd_comgr*`/`hiprtc-builtins*` 사전 로드, `AddDllDirectory` 항목을 그 한 디렉터리에서 가져온다. `[mlx-rocm] ROCm runtime: <dir> (<source>)`를 출력한다. |
| 오버라이드 의미 | `MLXCEL_ROCM_RUNTIME_DIR`가 설정되면 그것만 후보다. 상대 경로이거나 HIP 런타임이 없는 폴더면 다른 후보로 넘어가지 않고 오류를 낸다. |
| 후보 판정 | 바이너리 자신의 delay-import 표에 적힌 HIP 런타임 DLL(현재 `amdhip64_7.dll`)이 있는 폴더만 후보가 된다. 바이너리 디렉터리는 `GetModuleFileNameW(&__ImageBase)`로 얻는다. |
| 사전 로드 존재 검사 | 고른 폴더에 `amd_comgr*.dll`과 `hiprtc-builtins*.dll`이 있어야 한다. 없으면 폴더와 패턴을 밝히고 시작을 멈춘다. 로드 시점에 DLL이 없으면 기대한 경로를 밝히고 실패한다. |
| JIT 헤더 | `choose_rocm_include_dir()`에 후보가 하나 늘었다. `hip/hip_runtime.h`가 있으면 `<runtime dir>\include` 또는 `<runtime dir>\..\include`를 쓴다. 위치는 `ROCM_HOME`/`ROCM_PATH`/`HIP_PATH` 다음, 빌드 시점 SDK include 앞이다. |
| 번들 스크립트 | `package_windows_rocm_runtime.py`(표준 라이브러리만 사용): `--rocm-path`, 반복 가능한 `--arch`, `--out`(zip) 또는 `--stage-dir`, `--add-binary`, `--rocm-version`. 파일이 하나라도 없으면 실패하고, 출력은 결정적이다. |
| 테스트 | 호스트 C++ 테스트 `scripts/mlxcelverse/rocm_runtime_dir_test.{cpp,sh}`(`make verify-rocm-overlay`에 포함)가 순서, 오버라이드 오류, 사전 로드 검사, include 순서를 다룬다. `tests/test_package_windows_rocm_runtime.py`에 테스트 18개가 있다. |
| 문서 | `docs/installation.md`에 "Runtime directory and bundle" 절을 두고, `docs/environment-variables.md`에 `MLXCEL_ROCM_RUNTIME_DIR` 행을 추가했다. 설치 규칙은 llama.cpp 런타임과 다른 폴더 사용, 관리자만 쓸 수 있는 폴더, 130자 이하의 경로 길이, `HIP_PATH` 우선순위 안내다. `LOCAL_FIXES.md` 58번. |

## 3. 기술적 선택과 그 이유

### 3.1 프로세스당 디렉터리 하나, 대체 경로 없음

훅이 DLL마다 후보를 차례로 시도하게 할 수도 있었지만 그렇게 하지 않았다. 선택은 한 번만 하고, 고른 폴더에 DLL이 없으면 오류다. DLL 단위로 대체하면 #2604에서 조용히 실패한다고 확인된 설치 간 조합(디바이스 0개, 이어서 CPU 실행)이 그대로 생긴다. 같은 이유로 설정은 되었지만 잘못된 `MLXCEL_ROCM_RUNTIME_DIR`도 오류다. 오타 때문에 SDK나 exe 옆 번들이 조용히 대신 실행되어서는 안 된다.

### 3.2 실행 파일 디렉터리를 후보로 둔다

Windows는 암시적 import에서 이미 애플리케이션 디렉터리를 가장 먼저 검색하므로, 이를 신뢰해도 DLL을 심을 새 경로가 생기지 않는다. 압축을 푼 앱이 놓이는 자연스러운 위치이기도 하다. 후보 판정은 바이너리 자신의 delay-import 이름을 쓰므로, 나중에 HIP 메이저 버전이 다른 빌드가 나와도 맞는 파일을 확인한다.

### 3.3 평평한 폴더 대신 SDK 구조를 유지한다

이슈는 `mlxcel.exe` 옆의 평평한 폴더를 제안했다. `ROCM_KPACK_DEBUG=1`로 확인해 보니 rocBLAS, hipBLASLt, rocFFT는 디바이스 코드를 `<dll dir>\..\.kpack\<lib>_<arch>.kpack`에서 읽는다. 따라서 평평한 폴더에서는 GEMM이 모두 깨진다. 그래서 번들은 `bin\`, `.kpack\`, `include\`, `licenses\`를 유지한다. exe는 `bin\`에 넣거나, 오버라이드가 그 `bin\`을 가리킨다. JIT가 `<runtime dir>\..\include`를 보는 이유도 이것이다.

### 3.4 이름으로 로드되는 DLL의 존재를 요구한다

오케스트레이터 재검증에서 번들의 `amd_comgr.dll`을 지워 보았다. 기존 사전 로드는 아무것도 로드하지 않은 채 조용히 지나갔다. 그러자 HIP 런타임 자신의 `LoadLibrary("amd_comgr.dll")`가 `C:\Windows\System32\amd_comgr.dll`로 해석되었고, 프로세스는 CPU로 떨어진 뒤 0xC0000409로 종료했다. 이제 훅은 아무것도 로드하기 전에 고른 폴더의 목록을 확인한다. 이름으로 로드되는 두 계열 중 하나라도 없으면 메시지를 출력하고 delay-load 헬퍼의 `ERROR_MOD_NOT_FOUND` 예외를 일으킨다. 이는 #2604가 import된 DLL이 없을 때 하는 것과 같다. `hiprtc-builtins`도 필수로 했는데, 없으면 작업 디렉터리와 `PATH`에서 찾게 되기 때문이다.

### 3.5 경로를 UTF-8로 출력한다

MSVC CRT는 기본 C 로캘에서 `%ls` 변환을 첫 비ASCII 문자에서 멈춘다. 그래서 한국어 사용자 이름 아래에 둔 번들은 경로가 잘려 출력되었다. 이제 경로를 `WideCharToMultiByte(CP_UTF8)`로 변환해 `%s`로 출력한다. 디렉터리 목록은 예외를 던지지 않는 변환을 쓰므로, delay-load 헬퍼 밖으로 예외가 새지 않는다.

## 4. 검증

모든 실행은 Windows 11, Radeon 8060S(`gfx1151`), ROCm 10.1.0, eb1c47af 최종 빌드에서 했다. `C:\rocm\10.1.0`은 이름을 바꿔 치웠고, `PATH`는 System32만, ROCm 변수는 없음, JIT 캐시는 빈 상태였다. 작업 디렉터리에는 ROCm DLL 14개 이름으로 바꾼 `version.dll` 사본을 두었다.

- **커밋된 스크립트로 만든 번들**(`--add-binary` 사용): Qwen3-1.7B-bf16 `generate`가 "Paris"라고 답했다. 모듈 목록상 ROCm DLL 10개가 모두 번들의 `bin\`에서 로드되었다.
- **같은 폴더의 서버**: chat, 스트리밍(`[DONE]`), `/v1/messages` 모두 "Paris"를 반환했다.
- **번들을 가리키는 `MLXCEL_ROCM_RUNTIME_DIR`**(exe는 빌드 디렉터리에 둠): "Paris", 70 tok/s.
- **`amd_comgr.dll`, `hiprtc-builtins*.dll`, `rocblas.dll`을 각각 뺀 번들**: 모두 파일과 폴더를 밝히며 0xC06D007E로 종료했고, System32에서 로드된 것은 없다.
- **`정규`라는 이름의 junction으로 접근한 런타임 폴더**: 경로가 온전히 출력되었고 cold JIT `generate`가 동작했다. hiprtc가 UTF-8 비ASCII `-I`를 받아들인다는 것도 이로써 확인했다.
- **번들도 SDK도 없는 경우**: 오류가 두 후보를 모두 밝힌다. SDK가 있는 개발 빌드는 여전히 `C:\rocm\10.1.0\bin`을 쓴다.
- **검사**: `cargo test -p mlxcel-mlx-pin --lib`(49개), fmt, `--features rocm` clippy, overlay drift와 records, `check_env_var_docs.py`, 호스트 C++ 테스트, Python 패키징 테스트 모두 통과.
- **참고용 llama.cpp 시험**: llama.cpp b11889의 ROCm 10.0 `amdhip64_7.dll`, `amd_comgr.dll`, `rocm_kpack.dll`을 10.1 수학 라이브러리와 함께 쓰니 두 시험 모델이 정상 동작했다. 호스트 한 대의 결과이므로 문서는 여전히 폴더 공유를 금지한다.

## 5. 남은 공백

- **`HIP_PATH` 우선순위.** 이슈는 Linux와 CUDA에 맞춰 `ROCM_HOME`/`ROCM_PATH`/`HIP_PATH`를 번들 헤더보다 앞에 두도록 정했다. 그런데 AMD HIP SDK 설치 프로그램은 `HIP_PATH`를 시스템 전체에 설정한다. 그런 머신에서는 번들이 그 SDK의 헤더로 JIT 커널을 컴파일한다. 코드는 바꾸지 않고 문서로 안내했다. 런타임을 번들에서 가져온 경우 번들 헤더를 우선하는 변경은 후속 후보다.
- **실행하지 않은 것:**
  - 다른 GPU, 드라이버, 아키텍처(AOT 아키텍처 목록은 아직 gfx1151뿐)
  - Windows long-path 모드
  - FFT(hipFFT와 rocFFT는 번들에 들어 있지만 FFT는 실행하지 않음)
  - Linux(이번 변경은 Windows 전용 코드 경로라 영향이 없어야 함)
- **호스트 C++ 테스트를 CI가 돌리지 않는다.** `make verify-rocm-overlay`만 돌리며, 이 호스트에서는 `CXX`를 ROCm clang++로 지정해야 한다.
- **라이선스 파일 누락.** SDK tarball에 HIP 런타임, hipRTC, rocm_kpack의 라이선스 파일이 없어 번들에도 넣을 수 없다.
- **경로 길이.** long path가 꺼져 있으면 130자를 넘는 런타임 경로에서 rocBLAS/hipBLASLt 커널 로드가 깨진다. 훅은 경고만 하고 고치지는 않는다.

## 6. 학습 포인트

- 벤더 수학 라이브러리는 자기 DLL 기준 상대 경로(`..\.kpack`)로 데이터를 찾을 수 있다. 그래서 번들 구조는 import 표가 보여 주는 것보다 더 제약된다. 평평한 구조를 설계하기 전에 실제 파일 열기를 추적해야 한다.
- 조용히 아무것도 하지 않는 사전 로드는 시스템 검색 순서로 넘어가는 결과를 낳는다. "X를 전체 경로로 로드"에는 "X가 없으면 실패"가 함께 있어야 한다.
- MSVC에서 비ASCII 문자열에 `%ls`를 쓰면 기본 C 로캘에서 출력이 조용히 잘린다. 출력 전에 경로를 UTF-8로 변환해야 한다.
