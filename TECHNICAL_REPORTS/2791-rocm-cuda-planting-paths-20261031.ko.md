# PR #2791: ROCm과 CUDA의 DLL 및 커널 캐시 심기 경로 차단

**작성일**: 2026-10-31 (머지 전 작성)
**상태**: 리뷰 중
**언어**: C++, Rust, Shell
**위험도**: 중간 (모든 Windows 바이너리의 DLL 탐색과 ROCm 캐시 위치가 바뀌며, ROCm 경로는 호스트 테스트로만 실행했다)

## 요약

#2604와 #2607 이후에도 로컬 코드 심기 경로가 두 개 남아 있었다. Windows에서는 ROCm과 CUDA 라이브러리가 평문 이름으로 DLL을 열 때 현재 디렉터리를 탐색했다. ROCm에서는 커널을 공유 임시 디렉터리에 캐시하고 거기서 찾은 파일을 GPU에 로드했다. 이 PR은 Windows DLL 탐색에서 현재 디렉터리를 빼고, 정적 import도 이를 건너뛰도록 링크하며, ROCm 캐시를 사용자별 위치로 옮기고, 다른 사용자가 쓸 수 있는 디렉터리를 거부한다. RTX Spark N1X에서 작업 폴더에 평문 이름 10개로 둔 디코이는 로드되지 않았고, 모듈 집합은 변경 전 바이너리와 같았다. ROCm 바이너리는 실행하지 않았다.

## 1. 문제 정의

`amdhip64_7.dll`은 SDK나 드라이버에 없는 DLL 세 개를 이름으로 참조하고, CUDA 라이브러리도 `zlibwapi.dll`, `nvtileiras64.dll`, `cask_profile_interface.dll` 같은 이름을 참조한다. 사전 로드 훅은 자신이 아는 DLL만 처리하므로, 현재 디렉터리에 같은 이름의 파일이 있으면 프로세스에 로드될 수 있었다.

`hsaco_cache_dir()`의 기본값은 기본 권한의 `<tmp>/mlx/<version>/hsaco/<arch>`였고, `read_cached_hsaco()`는 거기서 찾은 파일을 로드했다. Linux에서는 다른 사용자가 이 디렉터리를 먼저 만들어 파일을 심을 수 있었다.

## 2. 기술적 선택과 그 이유

### 2.1 `SetDllDirectoryW(L"")`와, 이것이 닿지 못하는 부분의 링크 플래그

`SetDefaultDllDirectories`는 디스플레이 드라이버를 포함해 프로세스의 이후 모든 로드를 바꾸므로 계속 쓰지 않는다. `SetDllDirectoryW(L"")`는 표준 탐색에서 현재 디렉터리만 뺀다. 각 delay-load 훅의 맨 앞에서 호출하고, CUDA에서는 `#pragma init_seg(lib)`로 순서를 고정한 정적 초기화에서도 호출한다. nvcc가 정적 cudart를 링크하므로 첫 API 호출이 훅을 거치지 않고 `nvcuda.dll`을 열 수 있기 때문이다.

보안 점검에서 이 설정은 정적 import에 효과가 없다는 것을 측정했다. 작업 폴더의 디코이 `openblas.dll`이 `main` 이전에 로드됐다. `/DEPENDENTLOADFLAG:0xA00`은 이 import를 애플리케이션 디렉터리와 System32로 제한한다. 적용 후 같은 경우가 `0xC0000135`(찾을 수 없음)로 끝나고, 이전 바이너리는 `0xC0000139`로 끝난다.

사용자가 알아챌 변화도 있다. 작업 폴더에만 둔 DLL은 더 이상 로드되지 않고, `PATH`에만 있는 Visual C++ 런타임은 동작하지 않는다.

### 2.2 캐시는 CUDA의 신뢰 규칙을 따른다

POSIX 경로 검사는 #2539와 같다. 부모 소유자가 자식을 바꿔치기할 수 있으므로 `/`까지 모든 구성 요소가 사용자나 root 소유이고 그룹·기타 쓰기 권한이 없어야 한다. 첫 버전의 `umask 002` 복구는 중간 심볼릭 링크를 따라 chmod할 수 있었고, 신뢰 검사 전에 실행됐으며, world-writable 디렉터리를 조용히 받아들였다. 이제는 상위 경로가 통과한 뒤에 실행하고, 관리 경로를 `openat(O_NOFOLLOW)`로 따라가며, 그룹 쓰기만 지우고 기타 쓰기는 거부한다.

### 2.3 Windows에는 별도 검사가 필요했다

첫 버전은 `%TEMP%`가 사용자별이라는 근거로 Windows에 소유권 검사를 두지 않았다. 리뷰는 이를 HIGH로 판단했다. 모델 저장용으로 `MLXCEL_CACHE_DIR=D:\mlxcel`을 설정하면, GPU에 로드되는 캐시가 Authenticated Users 쓰기 권한을 상속하기 쉬운 폴더로 옮겨 간다. `hsaco_cache_dir_win.h`는 드라이브 루트 아래 각 구성 요소의 소유자와 DACL을 읽어, 다른 소유자, null DACL, 다른 SID의 쓰기 계열 허용 항목을 거부한다. 드라이브 루트는 기본 ACL이 Authenticated Users에게 폴더 생성을 허용하므로 제외한다. 루트가 자식에게 주는 권한은 그 바로 아래 구성 요소에서 드러난다.

### 2.4 HIP 의존이 없는 헤더에 로직 배치

ROCm 오버레이는 대부분의 호스트에서 컴파일되지 않고 CI도 `ROCm build and generate`를 건너뛴다. 경로와 신뢰 로직을 헤더로 분리한 덕분에 `overlay-host-tests`가 Linux에서 POSIX 경우를 실행하고, 이 호스트에서 실제 ACL로 Windows 경우를 실행했다.

## 3. 변경 요약

- DLL 탐색: 두 오버레이의 `delayload.cpp`, `build.rs` 두 곳.
- ROCm 캐시: 새 `hsaco_cache_dir.h`, `hsaco_cache_dir_win.h`. `jit_module.cpp`의 읽기·쓰기 경로에서 호출한다. 항목은 임시 파일과 rename으로 게시하고, 빈 `.hsaco`나 잘못된 `.txt`는 캐시 미스로 처리한다.
- 테스트: `rocm_hsaco_cache_test`, `tests/windows_dll_search_cwd.rs`, 호출을 지우면 CI가 실패하는 두 runtime-dir 스크립트의 소스 검사.
- 문서: `docs/installation.md`, `docs/environment-variables.md`, `LOCAL_FIXES.md` 항목 71.

## 4. 결과

| 확인 항목 | 위치 | 결과 |
|---|---|---|
| 작업 폴더에 평문 이름 디코이 10개, 개발자 빌드와 번들의 generate와 서버 | N1X | 로드되지 않음 |
| 리뷰 수정 전후 모듈 집합 | N1X | 네 실행 모두 동일 |
| 변경 전과 PR 바이너리, 캐시 있음과 없음, 8회 | N1X | 실행 파일 이름 외 동일 |
| `nvcuda.dll` | N1X | System32에서 로드 |
| 실행 파일 단독, 디코이 `openblas.dll` | N1X | `0xC0000135`, 이전 바이너리는 `0xC0000139` |
| 구성 요소에 Users, Everyone, Authenticated Users 쓰기 항목 | N1X, `icacls` | 거부, 파일 로드 없음 |
| World·그룹 쓰기, 다른 소유자, 심볼릭 링크 | CI, Linux | 거부, 파일 로드 없음 |

## 5. 트레이드오프와 남은 문제

- **ROCm 미실행:** Linux `make verify-rocm`, Windows ROCm 모듈 목록, ROCm 빌드의 DLL 테스트는 AMD 호스트가 필요하다. `jit_module.cpp`는 Linux ROCm용으로 컴파일하지 않았고, 새 `#pragma comment(lib)` 줄로 링크한 ROCm 바이너리도 없다. CI의 Windows 작업은 오버레이를 컴파일만 한다.
- **실제 요청 경로 없음:** DLL 테스트는 `LoadLibraryW`를 직접 호출한다. 없는 ROCm DLL을 요청하는 경로는 실행하지 않았다.
- **설계상 남은 경로:** `PATH`, `PATH`의 상대 항목, 실행 파일 자신의 디렉터리, `SearchPathW`는 그대로다. `ffmpeg`, `ffprobe`, IREE 컴파일러에는 설정이 상속되지 않아 각자의 탐색에 현재 디렉터리가 남는다.
- **드라이브 루트:** 이 호스트는 드라이브가 하나여서, 상속 ACE 경우를 실제 `D:\`가 아니라 `%TEMP%` 아래에 `icacls`로 만들었다.
- **Windows 항목:** 파일별 ACL 검사는 없고 디렉터리 검사에 의존한다.
- **여기서 고치지 않음:** CUDA PTX 캐시에도 같은 Windows 검사 누락과 `ptx_cache_dirs.rs` 복구 규칙이 있다. 빈 `MLXCEL_CACHE_DIR`은 헤더에서 미설정이지만 `lang_analyzer/cache.rs`에서는 상대 경로다.

## 6. 학습 포인트

- `SetDllDirectoryW`는 정적 import에는 너무 늦다. 훅에 넣는 수정은 훅 이전에 무엇이 로드되는지를 따져야 하고, 디코이로 측정해야 답이 나온다.
- 유닉스 규칙만 쓰는 검사는 Windows 쪽을 가정(`%TEMP%`는 사용자별)에 맡기게 되고, 별도 변수 하나가 그 가정을 조용히 깨뜨릴 수 있다.
- 모듈 목록에서 항목이 빠지면 변경을 탓하기 전에 이전 바이너리와 A/B로 확인해야 한다. `nvptxJitCompilera64.dll`은 이전 실행과 비교해 빠졌지만 이전 바이너리에서도 로드되지 않았다.
- 검사 스크립트의 `grep -v | grep -q`는 `pipefail`에서 일찍 끝난 것을 찾지 못한 것으로 읽어, 첫 푸시에서 CI를 실패시켰다.
