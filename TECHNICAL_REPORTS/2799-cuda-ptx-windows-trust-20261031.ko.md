# PR #2799: Windows에서 CUDA PTX 캐시 디렉터리 검사

**작성일**: 2026-10-31 (머지 전 작성)
**상태**: 리뷰 중
**언어**: C++, Rust, Shell
**위험도**: 중간 (Windows에서 `mlxcel`과 `mlxcel-server`의 기본 캐시 위치가 바뀌고, 지금까지 받아들이던 디렉터리에서 캐시가 꺼진다)

## 요약

`jit_module.cpp`는 캐시된 모듈을 GPU에 올리는데, `_WIN32` 분기는 캐시 디렉터리를 만들기만 하고 검사하지 않았다. 이 PR은 #2791의 ROCm 캐시에 쓴 소유자와 DACL 규칙을 CUDA 캐시의 `MLX_PTX_CACHE_DIR`, 영구 기본 경로, 임시 폴백에 적용한다. 모든 플랫폼에서 상대 캐시 경로를 거부하고, POSIX 복구에서 other-write를 지우지 않으며, 빈 `MLXCEL_CACHE_DIR`을 미설정으로 처리한다. 출발점을 측정해 보니 Windows에서 영구 기본 경로가 한 번도 적용된 적이 없었고, 이 PR에서 함께 고쳤다.

## 1. 문제 정의

문서대로라면 모델 저장을 위해 `MLXCEL_CACHE_DIR=D:\mlxcel`을 설정한 사용자는 Authenticated Users 쓰기 권한을 흔히 상속받는 폴더로 캐시를 옮기게 되고, 다른 로컬 계정이 거기에 모듈을 심을 수 있다.

`main` 바이너리의 실제 동작은 달랐다. `MLXCEL_CACHE_DIR`을 설정하면 `ensure_persistent_ptx_cache`가 `cuda-ptx/<commit>`을 만들지만 그 디렉터리는 비어 있었고, 엔트리는 `%TEMP%\mlx`에 쌓였다. Rust의 `set_var`는 `SetEnvironmentVariableW`를 호출하는데 `std::getenv`는 시작 시점에 복사한 C 런타임의 환경을 읽기 때문이다. 이 호스트의 `~/.cache/mlxcel/cuda-ptx/*`에도 이전 실행이 남긴 파일이 하나도 없었다.

## 2. 기술적 결정

### 2.1 Windows에서는 프로세스 환경을 직접 읽는다

오버레이는 이제 두 변수를 `GetEnvironmentVariableW`로 읽는다. 임시 폴백만 검사하면 문서와 다른 경로를 검사하는 셈이 되고, #2797은 영구 기본 경로를 검사 대상으로 명시한다. 사용자는 업그레이드 후 한 번 다시 컴파일하게 된다. `mlxcel`과 `mlxcel-server`는 더 이상 `%TEMP%\mlx`를 쓰지 않는다.

### 2.2 ROCm과 같은 규칙, 오버레이 자체 헤더

`ptx_cache_dir_win.h`는 `hsaco_cache_dir_win.h`의 규칙을 그대로 따른다. 소유자는 사용자, SYSTEM, Administrators 중 하나여야 하고, null DACL과 다른 SID에 대한 쓰기 계열 허용 항목을 거부한다. 드라이브 루트는 검사에서 뺀다. 기본 권한이 하위 디렉터리 생성뿐이고 다른 계정의 디렉터리를 지우거나 이름을 바꾸는 권한은 아니기 때문이다. 오버레이끼리 코드를 공유하지 않으므로 복사는 의도한 것이다.

### 2.3 `\\?\` 접두사는 경로 순회에서 떼고 호출에서 붙인다

`std::filesystem`은 `\\?\C:\x`의 루트 이름을 `\\?`로 해석하므로 루트가 드라이브가 아니고 `parent_path`가 `C:\`를 지나쳐 올라간다. 순회는 접두사 없는 형태를 쓰고, 안내 문구에도 그 형태가 나온다. 파일 시스템 호출마다 접두사를 다시 붙이므로 `MAX_PATH`를 넘는 경로도 동작하고, 반환 경로에는 업스트림 코드처럼 접두사가 남는다. 접두사가 `/`와 `..` 해석을 끄고 `cache_root`가 `.cache/mlxcel`을 슬래시로 잇기 때문에 `lexically_normal`을 먼저 적용한다.

### 2.4 상대 경로는 두 계층에서 막는다

`ptx_cache_env_refusal`은 상대 `MLX_PTX_CACHE_DIR`을 거부하고, 이 변수가 없으면 상대 `MLXCEL_CACHE_DIR`도 거부한다. `ensure_persistent_ptx_cache`도 상대 루트면 바로 반환한다. 그러지 않으면 현재 디렉터리 아래에 `rel-root\cuda-ptx\<commit>`을 만들고 `MLX_PTX_CACHE_DIR`을 거기로 설정하며, 안내 문구는 사용자가 설정한 적 없는 변수를 가리키게 된다.

### 2.5 other-write는 오버레이가 거부하도록 남긴다

`umask` 002로 설명되는 것은 group-write뿐이다. 누구나 쓸 수 있는 디렉터리에는 이미 다른 사람의 모듈이 있을 수 있어서, 그 비트를 지우면 그 모듈을 받아들이게 된다. `make_private_dir`는 이제 group-write만 지운다.

### 2.6 실제 ACL로 테스트

`tests/cuda_ptx_cache_trust_windows.rs`는 첫 실행이 컴파일한 엔트리를 검사 대상 디렉터리에 복사한다. 그 디렉터리를 신뢰하는 실행이라면 컴파일 없이 그 엔트리를 올린다. `planted_entries_load_from_a_private_dir`가 그 대조군이다. `ptx_cache_env_refusal`은 Linux의 `overlay-host-tests`에서도 실행되는데, CI에서 상대 경로 규칙을 실행하는 곳은 여기뿐이고, 이 스크립트에는 호출을 확인하는 소스 가드도 있다.

## 3. 변경 요약

- 오버레이: `ptx_cache_dir.h`와 `ptx_cache_dir_win.h`를 새로 추가하고, `jit_module.cpp`의 `ptx_cache_dir()`와 `device_cache_dir()`에서 호출한다.
- Rust: `cache_root`는 빈 값을 무시하고, `persistent_ptx_cache_dir`는 상대 루트에 `None`을 반환하며, `make_private_dir`는 group-write만 지운다.
- 테스트: `tests/cuda_ptx_cache_trust_windows.rs`와 `location.rs`, `tests/cuda_ptx_cache_trust/entries.rs`의 상대 경로 사례 2개, `cuda_runtime_dir_test`, `ptx_cache_dirs.rs`와 `lang_analyzer/cache.rs`의 단위 테스트.
- 문서: `docs/installation.md`, `docs/environment-variables.md`, 오버레이 `README.md`. CI의 `windows_build` 필터에 두 헤더를 추가했다.

## 4. 결과

| 검사 | 위치 | 결과 |
|---|---|---|
| 새 Windows 테스트, `main`의 오버레이 | N1X | 19개 중 15개 실패. 통과한 4개는 자식 프로세스, 비공개 디렉터리 히트, 대조군, 읽기 전용 항목 |
| 새 Windows 테스트, 이 PR | N1X | 19개 모두 통과 |
| 말단, 부모, 조부모에 Users, Everyone, Authenticated Users 쓰기 항목, 엔트리 심음 | N1X, `icacls` | 안내 한 줄로 거부, 컴파일, 엔트리 변경 없음 |
| 상속, 상속 전용, 관리 대상 이름의 정션, 공유 임시 디렉터리, `\\?\` 표기, `MAX_PATH` 초과 | N1X, `icacls`, `mklink /J` | 거부 |
| 상대 `MLX_PTX_CACHE_DIR`(`rel`, `.\rel`, `\rel`, `C:rel`)와 `MLXCEL_CACHE_DIR` | N1X | 거부, 현재 디렉터리에 생성 없음 |
| `mlxcel generate`, 기본 디렉터리 | N1X | 첫 실행 컴파일, 두 번째 히트, `~\.cache\mlxcel\cuda-ptx\7ca40033132d\sm121-nvrtc13.4`에 `.ptx` 4개 |
| `mlxcel generate`, Authenticated Users 쓰기를 상속한 `MLXCEL_CACHE_DIR`에 컴파일된 엔트리 심음 | N1X | `... is writable by S-1-5-11`, 컴파일 |
| `mlxcel-server`, 채팅 완성 1회, 공유 루트와 비공개 루트 2회 | N1X | 거부 후 컴파일, 컴파일 후 히트 |

## 5. 트레이드오프와 남은 항목

- **POSIX C++ 미컴파일:** 이 호스트에는 Linux 툴체인이 없어 `jit_module.cpp`의 `#else` 분기와 Linux 테스트를 여기서 빌드하거나 실행하지 못했다. 분기 컴파일은 CI가 맡는다. `entries.rs`의 새 사례 2개와 other-write를 다루는 `ptx_cache_dirs.rs` 테스트는 GB10에서 실행해야 한다.
- **실행하지 못한 경로:** 다른 소유자는 두 번째 계정이나 이 테스트가 가정하지 않는 권한이, null DACL은 FAT나 exFAT 볼륨이 필요하다. 둘 다 ROCm 헤더 코드를 그대로 가져왔다.
- **드라이브 루트:** 이 호스트는 드라이브가 하나뿐이라 상속 사례는 실제 `D:\`가 아니라 `%TEMP%` 아래에서 `icacls`로 구성했다.
- **Windows 엔트리:** ROCm과 마찬가지로 파일별 검사가 없다. 공유했다가 비공개로 바꾼 디렉터리는 남아 있는 엔트리를 그대로 올리므로, 문서에 그 엔트리를 지우라고 적었다.
- **빈 `MLX_PTX_CACHE_DIR`:** 지금까지처럼 두 플랫폼 모두에서 안내 없이 캐시를 끈다. ROCm은 자기 변수가 비어 있으면 미설정으로 본다. 이슈가 요청한 변경이 아니라서 두었다.

## 6. 배운 점

- 이슈가 설명한 경로를 바이너리는 타지 않았다. 변수를 설정하고 파일이 어디에 생기는지 보는 명령 하나로 수정 범위가 바뀌었다.
- Windows에서 `std::getenv`는 `SetEnvironmentVariableW`로 바꾼 값을 보지 못한다. C++ 코드에 넘기려는 Rust `set_var`는 모두 같은 문제가 있고, 오류 없이 실패한다.
- `\\?\`로 끝나는 주석 줄은 C++에서 다음 줄로 이어진다. MSVC `/W4 /WX`도 테스트도 잡지 못했고 grep으로 찾았다.
