# PR #2777: #2743의 PTX 캐시 신뢰 검사 잔여 항목 마무리

**날짜**: 2026-10-31 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: C++ (CUDA overlay), Rust, CMake
**위험도**: 낮음 (#2743이 이미 디렉터리를 보호하는 캐시에 대한 다층 방어다. 눈에 보이는 변화는 group-writable이거나 symlink인 캐시 항목이 한 번 다시 컴파일되어 교체된다는 점, 그리고 `--release` 바이너리가 `MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX`를 무시한다는 점이다)

## 요약

PR #2743(#2539를 닫음)은 CUDA JIT PTX 디스크 캐시가 현재 사용자나 root 소유이고 다른 누구도 쓸 수 없는 디렉터리만 신뢰하도록 만들었다. 그 구현 리뷰와 보안 리뷰에서 LOW 등급 잔여 항목 네 개가 남았고(#2765에 정리), #2743 리뷰에서 다섯 번째 항목이 추가되었다: 모든 빌드에 컴파일되는 테스트 전용 hook이다. 이 PR은 다섯 항목을 모두 닫는다. `MLX_PTX_CACHE_DIR`의 `..`는 디렉터리를 만들기 전에 해석한다. 각 캐시 항목 파일(`.ptx`와 `.txt`)은 symlink를 따라가지 않고 열며, descriptor에 대한 `fstat`이 private regular file을 보일 때만 로드한다. Rust 쪽 디렉터리 수리는 probe 파일 대신 `geteuid()`로 uid를 얻고 `O_DIRECTORY|O_NOFOLLOW|O_NONBLOCK`으로 열어서, 바꿔치기된 FIFO가 시작 과정을 멈추지 못하게 한다. suffix hook은 `debug`와 `test-fast` Cargo profile에서만 컴파일된다. 어느 잔여 항목도 그 자체로 코드 주입을 허용하지는 않았다. #2765를 닫는다.

## 1. 문제 정의

### `..`가 남는 디렉터리를 만듦

`jit_module.cpp` overlay의 `ptx_cache_dir()`는 없는 구성 요소를 `make_private_directories`로 하나씩 만들고, 경로의 canonical화는 그 다음 `cache_dir_is_trusted`에서만 한다. `<dir>/missing/../cache`이면 순회가 `<dir>/missing`을 만들고 `..`로 빠져나온 뒤 `<dir>/cache`를 만들었다. 캐시는 동작했지만 `<dir>/missing`이 디스크에 남았다.

### uid probe가 실패하거나 남을 수 있었음

`mlxcel-core`에는 `libc` 의존성이 없어서, `ptx_cache_dirs.rs`의 `current_euid`는 scope 디렉터리에 `.mlxcel-uid-probe-<pid>`를 만들고 그 소유자를 읽어 effective uid를 알아냈다. 생성과 삭제 사이에 crash가 나면 파일이 남았고, 소유자가 쓸 수 없는 scope 디렉터리(예: mode 0575)에서는 probe가 실패해 0700 수리가 조용히 아무것도 하지 않았다. #2743 리포트도 이 경우를 잔여 항목으로 적어 두었다.

### 캐시 항목을 파일 단위로 검사하지 않음

`read_cached_ptx`는 `<module>.ptx`와 `<module>.txt`를 `std::ifstream`으로 읽었다. `ifstream`은 symlink를 따라가고 소유자와 mode를 보지 않는다. #2743의 디렉터리 검사가 다른 사용자의 항목 심기를 막지만, 복원한 백업, 복사해 온 캐시, 이전 umask 때문에 신뢰된 디렉터리 안의 파일이 다른 사람에게 쓰기 가능할 수는 있다. 그러면 그 항목이 조용히 로드되었다.

### 0700 수리가 시작 과정을 멈출 수 있었음

`repair_dir`는 `symlink_metadata`로 경로를 검사한 뒤 `std::fs::File::open(path)`를 호출하고 device와 inode를 비교했다. 부모 디렉터리를 쓸 수 있는 사람은 두 호출 사이에 디렉터리를 FIFO로 바꿀 수 있고, FIFO를 읽기로 열면 writer가 나타날 때까지 block된다. `ensure_persistent_ptx_cache`는 `mlxcel`과 `mlxcel-server` 시작 시에 실행되므로, 모델을 로드하기도 전에 프로세스가 멈췄을 것이다.

### release 바이너리에 테스트 hook이 들어감

#2743은 `MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX`를 추가했다. publish 임시 파일의 suffix를 고정해서, 테스트가 publish가 여는 바로 그 경로에 symlink를 심을 수 있게 한다. 값은 `.tmp<hex>_<hex>` 형식으로 검증되어 경로 구분자를 담을 수 없었지만, 모든 빌드에 컴파일되었기 때문에 배포 바이너리가 테스트에만 필요한 환경 변수를 받아들였다. 이슈는 원래 이 항목을 범위 밖으로 두었으나, #2743 리뷰를 근거로 orchestrator가 이번 실행에 추가했다.

## 2. 기술적 선택과 그 이유

### 무엇이든 만들기 전에 `weakly_canonical`로 정규화

`ptx_cache_dir()`는 이제 `make_private_directories` 전에 절대 캐시 경로를 `std::filesystem::weakly_canonical(cache, error)`로 바꾼다. 이 호출은 존재하는 prefix를 신뢰 검사의 `canonical()`과 같은 방식으로 해석하고, 존재하지 않는 나머지는 어휘적으로 정규화한다. 그래서 만들어지는 디렉터리가 나중에 신뢰 검사가 승인하는 디렉터리와 같고, 검사는 여전히 그 결과에 대해 실행된다. 오류나 빈 결과는 다른 실패 분기처럼 기존의 "캐시 꺼짐" 값인 빈 경로를 반환한다. 대안은 `..`를 아예 거부하는 것이었는데, 지금 동작하는 설정을 깨면서 얻는 것이 없다. 신뢰 검사는 원래부터 정규화 전 경로를 본 적이 없다.

### 읽는 descriptor에서 항목을 검사

새 POSIX 전용 helper `read_trusted_cache_entry`는 각 항목 파일을 `O_RDONLY|O_NOFOLLOW|O_NONBLOCK|O_CLOEXEC`로 열고 descriptor를 `fstat`한 뒤, regular file이고 `geteuid()`나 root 소유이며 `S_IWGRP`와 `S_IWOTH`가 모두 없을 때만 그 descriptor로 읽는다. 경로가 아니라 descriptor를 검사하므로 읽은 파일이 곧 검사한 파일이다. `stat` 후 `ifstream`으로 다시 여는 방식은 그 틈을 다시 연다. `O_NOFOLLOW`는 symlink 항목의 open을 실패시키고, `O_NONBLOCK`은 항목 경로의 FIFO가 open을 block하지 못하게 한다(그 뒤 `S_ISREG` 검사가 거부한다). 읽기 루프는 `EINTR`을 재시도하며, short read, `fstat` 실패, 비정상적인 `st_size`에서 `resize`가 던지는 `std::bad_alloc`은 모두 miss다. RAII `FdCloser`가 예외 경로를 포함한 모든 경로에서 descriptor를 닫는다. 이것과 `.txt` 검사는 후속 commit d39ce2a3에서 들어왔다.

항목의 두 파일이 모두 통과해야 하고, 빈 `.ptx`는 #2492 때처럼 여전히 miss다. miss에는 새 코드가 필요 없다. `JitModule`이 다시 컴파일하고 `write_cached_ptx`가 기존 `O_CREAT|O_EXCL|O_NOFOLLOW` 임시 파일(mode 0600)과 `rename`으로 다시 publish하는데, 이 rename은 항목 경로의 symlink나 group-writable 파일을 그것을 통해 쓰지 않고 교체한다. Windows 분기는 기존 `file_size`와 `ifstream` 읽기를 유지한다.

### 이슈 문구와 달리 root 소유 항목도 허용

이슈는 "`geteuid()` 소유"를 요구했다. commit 7237b94b는 이를 현재 사용자 또는 root로 완화했다. `cache_dir_is_trusted`가 이미 root 소유 디렉터리 구성 요소를 허용하고, root는 어차피 어떤 파일이든 쓸 수 있으므로 root 소유 파일을 거부해도 보호가 늘지 않는다. 거부하면 root 소유의 미리 만든 캐시(예: container image에 포함된 캐시)는 non-root 사용자에게 영구적이고 조용한 miss가 된다. 모든 프로세스가 모든 kernel을 다시 컴파일하고, 쓸 수 없을지도 모르는 디렉터리에 다시 publish를 시도한다. PR 본문이 이 차이를 기록한다.

### `mlxcel-core`에서 `libc`를 직접 사용

`mlxcel-core`의 `[target.'cfg(unix)'.dependencies]`에 root crate와 같은 pin으로 `libc = "0.2"`를 추가했다. `Cargo.lock`에는 의존성 edge 하나만 늘고 새 package는 없다. `tighten_persistent_ptx_dirs`는 실패하지 않는 `libc::geteuid()`를 호출하고, probe 파일과 `current_euid`는 삭제되었다. 따라서 수리에 쓰기 가능한 scope 디렉터리가 더 이상 필요 없다. unit test도 같은 방식으로 uid를 읽는다.

### 수리용 open에 `O_DIRECTORY|O_NOFOLLOW|O_NONBLOCK`

`repair_dir`는 이제 `OpenOptions::custom_flags(libc::O_DIRECTORY | libc::O_NOFOLLOW | libc::O_NONBLOCK)`로 연다. 디렉터리가 아니거나 symlink이면 open이 실패하고, `O_NONBLOCK`은 FIFO 경우에도 즉시 반환하도록 보장한다. 앞선 `lstat`과의 device/inode 비교는 두 번째 방어선으로 남고, mode 변경은 여전히 descriptor(`fchmod`)를 통한다.

### test hook을 `PROFILE`이나 feature가 아니라 Cargo profile 디렉터리로 결정

hook은 이제 `#if defined(MLXCEL_PTX_CACHE_TEST_HOOKS)` 안에 있다. 언제 이 매크로를 정의할지가 가장 신경 쓴 부분이고, `build_support/ptx_cache_test_hooks.rs`의 모듈 주석이 뻔한 신호들이 왜 안 되는지 기록한다:

| 선택지 | 문제 |
|--------|------|
| `PROFILE` 환경 변수 | release를 상속하는 모든 profile에서 Cargo가 `release`로 설정한다. `test-fast`도 포함되므로 테스트 빌드가 hook을 잃는다 |
| `debug_assertions` | `test-fast`가 끈다 |
| Cargo feature | package metadata hash와 함께 `OUT_DIR`이 바뀌어, 테스트 빌드와 바이너리 빌드를 번갈아 하면 매번 MLX를 처음부터 다시 빌드한다 |
| **`OUT_DIR`에서 읽은 profile 디렉터리** (선택) | Cargo의 `<target>/[<triple>/]<profile>/build/<pkg>-<hash>/out` 구조에 의존한다. 인식하지 못하는 구조면 hook을 끈다(fail closed) |

`profile_dir_name`은 `OUT_DIR`의 조상 네 단계를 따라가며 `out`과 `build` 이름을 확인하고 profile 디렉터리 이름을 반환한다. `test_hooks_enabled`는 `debug`와 `test-fast`에서만 true다. 그 밖의 profile(`release`, `bench`, `dist`, 사용자 정의 profile)과 인식하지 못하는 구조는 모두 hook을 뺀다. `build.rs`는 이 결과를 `cfg!(feature = "cuda")`와 결합해 CMake에 `MLXCEL_PTX_CACHE_TEST_HOOKS=ON|OFF`를 넘기고 `cfg(mlxcel_ptx_cache_test_hooks)`를 내보낸다. `cuda_arch::PTX_CACHE_TEST_HOOKS`가 이 값을 노출하므로, hook이 없으면 `symlinked_temporary_is_not_written_through`가 메시지를 출력하고 skip한다. CMake에서는 정의를 `jit_module.cpp` 하나의 source property로만 설정하므로(`set_property(SOURCE ... TARGET_DIRECTORY mlx ...)`), 값을 바꿔도 MLX 전체가 아니라 파일 하나만 다시 컴파일된다.

판정 로직은 순수 함수로, `mlx_pin.rs`와 같은 방식으로 `build.rs`와 `mlxcel-mlx-pin` 양쪽에 path로 include된다. 그래서 unit test가 MLX를 빌드하지 않고 실행된다. CI의 `mlx_pin` path filter에 `build_support/ptx_cache_test_hooks*.rs`가 추가되었고, `scripts/ci/check_env_var_docs.py`는 `MLXCEL_PTX_CACHE_TEST_HOOKS`를 환경 변수가 아닌 빌드 내부 CMake option으로 allowlist에 넣었다.

## 3. 변경 요약

16개 파일, +676/-35. commit 세 개와 origin/main merge 두 개(77dff247, f04452d8):

| Commit | 내용 |
|--------|------|
| `9cca24d4` | 다섯 항목 전체: `weakly_canonical`, `.ptx` 항목별 검사, `libc::geteuid`, non-blocking 수리 open, profile 기반 test hook, 테스트와 문서 |
| `7237b94b` | 디렉터리 규칙에 맞춰 root 소유 항목 허용 |
| `d39ce2a3` | `.txt` kernel map도 검사, `resize` 예외 시 RAII guard로 descriptor 닫기, 항목 규칙 문서화 |

영역별:

- `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp`: `ptx_cache_dir()`에 `weakly_canonical`, 새 `read_trusted_cache_entry`, `read_cached_ptx`를 `_WIN32` 분기(변경 없음)와 두 파일을 helper로 읽는 POSIX 분기로 분리, `test_temp_suffix`를 `MLXCEL_PTX_CACHE_TEST_HOOKS`로 보호, `[mlxcel #2765]` 태그가 붙은 헤더 주석.
- `src/lib/mlxcel-core/src/ptx_cache_dirs.rs`: probe 제거, `geteuid`, flag를 준 open, unit test 2개 추가(총 6개).
- `src/lib/mlxcel-core/build.rs`, `build_support/ptx_cache_test_hooks.rs`(+ 테스트), `src/cuda_arch.rs`, `src/lib/mlxcelverse/CMakeLists.txt`, `src/lib/mlxcel-mlx-pin/src/lib.rs`: profile gate의 처음부터 끝까지.
- `src/lib/mlxcel-core/Cargo.toml`, `Cargo.lock`: unix 전용 `libc`.
- `tests/cuda_ptx_cache_trust.rs`와 새 `tests/cuda_ptx_cache_trust/entries.rs`: child process 시나리오 4개 추가(`dotdot_in_cache_dir_is_normalized`, `group_writable_cache_entry_is_not_loaded`, `symlinked_cache_entry_is_not_loaded`, `group_writable_kernel_map_is_not_loaded`). 각각 JIT 재컴파일, 첫 실행과 같은 출력, group/other 쓰기가 없는 regular file로 다시 publish되었는지를 확인한다. hook 테스트는 hook이 빠진 빌드에서 skip한다.
- `.github/workflows/ci.yml`, `scripts/ci/check_env_var_docs.py`: path filter와 allowlist.
- `docs/environment-variables.md`, `docs/installation.md`: `..` 해석, 항목별 규칙, hook의 profile 제한.

## 4. 트레이드오프와 잔여 항목

- **신뢰할 수 없는 항목은 재컴파일 한 번의 비용이 든다.** group-writable, symlink, 다른 사용자 소유, regular file이 아닌 항목은 miss다. kernel을 다시 컴파일하고 항목을 0600 파일로 교체하므로 나쁜 항목마다 한 번만 비용을 낸다. `docs/installation.md`는 이전 릴리스가 umask 002에서 쓴 항목이 한 번 재컴파일된다고 적는다. 실제로 해당되는 범위는 작다. #2743 이후 publish는 device 하위 디렉터리 안에 0600 임시 파일을 만들고, #2743 이전의 flat 항목은 아예 읽지 않는다. 현실적인 발생 경로는 복사하거나 복원하거나 손으로 준비한 캐시다.
- **사용자가 쓸 수 없는 디렉터리에 있는 root 소유 항목은 나쁘면 계속 miss다.** 다시 publish해도 교체할 수 없으므로 그 kernel은 매 실행마다 재컴파일된다. 이전과 같은 동작이고, 미리 만든 캐시가 망가졌을 때만 문제가 된다.
- **hook 테스트 커버리지는 테스트 profile에 달려 있다.** `release`나 사용자 정의 profile로 테스트를 돌리면 `symlinked_temporary_is_not_written_through`가 실패하지 않고 메시지를 출력하며 skip한다. CI와 `make verify-test*`는 `test-fast`를 쓰므로 gate에서는 계속 실행된다. 판정은 Cargo target 디렉터리 구조에도 의존한다. 향후 구조가 바뀌면 hook이 꺼지고(fail closed) 이 skip으로 드러난다.
- **release 빌드는 실행하지 않았다.** `--release` 바이너리가 변수를 무시한다는 근거는 profile 판정 unit test(허용 목록에 `release`를 추가하는 mutation이 `release_builds_leave_the_hooks_out`을 실패시키는 것 포함)와 source 단위 CMake 정의뿐이다. release 바이너리를 빌드해서 확인하지는 않았다.
- **Windows와 ROCm은 변경 없음.** `_WIN32` 읽기 경로는 여전히 파일별 검사 없이 `ifstream`을 쓴다. ROCm overlay(#2641)는 범위 밖이었다.
- **mode bit만 본다.** #2743과 마찬가지로 POSIX ACL은 group bit에 반영되는 범위 외에는 검사하지 않는다.
- **overlay README 항목은 갱신하지 않았다.** 이슈는 이를 범위에 넣었지만 diff는 건드리지 않았고, 대신 소스 헤더 주석이 `[mlxcel #2765]` 설명을 담는다.

## 5. 검증

PR 본문에 보고된 대로, GB10에서 `--profile test-fast --features cuda`로 실행했다.

**5c8ef80a에서 먼저 실패 확인** (수정 전에 작성한 테스트): `dotdot_in_cache_dir_is_normalized`(남는 `missing` 디렉터리가 존재), `group_writable_cache_entry_is_not_loaded`와 `symlinked_cache_entry_is_not_loaded`(재컴파일 없이 조용한 cache hit), `repair_runs_on_a_scope_dir_the_owner_cannot_write`(mode가 0575로 유지), `a_directory_swapped_for_a_fifo_does_not_block`(2초 안에 반환하지 않음. 테스트가 직접 FIFO를 쓰기로 열어 block된 thread를 풀고 hang 대신 실패로 끝남). hook gate에 대해서는 `PROFILES_WITH_TEST_HOOKS`에 `release`를 추가하는 mutation이 `release_builds_leave_the_hooks_out`을 실패시킨다.

**수정 후.** `cuda_ptx_cache_trust` 14/14(hook 테스트와 `group_writable_kernel_map_is_not_loaded` 포함). `CMakeCache`에 `MLXCEL_PTX_CACHE_TEST_HOOKS:BOOL=ON`이 있고 `jit_module.cpp`만 정의를 받는다. `cuda_custom_kernel_disk_cache` 1/1, `ptx_cache_dirs` 6/6, `mlxcel-mlx-pin`의 `ptx_cache_test_hooks` 4/4. `cargo fmt`, mlxcel-core, mlxcel, mlxcel-mlx-pin의 lib와 테스트에 대한 `clippy -D warnings`, `make verify-env-var-docs`, `cargo deny check`가 통과했다.

**검증하지 않은 것.** release 빌드. 이슈의 실제 바이너리 확인(home 기본값으로 granite-4.0-h-tiny-4bit에서 `mlxcel generate` 두 번 실행, 두 번째 실행에 JIT 안내 없음)은 PR 본문에 없다. root 소유 항목 분기는 root 소유 파일을 만들려면 권한이 필요해서 테스트되지 않는다. 테스트된 사용자 소유 분기와 같은 조건식을 공유한다. `repair_runs_on_a_scope_dir_the_owner_cannot_write`는 root로 실행하면 skip한다. 성능 측정은 하지 않았다. 항목당 추가 작업은 파일마다 `open` 한 번과 `fstat` 한 번이며, 기존의 `file_size`와 `ifstream` open을 대체한다.

## 6. 학습 포인트

- 조회한 이름이 아니라 읽는 객체를 검증한다. `open(O_NOFOLLOW)`와 descriptor에 대한 `fstat`은 같은 inode에 대해 검사와 읽기를 한다. `stat(path)` 후 다시 여는 방식은 그렇지 않다.
- 대상이 FIFO로 바뀌었을 수 있는 open에는, 기대하는 객체가 regular file이나 디렉터리여도 `O_NONBLOCK`을 붙인다. FIFO를 읽기로 여는 평범한 `open`은 writer가 나타날 때까지 block되어, 타입 혼동이 hang으로 바뀐다.
- 사용자 경로는 구성 요소 단위로 처리하기 전에 정규화한다. 디렉터리를 만들며 순회하는 코드는 `..`를 글자 그대로 실행하므로, 나중의 canonical화가 따라잡기 전에 부작용이 생긴다.
- 파일에 대한 신뢰 규칙은 그 파일을 담는 디렉터리의 규칙과 맞아야 한다. 디렉터리에는 root를 허용하고 파일에는 허용하지 않으면, 보호는 늘지 않고 정상적인 미리 만든 캐시가 영구 miss가 된다.
- Cargo의 `PROFILE` 변수는 profile 이름이 아니라 상속한 기본 profile이고, feature flag는 `OUT_DIR`을 바꾼다. build script가 실제 profile을 알아야 하면 `OUT_DIR` 구조가 쓸 수 있는 신호다. 엄격하게 파싱하고 인식하지 못하면 끄는 쪽으로 처리한다.
- CMake 정의는 그것이 필요한 source 하나로 범위를 좁힌다(`set_property(SOURCE ... TARGET_DIRECTORY ...)`). 그래야 값을 바꿔도 큰 의존성의 빌드 전체가 무효화되지 않는다.
