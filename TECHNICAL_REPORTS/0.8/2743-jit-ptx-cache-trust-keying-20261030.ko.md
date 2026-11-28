# PR #2743: JIT PTX 디스크 캐시의 신뢰 검사와 키 강화

**날짜**: 2026-10-30 (merge 전에 작성, 작성 시점의 PR은 open draft)
**상태**: 리뷰 중
**언어**: C++ (CUDA overlay), Rust
**위험도**: 중간 (사용자 간 코드 주입 경로를 막는다. 대신 group-writable인 명시적 캐시 디렉터리나 group-writable인 `~/.cache`에서는 이제 디스크 캐시가 꺼지고 안내 문구가 출력되며, 모드를 고칠 때까지 매 실행마다 NVRTC 전체 컴파일 비용을 낸다)

## 요약

CUDA JIT overlay(`jit_module.cpp`)는 컴파일된 모듈을 PTX 디스크 캐시에서 읽어 로드하는데, #2538 이후로는 mlxcel의 fused MoE와 SSM decode kernel도 여기에 포함된다. 그런데 캐시 디렉터리 자체는 한 번도 검사하지 않았다. 다중 사용자 호스트에서 `/tmp/mlx`를 먼저 만든 사람, 또는 캐시 디렉터리의 상위 경로 중 하나라도 쓸 수 있는 사람은 추측 가능한 이름으로 모듈을 심어 다른 사용자의 프로세스에 로드시킬 수 있었다. 이 PR은 canonical 경로의 모든 구성 요소가 leaf부터 `/`까지 현재 사용자나 root 소유이고 group/other 쓰기 권한이 없을 때에만 캐시 디렉터리를 쓰도록 한다(root 소유의 sticky 디렉터리, 즉 `/tmp`는 예외). 또한 POSIX fallback을 사용자별 `$TMPDIR/mlx-<euid>/` root로 옮기고, 항목을 `sm<cc>-nvrtc<major>.<minor>` 하위 디렉터리로 구분하며, 고아가 된 publish 임시 파일을 정리하고, 임시 파일을 `O_CREAT|O_EXCL|O_NOFOLLOW`로 연다. Rust 쪽에서는 `ensure_persistent_ptx_cache`가 디렉터리를 0700으로 만들고, 이전 릴리스가 umask 002에서 0775로 만든 디렉터리를 고쳐서 기존 설치가 업그레이드 후에도 영구 캐시를 유지한다. #2539를 닫으며, PR #2552 보안 리뷰에서 추가된 두 항목도 포함한다.

## 1. 문제 정의

### 캐시 디렉터리를 무조건 신뢰함

`ptx_cache_dir()`는 프로세스당 한 번 계산되는 static이었다. `MLX_PTX_CACHE_DIR`가 있으면 그것을, 없으면 `temp_directory_path() / "mlx" / version() / "ptx"`를 쓰고, `create_directories`로 만든 뒤 소유자나 모드를 확인하지 않았다. 이 fallback은 호스트의 모든 사용자가 공유했다. `ensure_persistent_ptx_cache`를 호출하는 것은 `mlxcel`과 `mlxcel-server`뿐이므로 `bench_decode`, `bench_engine`, `engine_parity`, `speculative_bench`, 모든 test binary, 모든 embedder가 이 공유 fallback을 썼다. 모듈 파일 이름은 kernel 이름, template 인자, 소스 해시로 정해져 추측할 수 있으므로, `/tmp/mlx`를 먼저 만든 사용자는 다른 사용자의 프로세스가 `cuModuleLoadDataEx`에 넘길 항목을 미리 채워 둘 수 있었다. PR #2538이 custom kernel을 캐시 대상으로 만들면서 위험이 커졌다. fused decode kernel은 이제 warm 실행마다 이 캐시에서 온다.

### 항목이 device나 컴파일러로 구분되지 않음

`compile()`은 `--gpu-architecture=sm_<major><minor>[a]`를 대상으로 하고 SASS를 낼 수도 있지만(`compiler_supports_device_sass`), 모듈 이름에도 디렉터리에도 compute capability나 NVRTC 버전이 들어 있지 않았다. 여러 GPU나 toolkit이 공유하는 캐시 디렉터리(NFS home, container image layer, 두 세대의 GPU가 있는 머신)에서는 한 device가 다른 device용으로 빌드된 모듈을 받을 수 있었다.

### 고아 임시 파일과 symlink를 따라가는 publish

`write_cached_ptx`는 임시 파일에 쓴 뒤 rename하는 방식으로 publish한다. 둘 사이에 프로세스가 죽으면 임시 파일이 영구히 남았다. PR #2552에서 이름을 바꾼 뒤로 임시 파일은 module 이름 없이 `.tmp<pid hex>_<tick hex>.<ext>` 형태의 dotfile이므로, 정리 로직은 module 이름 접두사가 아니라 이 형태로 맞춰야 한다. 임시 파일은 `std::ofstream`으로 열었는데, 이는 그 경로에 심어진 symlink를 따라가 대상에 쓴다.

### retry 분기가 테스트되지 않음

`JitModule::JitModule`에는 retry가 있다. driver가 캐시된 모듈을 거부하면 항목을 지우고, 다시 컴파일하고, 다시 publish한다. `tests/cuda_custom_kernel_disk_cache.rs`는 빈 `.ptx`와 잘못된 map으로만 이 분기에 도달하려 했는데, 둘 다 `read_cached_ptx`가 이미 miss로 처리하므로 `cuModuleLoadDataEx` 실패 경로 자체는 테스트에서 한 번도 실행되지 않았다.

## 2. 기술적 선택과 그 이유

### 강제는 Rust init이 아니라 overlay에서

모든 binary와 embedder가 `jit_module.cpp`를 공유한다. Rust 쪽 검사는 그것을 호출하는 곳에만 닿는데, 그 집합(`mlxcel`, `mlxcel-server`)은 이미 공유 temp fallback을 쓰지 않던 쪽이다. 검사를 `ptx_cache_dir()`와 `device_cache_dir()`에 두면 bench binary, test binary, embedder를 고치지 않고도 모두 적용된다. `_WIN32` 분기는 기존 동작을 유지한다. 이슈가 범위를 POSIX로 정했다.

### canonical 경로의 모든 구성 요소를 검사하고, 검사한 canonical 경로를 사용

`cache_dir_is_trusted`는 `std::filesystem::canonical(dir)`를 구한 뒤 leaf부터 `/`까지 각 구성 요소를 `lstat`한다. 구성 요소는 `geteuid()`나 root 소유이고 group/other 쓰기 권한이 없어야 통과한다. 단, root 소유이면서 sticky bit가 있으면(`/tmp`, 다른 사람이 내 항목을 rename하거나 지울 수 없는 곳) 예외다. leaf는 디렉터리여야 한다. leaf만 검사하는 방식은 기각했다. 쓰기 가능한 상위 디렉터리의 소유자는 leaf를 다른 곳으로 옮기고 자기 디렉터리를 그 자리에 둘 수 있기 때문이다. 이후에는 canonical 경로를 쓰므로 검사 뒤에 끼워 넣은 symlink 구성 요소가 이후의 읽기와 쓰기를 다른 곳으로 돌릴 수 없다. 또 그 시점에는 모든 구성 요소가 사용자나 root 소유이고 다른 사람에게 닫혀 있으므로, root가 아닌 다른 사용자는 프로세스가 끝날 때까지 해석된 경로를 바꿀 수 없다. `canonical()` 실패는 신뢰하지 않음으로 처리한다.

검사는 출처와 무관하게 적용된다. 명시적 `MLX_PTX_CACHE_DIR`, mlxcel binary가 이 변수로 설정하는 영구 기본값, temp fallback 모두 같다. 없는 구성 요소는 먼저 `::mkdir(..., 0700)`으로 만든다(`make_private_directories`). 이미 있는 구성 요소는 overlay가 chmod하지 않고 판정만 한다.

### mlxcel만 만드는 이름이 symlink이면 거부

사용자별 root인 `$TMPDIR/mlx-<euid>`는 공유 디렉터리 안에 있어서 다른 사용자가 먼저 그 이름을 만들 수 있고, device 하위 디렉터리 `sm<cc>-nvrtc<ver>`는 overlay만 만드는 이름이다. 이 둘에 대해 `make_owned_directory`는 `mkdir` 뒤에 경로를 `lstat`해서 symlink나 디렉터리가 아닌 것을 거부한다. 그런 링크를 따라가면 다른 사람이 이 사용자의 캐시 위치를 정하게 되고, 신뢰 검사는 링크 대상을 보게 되는데, 대상은 검사를 통과하면서도 private해지기 전에 쓰인 모듈을 여전히 담고 있을 수 있다(예: Rust 쪽 repair가 고치기 전까지 group-writable이던 디렉터리). 사용자가 지정한 `MLX_PTX_CACHE_DIR`에는 여전히 symlink가 있어도 된다. `canonical()`이 이를 해석하고, 해석된 구성 요소를 검사한다. 이 보완은 이전 PR head에 대한 리뷰에서 나왔다(commit 43e3d073).

### 거부하면 프로세스가 아니라 캐시를 끔

거부 시 `ptx_cache_dir()`나 `device_cache_dir()`는 빈 경로를 반환한다. 이는 `read_cached_ptx`와 `write_cached_ptx`가 이미 따르던 "캐시 꺼짐" 값이므로 kernel은 여전히 컴파일되고 실행된다. stderr에는 프로세스당 한 줄이 출력되며, 캐시 거부는 진행 상황 잡음이 아니라 설정 문제이므로 일부러 `MLXCEL_QUIET_JIT`로 끄지 못하게 했다: `mlxcel: CUDA kernel disk cache disabled: <component> <reason>`. reason은 `is not owned by the current user`, `is group- or other-writable`, `is a symbolic link`, `is not a directory`, `cannot be resolved (<error>)` 중 하나다. 프로세스를 실패시키는 방안은 고려하지 않았다. 캐시는 최적화이고, 하드 실패는 모드 비트 하나를 장애로 바꾼다.

### 기존 scoping 아래에 device와 컴파일러 단계를 추가

`device_cache_dir(device)`는 `ptx_cache_dir() / "sm<major><minor>-nvrtc<major>.<minor>"`(예: `sm121-nvrtc13.0`)를 반환한다. 0700으로 만들고, symlink면 거부하며, root와 같은 신뢰 검사를 한다. 프로세스마다 key별로 한 번, 누수시킨 mutex와 map 아래에서 해석한다(`get_jit_module`의 캐시처럼 누수시키는 이유는 static teardown 중에도 모듈이 빌드될 수 있기 때문이다). `JitModule::JitModule`은 `read_cached_ptx`, `write_cached_ptx`, retry의 `get_ptx_path`에 이 디렉터리를 쓴다. `ensure_persistent_ptx_cache`의 Rust 쪽 commit scoping(`cuda-ptx/<mlx-commit>`)은 그대로이고, 그 아래에 한 단계를 더한다. root에 이미 있는 flat 항목은 더 이상 읽지 않고 migration도 하지 않는다. 다음 pin 범위 디렉터리로 바뀔 때 함께 사라진다.

### 배타적이고 symlink를 따라가지 않는 임시 파일: 다층 방어

신뢰 검사를 통과한 뒤에는 다른 사용자가 캐시 디렉터리에 파일을 만들 수 없으므로, symlink가 심어진 경우는 주로 디렉터리가 private해지기 전에 남은 파일이나 같은 사용자의 실수에 해당한다. POSIX publish는 이제 임시 파일을 `O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC`, 모드 0600으로 열고 `write_all`(짧은 쓰기와 `EINTR` 재시도)로 쓴다. 실패하면(`EEXIST`, `ELOOP`) 이 프로세스에서는 해당 항목을 publish하지 않고 그 kernel의 캐시는 비어 있는 상태로 남는다. 컴파일된 코드는 그대로 쓰이고, 충돌한 경로는 우리 것이 아니므로 지우지 않는다.

나노초 tick을 예측하지 않고 이를 테스트하기 위해, overlay는 suffix를 고정하는 테스트 전용 hook `MLXCEL_PTX_CACHE_TEST_TMP_SUFFIX`를 읽는다. 값이 생성되는 형태 `.tmp<hex>_<hex>`일 때만 적용되므로 경로 구분자를 담을 수 없고, 다른 값은 무시된다. `docs/environment-variables.md`에 테스트 전용으로 문서화했다.

### 이름, 종류, 나이로 오래된 임시 파일 정리

프로세스에서 device 디렉터리를 처음 해석할 때 `sweep_stale_temporaries`는 최상위 항목 중 이름이 `.tmp<hex>_<hex>.{ptx,txt,cu}`와 맞고, `symlink_status`로 판정한 일반 파일이며(그래서 symlink는 따라가지도 지우지도 않는다), mtime이 한 시간보다 오래된 것(`kStaleTempAge`)을 지운다. 정상 publish는 수 밀리초 안에 rename하므로 한 시간이면 고아 파일과 진행 중인 쓰기를 넉넉히 구분한다. 오류는 무시한다. 정리는 housekeeping이다. 245자를 넘는 module 이름 때문에 생기는 중첩 디렉터리는 범위 밖이다.

### 이전 릴리스가 만든 디렉터리를 descriptor로 수리

umask 002(user private group, Ubuntu 일반 사용자의 기본값)에서 이전 릴리스는 `~/.cache/mlxcel/cuda-ptx/<commit>`를 0775로 만들었다. 수리하지 않으면 새 검사가 이런 기존 설치 전부에서, 즉 캐시가 원래 겨냥한 호스트에서 영구 캐시를 끈다. `ensure_persistent_ptx_cache`는 이제 `DirBuilder` 모드 0700으로 만들고 `ptx_cache_dirs::tighten_persistent_ptx_dirs`를 호출한다. 이 함수는 `cuda-ptx/<scope>`와 `cuda-ptx`의 group/other 쓰기 권한을 지우고, root는 `MLXCEL_CACHE_DIR`가 설정되지 않았을 때만 고친다. 명시적 `MLXCEL_CACHE_DIR`, 명시적 `MLX_PTX_CACHE_DIR`, root 위의 경로(`~/.cache` 등)는 사용자가 정한 그대로 두고, 검사에 실패하면 안내 문구와 함께 거부한다.

수리는 euid 소유이고 지울 쓰기 비트가 있는 실제 디렉터리에만 적용된다. 경로를 열고, 열린 객체의 device와 inode를 앞선 `lstat` 결과와 비교한 뒤, descriptor를 통해 모드를 바꾼다(`File::set_permissions`를 통한 `fchmod`). group/other 쓰기 외의 비트는 모두 유지된다(setgid도 남는다). 경로 기준 chmod는 상위 디렉터리를 쓸 수 있는 누군가가 끼워 넣은 symlink를 따라가서, 링크 대상(예: private 파일)에 0755를 적용할 수 있다. `mlxcel-core`는 libc 의존성이 없으므로, euid는 scope 디렉터리에 만든 probe 파일의 소유자로 읽고 곧바로 지운다. 디렉터리에 쓸 수 없으면 수리를 건너뛴다.

## 3. 변경 요약

9개 파일, +1520/-140, commit 네 개와 origin/main merge 하나(c242a692):

| Commit | 내용 |
|--------|------|
| `3a08816b` | 신뢰 검사, 사용자별 fallback, device 키, 오래된 임시 파일 정리, test hook을 포함한 `O_EXCL\|O_NOFOLLOW` publish, retry 분기 테스트, 공용 테스트 helper |
| `7031ca5f` | `ensure_persistent_ptx_cache`가 0700으로 만들고 group-writable 기본 디렉터리를 수리 |
| `43e3d073` | symlink인 `mlx-<euid>` root와 device 디렉터리 거부, device/inode 일치를 확인하는 descriptor 기반 수리 |
| `a8fc4840` | 모든 거부 사유와 정리 동작 문서화 |

- `src/lib/mlxcelverse/overlays/cuda/mlx/backend/cuda/jit_module.cpp`: `report_cache_disabled`, `make_owned_directory`, `make_private_directories`, `cache_dir_is_trusted`, `is_temp_suffix`, `is_publish_temporary`, `test_temp_suffix`, `write_all`, `sweep_stale_temporaries`, `device_cache_dir` 추가. `ptx_cache_dir()`의 POSIX 분기 재작성. `write_cached_ptx`는 임시 파일을 배타적으로 열고 중첩 디렉터리를 private으로 만든다. `JitModule::JitModule`은 읽기, publish, retry에 `device_cache_dir(device)`를 쓴다.
- `src/lib/mlxcel-core/src/lib.rs`: `ensure_persistent_ptx_cache`가 0700 `DirBuilder`를 쓰고 수리를 호출한다.
- `src/lib/mlxcel-core/src/ptx_cache_dirs.rs`(신규, unix 전용): `make_private_dir`, `repair_dir`, `current_euid`, `tighten_persistent_ptx_dirs`와 unit test 4개.
- `tests/common/cuda_kernels.rs`(신규): fused MoE 실행, `emit`/`parse_out`, `private_tempdir`(tempfile은 모드를 umask에 맡긴다), device 디렉터리 helper. 두 CUDA 캐시 테스트가 공유하며, `tests/common/mod.rs`가 `cfg(feature = "cuda")`로 노출한다.
- `tests/cuda_ptx_cache_trust.rs`(신규, 테스트 10개): group-writable 디렉터리 거부, 0777 상위 아래의 0700 디렉터리 거부, private 디렉터리 허용, `TMPDIR` 아래 사용자별 fallback(`$TMPDIR/mlx`는 만들지 않음), device와 NVRTC 키, 오래된 임시 파일 정리, symlink 임시 파일에 쓰지 않음(잘못된 hook 값 무시 포함), symlink device 디렉터리 거부, symlink fallback root 거부, child 진입점. `MLX_PTX_CACHE_DIR`와 `TMPDIR`는 프로세스당 한 번 읽히므로 각 시나리오는 test binary를 child로 다시 실행한다. parent는 CUDA를 건드리지 않는다.
- `tests/cuda_custom_kernel_disk_cache.rs`: 항목을 device 하위 디렉터리에서 찾는다. 새 단계는 gate-up `.ptx`를 `b"not a cuda module\n"`로 덮어쓰고 map은 유지한 뒤, child 성공, JIT 안내 문구, 비트 단위로 같은 출력, 전체 길이로 다시 publish된 `.ptx`, 바뀌지 않은 map을 요구한다.
- `docs/installation.md`, `docs/environment-variables.md`: 사용자별 fallback, device 하위 디렉터리, 신뢰 규칙, 모든 거부 사유, 수리, 정리, test hook.

## 4. 트레이드오프와 잔여 항목

- **group-writable 디렉터리의 동작 변경.** group-writable인 명시적 `MLX_PTX_CACHE_DIR`나 `MLXCEL_CACHE_DIR`, group-writable인 `~/.cache`나 home 디렉터리는 거부된다. 프로세스는 동작하지만 매 실행마다 모든 kernel을 컴파일한다(custom fused decode kernel만으로도 GB10에서 `mlxcel generate`마다 약 1초, #2492 기준). 안내 문구가 문제의 구성 요소를 알려 주며, 해결 방법은 그 구성 요소에 `chmod go-w`를 하거나 private한 `MLX_PTX_CACHE_DIR`를 지정하는 것이다. 한 그룹의 사용자들이 일부러 캐시를 공유하던 배포는 캐시를 잃는다. 그 공유가 바로 이 PR이 막는 공격 경로다.
- **다른 사용자 소유는 테스트하지 않음.** group-writable과 같은 판정식을 쓰지만, 다른 사용자 소유 디렉터리는 권한 없이 만들 수 없으므로 모드 분기만 실행된다.
- **모드 비트만 봄.** 검사는 `st_mode`와 `st_uid`만 읽고 POSIX ACL은 보지 않는다. Linux에서 특정 사용자나 그룹에 쓰기를 허용하는 ACL은 보통 group 쓰기 비트(ACL mask)로 드러나므로 거부되겠지만, ACL을 다루는 테스트는 없다.
- **root는 신뢰함.** root 소유 구성 요소는 통과하며, root는 언제든 무엇이든 바꿀 수 있다. 의도한 설계다.
- **Windows와 ROCm은 그대로.** `_WIN32` 분기는 소유자 검사 없이 공유 `%TEMP%\mlx\<version>\ptx`를 계속 쓴다. ROCm overlay는 범위 밖이었다.
- **예전 flat 항목은 디스크에 남음.** 읽지도 지우지도 않으며, pin 범위 디렉터리가 바뀔 때까지 공간을 차지한다.
- **수리는 쓰기 가능한 scope 디렉터리에 의존.** `current_euid`는 probe 파일을 만든다. 읽기 전용이거나 uid가 squash되는 파일시스템에서는 uid를 읽지 못하거나 일치하지 않아 수리가 조용히 아무것도 하지 않고, overlay가 안내 문구와 함께 거부한다.
- **overlay README는 갱신하지 않음.** `jit_module.cpp`를 언급하는 줄이 #2696과 겹쳐서 충돌을 피하려고 그대로 두었다.
- **이슈에서 범위 밖으로 정한 항목:** bench binary에서 `ensure_persistent_ptx_cache` 호출, publish 시 fsync, 긴 module 이름용 중첩 디렉터리 정리.

## 5. 검증

모두 GB10에서 `gpu-lock` 아래 실행했으며, PR 본문에 보고된 결과다.

**테스트 묶음.** `cuda_ptx_cache_trust` 10/10, `cuda_custom_kernel_disk_cache` 1/1, `ptx_cache_dirs` unit test 4/4. `cargo fmt`, `clippy -D warnings`(mlxcel과 mlxcel-core의 lib와 test, 두 test target), `make verify-env-var-docs` 통과.

**main 8385f40b에서 먼저 실패 확인.** 0770 디렉터리, 0777 상위, 사용자별 fallback, device 키, 정리, 수정된 disk-cache 테스트가 모두 실패했고 private 디렉터리 대조군은 통과했다. 이전 PR head에서는 symlink device 디렉터리, symlink fallback root, 경로가 바뀐 chmod가 실패했고, commit 43e3d073이 이를 고쳤다.

**main에서 실패시킬 수 없는 두 테스트의 mutation 근거.** test hook은 새로 추가됐고 retry는 원래 있었으므로 두 테스트 모두 main에서는 실패할 수 없다. 임시 파일 열기를 symlink를 따라가는 `O_CREAT|O_TRUNC`로 되돌리고 retry catch 맨 앞에 `throw;`를 넣자, symlink 테스트가 실패했고 garbage `.ptx` 단계가 abort했다.

**실제 binary.** granite-4.0-h-tiny-4bit에서 `mlxcel generate`의 두 번째 실행은 새 캐시에서도, 실제 home 기본값에서도 JIT 안내 문구를 출력하지 않았다. home 기본값에서 0775 디렉터리는 0755가 되었고, `sm121-nvrtc13.0`은 0700으로 만들어졌다.

**검증하지 않은 것.** GB10의 Linux에서만 실행했고, umask나 `TMPDIR` 관례가 다른 배포판은 실행하지 않았다. NFS 위의 home 디렉터리, 한 호스트에서 두 세대 GPU가 공유하는 캐시, Windows 분기는 실행하지 않았다. 성능 측정은 하지 않았다. 추가 작업은 해석한 디렉터리마다 프로세스당 한 번, `canonical()` 한 번과 경로 구성 요소당 `lstat` 한 번이다.

## 6. 학습 포인트

- 경로에 대한 신뢰 검사는 leaf가 아니라 `/`까지의 모든 구성 요소를 다뤄야 한다. 어느 상위 디렉터리든 쓸 수 있다면 leaf를 바꿔치기할 수 있다. root 소유 sticky 예외 덕분에 `/tmp` 자체는 통과하고, 다른 사용자의 `/tmp/mlx`는 실패한다.
- 검사한 뒤에는 검사가 승인한 canonical 경로를 쓴다. 나중에 원래 경로를 다시 해석하면 검사가 닫은 틈이 다시 열린다.
- 공유 디렉터리 안에서 자기 코드만 만드는 이름에 symlink가 있다면, 그것은 따라갈 설정이 아니라 조작의 증거다. 링크를 해석해서 대상을 검사하면, 예전에 심어진 내용을 아직 담고 있는 디렉터리를 승인할 수 있다.
- 모드를 안전하게 바꾸려면, 검사한 것과 device와 inode가 같은 descriptor에 대해 작업해야 한다. `chmod(path)`는 그 순간 경로에 있는 무엇이든 따라간다.
- 기존 설치가 기본적으로 실패하게 되는 보안 강화에는 같은 PR 안에 migration이 있어야 한다. 여기서는 umask 002 디렉터리 수리가 업그레이드 후에도 영구 캐시를 유지시킨다. 그것이 없었다면 이 수정은 자신이 보호하는 기능을 조용히 꺼 버렸을 것이다.
- 결정적으로 도달할 수 없는 코드 경로(나노초 tick으로 정해지는 임시 파일 이름)에는, 엄격하게 검증되는 test hook과 mutation 실행이 옛 코드와 새 코드에서 모두 통과하는 테스트보다 강한 근거다.
