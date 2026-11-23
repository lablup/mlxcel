# PR #3289: XLA 아티팩트 캐시를 사용자 전용으로 만들기

**날짜**: 2026-11-23
**상태**: PR 열림(#3248을 닫음). ROCm gfx1151에서 IREE `hip`으로 검증했다. 같은 게이트와 greedy 비교를 GB10 / CUDA에서 돌리는 작업은 아직 남아 있다.

## 요약

OpenXLA/IREE가 컴파일한 아티팩트 캐시(텍스트 VMFB 번들, Phi4MM과 Gemma3n 오디오, 비전 타워 캐시 여섯 개, ROCm 튜닝 스펙, 네이티브 GEMV HSACO 오브젝트, aux smoke와 numeric probe)는 모두 공유 임시 디렉터리 아래 고정된 이름의 디렉터리였고, 캐시된 아티팩트를 읽을 때 소유자를 전혀 확인하지 않았다. 다른 로컬 사용자가 디렉터리를 먼저 만들어 두거나, 조건이 맞는 manifest와 함께 VMFB를 심어 둘 수 있었다. manifest의 모든 필드는 공개된 입력에서 계산할 수 있기 때문이다. CPU 타깃 VMFB는 네이티브 코드이므로 결과적으로 피해 사용자 권한으로 코드가 실행된다.

이제 캐시는 `${MLXCEL_CACHE_DIR:-$HOME/.cache/mlxcel}/xla/<cache>` 아래에 있다. 캐시 디렉터리들과 파일이 현재 사용자 전용이고(`O_NOFOLLOW` 디스크립터에서 확인), 캐시 루트 위의 어떤 디렉터리도 다른 사용자가 루트를 바꿔치기할 수 없는 경우에만 아티팩트를 재사용한다. 검사에 실패한 위치는 건드리지 않고 stderr 경고 한 줄로 알린 뒤, 새 `mkdtemp` 디렉터리에 컴파일한다. 파일은 원자적 rename으로 `0600` 권한으로 게시한다. 검사 로직은 `mlxcel-core`의 새 모듈 두 개(`private_dir`, `private_group`)에 있고, `mlxcel-xla`의 새 모듈 `cache_dir`이 모든 프로덕션 지점을 이 검사로 보낸다.

PR은 커밋 다섯 개를 거쳤다. 리뷰에서 상위 디렉터리 검사가 추가됐고, 그룹 쓰기 가능한 상위 디렉터리에 대해서는 사용자가 fail-closed 정책을 골랐으며, 보안 리뷰에서 HIGH(fallback 루트를 다시 검증하지 않음)와 MEDIUM(nsswitch 파서의 fail-open)을 찾아 둘 다 고쳤다.

## 1. 문제 정의

#3248(#3242 보안 리뷰 중 발견, 기존부터 있던 문제)은 다음 지점을 짚었다. `iree.rs`, `tuning.rs`, `phi4_audio.rs`는 `temp_dir()/mlxcel-xla-vmfb`를 썼고, 비전과 오디오 런타임은 같은 임시 디렉터리 아래 계열별 이름을 썼으며, `native_qmv.rs`는 `temp_dir()/mlxcel-xla-qmv4`를 썼다. 모든 지점이 `create_dir_all`을 호출한 뒤 이름이 이미 있으면 거기 있는 것을 그대로 읽었다.

보조 manifest(`aux_manifest.rs`)가 증명하는 것은 아티팩트가 체크포인트 설정, 컴파일러 버전, 플래그, MLIR 텍스트로 만든 계약과 일치한다는 사실뿐이다. 이 중 비밀은 없으므로 manifest는 파일을 누가 썼는지에 대해 아무것도 말해 주지 않는다. JIT PTX 캐시도 같은 종류의 문제가 있었고 #2743과 #2777에서 강화됐지만, XLA 캐시는 그런 처리를 받은 적이 없었다.

옆에 작은 결함이 두 개 더 있었다. `compile_one`과 튜닝 스펙은 최종 이름에 바로 썼기 때문에, 도중에 프로세스가 죽으면 불완전한 VMFB가 남고 다음 로드가 그것을 재사용했다. 또 `umask`가 `002`이면 모든 캐시 파일이 그룹 쓰기 가능 상태가 됐다.

## 2. 변경 요약

| 파일 | 변경 |
|---|---|
| `mlxcel-core/src/private_dir.rs` (신규, public) | `ensure_private_dir`, `verify_private_dir`, `verify_trusted_ancestors`, `open_private_file`, `seal_private_file`, `create_fresh_private_dir`, `current_euid`, `clear_group_write` |
| `mlxcel-core/src/private_group.rs` (신규) | 프로세스 egid가 단독 멤버 그룹임을 `/etc/nsswitch.conf`, `/etc/group`, `/etc/passwd`로 증명 |
| `mlxcel-xla/src/cache_dir.rs` (신규) | `XlaCache` enum, `xla_cache_dir`, `artifact_slot`, `read_private`, `write_private_atomic`, `publish_private`, `mkdtemp` fallback |
| `iree.rs`, `tuning.rs`, `native_qmv.rs`, `aux_manifest.rs` | `artifact_slot`으로 캐시 조회, 임시 형제 파일에 컴파일한 뒤 rename으로 `0600` 게시, GEMV HSACO는 검사한 디스크립터에서 읽음 |
| 비전/오디오 런타임, `aux_smoke.rs`, `numeric_probe.rs` | `temp_dir().join(...)`과 `create_dir_all`을 `xla_cache_dir(XlaCache::...)`로 교체 |
| `production_tests.rs` | 테스트 디렉터리를 프로덕션과 같이 `0700`으로 생성. umask 002에서 `create_dir_all`로 만들면 이제 거부된다 |
| `docs/installation.md`, `docs/environment-variables.md`, `CHANGELOG.md` | 캐시 위치, 상위 디렉터리와 그룹 규칙, 거부 메시지, 캐시 비우는 방법 |
| `scripts/xla/fusion_spike.sh` | decode MLIR을 새 위치에서 찾음 |

### 디렉터리와 파일 검사

`verify_private_dir`과 `open_private_file`은 경로를 `O_NOFOLLOW | O_NONBLOCK`으로 열고 디스크립터를 `fstat`한다. 실제 디렉터리나 일반 파일이어야 하고, 소유자가 `euid`여야 하며, 그룹이나 other 쓰기 비트가 없어야 한다. `O_NONBLOCK`은 파일 자리에 FIFO가 바꿔치기돼도 open이 멈추지 않게 한다(#2765와 같은 교훈). GEMV 캐시는 같은 디스크립터에서 바이트를 읽기 때문에 검사와 읽기 사이에 틈이 없다.

`ensure_private_dir(root, subpath, repair_root, euid)`는 상대 경로 루트나 `..`가 들어간 루트를 거부하고, 경로 체인을 `0700`으로 만들고, mlxcel이 관리하는 디렉터리의 그룹 쓰기를 지운 뒤(루트는 기본값일 때만 지우고, 명시적 `MLXCEL_CACHE_DIR`이면 건드리지 않음), 체인의 모든 디렉터리를 검증한다. 그룹 쓰기는 PTX 캐시에 이미 있던 `make_private_dir`(디스크립터를 통한 fchmod와 dev/inode 재확인)로 지운다. 복사하지 않고 재사용했으며 PTX 동작은 바뀌지 않았다.

### 상위 디렉터리

IREE는 VMFB를 경로로 열기 때문에 디스크립터 검사만으로는 부족하다. 루트 위의 디렉터리에 쓸 수 있는 사람은 검사와 로드 사이에 검증된 루트를 rename으로 치우고 자기 것으로 바꿀 수 있다. `ensure_private_dir`은 루트의 부모를 한 번 canonicalize하고, 거기서 `/`까지 `verify_trusted_ancestors`를 돌린다(각 구성 요소는 실제 디렉터리이고, 소유자가 사용자나 root이며, `/tmp`처럼 root 소유의 sticky 디렉터리가 아니면 other 쓰기가 없어야 함). 그리고 leaf 경로를 canonical 부모를 통해 다시 써서 돌려주므로, 검사 뒤에 루트 위의 심볼릭 링크가 바뀌어도 이후의 경로 open이 엉뚱한 곳으로 가지 않는다. 루트 위는 아무것도 수정하지 않는다. #2539의 PTX overlay 상위 디렉터리 규칙과 같다.

### 그룹 쓰기 가능한 상위 디렉터리

`umask` `002`와 사용자 전용 그룹(user private group) 환경에서는 `~/.cache`가 `0775`이고 그 그룹에는 소유자만 들어 있다. 엄격한 규칙을 그대로 적용하면 이런 디렉터리가 거부되고, 그런 호스트는 매번 fallback으로 가게 된다. `private_group`은 다음 조건이 모두 증명될 때만 그룹 쓰기 가능한 상위 디렉터리를 받아들이고, 하나라도 아니면 엄격한 규칙과 똑같이 거부한다.

- Linux이고, 디렉터리의 그룹이 프로세스 egid이며, other 쓰기가 없다.
- `/etc/nsswitch.conf`에서 `passwd`, `group`, `initgroups`, `passwd_compat`, `group_compat`을 지정하는 모든 줄이 `files`, `systemd`, `compat`만 나열한다. glibc의 `process_line`과 같은 방식으로 파싱한다(콜론은 선택, 뒤의 줄이 앞의 줄을 덮어씀). `passwd`와 `group`은 반드시 있어야 하고, 파일이 없으면 거부한다.
- `/etc/passwd`와 `/etc/group`에 `+` / `-` compat 줄이나 형식이 깨진 줄이 없다(앞쪽 공백이 있어도 마찬가지).
- `/etc/group`에 그 gid의 다른 멤버가 없고, `/etc/passwd`에 그 gid를 주 그룹으로 쓰는 다른 계정이 없다.

파일은 `getpwent`를 쓰지 않고 일반 텍스트로, 4 MiB 상한을 두고, 프로세스당 한 번 읽는다. 거부된 상위 디렉터리는 `GroupWritable` 오류가 되고, 메시지에 경로와 해결 방법(`chmod g-w <path>` 또는 전용 `MLXCEL_CACHE_DIR`)이 들어간다.

### Fallback

거부된 위치는 읽지도, 옮기지도, 지우지도 않는다. `artifact_slot`은 같은 파일 이름을 프로세스별 `mkdtemp` 루트(임시 디렉터리 아래 `mlxcel-xla-XXXXXX`)의 새 하위 디렉터리에 붙여 돌려준다. 그 전에 임시 디렉터리 자체가 상위 디렉터리 검사를 통과해야 한다. 이 루트는 `Mutex`에 보관하고, 하위 디렉터리를 만들기 전과 후에 다시 검증한다. 루트가 사라졌거나, 남의 소유이거나, 쓰기 가능하거나, 심볼릭 링크이면 버리고 새로 만든다.

## 3. 기술적 선택과 그 이유

**`XDG_CACHE_HOME`이 아니라 기존 캐시 루트를 재사용.** 이슈는 `$XDG_CACHE_HOME/mlxcel`을 제안했다. PR은 tokenizer 스크립트 캐시와 autotune tactic 저장소가 이미 쓰는 `mlxcel_core::cache_root()`를 썼다. 그래서 mlxcel의 모든 캐시가 하나의 루트와 하나의 override(`MLXCEL_CACHE_DIR`) 아래 있다. 새 환경 변수는 추가하지 않았다.

**남의 위치는 고치지 않고 거부한 뒤 다시 컴파일.** 거부된 디렉터리에는 다른 사용자의 파일이 들어 있을 수 있다. 그것을 지우거나 다시 쓰는 것은 공격자가 통제하는 경로에 손을 대는 일이므로, 코드는 그대로 두고 다른 곳에 컴파일한다. 대가는 거부된 로드마다 한 번의 컴파일과 경고 한 줄이고, 로드 자체는 성공한다.

**디스크립터로 검사하고, 경로는 상위에서 지킨다.** 파일은 `O_NOFOLLOW` 디스크립터에서 `fstat`으로 검사하므로 파일 자체의 심볼릭 링크와 바꿔치기 경쟁은 사라진다. IREE가 경로로 다시 열기 때문에 남는 경쟁은 디렉터리 수준이고, 이것은 상위 디렉터리 검사가 막는다. 두 부분은 서로에 의존하며 어느 한쪽만으로는 충분하지 않다.

**그룹 쓰기 정책은 사용자가 골랐고, 조건은 절대 fail-open하지 않는 것.** 첫 리뷰 수정(9ab6dbf0)은 그룹이 프로세스 egid인 그룹 쓰기 가능 상위 디렉터리를 모두 받아들였다. `users`처럼 주 그룹을 공유하는 호스트에서는 그 그룹의 모든 멤버를 신뢰하는 셈이 된다. 선택지는 세 가지였다. egid 지름길을 유지하기, 엄격한 PTX 규칙으로 그룹 쓰기 가능 상위 디렉터리를 모두 거부하기, 로컬 NSS 파일로 단독 멤버 그룹임이 증명될 때만 받아들이기. 사용자는 세 번째를 골랐고, 조건은 의심스러운 경우 받아들이지 말고 거부하라는 것이었다. 그래서 glibc라면 `files`로 기본 처리할 상황에서도 `nsswitch.conf`가 없으면 거부하고, 원격 소스(`sss`, `ldap`, `nis`, `winbind`)가 하나라도 있으면 거부하며, glibc라면 건너뛸 파싱 불가 줄도 거부한다. 엄격한 규칙이 바닥이고, 완화 규칙은 안전이 증명된 호스트만 추가한다.

**NSS 파일을 일반 텍스트로 파싱.** `getpwent` / `getgrent`는 NSS를 거치므로 원격 소스를 열거하거나 네트워크에서 멈출 수 있다. 세 파일을 직접 읽으면 크기가 제한되고, 결과가 결정적이며, 테스트하기 쉽다. 파서가 파일 내용, uid, gid, 이름을 인자로 받으므로 root 없이, 그리고 호스트의 계정 구성과 무관하게 테스트가 돈다. `verify_trusted_ancestors_with`가 그룹 판단을 클로저로 받는 것도 같은 이유다.

**모든 곳에서 원자적 `0600` 게시.** VMFB, HSACO, MLIR 입력, 튜닝 스펙, manifest는 고유한 `O_EXCL` 임시 형제 파일에 쓰고 `0600`으로 봉인한 뒤 rename한다. 크래시 뒤 불완전한 VMFB를 재사용하던 문제가 사라지고, 게시된 파일은 umask와 상관없이 전용 파일 검사를 통과한다.

## 4. 리뷰 결과

**구현 리뷰**(PR 코멘트 7088994696): HIGH 하나. 첫 커밋은 캐시 루트와 그 아래 디렉터리만 검사했는데 IREE는 VMFB를 경로로 연다. 따라서 쓰기 가능하거나 남이 소유한 부모 디렉터리(또는 그런 곳을 가리키는 심볼릭 링크 부모)가 있으면 검사 뒤에 루트를 바꿔치기할 수 있었다. 9ab6dbf0에서 상위 디렉터리 검사와 canonical 부모를 통한 경로 재작성으로 고쳤다. LOW 두 개는 남겼다. `emitter/gemma3n_qmv.rs::run_cuda_stablehlo_probe`는 여전히 임시 디렉터리 아래 예측 가능한 pid별 파일을 쓰지만 `diagnostics` 기능과 CUDA IREE 빌드 뒤에 있어 프로덕션에는 들어가지 않는다. 그리고 fallback 디렉터리를 지우지 않는다(문서화됨). 리뷰어는 소유자, 모드, 심볼릭 링크, 상위 디렉터리 검사를 빼면 `cache_dir` 테스트 11개 중 7개가 실패하는 것도 확인했다.

**그룹 정책**(85d1f434, 951faf46): 9ab6dbf0의 egid 지름길을 3장에서 설명한 단독 멤버 증명 규칙으로 바꿨다.

**보안·성능 리뷰**(PR 코멘트 7089410163)는 로컬 다중 사용자 공격자 관점에서 진행했다. 상위 디렉터리 검사, 재작성된 leaf 경로, 디스크립터 검사, VMFB와 manifest 쌍 검사, GEMV 디스크립터 읽기는 문제가 없었다. 하드 링크는 `nlink` 검사가 필요 없다. 검증된 디렉터리에 항목을 만들 수 있는 것은 사용자뿐이고, 링크된 inode는 소유자와 모드를 그대로 유지하기 때문이다.

- HIGH, 42f96db5에서 수정: `mkdtemp` fallback 루트를 `OnceLock`에 캐시하고 다시 검사하지 않았다. 오래 떠 있는 서버 아래에서 systemd-tmpfiles 같은 임시 파일 정리기가 이 루트를 지우면, 다른 사용자가 `/tmp`에서 보이는 같은 이름을 다시 만들고 컴파일과 로드 사이에 로드별 하위 디렉터리를 바꿔치기할 수 있었다. 이제 하위 디렉터리를 만들기 전과 후에 루트를 다시 검증하고, 실패하면 교체한다.
- MEDIUM, 951faf46에서 수정: `nsswitch_is_local`은 콜론을 요구했고 각 줄을 따로 봤다. 그래서 `passwd sss`는 무시됐고, `passwd: files` 뒤에 `passwd sss`가 오는 설정도 통과했다. 원격 소스에서 보조 멤버를 추가할 수 있는 `initgroups`도 보지 않았다. 이제 파서는 glibc의 `process_line`을 따르고, 검사 대상인 모든 줄이 로컬이어야 한다. 리뷰는 root가 쓴 설정에서만 fail-open한다는 이유로 MEDIUM으로 매겼지만, 사용자의 "절대 fail-open하지 않는다" 조건 때문에 고쳐야 했다.
- LOW, 951faf46에서 수정: 앞에 공백이 있는 `+` / `-` compat 줄을 잡지 못했다.
- MEDIUM, 문서화: fallback 디렉터리를 지우지 않으므로, 거부가 일어난 실행마다 컴파일된 아티팩트가 `/tmp`(Debian 13에서는 tmpfs)에 남는다.
- LOW, 미수정: `write_private_atomic`과 `write_owner_only_new`는 `O_EXCL` open이 실패하면 그 이름을 지운다. 그 이름은 검증된 전용 디렉터리 안의 고유한 형제 파일이므로, 만들 수 있는 사람은 사용자 자신뿐이다.

성능: 상위 디렉터리 검사는 모델 로드 때 캐시 해석마다 한 번, 그룹 판단은 프로세스당 한 번 돈다. 토큰마다 도는 것은 없다.

## 5. 테스트와 검증

새 테스트:

- `mlxcel-xla` `cache_dir::tests`(11개): 기본 위치를 소유자 전용으로 만들고 재사용, 다른 uid(euid 불일치), 그룹/world 쓰기 가능 디렉터리, 심볼릭 링크 캐시 디렉터리, other 쓰기 가능 디렉터리 아래의 루트, 상대 경로 또는 없는 루트, 심볼릭 링크 아티팩트, 전용 디렉터리 안의 그룹 쓰기 가능 아티팩트, 유효한 manifest와 함께 심어 둔 VMFB(world 쓰기 가능 디렉터리와 다른 uid), 컴파일 정확히 한 번으로 소유자 재사용, 원자적 소유자 전용 쓰기, 생성 뒤 교체된 fallback 루트.
- `mlxcel-core` `private_dir` 테스트: 상대 루트와 밖으로 나가는 subpath, 다른 소유자, 쓰기 가능 디렉터리, 관리 체인은 그룹 쓰기를 잃고 명시적 루트는 그대로인 것, 체인 어디서든 심볼릭 링크, 파일의 디스크립터 검사, 새 디렉터리, other 쓰기 가능 상위 디렉터리, 그룹 판단을 양쪽으로 몰아 본 그룹 쓰기 가능 상위 디렉터리, canonical 부모를 통한 leaf.
- `mlxcel-core` `private_group` 테스트: 단독 멤버는 허용(`compat` 포함). 다른 멤버, 다른 주 그룹 계정, 원격 소스, 덮어쓰는 줄, 선택적 콜론, `initgroups`, 대괄호 action, 앞에 공백이 있는 compat 줄, 깨진 줄, 없는 파일은 거부.

최종 head 결과:

- `cargo test --profile test-fast -p mlxcel-xla --lib`: 447개 통과.
- `cargo test --profile test-fast -p mlxcel-core --lib -- private_dir private_group ptx_cache_dirs`: 28개 통과.
- `mlxcel-xla`(기능 없음, `iree`, `iree,micro-oracle,diagnostics`)와 `mlxcel-core`의 lib와 테스트에 clippy `-D warnings`, `cargo fmt --check`, `scripts/ci/check_used_by.py`, `scripts/ci/check_env_var_docs.py`.
- ROCm gfx1151, IREE `hip`: 최종 head에서 `xla_greedy_timing` 3/3, 거부 경고 없음. 이 호스트는 `~/.cache`가 `0775 devops:devops`이고, 단독 멤버 그룹이며, `passwd: files systemd` / `group: files systemd`이므로 완화 규칙이 실제로 쓰이는 환경이다. 9ab6dbf0 뒤에 `xla_decode_scalar_reuse` 4/4, `xla_prepared_prefill` 4/4.
- 실제 모델 greedy 토큰 id(`MLXCEL_BACKEND=xla MLXCEL_PRINT_TOKEN_IDS=1 mlxcel generate --temp 0 -n 24`): c362e290에서 llama-3.2-1b-instruct-4bit, Qwen3-VL-2B-Instruct-4bit, Molmo-7B-D-0924-4bit(이미지, `MLXCEL_XLA_VISION_BACKEND=iree`)가 콜드/웜 캐시 모두 main과 동일. 85d1f434 뒤 캐시를 비우고 llama-3.2-1b를 다시 확인: 동일, 콜드 11초, 웜 4초, 거부 없음.
- `xla_gemma3n_prefill`은 367번째 줄에서 실패한다(`[20, 20, 20]` 대 `[5, 5, 5]`). 수정하지 않은 main(a9860ece)에서도 똑같이 실패하므로 기존 문제이고 이 PR 때문이 아니다.

## 6. 남은 작업과 한계

- **GB10 / CUDA 검증이 남아 있다.** #3248의 마지막 수용 기준(GB10 호스트에서 텍스트와 멀티모달 실제 모델 로드 결과 불변)은 ROCm에서만 확인했다. CUDA IREE 경로와 CUDA GEMV 오브젝트 경로는 새 캐시 코드로 돌려 보지 않았다.
- **fallback 디렉터리는 종료 때 지우지 않는다.** 거부된 위치마다 임시 디렉터리 아래 `mlxcel-xla-XXXXXX` 디렉터리가 남는다. `docs/installation.md`에 문서화돼 있다. 캐시가 매번 거부되는 호스트에서는 임시 파일 정리기가 지울 때까지 쌓인다.
- **업그레이드 뒤 첫 로드는 다시 컴파일한다.** 예전 `/tmp/mlxcel-xla-*` 디렉터리는 더 이상 읽지 않으므로 직접 지워도 된다.
- **그대로 둔 것:** `#[cfg(test)]` pid별 임시 경로, 그리고 `run_cuda_stablehlo_probe`(diagnostics 기능, CUDA IREE 빌드 전용).
- **거부가 일어날 호스트:** 주 그룹을 공유하는 호스트의 그룹 쓰기 가능 상위 디렉터리, 원격 네임 서비스, NIS compat 줄. 경고에 경로와 해결 방법이 나오고, 로드는 fallback으로 성공한다.

## 7. 학습 포인트

- 모든 필드가 공개 입력에서 나오는 내용 manifest는 진위 확인이 아니다. 캐시에 대한 신뢰는 누가 그곳에 쓸 수 있는지에서 나와야 한다.
- 디스크립터 검사는 파일을 지키지만, 다른 구성 요소가 나중에 경로로 다시 여는 것까지 지키지는 못한다. 라이브러리(여기서는 IREE 런타임)가 경로로 연다면 `/`까지의 모든 디렉터리가 신뢰 경계에 들어간다.
- 임시 디렉터리에 대한 일회성 검사는 시간이 지나면 무효가 된다. 임시 파일 정리기는 오래된 디렉터리를 지우고, 공유 `/tmp`에서 보이는 이름은 누구나 다시 만들 수 있다. 오래 떠 있는 프로세스가 보관하는 상태는 쓸 때마다 다시 검증해야 한다.
- 보안 판단을 위해 설정 파서를 다시 구현하면 실제 파서의 특이점(선택적 콜론, 마지막 줄 우선)까지 맞춰야 한다. 재구현이 달라질 수 있는 지점에서 모호한 경우를 전부 거부하면 그 차이가 fail-closed 쪽으로 남는다.
- uid, gid, 파일 내용, 정책 판단을 인자로 넘기면 root 없이도 다른 사용자의 디렉터리와 다른 호스트의 계정 데이터베이스를 테스트에서 재현할 수 있다.
