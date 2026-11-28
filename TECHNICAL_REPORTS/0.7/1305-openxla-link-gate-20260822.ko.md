# 기술 보고서: PR #1305 - 링크 전용 회귀가 main에 들어오지 못하도록 CI에서 OpenXLA 바이너리를 링크

**날짜**: 2026-08-22
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어/기술**: YAML (GitHub Actions)
**위험도**: 낮음 (CI 설정만 변경, 런타임 코드 변경 없음)

---

## 요약

CI는 OpenXLA 기능 조합을 컴파일하기만 했고 링크한 적이 없다. `cargo check`는 링커를 부르지 않으므로, `build.rs`의 IREE 링크 레시피가 깨져도 모든 체크가 초록인 채로 `main`에 들어갈 수 있었다. 가정이 아니라 실제로 일어난 일이다. 이슈 #1274가 정확히 그 실패였고, 통합 테스트 하나조차 링크하지 못하는 트리에서 `cargo check --features cuda,xla-iree --all-targets`는 통과했다.

이 PR은 `.github/workflows/ci.yml`에 `xla-link` job을 추가한다. 셀프호스티드 GB10 러너에서 `--features cuda,xla-iree`로 실제 타깃을 링크하며, 링크 라인을 실제로 움직일 수 있는 경로만 좁게 필터링해 트리거된다. 게이트가 동작한다는 것은 레시피를 일부러 깨서 확인했다. 같은 트리에서 `cargo check`는 59초 만에 통과하는데 링크는 실패한다.

리뷰 과정에서 이 변경 자체의 주석에 있던 사실 오류 세 건이 정정되었다. 그중 하나는 이 보고서가 핵심 학습 포인트로 다루는 것으로, `nm`이 전혀 다른 명령의 alias로 잡혀 있는 쉘에서 `nm`을 실행해 만들어낸 주장이었다.

## 1. 문제 정의

### 1.1 배경

PR #1282가 추가한 `xla-compile` job은 GB10 러너에서 두 명령을 실행한다.

```
cargo check --features cuda,xla-iree --all-targets
cargo check --no-default-features --features xla-diagnostics --all-targets
```

CI의 다른 어떤 것도 XLA 기능을 컴파일하지 않는다. `pipeline-parallel-ci.yml`은 기본 기능으로 clippy를 돌리고, `nightly-verify.yml`은 `metal,accelerate`로 `make verify`를 돌리며, `release.yml`은 `xla-iree`를 빌드하지 않는다. 그 job의 주석 자체가 링크 공백을 "의도적으로 열어둔 것"으로 기록하며 이 작업을 예고하고 있었다.

### 1.2 기존 문제

`cargo check`는 타입 검사에서 멈추고 링커를 실행하지 않는다. 따라서 `build.rs`가 IREE 런타임을 위해 내보내는 `cargo:rustc-link-arg` 레시피 전체가 CI에서 한 번도 검증되지 않는다. #1274가 대표 사례다.

```
/usr/bin/ld: libiree_runtime_unified.a(call.c.o): undefined reference to symbol '__stack_chk_guard@@GLIBC_2.17'
/usr/bin/ld: /lib/ld-linux-aarch64.so.1: error adding symbols: DSO missing from command line
```

이것이 `main`에 들어갔고, 무관한 PR을 검증하다가 사람 손으로 발견되었다. 수정 PR #1275는 IREE 아카이브 뒤에 `-lc`를 한 번 더 붙인다. rustc가 자기 `-lc`를 `cargo:rustc-link-arg`가 덧붙일 수 있는 어떤 것보다 먼저 내보내기 때문이다.

### 1.3 위험 평가

노출 범위는 제한적이지만 실재한다. OpenXLA 경로는 기본 기능이 아니므로 링크가 깨져도 배포되는 기본 빌드는 멀쩡하다. 대신 그 경로를 작업하는 모든 개발자가 막히고, 발견 시점이 최악이다. 사후에, 사람 손으로.

## 2. 변경 요약

파일 하나, `.github/workflows/ci.yml`이다.

| 변경 | 내용 |
|---|---|
| `changes` 출력 추가 | `xla_link`, `dorny/paths-filter`의 형제 필터 |
| 필터 경로 | `build.rs`, `src/lib/mlxcel-xla/build.rs`, `src/lib/mlxcel-xla/csrc/**`, `scripts/iree/**`, `rust-toolchain.toml`, `.github/workflows/ci.yml` |
| 새 job | `xla-link`, `xla-compile` 뒤, `runs-on: GB10`, `timeout-minutes: 120`, `permissions: contents: read` |
| 링크 명령 | `cargo test --release --features cuda,xla-iree --test xla_prepared_prefill --no-run` |
| 타깃 디렉터리 | `$HOME/.cargo-target/mlxcel-xla-link-ci`, `xla-compile`/릴리스 job과 분리 |
| 동시성 | PR 단위 job 스코프 그룹, `cancel-in-progress: true` |
| 주석 정정 | `clippy`의 fork 가드 설명, `xla-compile`의 미커버 목록이 새 job을 가리키도록 |

## 3. 기술적 선택과 그 이유

### 3.1 릴리스 프로파일은 선택이 아니라 강제

이 호스트에서 디버그 프로파일은 이 타깃들을 아예 링크하지 못한다. 최적화되지 않은 바이너리가 AArch64 직접 분기 범위를 넘어서, 평범한 `libstd`와 `compiler_builtins` 심볼에 대해 `relocation truncated to fit: R_AARCH64_CALL26` 오류가 수백 개 난다. 값싼 디버그 링크는 애초에 선택지가 아니었고, 그래서 이 job은 초 단위가 아니라 분 단위이며 `xla-compile`에 스텝 하나를 덧붙이는 대신 별도 job이 되었다.

### 3.2 배포 바이너리가 아니라 가장 작은 타깃을 링크

`build.rs`는 레시피를 `cargo:rustc-link-arg`로 내보내고, 이는 해당 크레이트가 링크하는 모든 산출물에 적용된다. 따라서 어느 타깃을 링크하든 검증되는 레시피는 동일하며, #1274가 통합 테스트 링크 실패였으므로 통합 테스트가 직접적인 재현자다.

리뷰가 여기 근거를 정정했다. cargo는 통합 테스트가 선택되면 해당 패키지의 `[[bin]]` 타깃도 함께 빌드한다. 그래서 이 명령은 테스트 바이너리 외에 `mlxcel`, `mlxcel-server`, `speculative_bench`, `mlxcel-bench-decode`까지 링크한다. 기각한 `cargo build --release --bin mlxcel-server` 대안은 더 비싼 선택지가 아니라 선택된 명령의 진부분집합이고, 아래 측정치에는 이미 바이너리들이 포함되어 있다. 타깃 선택 자체는 옳았지만, 처음 적어둔 이유는 틀렸다.

### 3.3 매 Rust PR도 스케줄도 아닌 경로 필터

스케줄은 실패를 그 실패를 만든 PR과 분리시킨다. #1274가 빠져나간 방식이 정확히 그것이다. 매 Rust PR 실행은 링크 라인에 영향을 줄 수 없는 대다수 PR에 공유 러너의 몇 분을 쓰는 일이다.

필터는 인과 표면을 덮는다. 루트 `build.rs`가 레시피를 갖고 있고, `scripts/iree/**`가 레시피에 이름이 적힌 아카이브 집합을 가진 배포판을 고정하며, `mlxcel-xla`의 빌드 스크립트와 `csrc/**` 소스가 그 아카이브들이 해결해주는 미정의 심볼을 가진 shim 오브젝트를 만든다. `rust-toolchain.toml`이 같은 경로에 있는 이유는, #1274가 전적으로 rustc가 자기 `-lc`를 덧붙인 인자들 대비 어디에 두느냐의 문제였고 툴체인 상향이 그것을 옮길 수 있기 때문이다.

뒤의 `mlxcel-xla` 경로 둘은 리뷰 중 추가되었다. 최초 필터는 `build.rs`와 `scripts/iree/**`만 지정했는데, picomatch 패턴으로서 `build.rs`는 루트 빌드 스크립트만 매치한다. C shim 변경은 `cargo check`를 통과하면서 이 job도 시작시키지 못했을 것이다.

### 3.4 `RUSTFLAGS`를 의도적으로 설정하지 않음

`xla-compile`이 `RUSTFLAGS: "-D unused_imports"`를 설정하며 린트 정책을 소유한다. 여기서 설정하지 않으면 빨간 실행은 명백히 링크 실패이지, 링크 job 이름을 쓴 린트 실패가 아니다. 이슈 #1304가 별도로 dead-code 백로그를 정리하고 `xla-compile`을 `-D warnings`로 확대한다.

### 3.5 GB10 위 첫 느린 job이므로 동시성 그룹

GB10의 다른 job은 웜 상태에서 전부 초 단위다. 이것만 분 단위이고 `timeout-minutes: 120`이다. `ci.yml`에는 동시성 제어가 전혀 없어서, `build.rs`를 건드리는 PR에 세 번 푸시하면 러너 하나에 최대 120분짜리 링크 job 세 개가 쌓이고, 그 러너는 모든 Rust PR의 clippy와 릴리스 빌드도 처리한다. 그룹은 PR 단위로 키를 잡아 PR끼리는 서로를 취소하지 않으며, `pipeline-parallel-ci.yml`이 이미 쓰는 패턴을 따른다.

## 4. 리뷰에서 나온 지적

### 4.1 HIGH: 쉘 alias가 만들어낸 사실이 주석에 기록됨

초기 리비전은 실패한 `-lc` 대조 실험을, 고정된 IREE 런타임이 스택 프로텍터 없이 빌드되어 `libiree_runtime_unified.a`에 `__stack_chk_guard` 참조가 전혀 없다("`nm`이 0을 보고한다")고 설명했다.

실제로는 176개가 있고, #1274에 이름이 나오는 `call.c.o`도 포함된다. 0이 나온 이유는 이 쉘이다.

```
nm is an alias for mosh --ssh="ssh -i ~/.ssh/nubimaru.pem" ubuntu@<host>
```

`nm <archive> | grep -c __stack_chk_guard`는 아카이브를 들여다본 적 없는 프로그램의 출력을 센 것이고, `2>/dev/null`이 그 사실을 드러냈을 `command not found`를 가렸다. `/usr/bin/nm`으로 보면 `build.rs` 주석이 기록한 전제가 오늘도 전부 성립한다. 아카이브의 176개 오브젝트에서 미정의, `libc.so.6`에서도 미정의, 정의는 `ld-linux-aarch64.so.1`에만 있다.

### 4.2 HIGH: 경로 필터가 인과 표면의 일부를 빠뜨림

3.3 참조. `src/lib/mlxcel-xla/build.rs`와 `src/lib/mlxcel-xla/csrc/**` 추가로 수정.

### 4.3 MEDIUM: "무엇을 링크하는가" 근거가 바이너리에 대해 틀림

3.2 참조. 주석이 바이너리들도 여기서 링크된다는 사실을 기록하도록 수정.

### 4.4 보안: fork 가드 주석이 사실과 다르고 전파되고 있었음

`clippy` job 위 주석은 `if: github.repository == 'lablup/mlxcel'`이 fork PR을 셀프호스티드 러너에 올리지 않는다는 뜻이라고 단언했다. fork에서 이 저장소로 연 `pull_request` 이벤트에서는 `github.repository`가 base 저장소이므로 가드가 참이고 job이 실행되며, PR의 `build.rs`와 `scripts/iree/**`가 러너 사용자 권한으로 실행된다. 가드의 실제 효과는 GB10 러너가 없는 *이 저장소의 fork*에서 job이 영원히 대기하는 것을 막는 것이다. fork 케이스를 실제로 통제하는 것은 저장소의 Actions fork-PR 승인 정책이며 현재 `first_time_contributors`다.

이슈 #1303이 이 job을 명세하면서 그 잘못된 해석을 그대로 옮겨 적었으므로 오해가 번지고 있었다. 주석은 이제 가드가 실제로 하는 일을 기록한다. 이 PR이 새로 만드는 노출은 없다. `clippy`와 `xla-compile`이 더 넓은 필터로 게이트되며 이미 fork PR 코드를 그 러너에서 실행한다.

## 5. 검증

모두 IREE 런타임이 준비된 GB10 호스트에서 실행했다.

| 실행 | 결과 |
|---|---|
| 콜드 링크, `CARGO_TARGET_DIR` 비움 | exit 0, 15분 40초, `Executable tests/xla_prepared_prefill.rs`로 종료 |
| 웜 링크 (2회) | exit 0, 6분 01초 / 7분 24초 |
| 링크가 깨진 트리에서 `cargo check` | **exit 0, 59초** |
| 같은 트리에서 링크 | **exit 101**, `error: linking with cc failed`, `flatcc_verify_*` 미정의 참조 |
| 아카이브 복원 후 링크 | exit 0 |
| 이 PR 자체에서의 job | 커밋 4개 모두 성공 |

깨뜨린 방법은 `build.rs`의 `IREE_CUDA_HOME` 분기에서 `-l:libflatcc_parsing.a`를 제거한 것이다. 가운데 두 줄이 이 변경의 전부다. 같은 트리에서 한 명령은 초록, 다른 하나는 빨강이고, 그것이 #1274의 모양이다. 이 PR은 `build.rs`를 건드리지 않으며 모든 대조 실험은 브랜치 밖에서 실행하고 되돌렸다.

콜드 수치의 대부분은 링크가 아니라 MLX의 CUDA 소스를 처음부터 컴파일하는 시간이다. 의미 있는 값은 웜 쪽이다. 트리거가 `build.rs`에서 걸리는데, 그것은 `mlxcel` 크레이트를 무효화하지만 `mlxcel-core`의 MLX 빌드는 건드리지 않기 때문이다.

## 6. 검증되지 않은 채로 남은 것

- **`-lc` 대조 실험이 재현되지 않는 이유.** 이슈의 인수 조건대로 먼저 시도했고, 링크는 성공했다. 내보낸 빌드 스크립트 출력에 `rustc-link-arg=-lc`가 없음을, 그리고 IREE 아카이브들이 링크 라인에 있음을 각각 확인했다. 스택 프로텍터 설명은 틀렸다(4.1). 추가 가설 둘도 검증 후 배제했다. 이 glibc에서 `-lpthread`와 `-ldl`은 libc를 뒤늦게 끌어오는 링커 스크립트가 아니라 스텁 아카이브로 해석되고, `libm.so`는 `libm.so.6`만 그룹한다. 해당 항목은 남기며, 재현에 실패한 대조 실험 하나를 근거로 제거하지 말라고 주석에 적었다.
- **`IREE_DIST` 레시피와 macOS `IREE_MACOS_HOME` 레시피**는 어느 기계에서도 링크로 검증된 적이 없다. 두 배포판을 가진 러너가 없다.
- **초록 실행이 링크가 일어났다는 증거는 아니다.** 타깃 디렉터리가 PR 간에 유지되므로 cargo가 할 일을 못 찾을 수 있다. 이 job의 첫 실행은 `Finished release profile in 0.14s`와 함께 12초에 끝났다. 이는 정상 동작이다. cargo는 fingerprint가 움직일 때 정확히 재링크하고, `build.rs`는 IREE 환경변수 세 개 모두에 `rerun-if-env-changed`를 선언한다. 좁은 구멍은 `IREE_VERSION`을 바꾸지 않으면서 런타임 빌드 방식만 바꾸는 `scripts/iree/**` 수정이다. 스크립트가 멱등이라 아무것도 다시 빌드되지 않는다.
- **`Cargo.toml`과 `Cargo.lock`은 필터 밖**이라 의존성 변경으로 인한 링크 변화는 job을 트리거하지 않는다. 넓히면 모든 의존성 상향에서 실행된다.

## 7. 학습 포인트

1. **진단 도구가 정말 그 도구인지 확인할 것.** 쉘 alias가 `nm`을 SSH 클라이언트로 바꿔놓았고, stderr를 버린 탓에 `command not found`가 자신 있는 빈 결과로 둔갑했다. 0건을 반환한 검색은 결론을 세우기 전에 양성 대조 하나를 거칠 자격이 있다.
2. **`cargo check`는 링크 게이트가 아니며 `--all-targets`도 그걸 바꾸지 않는다.** 이 플래그는 타입 검사 범위를 넓힐 뿐 링크 범위를 넓히지 않는다.
3. **`cargo:rustc-link-arg`는 링크되는 모든 산출물에 적용된다.** 따라서 어떤 타깃이든 같은 레시피를 검증한다. 가장 작은 것을 고르는 일은 커버리지 결정이 아니라 비용 결정이다.
4. **통합 테스트를 선택하면 패키지의 바이너리들도 함께 빌드된다.** `cargo test --test X --no-run`은 보이는 것만큼 최소한의 링크가 아니며, 그래서 예상보다 나은 게이트이고 `--bin` 대안은 절약이 아니라 부분집합이다.
5. **경로 필터는 인과에 대한 주장이다.** 필터를 쓰는 일은 검증 대상 산출물을 실제로 무엇이 움직이는지 묻게 만들며, 여기서 초안은 아카이브가 해결해주는 바로 그 심볼을 만들어내는 C shim을 놓쳤다.

## 8. 후속 작업

- **fork PR은 실제로 GB10 러너에 도달한다**(4.4). 이 PR이 만든 것이 아니라 기존 job들에서 물려받은 상태다. GB10 job 세 개 전부에 head 저장소 검사를 넣거나, 승인 정책을 조이거나, 전용 러너 계정을 두는 하드닝 이슈가 필요하다.
- **디스크 압박.** 93퍼센트 찬 볼륨에 네 번째 영구 타깃 디렉터리가 추가된다. `~/.cargo-target`은 이미 39GB이고 정리하는 주체가 없으며, 같은 호스트에서 릴리스를 만든다.
- **`-lc` 근본 원인**은 미해결이다(6절). 이를 해결하는 사람은 `build.rs` 주석의 "Measured, not assumed" 문구도 함께 다시 볼 것.
- **이슈 #1304**가 dead-code 백로그를 정리하고 `xla-compile`을 `-D warnings`로 확대한다.

## 참고

- 이슈 #1303(이 작업), #1274(링크 실패), #1275(`-lc` 수정), #1282(이 게이트가 확장하는 컴파일 게이트), #1304(같은 제외 목록의 린트 절반)
- `.github/workflows/ci.yml`, `build.rs:143-176`, `src/lib/mlxcel-xla/build.rs`
