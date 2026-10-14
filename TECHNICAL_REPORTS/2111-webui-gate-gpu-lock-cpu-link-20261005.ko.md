# 기술 보고서: Issue #2111 - WebUI Activity gate의 host GPU lock 대기와 CPU 전용 link job

**날짜**: 2026-10-05

**상태**: 구현 완료. 실제 `gpu-lock` script를 별도 lock directory로 돌려 control flow를 로컬에서 검증했습니다. origin/main `6ddbb579` 기준, PR 자체의 GB10 CI 실행과 머지 대기 중.

**언어**: GitHub Actions YAML과 bash (`.github/workflows/ci.yml`), Markdown (`docs/webui-integration-matrix.md`, `docs/installation.md`)

**위험도**: 낮음 (CI 전용 변경입니다. gate는 여전히 fail closed이고, `gpu-lock`이 없는 runner에서는 바뀌지 않은 본문이 그대로 실행됩니다)

## 요약

GB10 runner의 `WebUI installed artifact` job은 환경 요인 두 가지로 실패했습니다. Activity gate가 같은 host의 개발 세션이 쓰는 `gpu-lock`을 무시했기 때문에, 세션의 GPU 작업이 gate의 fail-closed `nvidia-smi` 검사에 걸렸습니다. 또 persistent `test-fast` target directory에 남은 incremental state가 `serde_json` generic을 찾지 못하는 link 오류를 냈고, 수동으로 지운 뒤에야 통과했습니다. 이제 gate는 cooperative CI flock과 `nvidia-smi` 검사 사이에서 `gpu-lock`을 최대 600초 기다리고, job은 `CARGO_INCREMENTAL=0`으로 build합니다. #2108에서 넘어온 범위로, 새 `CPU-only link` job이 GPU feature 없이 `mlxcel-core` test binary를 link합니다. #2108의 guard 누락이 드러난 바로 그 구성입니다.

## 1. 문제

- Lock 충돌: run 36982300183 attempt 1 (PR #2093)은 개발 세션이 GPU를 쓰는 동안 `nvidia-smi` 검사에서 실패했고, `gpu-lock`을 잡은 상태로 다시 돌린 run은 통과했습니다. runner는 uid 1000, `PrivateTmp=no`로 돌기 때문에 `/tmp/gpu-lock-1000/lock`을 볼 수 있습니다.
- 오래된 incremental state: run 36988516810 (PR #2094)은 attempt 1-5에서 link에 실패했고 (`$HOME/.cargo-target/mlxcel-webui-installed-ci`의 codegen unit 하나에서 `serde_json` generic undefined), incremental state를 지운 attempt 6에서만 통과했습니다.
- CPU 전용 build를 link하는 CI job이 없었습니다. GB10 job은 모두 `--features cuda`를 쓰고, `cargo check`는 link하지 않습니다 (#2108).

## 2. 변경 사항

| 영역 | 변경 |
|---|---|
| `webui-installed-artifact` job env | 이 job에만 `CARGO_INCREMENTAL: "0"` |
| `Run Activity performance gate...` step | 본문 (nvidia-smi 검사와 harness, 내용은 그대로)을 `$RUNNER_TEMP/webui-activity-gate.sh`로 씁니다. 기존 flock 안에서 `gpu-lock`이 `PATH`에 있으면 `gpu-lock run --tag webui-activity --wait 600 -- bash <script>`로, 없으면 직접 실행합니다. 600초 대기 두 번을 감안해 step timeout을 45분에서 65분으로 늘렸습니다 |
| `changes` job | 새 `cpu_link` filter: `**/*.rs`, `**/Cargo.toml`, `Cargo.lock`, `build.rs`, `rust-toolchain.toml`, `src/lib/mlx-cpp/**`, `src/lib/mlxcel-core/cpp/**`, `src/lib/mlxcel-core/build_support/**` |
| 새 `cpu-link` job (`CPU-only link`) | GB10, repository guard, `cpu_link == 'true'` 또는 `ci:full`; target dir `$HOME/.cargo-target/mlxcel-cpu-link-ci`; `cargo test -p mlxcel-core --profile test-fast --lib --no-run` 실행 |
| `concurrency` 주석 | GB10 job 목록을 넷에서 일곱으로 갱신 |
| 문서 | `docs/webui-integration-matrix.md`에 runner 메모, `docs/installation.md`는 `CPU-only link` job을 가리킴 |

## 3. 설계 결정

- Lock 순서는 CI flock, `gpu-lock`, `nvidia-smi` 순입니다. cooperative fd 9는 `gpu-lock`의 `exec`을 거쳐 상속되므로 본문 전체 동안 두 lock이 모두 유지됩니다. quiet-host 대기 (#1949)는 본문 script 맨 앞에 끼워 넣으면 됩니다.
- Lock timeout과 본문 실패는 exit code 75가 아니라 본문이 가장 먼저 쓰는 marker file로 구분합니다. 본문의 exit status는 그대로 전달되고, marker 없이 0이 아닌 값으로 끝나면 `::error::`와 `gpu-lock status`를 출력하고 step이 실패합니다.
- 대기 전에 holder와 시각을, 본문 시작 시 대기한 초를 출력하므로 job log에서 대기를 확인할 수 있습니다.
- link 실패 시 clean-and-retry 대신 `CARGO_INCREMENTAL=0`을 씁니다. issue에서 원인을 숨기고 전체 rebuild 비용을 낸다는 이유로 기각된 방식입니다. 다른 GB10 target directory는 profile 기본값을 유지합니다.
- #2108이 고친 코드 (`src/lib/mlx-cpp/turbo/kv_inplace_write.cpp`)는 `rust` filter에 걸리지 않으므로 `cpu-link`는 별도 filter를 씁니다. warm 실행이 몇 초라 overlay 제외 규칙은 두지 않았습니다. 6초라는 warm 수치가 incremental로 측정된 값이므로 이 job은 incremental을 유지합니다.

## 4. 검증

- `actionlint` 1.7.7 (`workflow-lint`가 고정한 버전), `-shellcheck='shellcheck --severity=error'`: 문제 없음. 기본 severity에서도 새 job과 step은 finding을 추가하지 않습니다.
- YAML parse 성공, `runs-on: GB10` job 7개.
- YAML에서 추출한 step script를 `--wait` 3초, stub `nvidia-smi`/`python3`로 일곱 경우에 실행했습니다: `gpu-lock` 없음 (본문 실행, exit 0, oom 조정 없음); timeout하는 stub `gpu-lock` (holder와 함께 exit 1); 별도 lock directory를 가리키게 한 실제 `gpu-lock` script로 lock이 비어 있을 때 (exit 0, `oom_score_adj=1000`), 2초 점유 (약 1초 대기 후 통과), 대기 시간보다 길게 점유 (holder와 함께 exit 1); lock 아래에서 본문 자체가 75로 끝날 때 (lock timeout으로 보고하지 않고 exit 75 전달); `gpu-lock` 없이 본문 실패 (exit 3 전달).
- 로컬에서 검증하지 못한 것: GB10 실행 자체와 `CARGO_INCREMENTAL=0`의 warm build 시간. 변경 후 첫 실행은 workspace crate를 다시 build하므로 warm 수치는 두 번째 실행에서 얻습니다.

## 5. 조율

#1949는 같은 precondition block에 quiet-host 대기를 제안하고, #1925도 관련이 있습니다. 여기서는 둘 다 구현하지 않았고, 위 순서 덕분에 #1949는 한 곳에만 끼워 넣으면 됩니다.
