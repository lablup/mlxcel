# 기술 보고서: PR #2161 - Host 전역 lock으로 rocm_gpu_guard.sh 실행 직렬화

**날짜**: 2026-10-06

**상태**: gfx1151 호스트에서 구현 및 검증 완료. Head `06e0abff` (origin/main `1e561f1e`과 최신 상태 일치), PR open, 머지 대기 중.

**언어**: Bash (`scripts/rocm_gpu_guard.sh`), Python (`tests/test_rocm_decode_profile.py`), Markdown (`docs/benchmarks.md`)

**위험도**: 낮음 (benchmark 도구만 바뀝니다. Rust, HIP, Metal, CUDA 코드 변경 없음. 호출자에게 보이는 동작 변경은 하나: `--max-wait`가 이제 guard 실행 전체에 대한 단일 예산입니다)

## 요약

Issue #2146 (epic #1801의 일부)은 동시에 시작한 두 `scripts/rocm_gpu_guard.sh` 실행이 서로를 거부하다가 둘 다 75로 끝난다고 보고했습니다. #2061에서 추가된 guard는 `--idle-secs` 동안 GPU가 idle이기를 기다린 뒤 COMMAND를 실행하고, 1 Hz monitor가 COMMAND나 그 자손이 아닌 GPU process를 보면 그 시도를 거부합니다. Idle window가 같은 두 guard는 같은 초에 대기를 끝내고 함께 command를 시작하며, 각 monitor는 상대의 command를 외부 GPU 사용자로 봅니다. #1814 ROCm port 작업 중 병렬 unit들은 이를 피하려고 `--idle-secs`를 손으로 서로 다르게 골라야 했습니다 (45, 55, 60, 75, 90, 120초).

이 PR은 `flock`으로 guard를 host 전역에서 직렬화합니다. Guard는 첫 idle 대기 전에 lock을 잡고 종료할 때까지 유지하므로 guard들이 차례로 실행됩니다. 단순한 lock이 만들 수 있는 예외 상황도 함께 막습니다. Lock 파일은 없을 때만 만들고 read-only로 열며, regular file이 아닌 lock 경로는 거부하고, `--max-wait`는 lock 대기와 idle 대기를 합친 하나의 예산이 되며, 대기 중에도 INT와 TERM이 동작하고, COMMAND는 lock descriptor 없이 실행되며, 중첩된 guard는 `ROCM_GPU_GUARD_LOCK_HELD`가 자신의 조상을 가리킬 때만 lock을 건너뜁니다.

새 test는 8개입니다. 그중 7개는 main의 script에서 실패합니다. 나머지 하나 (COMMAND가 남긴 daemon이 lock을 쥐고 있으면 안 됨)는 lock이 없는 main에서는 통과하고, 이 branch에서 descriptor close를 제거하면 실패합니다. 실제 KFD 목록 위에서 같은 초에 시작한 두 guard가 모두 `CLEAN`으로 끝났고 `CONTENDED`나 `REJECTED` 줄은 없었습니다. 두 번째 guard는 779초 뒤에 lock을 얻었습니다.

## 1. 문제 정의

### 1.1 두 guard가 서로를 거부한 이유

`wait_idle`은 `/sys/class/kfd/kfd/proc`과 process 목록에서 연속된 idle sample 수를 셉니다. 두 guard 사이에는 아무 조율이 없었습니다. Guard 자체는 bash process라서 KFD 목록에 나타나지 않으므로, 한 guard가 대기하는 동안 다른 guard도 GPU를 idle로 봅니다. `--idle-secs`가 같으면 두 streak이 같은 초에 끝나고, 둘 다 COMMAND를 시작하며, 두 monitor 모두 자기 COMMAND tree가 아닌 KFD 항목을 보게 됩니다. 두 시도 모두 `REJECTED (contended)`가 되고, 같은 window로 같은 idle 대기에 돌아가며, `--max-attempts` (기본 5)가 소진될 때까지 같은 박자로 반복한 뒤 둘 다 75로 종료합니다. 어느 COMMAND의 출력도 쓸 수 없습니다.

### 1.2 강요된 우회책

2026-10 ROCm port 작업 (#1814)은 여러 benchmark unit을 병렬로 돌렸습니다. 각 unit은 streak이 다른 시각에 끝나도록 자기만의 `--idle-secs` (45, 55, 60, 75, 90, 120초)를 골라야 했습니다. 이 방법은 충돌 확률을 낮출 뿐이고, 모든 호출자를 host 전체의 window 배정 규칙에 묶으며, 긴 window를 고른 실행은 그만큼 길어집니다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| `scripts/rocm_gpu_guard.sh` | `ROCM_GPU_GUARD_LOCK` (기본 `/tmp/mlxcel-rocm-gpu-guard.lock`)에 `flock`을 거는 `acquire_lock`; bash `SECONDS`에서 COMMAND 실행 시간을 뺀 단일 `--max-wait` 예산; `ROCM_GPU_GUARD_LOCK_HELD` 조상 검사; COMMAND와 monitor에서 lock descriptor 닫기; `flock`이 없으면 exit 2; header와 `usage()` 범위 갱신 |
| `tests/test_rocm_decode_profile.py` | `run_guard`가 `ROCM_GPU_GUARD_LOCK`을 test별 임시 경로로 지정하고 상속된 `ROCM_GPU_GUARD_LOCK_HELD`를 제거; `hold_lock` helper; 신규 `GuardTests` 8개 |
| `docs/benchmarks.md` | Lock, 예산, 중첩, `sudo`, 신뢰할 수 없는 사용자와 공유하는 host에서 lock 경로를 옮기는 방법을 설명하는 문단 하나 |

Commit은 세 개입니다. Lock 도입 (`fc0bc3c2`), 조상 guard가 보낸 `ROCM_GPU_GUARD_LOCK_HELD`만 신뢰 (`1f6ff1ad`), regular file이 아닌 lock 경로 거부 (`06e0abff`). origin/main 대비 diff는 3개 파일, 274줄 추가, 15줄 삭제입니다.

## 3. 설계

### 3.1 Backoff가 아니라 lock인 이유

Issue는 무작위 backoff를 기각했습니다. 충돌 확률은 줄지만 충돌할 때마다 시도 하나를 통째로 잃고, 동시에 시작한 실행들은 계속 상관관계를 가집니다. Lock은 경쟁 자체를 없앱니다. "idle 대기"와 "COMMAND 완료" 사이에는 항상 guard 하나만 있습니다. Lock은 첫 idle 대기 전에 잡아 모든 시도 동안 유지하므로, guard가 아닌 GPU 사용자 때문에 시도가 거부된 guard도 다음 guard와 다시 경쟁하지 않고 자기 차례를 유지합니다.

### 3.2 Lock 파일 생성과 열기

```bash
[[ -e "$LOCK" ]] || { (umask 000; : >>"$LOCK") 2>/dev/null || true; }
[[ -f "$LOCK" ]] || { echo "... is not a regular file" >&2; exit 2; }
{ exec {LOCK_FD}<"$LOCK"; } 2>/dev/null || { echo "... cannot open guard lock" >&2; exit 2; }
```

- **없을 때만, 모두가 쓸 수 있게 생성.** 모든 사용자의 guard가 같은 기본 파일을 공유할 수 있습니다. 다른 guard가 먼저 만들어 생성이 실패해도, 이어지는 open이 성공하면 문제없습니다.
- **Read-only로 열기.** `flock`은 read-only descriptor에서도 동작합니다. Sticky `/tmp`에서 다른 사용자가 소유한 파일을 `O_CREAT`로 열면 `fs.protected_regular` 때문에 실패하므로, 파일이 있으면 guard는 `O_CREAT`로 열지 않습니다.
- **Regular file이 아닌 경로 거부.** FIFO를 읽기로 열면 writer가 나타날 때까지 막히므로, 예산 코드가 돌기도 전에 `--max-wait`를 넘겨 멈춥니다. Guard는 대신 exit 2로 끝납니다.

### 3.3 하나의 `--max-wait` 예산

인자 파싱 뒤에 `SECONDS=0`을 두고, `run_secs`에 COMMAND 실행 시간을 누적합니다. 사용한 예산은 `SECONDS - run_secs`이므로, lock 대기와 모든 idle 대기는 포함되고 측정 자체는 빠집니다.

- Lock 대기는 `--max-wait > 0`이면 `flock -w <남은 시간>`, 아니면 blocking `flock`입니다. 시간이 다 되면 `gave up waiting for guard lock <path> after <N>s`를 남기고 COMMAND 없이 75로 종료합니다.
- Lock을 얻은 뒤 남은 예산이 `--idle-secs`보다 짧으면, 끝낼 수 없는 idle 대기 동안 lock을 쥐고 뒤에 줄 선 guard를 막는 대신 75로 종료합니다.
- `wait_idle`도 같은 예산을 보며, streak이 아직 완성되지 않았을 때만 검사하므로 마감 시각에 정확히 완성된 streak은 실행됩니다.

### 3.4 Lock 대기 중 signal

Bash는 foreground 자식이 끝날 때까지 trap 실행을 미룹니다. 그래서 foreground `flock`을 쓰면 lock 대기 내내 INT와 TERM에 반응하지 않습니다. `acquire_lock`은 상속된 descriptor로 `flock`을 background에서 실행하고 `wait`을 호출하는데, trap이 걸린 signal은 `wait`을 깨웁니다. Lock은 open file description에 붙으므로 background `flock`이 잡은 lock은 guard shell의 lock이기도 합니다. `on_signal`은 COMMAND와 monitor와 함께 `lock_pid`도 종료하므로 guard보다 오래 사는 `flock`이 없고, guard는 130 또는 143으로 끝납니다.

### 3.5 COMMAND와 monitor에서 descriptor 닫기

COMMAND는 `"$@" {LOCK_FD}<&- &`로, monitor subshell도 같은 redirection으로 시작합니다. 그렇지 않으면 COMMAND가 남긴 daemon이 descriptor를 상속해 guard가 끝난 뒤에도 lock을 쥐고, 이후의 모든 guard를 무기한 막습니다. Monitor의 마지막 `sleep 1`도 마찬가지입니다.

### 3.6 중첩

Guard는 COMMAND에 `ROCM_GPU_GUARD_LOCK_HELD=$$`를 export합니다. 이 변수를 가지고 시작한 guard는 값이 양의 정수이고, 자기 pid가 아니며, `/proc/<pid>/stat`의 부모를 따라 올라가는 탐색 (pid 1 전에서 멈춤)에서 조상으로 확인될 때만 lock을 건너뛰고 `lock held by outer guard <pid>`를 남깁니다. 조상 검사가 없으면 (첫 commit은 어떤 값이든 믿었음) 재사용된 pid, daemon이 들고 다니는 export, pid 1이 무관한 guard를 lock 없이 실행시킬 수 있습니다. 조상 검사로 중첩은 deadlock이 나지 않고, 오래된 값은 lock을 우회하지 못합니다. 바깥 guard가 lock을 쥔 경우 안쪽 guard는 `/dev/null`을 placeholder descriptor로 열어, COMMAND와 monitor의 닫기 redirection을 조건 없이 유지합니다.

두 가지 한계는 header와 `docs/benchmarks.md`에 적혀 있습니다. 같은 바깥 guard 아래에 중첩된 guard끼리는 직렬화되지 않고, 환경이 초기화되면 (예: `--preserve-env=ROCM_GPU_GUARD_LOCK_HELD` 없는 `sudo`) 중첩된 guard가 자기 조상의 lock을 기다리게 됩니다.

### 3.7 `flock`이 없을 때

바깥 guard가 lock을 쥐고 있지 않은데 `flock`이 없으면 `rocm_gpu_guard: flock not found (util-linux)`와 함께 exit 2로 끝납니다. Lock 없이 실행하면 경고 없이 같은 박자 문제가 돌아옵니다.

## 4. 동작 변경: `--max-wait`

이 PR 전에는 `wait_idle`이 자체 counter를 가졌기 때문에 `--max-wait`는 시도마다 idle 대기를 따로 제한했고, 시도가 5번이면 최대 `--max-wait`의 5배까지 기다릴 수 있었습니다. 이제 `--max-wait`는 guard 시작부터 COMMAND 실행 시간을 뺀, lock 대기와 모든 idle 대기를 합친 전체 대기를 제한합니다. 포기 메시지는 `gave up after <MAX_WAIT>s of waiting without <IDLE_SECS>s of idle`로 바뀌었습니다. 저장소 안의 어떤 script도 `--max-wait`를 넘기지 않으므로 내부 호출자의 동작은 바뀌지 않습니다. 시도별 의미에 기대던 외부 호출자는 전체 대기가 더 짧아집니다.

## 5. 검증

### 5.1 신규 test와 main에서 실패하는 test

신규 test는 가짜 KFD 디렉터리와 test별 lock 경로를 쓰므로 host lock을 건드리지 않습니다. 이 보고서를 위해 branch의 test 파일을 main의 `rocm_gpu_guard.sh`로 (scratch 복사본에서) 실행한 결과입니다.

| Test | main에서 |
|---|---|
| `test_two_guards_started_together_run_one_after_the_other` | 실패: 두 guard 모두 75로 종료 (issue의 버그) |
| `test_a_held_lock_counts_against_max_wait_and_the_command_never_runs` | 실패 |
| `test_a_guard_inside_a_guard_does_not_wait_for_the_lock` | 실패 |
| `test_a_held_variable_naming_a_non_ancestor_does_not_skip_the_lock` | 실패: 75를 기대하는데 main은 0으로 종료. PR에 따르면 첫 commit에서도 실패 |
| `test_a_lock_path_that_is_not_a_regular_file_is_refused` | 실패 |
| `test_sigterm_while_waiting_for_the_lock_stops_the_guard` | 실패 |
| `test_a_nested_guard_command_completes` | `lock held by outer guard` assertion에서 실패 (중첩 command 자체는 main에서도 완료됨) |
| `test_a_daemon_left_by_the_command_does_not_keep_the_lock` | 통과 (main에는 lock이 없음). PR에 따르면 이 branch에서 `{LOCK_FD}<&-` close를 제거하면 실패 |

main의 script로는 26개 test 중 7개가 실패했습니다. 두 guard test는 둘 다 0으로 끝나고, `CONTENDED` 줄이 없으며, 네 event가 순서대로 (start, CLEAN, start, CLEAN) 나오고, 한 guard가 `waiting for guard lock`을 남기는지를 확인합니다.

### 5.2 Gate

Orchestrator가 head `06e0abff` (origin/main과 최신 상태 일치)에서 실행한 결과:

- `bash -n scripts/rocm_gpu_guard.sh`: OK.
- `python3 -m unittest tests/test_rocm_decode_profile.py`: 26개 test OK (guard test 17개).
- `make verify-versions verify-kernel-dtype-keys verify-kernel-port-dispatch verify-llama-compat verify-rocm-overlay verify-fmt`: exit 0.

PR 기록: `cargo test --features rocm --test dead_doc_pointers` 통과. 수동 lock 대기 중 SIGTERM은 `flock`을 남기지 않고 143으로 종료. `PATH`에서 `flock`을 숨기면 위 메시지와 함께 exit 2. 변경이 script, test, 문서뿐이므로 전체 `make verify-rocm` gate는 실행하지 않았습니다.

### 5.3 gfx1151 end-to-end

두 guard (commit `1f6ff1ad` 기준. 마지막 commit은 regular file 검사와 주석만 추가), 기본 lock 경로, `--idle-secs 20 --max-attempts 2`, 공유 `--log` 하나로, 같은 초에 시작해 각각 실제 KFD 목록 위에서 1-token `mlxcel generate -m models/mlx/Qwen3-0.6B-4bit`를 감쌌습니다.

- Guard A는 `acquired guard lock ... after 0s`, guard B는 `waiting for guard lock`을 남겼습니다.
- A의 idle 대기는 다른 unit의 GPU 작업과 cargo build가 끝날 때까지 이어졌고, 22:05:25에 `attempt 1: CLEAN, exit 0`을 남겼습니다.
- B는 같은 초에 `acquired guard lock ... after 779s`를 남기고, 자기 20초 idle window를 기다린 뒤 22:05:57에 `attempt 1: CLEAN, exit 0`을 남겼습니다.
- 두 실행 모두 `CONTENDED`나 `REJECTED` 줄이 없습니다.

779초는 설계의 비용도 보여 줍니다. Lock을 idle 대기 동안에도 쥐고 있으므로, A가 무관한 GPU 사용자를 기다리는 동안 B도 그만큼 줄을 섰습니다.

## 6. 기술적 선택과 그 이유

- **경쟁을 견디지 않고 직렬화.** Guard는 KFD 목록만으로 다른 guard의 command와 다른 GPU 사용자를 구별할 수 없으므로, monitor에게 다른 guard를 가르치는 대신 목록 밖에서 조율합니다.
- **첫 idle 대기부터 종료까지 lock 유지.** COMMAND 주변에서만 잡으면 두 번째 guard의 idle window가 첫 COMMAND 실행 전에 끝나, 새 idle window 없이 첫 실행 직후에 측정하게 됩니다.
- **없을 때만 만들고 read-only로 열기.** 이 방식 덕분에 `fs.protected_regular` 아래에서도 하나의 기본 경로를 모든 사용자가 쓸 수 있습니다.
- **시도별 제한 대신 하나의 예산.** Lock 대기가 생기면 단계별 제한은 실제 상한을 여러 제한에 시도 횟수를 곱한 합으로 만듭니다. 시계 하나가 "이 guard는 얼마나 기다려도 되는가"에 바로 답합니다.
- **Background `flock`과 `wait`.** pid 하나를 추적하는 비용으로 새 대기 구간에서도 기존 INT, TERM 계약을 유지합니다.
- **중첩에 조상 검사.** 환경 변수만으로는 pid 재사용이나 daemon 때문에 우연히 위조될 수 있습니다. 부모 탐색이 lock 건너뛰기를 실제 process tree에 묶습니다.
- **`flock`이 없으면 실패.** 경쟁을 조용히 되살리는 대신 exit 2로 끝납니다.

## 7. 남은 위험과 후속 작업

- **공유 기본 경로.** 어떤 로컬 사용자든 `/tmp/mlxcel-rocm-gpu-guard.lock`을 열어 쥐거나 읽을 수 없게 미리 만들 수 있으므로, 악의적인 사용자는 모든 guard를 막을 수 있습니다. 문서는 신뢰할 수 없는 사용자와 공유하는 host에서 `ROCM_GPU_GUARD_LOCK`을 benchmark 사용자만 접근할 수 있는 경로로 지정하라고 안내합니다.
- **대기자 간 공정성 없음.** `flock`은 FIFO 순서를 보장하지 않으므로 guard가 셋 이상 줄을 서면 시작 순서와 획득 순서가 다를 수 있습니다. End-to-end로는 두 guard만 실행했습니다.
- **선두 차단.** Guard가 아닌 GPU 사용자를 기다리는 guard는 그동안 lock을 계속 쥡니다 (위 실행에서 779초). 상한이 필요한 호출자는 이제 전체 예산인 `--max-wait`를 넘겨야 합니다.
- **사용자 간 공유는 실행해 보지 않음.** `fs.protected_regular` 관련 논리는 host에서 서로 다른 두 사용자로 시험하지 않았습니다.
- **같은 바깥 guard 아래의 중첩 guard**끼리는 직렬화되지 않습니다. 문서화만 되어 있고 강제되지 않습니다.
- **Metal과 CUDA**는 실행하지 않았습니다. Guard는 ROCm 전용이고 이 변경은 Metal이나 CUDA 경로를 건드리지 않습니다.

## 8. 학습 포인트

- **Guard가 스스로 경쟁 상대가 될 수 있습니다.** 모든 외부 GPU 사용자를 거부하는 monitor에는 동료 guard가 그 외부 사용자가 되지 않게 하는 장치가 필요하고, process 목록 필터링만으로는 안 됩니다.
- **Bash trap은 foreground 자식을 기다립니다.** Signal 처리를 약속하는 script에서 오래 막히는 호출은 background로 돌리고 `wait`해야 합니다.
- **상속된 descriptor는 주인보다 오래 삽니다.** File descriptor로 쥐는 lock은 주인보다 오래 살 수 있는 모든 자식에서 닫아야 합니다. 그렇지 않으면 남은 daemon이 lock을 소유합니다.
- **`/tmp`에는 고유한 open 규칙이 있습니다.** `fs.protected_regular`는 다른 사용자 파일에 대한 평범한 `>>`를 실패로 바꿉니다. 한 번 만들고, 그 뒤에는 `O_CREAT` 없이 엽니다.
- **경로를 여는 것만으로 막힐 수 있습니다.** Open 전에 `-f`를 확인하면 FIFO가 script를 자체 timeout 너머로 멈추게 하는 일을 막습니다.

Refs: #2146, #2061, #2065, #1814, #1801.
