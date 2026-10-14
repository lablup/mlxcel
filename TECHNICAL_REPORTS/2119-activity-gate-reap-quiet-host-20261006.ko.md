# 기술 보고서: PR #2119 - 누수된 Activity gate 서버를 정리하고 조용한 호스트를 기다림

**작성일**: 2026-10-06

**상태**: GB10 CI runner에서 구현하고 검증했으며 머지 대기 중이다.

**언어**: Python(게이트 호스트 helper, verifier), YAML(workflow)

**위험도**: 낮음. CI 잡 하나만 바뀐다. 신호는 이 잡의 identity를 가지고 제어 터미널이 없는 프로세스에만 보내고, 통과 경로의 정상 종료 요건은 그대로다.

## 요약

GB10 runner의 WebUI Activity 성능 게이트는 verifier가 죽거나 정리가 중간에 포기하면 서버를 남겼다. 남은 서버는 이후 모든 실행을 GPU 사전 검사에서 실패시켰는데, 그 검사는 개수만 출력했다. 또 다른 작업이 호스트에 부하를 주는 중에도 채점했다. 이제 새 helper `scripts/webui/activity_gate_host.py`가 채점 전에 앞선 실행의 누수를 정리하고 호스트가 조용해질 때까지 기다리며, 항상 실행되는 단계가 실패한 실행이 남긴 것을 거둔다(#1949). 이 PR의 첫 CI 실행이 runner에서 PR #2117을 다섯 번 실패시킨 실제 누수를 찾아 정리했다.

## 1. 문제 정의

**결함 1: 누수.** verifier는 서버, Xvfb, openbox, node, 그리고 Playwright를 통해 Chromium을 띄우는데, 각각 자기 세션에서 돈다. 그래서 단계 timeout, 취소, verifier 강제 종료 어느 것도 이들에게 닿지 않는다. 유일한 정리는 프로세스 안의 `finally`였고, 여기에는 세 가지 빈틈이 있었다.
- verifier가 kill되면 아예 실행되지 않는다.
- SIGKILL 대기에 예외 처리가 없어서, 그 대기보다 오래 사는 프로세스가 있으면 `finally` 밖으로 예외가 빠져나갔다.
- `process_group_empty: false`를 기록만 하고 아무 조치도 하지 않았다.

GPU 메모리 4~5 GiB를 잡은 누수 서버는 다음 실행의 `nvidia-smi` 사전 검사를 개수만 출력한 채 실패시켰다. 2026-10-05에 `main` 680f64ee가 채점에서 실패했고, 이어진 PR #2117의 다섯 실행이 모두 그 사전 검사에서 실패했다.

**결함 2: 부하 중 채점.** WebUI 코드를 건드리지 않은 앞선 두 PR이, 같은 호스트에서 다른 잡이 컴파일하는 동안 decode 저하를 2.8%와 3.9%로 보고했다. 예산은 2%다. 둘 다 조용한 호스트에서 다시 돌리자 통과했다. GPU가 비어 있는 것만으로는 부족하고, 호스트 자체가 조용해야 한다.

## 2. 변경 요약

- **`activity_gate_host.py precheck`**: CI flock과 `gpu-lock` 다음, 게이트 본문의 첫 명령으로 실행된다.
  - 앞선 실행이 남긴, 이 잡의 identity를 가진 프로세스를 정리한다.
  - quiet 대기 전후에 남은 GPU compute 프로세스가 있으면 fail closed한다.
  - 호스트가 조용해질 때까지 최대 600초 기다린다.
  - 관찰한 호스트 상태를 기록한다.
- **`activity_gate_host.py reap`**: 새 `if: always()` 단계다. verifier의 pid 파일에 있는 것과 그 밖의 identity 프로세스를 정리하고, 살아남은 것이 있으면 잡을 실패시킨다.
- **`verify_activity_performance.py`**:
  - `--pid-file`로 소유 프로세스를 모두 기록한다.
  - `--host-state`로 precheck가 관찰한 상태를 전체 evidence에 넣고, activity 후 부하 샘플도 함께 남긴다.
  - `terminate_owned`가 더는 `finally` 밖으로 예외를 던지지 않고, 그룹에 남은 멤버를 정리한다.
  - Xvfb와 openbox를 run 디렉터리에서 띄운다.
- **workflow**: 게이트 단계 timeout을 65분에서 75분으로, 잡 timeout을 90분에서 120분으로 늘렸다. 새 테스트는 `make verify-webui-helper-tests`에 들어간다.

## 3. 기술적 선택과 그 이유

**혈통이 아니라 identity로 찾는다.** 누수된 프로세스는 부모를 잃었기 때문에 프로세스 계보로는 찾을 수 없다. 그래서 잡이 통제하는 특징으로 알아본다.
- 실행 파일이 `$RUNNER_TEMP/mlxcel-webui-installed/`에 있다.
- 작업 디렉터리가 verifier run 디렉터리다.
- `HOME`이 verifier run 디렉터리 안에 있다.

Chromium에 닿는 단서는 `HOME`뿐이다. Playwright가 브라우저를 `detached`로 띄우므로 브라우저는 자기 프로세스 그룹에 있고 작업 디렉터리도 저장소다. 하지만 환경은 verifier에게서 물려받고, 거기에 `HOME=<run dir>/home`이 들어 있다. `clean_env`가 `GITHUB_*`를 지우기 때문에 그 변수들은 단서가 될 수 없다. 누수 판정은 고아가 되었거나 run 디렉터리가 사라졌는지로 한다. runner는 잡 사이에 `RUNNER_TEMP`를 비운다. 그래서 누수 서버의 실행 파일과 cwd가 둘 다 `(deleted)`로 읽혔다.

**공유 호스트이므로 신호는 좁게 보낸다.** 개발자도 같은 uid로 runner에 로그인하기 때문에 EPERM은 아무것도 막아 주지 않는다. 보안 리뷰에서, 첫 버전은 run 디렉터리에 머문 개발자 셸의 프로세스 그룹 전체를 SIGKILL할 수 있었다는 점이 드러났다. 이를 세 가지 규칙으로 막는다.
- 제어 터미널이 있는 프로세스는 건너뛴다.
- 디렉터리나 `HOME`으로만 매칭된 프로세스에는 하나씩 신호를 보낸다.
- 그룹 전체 신호는 잡이 띄운 것이 확실한 세션 리더에게만 보낸다. pid 파일 기록과, 설치 산출물 실행 파일을 돌리는 리더가 그 경우다.

모든 신호는 pidfd를 연 뒤 시작 시각을 다시 확인하고 보내므로, 재사용된 pid에는 신호가 가지 않는다. pid 파일 기록에는 boot id도 남긴다.

**load average가 아니라 runnable task 수로 판단한다.** `/proc/loadavg`의 순간 runnable 수는 1초 안에 반응한다. 반면 1분 load average와 `ps %cpu`는 막 시작한 컴파일을 유휴로 읽는다. 기준은 1초 간격 10개 샘플의 평균이 CPU 수의 4분의 1 이하인지다. GB10에서는 이 값이 5이고, 유휴 runner(측정값 1.1~1.4)와 컴파일이나 CPU당 busy loop 하나(23.8)를 가른다. 창이 다 차기 전에는 거부하지 않는다. 그렇게 거부하던 버전을 GB10 시뮬레이션이 잡아냈다.

**대기 전후에 GPU를 확인한다.** 대기 뒤에만 확인하면 GPU 충돌이 최대 10분의 호스트 부하 뒤에 가려지고, 그동안 두 lock을 계속 쥐게 된다.

**공개 저장소이므로 정보를 가린다.** lablup/mlxcel의 로그와 아티팩트는 누구나 읽을 수 있다. 그래서 이 잡의 것이 아닌 GPU 프로세스는 pid, 프로세스 이름(basename), 메모리, 부모, uid만 보고한다. 호스트 스냅샷에는 컴파일러 이름과 pid만 넣고, 임의의 명령 이름은 넣지 않는다.

## 4. 검증

- helper 테스트:
  - host helper 테스트 18개. 잡이 서버를 두는 위치에 `sleep` 복사본을 놓고 실제 프로세스로 돌린다.
  - verifier 테스트 21개. 옛 코드에서 실패하는 `terminate_owned` 테스트 두 개를 포함한다.
  - summary 테스트 10개. `host_state`가 요약을 막지도, 요약으로 새어 나가지도 않는지 확인하는 테스트를 포함한다.
- runner `lablup-dgxspark21`(spark-101):
  - run 37387059748이 실제 누수(pid 1453937, PPID 1, 실행 파일과 cwd 모두 `(deleted)`)를 정리한 뒤 통과했다.
  - run 37388347876(최종 head)은 평균 runnable 1.4로 통과했고, 정상 종료도 그대로였다.
  - 일회성 probe run 37388422039:
    - CPU당 busy loop 하나를 돌리자 평균 23.8에서 거부했다.
    - 부하를 걷어내자 1.1에서 통과했다.
    - activity 도중 verifier를 SIGKILL하자(서버 4604 MiB) reap이 Xvfb, openbox, 서버, node, Chromium을 정리했고, 이후 `nvidia-smi`와 `pgrep` 결과가 모두 비어 있었다.
- spark-102에서 `gpu-lock`을 잡고 실제 서버로 GB10 시뮬레이션:
  - 누수 서버를 정리했다.
  - 외부 GPU 프로세스는 경로 없이 이름만 대고 즉시 거부했으며, 건드리지 않고 두었다.
  - 바쁜 호스트를 거부했다.
  - verifier를 SIGKILL한 뒤 남은 서버를 pid 파일로 거뒀다.

## 5. 남은 위험

- 이 잡의 프로세스 두 벌이 동시에 돌면 서로를 자기 것으로 본다. runner 하나는 잡을 순차로 돌리고, 다른 runner 등록은 각자 `_work/_temp`를 쓴다.
- quiet 검사는 게이트 시작 시점만 본다. 측정 도중 시작한 부하는 여전히 점수를 움직인다. `after_activity` 샘플로 사후에 드러날 뿐 막지는 못한다. 판정 규칙 자체는 #1925 범위다.
- sweep과 정지 사이에 스스로 종료한 프로세스(openbox, node, Chromium은 Xvfb와 함께 죽는다)에 대해 reap 로그는 `signals none`을 출력한다. JSON 보고서에서는 모호함이 없다.
- `gpu-lock`은 그것을 쓰는 개발 세션만 덮는다. 그 밖에서 시작한 GPU 작업은 여전히 게이트를 실패시키며, 이제는 pid와 소유자가 이름으로 나온다.

## 6. 학습 포인트

- **자기 세션을 여는 프로세스는 그것을 띄운 프로세스 밖에서 정리해야 한다.** 프로세스 안의 `finally`는 SIGKILL을 넘어서지 못한다. pid 파일을 쓰는 항상 실행 단계가 그 틈을 메우고, identity sweep이 pid 파일이 보지 못한 것까지 덮는다.
- **모든 kill이 공유 호스트에서 어디까지 미치는지 확인한다.** 한 uid를 여럿이 쓰는 runner에서는 권한이 아무것도 막지 않는다. 의미 있는 필터는 제어 터미널 확인, 프로세스별 개별 매칭, pidfd다.
- **"조용함"은 순간값으로 잰다.** load average와 누적 `%cpu`는 바로 게이트가 봐야 할 사건에서 늦는다.
- **실제 호스트 시뮬레이션은 단위 테스트가 못 찾는 것을 찾는다.** 창과 timeout의 버그, 그리고 Chromium이 node 프로세스 그룹을 벗어난다는 사실은 둘 다 실제로 돌려 보고서야 나왔다.

## 7. 관련 항목

- 이슈 #1949.
- PR #2117(누수에 다섯 번 걸린 실행).
- #1925(판정 규칙, 범위 밖).
- PR #2114(`gpu-lock` wrapper).
- probe PR #2120(닫힘).
