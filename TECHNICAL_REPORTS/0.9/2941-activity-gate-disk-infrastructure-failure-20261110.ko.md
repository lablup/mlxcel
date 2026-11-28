# PR #2941: Activity 게이트의 디스크 고갈을 인프라 실패로 보고

**날짜**: 2026-11-10
**상태**: 완료 (draft PR로 열려 있음, #2940 해결)

## 요약

셀프호스티드 GB10 러너(spark-101)에서 도는 WebUI Activity 성능 게이트가 호스트 디스크 때문에 실패하면 이제 "infrastructure"라고 표시합니다. 두 가지 장치가 이를 담당합니다. 호스트 precheck는 `$RUNNER_TEMP`의 여유 공간이 `--min-free-gb`(기본 5 GB)보다 작으면 시작을 거부하고, verifier는 activity 스크립트가 실패했을 때 로그 끝부분에 ENOSPC, EDQUOT, EIO 또는 처리되지 않은 스트림 `'error'` 이벤트가 보이면 `activity performance verification failed` 대신 `activity gate infrastructure failure`로 분류합니다. 두 경로 모두 여전히 exit 1로 끝납니다. 성능 예산, verdict 정책, verdict 로직은 바뀌지 않았습니다. Python CI 헬퍼만 변경했으며 6개 파일, +259/-8입니다.

## 1. 문제 정의

CI 실행 42753658537(main `6cd038eb`)의 `WebUI installed artifact` 잡이 `activity performance verification failed: activity-performance.mjs exited 1`로 실패했습니다. 진단 아티팩트의 `activity-performance.log` 끝에 실제 원인이 있습니다. `webui/scripts/activity-performance.mjs`의 pair 루프 안 `process.stdout.write`에서 Node의 `Error: ENOSPC: no space left on device, write`가 발생했고, `SyncWriteStream`의 처리되지 않은 `'error'` 이벤트로 터졌습니다. 크래시 전 샘플은 정상이었고(off와 one-visible 모두 약 164~167 tok/s), verdict는 실행되지도 않았습니다.

러너 루트 파일시스템이 가득 찬 상황인데 메시지는 성능 회귀를 가리켰습니다. 원인 파악에 시간이 들고, 2% 예산을 완화하는 잘못된 수정을 유도할 수 있습니다. 이슈는 잡의 실패 여부는 그대로 두고 실패에 올바른 이름을 붙이도록 요청했습니다.

## 2. 변경 요약

### 2.1 여유 공간 precheck (`scripts/webui/activity_gate_host.py`)

- `precheck`에 `--min-free-gb` 옵션을 추가했습니다(float, 기본 5.0, 10진 GB = 1e9 바이트). `parse_args`는 음수와 유한하지 않은 값을 거부합니다. `nan`은 어떤 비교에서도 false가 되어 검사를 조용히 꺼 버리기 때문입니다.
- `precheck()`는 누수 프로세스 정리, GPU 검사, 600초 quiet 대기보다 먼저 `shutil.disk_usage(args.run_root)`를 호출합니다. 디스크가 가득 찬 경우 10분을 기다리지 않고 즉시 실패합니다. `GateRefused("infrastructure: only N GB free on <path>, need M GB")`를 던지고, 기존 거부 처리 경로(`::error::` 주석과 exit 1)를 그대로 탑니다.
- 관찰 결과는 evidence에 포함되는 host-state JSON에 `state["disk"]`(경로, 여유 바이트, 최소값)로 기록됩니다.
- `disk_usage`는 기존 `gpu_query`, `read_runnable`과 같은 방식의 주입 가능한 키워드 인자입니다. 테스트는 이를 stub으로 바꿔서 테스트 호스트의 디스크 상태에 의존하지 않습니다.
- `finally` 블록의 host-state 기록은 이제 best effort입니다. 디스크 부족으로 거부한 직후에는 이 기록 자체가 ENOSPC를 낼 수 있고, 잡히지 않으면 거부 메시지를 traceback이 덮어씁니다. 실패하면 `::warning::`을 내고 exit code는 1로 유지됩니다.
- `.github/workflows/ci.yml`은 `--min-free-gb 5`를 명시적으로 넘기고, 스텝 주석에 두 동작을 설명합니다.

### 2.2 실패 분류 (`scripts/webui/verify_activity_performance.py`)

- `classify_environment_failure(log_path)`는 activity 로그의 마지막 64 KiB로 seek한 뒤, ENOSPC, EDQUOT, EIO를 완전한 토큰으로 찾으면(마지막 일치가 우선) 또는 처리되지 않은 스트림 오류 이벤트를 찾으면 짧은 사유를 반환합니다. 반환값은 고정 문자열뿐이고 로그 텍스트를 그대로 옮기지 않으므로, 로그의 신뢰할 수 없는 내용이 CI 메시지로 흘러가지 않습니다.
- `run_activity_script()`는 0이 아닌 종료일 때만 이 함수를 호출합니다. 일치하면 사유, 종료 코드, 로그 경로를 담은 `ActivityInfrastructureError`(`AssertionError`의 하위 클래스라 기존 `except` 경로가 모두 실패로 취급)를 던집니다. 일치하지 않으면 기존 `AssertionError("activity-performance.mjs exited N; log=...")`가 그대로입니다.
- `run()`은 이 오류에 대해 evidence에 `failure_class: infrastructure`를 기록합니다.
- `main()`은 예외 자체나 `__cause__`/`__context__` 체인 안에 `ActivityInfrastructureError` 또는 errno가 ENOSPC, EDQUOT, EIO인 `OSError`가 있으면 인프라 실패로 표시합니다. 두 번째 경우는 verifier 자신의 기록(evidence, run 디렉터리)에서 나는 디스크 오류를 다룹니다. 그 외에는 기존 메시지를 유지하며, 모두 1을 반환합니다.
- 실패 핸들러와 정리용 `finally`의 evidence flush는 `Harness.flush_best_effort()`를 거칩니다. 기존 `h.flush()`는 디스크가 가득 찬 상황에서 스스로 ENOSPC를 내며 원래 예외를 덮어써서, 이 PR이 겨냥한 바로 그 경우에 예전 메시지가 출력됐을 것입니다.

### 2.3 문서

`docs/webui-integration-matrix.md`의 host precheck 문단에 디스크 검사와 기본값을 설명했습니다.

## 3. 기술적 선택과 그 이유

### 3.1 이슈의 20 GB 대신 5 GB를 고른 이유

임계값은 실행이 실제로 끝날 수 없을 때만 거부해야 합니다. 게이트가 실행마다 `$RUNNER_TEMP`에 직접 쓰는 양은 작습니다. 336 MB qwen3-0.6b-4bit 픽스처의 reflink/복사 뷰를 담은 전용 run 디렉터리, run 디렉터리의 `HOME` 아래 Chromium 프로필, 그리고 각각 수십 KB인 server, Xvfb, openbox, activity 로그와 evidence JSON입니다(실패한 실행의 진단 파일 합계가 약 40 KB). 설치된 바이너리(672 MB)와 픽스처 캐시는 precheck 이전에 이미 디스크에 있습니다. 1 GB에 한참 못 미치므로, 5 GB면 10~45분 동안 같은 파일시스템에 쓰는 다른 작업을 위해 몇 배의 여유가 남습니다.

작성 시점 spark-101의 여유 공간은 약 26~28 GB였습니다. 20 GB 하한은 지금은 통과하겠지만, 정상적으로 끝날 실행을 거부하기까지 여유가 거의 없어 디스크 사용량 추세가 그대로 게이트 불안정성으로 이어집니다.

### 3.2 임계값보다 분류가 더 중요한 이유

실패한 실행은 약 48 GB 여유로 시작했고, 게이트 도중 동시에 돌던 작업의 쓰기로 디스크가 찼습니다. 48 GB 미만의 어떤 시작 시점 임계값도 이를 잡지 못했을 것입니다. precheck는 시작 시점에 이미 부족한 경우를 다루고, 실제로 일어난 실행 도중의 경우는 로그 분류가 이름을 붙입니다.

### 3.3 실패는 실패로 유지

두 장치 모두 메시지만 바꿉니다. 거부나 분류된 실패도 exit 1로 잡을 실패시킵니다. 인프라 실패를 skip으로 바꾸면 고장 난 러너가 무기한 green을 보고할 수 있고, 게이트의 fail-closed 설계(#1949)는 바로 그것을 막기 위해 존재합니다.

## 4. 리뷰 발견 사항과 수정

- **디스크가 가득 찼을 때 메시지가 가려짐 (MEDIUM, 수정).** 보안 리뷰에서 `run()` 실패 핸들러의 evidence flush가 같은 디스크에서 ENOSPC를 내며 `ActivityInfrastructureError`를 덮어쓴다는 점을 찾았습니다. `flush_best_effort()`와 `main()`의 원인 체인 검사로 고쳤습니다. 모든 evidence flush가 ENOSPC를 내도록 만든 테스트에서 인프라 오류가 그대로 전파되는지 확인합니다.
- **유한하지 않은 `--min-free-gb` (LOW, 수정).** `nan`이 검사를 꺼 버렸는데, 이제 거부합니다.
- **`reap` 파싱 크래시 (머지 전 수정).** 유한성 검사의 첫 버전에 괄호가 빠져(`A and B or C`) `--min-free-gb`가 없는 `reap`에서 `args.min_free_gb`를 평가하다 `AttributeError`가 났습니다. 매 실행 항상 도는 `Reap Activity gate processes` 스텝이 깨졌을 것입니다. 수정했고 `reap` 명령줄을 파싱하는 테스트로 보호합니다.
- **넓은 스트림 오류 매칭 (LOW, 수용).** 디스크 errno 없는 처리되지 않은 `'error'` 이벤트(예: 소켓 ECONNRESET)도 인프라로 표시됩니다. 이슈가 요청한 동작이고, 종료 코드는 영향받지 않으며, 현재 샘플 줄에는 그런 텍스트가 없습니다.

## 5. 검증

- `python3 scripts/webui/activity_gate_host_tests.py`: 22개 테스트 통과(여유 공간과 경로를 명시한 디스크 부족 거부, GPU 조회와 quiet 대기에 도달하지 않음, 통과 사례, host-state 기록 실패 시에도 거부 유지, 인자 검증, `reap` 파싱).
- `python3 scripts/webui/verify_activity_performance_tests.py`: 32개 테스트 통과. 픽스처는 실행 42753658537 로그의 실제 끝부분을 러너 경로만 중립화해 잘라 낸 것입니다. 픽스처를 출력하고 exit 1 하는 가짜 `node_bin` 셸 스크립트로 `run_activity_script`를 거치고, `run`을 patch한 `main`도 거칩니다. 예산 형태 실패는 기존 메시지를 유지하고, 체인된 ENOSPC와 EIO는 인프라로 표시되며 EACCES는 그렇지 않습니다.
- `python3 -m unittest discover -s tests -p 'test_*.py'`, `scripts/insert_apache_header.py --check`, `scripts/ci/check_env_var_docs.py`, `cargo fmt --all -- --check`: 이상 없음.
- 실제 호스트에서 `precheck --min-free-gb 100000`은 exit 1과 `infrastructure: only 1588.5 GB free on <path>, need 100000 GB`로 거부했고, `--min-free-gb 1`은 디스크 검사와 나머지 precheck를 통과했으며, pid 파일 없는 `reap`은 exit 0이었습니다.

## 6. 학습 포인트

- 디스크에 쓰는 실패 핸들러는 자신이 보고하려는 바로 그 실패에 무너질 수 있습니다. 오류 종류가 "디스크 가득 참"이면 보고 경로의 모든 쓰기는 best effort여야 메시지가 살아남습니다.
- 한 서브커맨드에만 적용돼야 하는 가드에서 `A and B or C`는 흔한 우선순위 실수입니다. argparse 서브커맨드 옵션은 다른 서브커맨드의 namespace에 존재하지 않으므로, 이 실수는 엉뚱한 명령의 `AttributeError`로 드러납니다.
- 실행 시작 시점의 자원 검사와 사후 분류기는 서로 다른 질문에 답합니다. 앞의 것은 고장 난 호스트에서 시작하지 않게 하고, 뒤의 것은 실행 도중 생긴 실패에 이름을 붙입니다. 이번 변경의 계기가 된 사고는 뒤의 경우였습니다.

## 7. 다루지 않은 부분

- spark-101의 공간 확보는 저장소 밖의 운영 작업입니다.
- 분류기는 기록된 ENOSPC 로그와 합성한 EDQUOT/EIO 로그로만 검증했습니다. 러너에서 Node가 실제로 낸 EDQUOT나 EIO는 아직 관찰되지 않았습니다.
