# 기술 리포트: PR #1780 - chore(ci): add a concurrency group so superseded runs stop stacking

**날짜**: 2026-09-11
**작성자**: mlxcel maintainers
**리뷰어**: 구현 리뷰 사이클
**상태**: 완료
**언어**: YAML(GitHub Actions 워크플로)
**위험도**: 중간(모든 기여자가 의존하는 공용 CI 인프라; 제품 코드는 건드리지 않음)

---

## 요약

`.github/workflows/ci.yml`에는 워크플로 수준 `concurrency` 키가 없었다. 그래서 푸시할 때마다 CI 실행이 통째로 큐에 쌓였고, 무엇도 무엇을 대체하지 않았다. 보통은 시간과 비용 문제로 끝나지만, 이 파일의 잡 넷은 `GB10`에서 돈다. 릴리스 빌드까지 함께 맡는 단 하나의 자체 호스팅 Linux 러너다. 그래서 아무도 더는 기다리지 않는 커밋의 실행이 그냥 놀고 있는 데 그치지 않고, 정작 필요한 실행보다 앞선 자리를 차지했다.

이 변경은 워크플로와 ref를 키로 삼는 워크플로 수준 `concurrency` 블록 하나를 더한다. `cancel-in-progress`는 조건식으로 적어, PR ref는 앞선 실행을 대체하되 `main`에서 진행 중인 실행은 절대 취소되지 않게 했다. 다른 워크플로도, 다른 잡도 건드리지 않는다.

---

## 문제 정의

2026-09-10 KST 기준으로 `gh run list`에서 측정한 값이다. 24시간 동안 CI 실행 113건이 만들어졌고, 2026-09-10T03:18:40Z에는 열넷이 동시에 열려 있었다. 그중 열하나가 브랜치 네 개에 몰려 있었다. 같은 구간에서 여섯 건이 `cancelled`로 끝났는데, 모두 자기 브랜치의 새 푸시에 밀린 뒤 러너를 비우려고 사람이 직접 취소한 것이다.

기록해 둘 만한 이차 효과가 있다. 이것이 처리량만의 문제가 아니라 정확성 문제이기도 한 이유다. 2026-09-10 02:15:54 KST에 러너 서비스가 잡 수행 도중 OOM으로 종료됐고, 실행 `34381499951`의 `cargo-clippy` 잡은 GitHub에 스텝 0개, 로그 없음으로 기록돼 있다. 서비스 저널을 읽기 전까지는 진짜 린트 결과와 구별되지 않는다. 그 clippy 잡은 실행이 만들어지고 8시간 38분이 지나서야 시작했고, 뒤에서 브랜치 네 개가 기다리고 있었다.

같은 파일의 잡 둘은 이미 국소적으로 이 문제를 풀어 두었고, 그래서 이 패턴은 여기서 받아들여진 방식이다. `xla-link`와 `cuda-sm70-compile`은 각각 PR 단위 `concurrency` 그룹을 달고 있다. GB10을 오래 붙잡는 잡이라는 이유에서다. `clippy`와 `xla-compile`은 어디에도 걸리지 않았고, 실행 전체를 덮는 것도 없었다.

---

## 변경 요약

`on:`과 `permissions:` 사이에 블록 하나를 넣고, 주석이 촘촘한 이 파일의 문체가 요구하는 근거 주석을 함께 달았다.

- **그룹**: `ci-${{ github.workflow }}-${{ github.event.pull_request.number || github.ref }}`. `pull_request.number || ref` 꼴은 `pipeline-parallel-ci.yml`과 이 파일의 잡 수준 그룹 둘이 이미 쓰는 형태이고, 세 번째 형태를 새로 만들지 않았다. `main`으로 가는 `push`에서는 왼쪽 피연산자가 null이라 `github.ref`로 떨어지고, 그래서 `main` 푸시 실행은 모든 PR 그룹과 구별되는 그룹에 들어간다.
- **취소**: `cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`. `cancel-in-progress`에 식을 쓰는 것은 문서화된 동작이며, GitHub 문서 자체가 `${{ !contains(github.ref, 'release/') }}`를 예시로 든다.
- **손대지 않은 것**: `xla-link`와 `cuda-sm70-compile`의 잡 수준 그룹. 워크플로 수준 그룹과 잡 수준 그룹은 합성되고, 저 둘은 의도적으로 다르게 키를 잡았다. 두 체계를 하나로 맞추는 일은 별개 변경이다.

---

## 기술적 판단

**`main`의 진행 중 실행을 취소하지 않기로, 기본값에 맡기지 않고 명시적으로 정했다.** 이슈가 남긴 단 하나의 열린 질문이었고, 이 식은 한쪽 편을 든다. `main` 푸시 실행은 낡은 PR 베이스가 가려 둔 것을 잡아내는 실행이고, 그 커밋에서 릴리스가 빌드되기 전의 마지막 관문이다. 다음 머지가 90초 뒤에 들어왔다는 이유로 그것을 죽이면, 다른 무엇도 만들어 주지 않는 유일한 신호를 버리는 셈이다. 반대편 논거도 실재한다. 이 저장소는 연달아 머지가 들어올 만큼 자주 머지하고, 그러면 GB10 잡이 큐에 쌓인다. 다만 그 비용은 무한정 늘지 않고 묶여 있다. *대기 중* 실행을 취소하는 것은 그룹핑에 본래 딸린 동작이고 `cancel-in-progress`는 *실행 중*인 쪽만 관장하므로, `main`은 머지 빈도와 무관하게 진행 중 하나에 대기 중 최대 하나를 유지한다.

남는 대가는 나중 독자가 놀라며 다시 발견하지 않도록 주석에 적어 두었다. 중간에 낀 `main` 커밋은 대기 중인 채로 실행을 잃을 수 있다. `main`의 머리에 있는 커밋은 언제나 그룹에서 가장 새것이라 항상 실행을 받고, 릴리스를 끊기 전에 중요한 성질은 그쪽이다. `nightly-verify.yml`도 같은 이유로 같은 선택을 했다.

**`release.yml`을 그대로 둔 것은 조용히 정하지 않고 밝혀 적는다.** `release.yml`은 `build-linux-cuda` 잡을 통해 GB10 큐에 기여하는 유일한 다른 워크플로다. `release: published`와 `workflow_dispatch`에서만 걸리므로 푸시로 쌓이지 않고, 산출물을 만드는 릴리스 빌드에 그룹을 씌우면 배포되는 결과물을 내는 실행을 취소할 위험이 생긴다. 이 변경으로 굶지도 않는다. 취소는 그 앞선 큐에서 CI 작업을 덜어낼 뿐이기 때문이다.

**`python.yml`과 `update_homebrew_formula.yml`도 그대로 두었다.** 둘 다 concurrency 그룹이 없지만 GB10에 닿지 않으므로, 이 이슈가 말하는 경합에 참여하지 않는다. 둘에 그룹을 씌우는 것은 호스팅 러너 분(minutes) 문제로서 따로 판단할 일이다.

---

## 워크플로 목록

| 워크플로 | 이번 변경 전 워크플로 수준 `concurrency` | GB10 잡 |
|---|---|---|
| `ci.yml` | 없음(잡 수준만, `xla-link`와 `cuda-sm70-compile`) | 4개: `clippy`, `xla-compile`, `xla-link`, `cuda-sm70-compile` |
| `release.yml` | 없음 | 1개: `build-linux-cuda` |
| `nightly-verify.yml` | `group: nightly-verify`, `cancel-in-progress: false` | 없음(자체 호스팅 macOS) |
| `pipeline-parallel-ci.yml` | `group: pp-ci-...`, `cancel-in-progress: true` | 없음(`pp-three-host`) |
| `python.yml` | 없음 | 없음(ubuntu-latest) |
| `update_homebrew_formula.yml` | 없음 | 없음(macos-latest) |

---

## 검증

로컬에서 확인한 것:

- 파일이 파싱되고, 블록이 의도한 그룹 문자열과 식으로 풀린다. 다른 것이 움직이지 않았음을 확인하려고 `GB10` 잡 넷과 살아남은 잡 수준 그룹 둘을 파싱된 트리에서 다시 읽었다.
- `actionlint` 1.7.7이 변경 전후로 완전히 같은 지적을 낸다. 기존 8건(알 수 없는 자체 호스팅 러너 레이블, `run:` 스크립트 안의 shellcheck `info`), 새로 생긴 것 0건. 이 변경은 순수 삽입이므로 줄 번호를 지운 뒤 `origin/main:.github/workflows/ci.yml`과 비교했다.
- 그 '깨끗함'이 헛돈 결과가 아니라 실제로 무언가를 확인한 결과임을 따로 증명했다. 같은 식 둘에 `github.reff`와 없는 컨텍스트 필드를 넣은 음성 대조군을 만들면 `actionlint`가 concurrency 블록에서 실패한다. 즉 식 타입 검사기가 이 구문을 들여다본다.
- `git diff --name-only origin/main`이 `.github/workflows/ci.yml` 하나만 돌려주어, "`release.yml`, `nightly-verify.yml`, `pipeline-parallel-ci.yml`은 변경 없음" 기준을 확인했다.

**경쟁을 연출하지 않고 실제로 확인했다.** 연출된 이중 푸시는 비용 때문에 접었다. `.github/workflows/ci.yml`은 `changes`의 경로 필터 넷(`rust`, `mlx_pin`, `xla_link`, `cuda_arch`) 모두에 들어 있어서, 이 브랜치에 푸시할 때마다 GB10 잡 넷이 전부 시작되고 그중 둘은 `timeout-minutes: 120`을 달고 있다. 부하를 덜어 준다는 주장을 증명하겠다고 단 하나뿐인 공용 러너에 가능한 가장 무거운 부하를 만들어 내는 것은 잘못된 거래다. 이슈가 제시한 검증 절차는 이 필터 적용 범위를 셈에 넣지 않았다.

그런데도 증거는 확보됐다. 이 리포트를 커밋하는 일 자체가 어차피 해야 할 작업으로서 브랜치에 두 번째 푸시를 일으켰기 때문이다. 그 결과는 수용 기준을 그대로 재현한다.

| 실행 | 헤드 | 생성 | 최종 상태 |
|---|---|---|---|
| `34562655380` | `ab7fd234`(첫 푸시) | 04:33:41Z | `completed` / **`cancelled`**, 04:36:33Z |
| `34562815627` | `bc4ac8c8`(리포트 푸시) | 04:36:13Z | `queued`, 취소되지 않음 |

이 브랜치에서 `cancelled`가 아닌 실행은 정확히 하나이고, 세션 어느 시점에도 `gh run cancel`을 쓰지 않았다. 새 푸시가 등록되고 20초 뒤에 concurrency 그룹이 이전 실행을 취소했다. 취소된 실행의 잡 내역은 이 변경의 값어치를 축소판으로 보여 준다. `cargo-clippy`와 `OpenXLA feature compile`은 `lablup-dgxspark21`에서 이미 끝나 있었고, `OpenXLA feature link`는 **`lablup-dgxspark21`에서 실행 중인 채로 취소**됐으며, `CUDA sm_70 compile`은 `runner_name`이 빈 채로 취소됐다. 러너를 차지하기 전에 풀려났다는 뜻이다. 뒤의 둘이 `timeout-minutes: 120`을 달고 있는 그 한 쌍이다.

**여전히 확인하지 못했고, 주장하지도 않는 것.** 식의 `main` 쪽 절반이다. `refs/heads/main`에서 `cancel-in-progress`가 false로 풀려 머지 후 진행 중인 실행을 건드리지 않는다는 것은, `main`에 머지가 일어나고 그 실행이 아직 도는 동안 두 번째 머지가 들어와야만 관찰된다. 이 PR의 무엇도 그것을 보여 주지 않으며, 위 실행은 `github.ref`가 `refs/pull/1780/merge`인 풀 리퀘스트 가지만 밟았다. `main` 동작은 관찰이 아니라 식의 문서화된 의미와 그것을 읽은 결과에 기대고 있다.

---

## 후속 과제

- `actions.runner.lablup.lablup-dgxspark21.service`의 systemd 드롭인에 `MemoryMax=`와 `MemoryAccounting=yes`로 러너 서비스를 묶어 두면, OOM이 에이전트 전체가 거둬지는 대신 러너 안에서 로그를 남기고 죽는 잡이 된다. 저장소 파일이 아니라 호스트 설정이므로 러너 런북에 들어갈 일이다. 이슈도 짚어만 두고 여기에 묶지 않기로 했다.
- PR 시점에 `clippy`가 GB10에서 돌아야 하는가는 #1283에서 따로 다룬다.
- 워크플로 수준 그룹과 잡 수준 그룹 둘을 하나의 체계로 맞추는 일은 뒤로 미룬 정리 과제로 남는다.
