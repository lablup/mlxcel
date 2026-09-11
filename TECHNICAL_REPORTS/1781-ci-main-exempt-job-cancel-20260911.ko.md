# job 단위 `cancel-in-progress` 두 곳에서 `main` 예외 처리

## 배경

#1774(PR #1780)는 `.github/workflows/ci.yml`에 워크플로 단위 concurrency 그룹을 추가해, 릴리스 빌드까지 겸하는 단일 self-hosted 러너 GB10에 중복 실행이 쌓이지 않게 했다. 그 그룹은 `main`을 의도적으로 제외한다(`cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}`). 근거는 main 푸시 실행이 오래된 PR 베이스가 가렸던 문제를 잡아내는 지점이고, 릴리스가 그 커밋에서 빌드되기 직전의 마지막 게이트라는 것이다.

이 변경보다 먼저 있던 job 단위 그룹 두 개는 조건 없는 플래그를 유지하고 있었다.

## 결함

`xla-link`와 `cuda-sm70-compile`은 각각 다음을 갖고 있었다.

```yaml
concurrency:
  group: <key>-${{ github.event.pull_request.number || github.ref }}
  cancel-in-progress: true
```

`main` 푸시에는 `github.event.pull_request.number`가 없으므로 키가 `github.ref`, 즉 `refs/heads/main`으로 떨어지고 모든 머지가 한 그룹을 공유한다. 플래그가 무조건이면 두 번째 머지가 직전 머지의 작업을 실행 도중에 취소한다.

이는 PR 단위 설계의 실수가 아니다. 각 job의 주석이 밝히듯 그룹의 목적은 PR끼리 서로를 취소하지 않게 하는 것이고, PR 번호를 키로 쓰면 그 목적은 정확히 달성된다. 고려되지 않은 것은 `main` 폴백 경로이며, 같은 식이 거기서는 다른 결과를 낳는다.

하필 이 두 job이 가장 나쁜 자리다. 둘 다 `timeout-minutes: 120`으로 파일 내 최대치이고(`xla-compile`과 공유), 둘 다 GB10에서 돈다. 릴리스가 빌드될 브랜치에서 120분짜리 링크나 컴파일을 취소하는 것이야말로 워크플로 단위 예외가 막으려던 상황이다.

## 변경

job 단위 플래그 두 개를 워크플로 단위와 같은 조건으로 맞췄다.

```yaml
cancel-in-progress: ${{ github.ref != 'refs/heads/main' }}
```

PR에서는 식이 그대로 true로 평가되므로 PR 격리 동작은 달라지지 않는다. 근거는 각 플래그 옆에 기록했고, 이 파일이 이미 쓰고 있는 설명적 주석 방식을 따랐다.

## 검증

검증한 것: 워크플로가 정상 파싱되고 세 concurrency 블록(최상위, `xla-link`, `cuda-sm70-compile`)이 모두 같은 조건으로 해석된다. `actionlint` 지적이 변경 전후 완전히 동일하다. 기존 8건(커스텀 GB10 라벨에 대한 `runner-label` 4건, `shellcheck` 4건)이며 새로 유입된 것은 없다.

검증하지 못한 것: `main` 동작 자체. 증명하려면 `main`에 짧은 간격으로 두 번 머지가 일어나고 두 번째도 해당 job의 경로 필터를 통과해야 하는데, 공용 인프라에서 이를 안전하게 연출할 수 없다. PR #1780도 같은 이유로 같은 기준을 미체크로 남겼고, 이 변경은 그 한계를 해소하지 못한 채 그대로 물려받는다.

기록해 둘 경계 사례: `main`에서의 취소는 두 번째 머지가 해당 job의 `if:` 경로 필터에도 걸릴 때만 발생했다. 매칭되는 경로를 건드리지 않는 머지는 job 자체를 건너뛰므로 그룹에 합류하지 않는다.
