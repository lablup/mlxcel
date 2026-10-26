# PR #2645: fork에서 연 pull request에 대해 self-hosted CI job 건너뛰기

**날짜**: 2026-10-27
**상태**: 완료
**언어**: GitHub Actions YAML, Markdown
**위험도**: 중간

## 요약

공유 self-hosted runner에서 도는 `ci.yml` job 여덟 개(`GB10`의 일곱 개와 `rocm-gfx1151`의 `rocm-build`)가 fork에서 연 pull request의 코드를 runner 자신의 사용자로 실행했습니다. 이제 각 job은 PR의 head 저장소가 base 저장소와 같을 것도 요구합니다. `pipeline-parallel-ci.yml`이 이미 쓰던 보호 조건이며, workflow를 건드리지 않은 fork PR은 이 job들을 건너뜁니다. workflow 자체를 수정하는 fork PR은 이 보호 조건으로 막을 수 없고, 저장소의 Actions fork-PR 승인 정책만이 그 경우를 막습니다. 이 PR은 그 정책을 문서화했지만 바꾸지는 못합니다 (이슈 #2638).

## 1. 문제

모든 self-hosted job의 조건은 `github.repository == 'lablup/mlxcel'`이었습니다. `pull_request` 이벤트에서 `github.repository`는 base 저장소이므로, fork에서 연 PR에서도 이 조건은 참입니다. 이 보호 조건이 하는 일은 `GB10` runner가 없는 이 저장소의 fork 안에서 job이 영원히 대기하지 않게 하는 것뿐입니다.

조건을 통과하면 checkout은 PR merge commit을 가져오고, job은 PR 작성자가 제어하는 코드를 실행합니다. `cargo clippy`, `cargo check`, `cargo test --no-run`은 모든 crate의 `build.rs`를 실행하고 (`mlxcel-core`의 것은 cmake로 MLX를 빌드합니다), `xla-compile`과 `xla-link`는 `scripts/iree/setup-cuda.sh`를, `webui-installed-artifact`는 `scripts/webui/*.py`와 `pnpm --dir webui install`을 실행합니다. 이 runner는 오래 살아 있으며, gpu-lock으로 관리되는 개발 세션 및 release 빌드와 같은 uid를 씁니다. `release.yml`의 release job은 `$HOME/.cargo-target/mlxcel-cuda-*`, 공유 rustup, `~/.cargo/bin`을 재사용하고, purge 여부는 같은 uid의 프로세스가 쓸 수 있는 marker 파일로 결정됩니다. 따라서 fork의 build script가 이후 release가 링크할 산출물을 심을 수 있었습니다.

trigger는 `pull_request_target`이 아니라 `pull_request`이므로, fork 실행은 원래부터 읽기 전용 `GITHUB_TOKEN`을 받고 secret은 받지 못합니다. 노출 대상은 저장소 자격 증명이 아니라 runner 호스트였습니다.

## 2. 기술적 결정

**보호 조건의 형태.** 여덟 job 각각에 다음 조건을 추가했습니다.

```yaml
&& (github.event_name != 'pull_request'
    || github.event.pull_request.head.repo.full_name == github.repository)
```

저장소 보호 조건 뒤, 기존 path filter 및 `ci:full` 조건 앞에 AND로 붙습니다. `pipeline-parallel-ci.yml`이 자기 job에 쓰는 식과 같아서, 저장소 안의 fork 보호 방식이 하나로 통일됩니다. `event_name` 조건이 필요한 이유는 `push`(현재 `ci.yml`의 유일한 다른 trigger), `schedule`, `workflow_dispatch`에서 `github.event.pull_request`가 null이기 때문입니다. 이 조건이 없으면 비교가 `null == 'lablup/mlxcel'`이 되어 `main` push에서 job이 돌지 않게 됩니다.

**`ci:full`과의 순서.** label 조건은 마지막 `&&` 항 안에 있으므로, label은 같은 저장소 PR의 self-hosted job을 강제로 실행할 수는 있어도 fork PR을 포함시킬 수는 없습니다. 누구든 리뷰 스레드에서 label을 요청할 수 있는데, 보호 조건은 maintainer가 그 label을 붙이지 않기를 기대하는 데 의존하지 않습니다.

**삭제된 head 저장소.** PR이 열려 있는 동안 fork가 삭제되면 `head.repo`가 null이 되고 `full_name`은 base와 같지 않으므로 job은 건너뜁니다. 실패하면 닫히는 쪽입니다.

**연쇄 효과 없음.** `ci.yml`을 YAML로 훑으면 `runs-on`에 `GB10`이나 `self-hosted`가 있는 job은 정확히 이 여덟 개입니다. 각각 hosted `changes` job에만 의존하고, 보호된 job을 `needs`에 둔 job은 없으며, `if: always()`로 결과를 모으는 job도 없습니다. 그래서 fork PR에서 이들을 건너뛰어도 다른 것은 바뀌지 않습니다. hosted job(fmt, deny, contract 검사들, `webui-bundle`, `workflow-lint`)은 fork PR에서도 그대로 돕니다. `ci.yml` 밖에서 PR로 trigger되는 self-hosted job은 `pipeline-parallel-ci.yml`의 `three-host-real-model` 하나뿐이고, 이미 보호 조건이 있었습니다.

**안내 메시지.** `rocm-ci-status`는 ROCm job이 판정 기준이라고 말했지만, fork PR에서는 그 job이 돌지 않습니다. 이제는 maintainer가 ROCm 빌드를 따로 확인해야 한다는 warning을 내고 0으로 종료합니다. 이벤트 이름, head 저장소, base 저장소는 `run:` 안의 `${{ }}` 치환이 아니라 `env:`로 script에 전달됩니다. head 저장소 이름은 fork가 정하는 텍스트이고, 환경 변수로 넘기면 shell 소스에 섞이지 않습니다. `self-hosted-gate-advisory`와 `CONTRIBUTING.md`는 이제 `ci:full`이 같은 저장소 PR에만 적용된다고 말합니다.

**보호 조건이 다루지 않는 것.** `pull_request` 실행은 workflow 파일을 `main`이 아니라 PR의 merge commit에서 가져옵니다. 악의적인 fork PR은 이 `if:` 줄을 지우거나 `runs-on: GB10`인 새 job을 추가할 수 있고, 그 실행은 수정된 workflow를 그대로 실행합니다. `ci.yml` 안에 무엇을 쓰든 이를 막을 수 없습니다. 유일한 장벽은 저장소의 Actions fork-PR 승인 정책이며, REST API(`actions/permissions/fork-pr-contributor-approval`)로 확인한 현재 값은 `first_time_contributors`입니다. 이 설정은 그 기여자의 기여가 하나라도 merge되기 전까지만 fork PR 실행을 붙잡습니다. 그 뒤로는 workflow를 다시 쓰는 PR을 포함해 그 사람의 fork PR이 승인 없이 실행됩니다. `all_external_contributors`로 바꾸면 모든 fork PR 실행이 승인을 기다립니다. 이 변경은 저장소 관리자 권한이 필요하며 이 PR의 범위가 아니었습니다. 다시 쓴 `clippy` 주석은 이 한계를 적고, fork PR의 실행을 승인하기 전에 `.github/workflows/**` diff를 읽으라고 요청합니다. PR의 앞선 commit 하나는 이 주석에서 승인 정책 언급을 빼고 fork 코드가 runner에 닿지 않는다고 주장했는데, 후속 commit(4f5beac6)이 정확한 설명을 되살렸습니다.

**수용 기준 3은 강제가 아닌 문서화.** PR 단계 CI를 release 입력에서 격리하려면 runner 호스트에서 job 종류별로 별도 uid나 `HOME`이 필요하고, 이는 workflow 수정이 아닌 호스트 변경입니다. 같은 저장소 PR(쓰기 권한이 있는 작성자)은 여전히 runner의 uid로 실행됩니다. job별 `$HOME/.cargo-target/mlxcel-*-ci` 디렉터리는 release 빌드의 `mlxcel-cuda-*`와 이름으로만 구분되고, rustup과 `~/.cargo/bin`은 공유됩니다. `clippy` 주석은 이를 남은 노출로 기록합니다.

## 3. 변경 요약

- `.github/workflows/ci.yml`: `clippy`, `cpu-link`, `webui-installed-artifact`, `xla-compile`, `xla-link`, `cuda-sm70-compile`, `cuda-blockfloat`, `rocm-build`(`vars.ROCM_CI_ENABLED` 조건 유지)에 head 저장소 보호 조건 추가. `rocm-ci-status`에 fork 분기 추가. `clippy` 주석을 두 보호 조건, 악의적 PR에 대한 한계, 같은 uid 노출을 설명하도록 다시 쓰고 `cpu-link`가 이를 참조. `self-hosted-gate-advisory`의 주석과 warning 문구를 같은 저장소 PR로 한정.
- `CONTRIBUTING.md`: self-hosted job은 fork PR에서 절대 돌지 않으며 `ci:full`로도 바뀌지 않는다는 한 문장.

## 4. 트레이드오프

fork PR은 이제 self-hosted 신호 없이 merge됩니다. clippy도, CUDA나 XLA 링크 검사도, 설치된 산출물 기반 WebUI 검사도 없습니다. job 수준 `if:`로 건너뛴 job은 skipped로 보고되고, GitHub는 이를 required check 통과로 취급하므로 branch protection도 이런 PR을 막지 않습니다. 그 검사가 필요하면 maintainer가 fork의 commit을 이 저장소의 branch로 push해서 CI를 돌려야 합니다. PR은 이 비용을 받아들였습니다. 대안은 release 빌드와 공유하는 호스트에서 신뢰할 수 없는 build script를 실행하는 것이었습니다.

## 5. 검증

- `yaml.safe_load`가 `ci.yml`을 파싱합니다.
- job별 script가 `runs-on`에 `GB10`이나 `self-hosted`가 있는 모든 job에 `event_name != 'pull_request'` 조건과 head 저장소 비교가 함께 있는지 확인합니다 (여덟 개 모두 OK).
- `.github/actionlint.yaml`을 쓴 `actionlint` 1.7.7은 `ci.yml`에 대해 `main`과 마찬가지로 아무것도 보고하지 않습니다.

미검증: 실제 fork PR은 열지 않았으므로, fork PR에서 job이 건너뛰어지는 것과 `rocm-ci-status`의 fork 분기는 실행을 관찰한 것이 아니라 식을 읽어서 확인한 것입니다. 승인 정책 권고는 적용되지 않았으므로, 관리자가 바꾸기 전까지 위에서 설명한 악의적 PR 경로는 열려 있습니다.

## 6. 학습 포인트

- `pull_request`에서 `github.repository`는 base 저장소를 가리킵니다. 이 값으로 쓴 보호 조건은 "이 저장소의 fork 안에서 실행"과 "여기서 실행"을 구분할 뿐, "fork에서 온 PR"과 "branch에서 온 PR"을 구분하지 못합니다. head 저장소는 `github.event.pull_request.head.repo.full_name`입니다.
- workflow 파일 안의 `if:`는 그 workflow를 실행하는 PR에 대한 보안 경계가 아닙니다. `pull_request` 실행은 PR이 가진 workflow 사본을 쓰기 때문입니다. 악의적인 workflow 수정에 대한 보호는 파일 밖에서 와야 합니다: fork-PR 승인 정책, 특정 workflow나 저장소로 제한한 runner group, ephemeral runner.
- fork가 제어하는 이벤트 필드(head 저장소 이름, branch 이름, PR 제목)는 `run:` 안의 `${{ }}`가 아니라 `env:`로 shell step에 전달해야 합니다.
