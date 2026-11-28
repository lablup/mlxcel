# 기술 보고서: PR #2011, deploy_webpage.sh를 set -euo pipefail로 강화

**작성일**: 2026-09-28

**상태**: 구현 및 문법 검사 완료. 머지 대기 중. 스크립트는 실행하지 않았다. 실제로 실행하면 `mlxcel-releases` 원격 저장소로 push되기 때문이다.

**언어**: Bash

**위험도**: 낮음

## 요약

`scripts/deploy_webpage.sh`는 `scripts/` 디렉터리에서 `run_quality_gate.sh`, `bench_all_models.sh` 등 이미 더 엄격한 `set -euo pipefail`을 쓰는 나머지 스크립트와 달리 단순한 `set -e`로 실행되고 있었다. 이 PR은 6번째 줄을 `set -euo pipefail`로 바꿔 issue #1660을 닫는다. 변경 자체는 한 줄짜리 diff이며, 이 작업의 핵심은 편집이 아니라 새 strictness 아래에서 스크립트가 안전한지 확인하는 데 있다.

## 문제 정의

단순한 `set -e`는 `set -euo pipefail`이 잡아내는 두 종류의 실수를 잡지 못한다: 설정되지 않은 변수를 확장하는 경우(`-u`)와 파이프라인 마지막 단계가 아닌 곳에서 발생하는 실패(`pipefail`)다. 이 배포 스크립트는 빌드 결과물을 별도의 GitHub Pages 저장소로 강제 push(`git push -f`)하므로, 앞부분에서 조용히 삼켜진 실패가 있으면 오류 표시 없이 오래되었거나 불완전한 사이트가 배포될 수 있다. issue #1660은 이 이유로 스크립트를 `scripts/`의 나머지와 맞추도록 요청했다.

## 변경 요약

- `scripts/deploy_webpage.sh:6`: `set -e`를 `set -euo pipefail`로 변경. 다른 줄은 변경되지 않았다.

88줄짜리 스크립트를 두 가지 새 strictness 모드 관점에서 읽어봤다:

- **`-u` (nounset):** 스크립트는 다섯 개의 변수를 확장한다: `SCRIPT_DIR`, `PROJECT_ROOT`, `WEBPAGE_DIR`(8-10번째 줄)과 `REPO_URL`, `BRANCH`(13-14번째 줄). 다섯 개 모두 16번째 줄 이후 첫 사용 이전에 무조건적으로 할당되므로 확장 시점에 미설정 상태일 수 없다. 리다이렉트용 `index.html`을 작성하는 heredoc(32-70번째 줄)은 delimiter를 quote한 형태(`<< 'EOF'`)라서 그 안의 `$`처럼 보이는 내용(인라인 JavaScript의 `lang` 변수)은 shell 확장이 아니라 그냥 문자열이다. 따라서 `-u`는 현재 작성된 스크립트에는 관찰 가능한 효과가 없다. 이는 나중에 미할당 또는 조건부 할당 변수를 추가하는 편집이 들어올 때를 대비한 guard다.
- **`pipefail`:** 스크립트 어디에도 `|` 파이프라인이 없다. 따라서 `pipefail`도 현재는 마찬가지로 아무 효과가 없다. `-u`와 마찬가지로 이 값의 의미는 미래를 향한 것이다: `scripts/run_quality_gate.sh:3`을 비롯해 PR 본문에 열거된 다른 스크립트들의 기존 관례와 맞추고, 나중에 빌드나 git 명령을 `grep`/`tee` 등으로 파이프하는 편집이 들어왔을 때 해당 명령의 종료 코드가 조용히 버려지는 것을 막는다.

읽는 과정에서 관찰했지만 이번 issue 범위 밖이고 strictness 변경의 영향을 받지 않는 두 가지:

- 20번째, 25번째 줄의 `cd`(`cd "$WEBPAGE_DIR"`, `cd out`)는 subshell이 아니라 직접 실행되므로 작업 디렉터리 변경이 스크립트 실행이 끝날 때까지 유지된다. issue #1660은 이를 별개의 기존 특성으로 명시적으로 언급했다.
- `$BRANCH`와 `$REPO_URL`은 두 사용처(`git branch -m $BRANCH` 75번째 줄, `git push -f $REPO_URL $BRANCH` 86번째 줄)에서 quote되지 않았다. 둘 다 공백이나 glob 문자가 없는 상수 리터럴이라 현재는 안전하지만, `-u`는 quote가 막아주는 word-splitting을 막아주지는 않는다.

## 검증

- `bash -n scripts/deploy_webpage.sh`: 통과.
- 스크립트의 모든 변수 확장을 수동으로 읽어 확인함(위에 요약). 어떤 확장도 미설정 변수가 빈 문자열로 처리되는 데 의존하지 않으므로 `-u`가 스크립트를 깨지 않는다.
- 종단 간 실행은 하지 않음: 스크립트는 `git@github.com:lablup/mlxcel-releases.git`로 `git push -f`를 수행하므로, 검증을 위해 실제로 실행하면 해당 저장소에 배포가 이루어진다. 이는 의도적으로 실행하지 않았다.

## 남은 작업

이번 issue에서 추가로 필요한 작업은 없다. 위의 두 관찰 사항(유지되는 `cd`, quote되지 않은 `$BRANCH`/`$REPO_URL`)은 issue #1660 범위 밖의 기존 스크립트 특성이며, 이 스크립트를 다시 다루게 될 때 함께 정리할 수 있는 후속 개선 후보다.
