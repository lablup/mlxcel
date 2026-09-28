# 기술 리포트: PR #2021 - CONTRIBUTING의 PR 시점 CI 게이트 목록을 ci.yml에 맞게 수정

**작성일**: 2026-09-28
**상태**: 완료
**언어**: Markdown
**위험도**: 낮음

## 요약

`CONTRIBUTING.md`는 clippy가 "PR 시점에 게이트되지 않는다"고, PR 시점 CI가 "저렴한 게이트만" 실행한다고 안내했다: `cargo fmt`, `cargo deny`, 크레이트 버전 검사. 이 서술은 `#1285`보다 이전 상태를 가리킨다. `#1285`는 `.github/workflows/ci.yml`에 `clippy` 잡을 다시 추가하고 Rust 파일을 건드리는 PR에 게이트했으며, 그 문단이 전혀 언급하지 않은 약 열두 개의 잡이 함께 추가되었다. PR #2021은 로컬 clippy 명령의 인라인 주석과 PR 시점 CI 문단을 현재 워크플로 파일에 맞게 다시 써서, 어떤 잡이 무조건 실행되는지, 어떤 잡이 경로 필터링되는지, 어떤 잡이 공유 셀프 호스티드 `GB10` 러너에서 도는지, 어떤 잡이 권고성(advisory)에 그치는지를 나열한다.

## 1. 문제 정의

### 1.1 배경

`CONTRIBUTING.md`의 해당 문단이 마지막으로 정확했던 시점 이후 `ci.yml`은 크게 성장했다: PR 시점에 `clippy` 잡이 다시 추가되었고(`#1285`, 비용 문제로 제거되었던 `#21`/`#23` 이후), CUDA/MLX 오버레이 컴파일, WebUI 번들과 설치된 아티팩트 스모크 테스트, 고정된 MLX 커밋 파서, 대기 중인(parked) ROCm 게이트, 여러 툴체인 불필요 일관성 검사를 각각 별도 잡이 담당하게 되었다. 이런 성장은 기여자용 문서에 전혀 반영되지 않아, 그 문단을 신뢰한 기여자는 자신의 PR에서 빨간 `clippy` 체크를 보고 놀라게 된다.

### 1.2 기존 문제점

- **오래된 게이트 목록**: 문단은 세 개의 게이트(`fmt`, `deny`, 크레이트 버전 검사)만 언급했지만, 워크플로는 이제 약 스무 개의 잡을 실행하며 그중 다수가 실제로 게이트한다.
- **잘못된 clippy 서술**: 문서는 clippy가 강제되지 않는다고 했지만, `ci.yml`의 `clippy` 잡은 Rust를 건드리는 PR에서 `cargo clippy -p mlxcel --lib --tests -- -D warnings`를 강제한다.
- **경로 조건부·권고성 잡에 대한 가시성 부재**: 이전 문단을 읽은 기여자는 자신의 diff에 실제로 어떤 잡이 실행될지, 일부 잡(`cross-repo-refs`, `rocm-ci-status`, `self-hosted-gate-advisory`)이 PR을 절대 실패시키지 않고 경고만 남긴다는 사실을 알 방법이 없었다.

### 1.3 위험성 평가

낮음. 이 변경은 문서 전용이며 소스, 빌드 스크립트, 워크플로 파일은 전혀 건드리지 않고 어떤 게이트의 실제 동작도 바뀌지 않는다. 위험은 오로지 이 서술이 다시 틀려지는 것뿐이며, 그래서 실제 워크플로 파일을 기준으로 검증하고 전체 리뷰 사이클을 거쳤다.

## 2. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 1 |
| 추가된 줄 | 2 |
| 삭제된 줄 | 2 |

- `CONTRIBUTING.md:46`: `cargo clippy --workspace --all-targets --features metal,accelerate -- -D warnings`의 인라인 주석이, CI에서는 더 좁은 `-p mlxcel --lib --tests` 슬라이스만 게이트되며 전체 실행은 여전히 기여자 본인의 몫이라고 말하도록 바뀌었다.
- `CONTRIBUTING.md:50`: PR 시점 CI 문단을 현재 `ci.yml` 기준으로 다시 열거했다: 경로 필터를 담당하는 `changes` 잡; 무조건 모든 PR을 게이트하는 툴체인 불필요 검사 일곱 개(`crate-versions`, `kernel-dtype-keys`, `binary-assets`, `cuda-arch-lists`, `license-headers`, `llama-compat-manifest`, `webui-contract`); Rust 변경에 게이트되는 `fmt`/`deny`/`clippy`; 각자 자신의 경로 필터에 게이트되는 추가 `GB10` 잡들(`xla-compile`은 Rust 변경, `xla-link`는 IREE 링크 레시피, `cuda-sm70-compile`/`cuda-blockfloat`는 CUDA 아키텍처 경로, `webui-installed-artifact`는 Rust 또는 WebUI 변경); 호스티드 `webui-bundle`과 `mlx-pin`의 경로 필터; 대기 중인 `rocm-build` 게이트와 그 권고성 `rocm-ci-status`; `workflow-lint`와 권고성 `self-hosted-gate-advisory`; 항상 실행되지만 권고성인 `cross-repo-refs`. 기존 `pipeline-parallel-ci.yml`, `nightly-verify.yml` 언급은 그대로 유지하고 여전히 정확한지 재검증했다.
- 마지막 문장은 원래 "CUDA verification is not gated at PR time either; that stays exclusive to `release.yml`"였으나 수정했다: `release.yml`에는 `cargo test` 단계가 없으므로, 전수 CUDA·ROCm 테스트 스위트는 `release.yml`이 아니라 로컬의 `make verify-test-cuda`와 `make verify-rocm`에 의해서만 게이트된다.

## 3. 기술적 선택과 그 이유

### 3.1 이슈 자체의 목록이 아니라 실제 워크플로 파일을 기준으로 검증

**컨텍스트**: 이슈 #1702의 "Implementation Notes"는 이미 여러 잡(`crate-versions`, `kernel-dtype-keys`, `llama-compat-manifest`, `mlx-pin`, `cross-repo-refs`, `xla-compile`, `xla-link`, `cuda-sm70-compile`)을 열거했지만, 이슈는 2026-09-08에 등록되었고 그 이후 최근 커밋들이 CI 게이팅을 바꿨다(GB10 러너 범위 조정, 새 ROCm 게이트).

**선택 이유**: 작업 지시 자체가 이슈의 잡 목록이 오래되었을 수 있다고 명시했다. 모든 잡 이름, `if:` 조건, `runs-on` 값, 경로 필터를 `changes` 잡의 `dorny/paths-filter` 정의까지 포함해 현재 `main`의 `.github/workflows/ci.yml`에서 직접 다시 도출했으며, 이슈 본문을 그대로 믿지 않았다.

**결과**: 그럼에도 첫 초안은 직접 검증이 아니라 유추로 한 가지를 틀렸다: `cuda-sm70-compile`과 `cuda-blockfloat`를 `clippy`, `xla-compile`과 함께 "Rust 경로 게이트" 잡으로 묶었다. `pr-reviewer`가 워크플로 파일을 대조해 이를 잡아냈다: 두 잡은 실제로는 `cuda_arch` 필터(`src/lib/mlx-cpp/**`에서 ROCm·Metal 서브트리를 제외하고 MLX 빌드 스크립트를 더한 것)에 게이트되며, 이 필터에는 `*.rs` 글롭이 전혀 없어 일반 Rust 소스 변경으로는 트리거되지 않는다. PR 본문을 최종 상태에 맞게 갱신하기 전, 후속 커밋에서 이를 수정했다.

### 3.2 요약이 아니라 열거

**컨텍스트**: "CI는 이제 이전보다 훨씬 많은 게이트를 실행하며, 그중에는 Rust로 트리거되는 clippy 잡도 포함된다"처럼 각 잡을 나열하지 않는 더 짧은 재작성도 가능했다.

**선택 이유**: 이 문단은 요약이 조용히 낡으면서 한 번 이미 썩은 전력이 있다. 각 잡을 실제 식별자로 명시하면 향후 드리프트(잡 이름 변경, 경로 필터 변경)가 `ci.yml`에 대한 grep만으로 탐지 가능해지고, 잡과 `#N` 이슈 참조를 본문에 직접 이름으로 언급하는 파일 자체의 기존 스타일과도 맞는다.

**트레이드오프**: 다시 쓴 문단은 원문보다 약 2.5배 길어졌다(하나의 물리적 줄로서 1104자에서 약 2530자로, 파일의 no-hard-wrap 관례는 그대로 유지). 리뷰 과정에서 간결함과 이 선택을 저울질했고, 열거된 각 잡이 기여자가 실행에 옮길 수 있는 사실(자신의 diff에서 어떤 체크가 돌지)이라는 점에서 장식이 아니라고 판단해 유지했다.

## 4. 검증

- `python3 scripts/ci/check_cross_repo_refs.py`: 통과; `#1285`는 이 환경에 `GH_TOKEN`이 없어 수동 확인 대상으로 표시되었으며, `gh pr view 1285`로 독립적으로 실재함을 확인했다(머지됨, 제목 "fix(lint): clear the err_expect on main and gate clippy at PR time").
- `python3 "$HOME/.claude/skills/commit-conventions/scripts/validate_body.py" CONTRIBUTING.md`: 새로운 위반 없음; 수정된 두 줄 모두 하나의 물리적 줄로 유지된다. 파일의 다른 곳(109-121행)에 있는 기존 하드랩 위반은 이 변경과 무관하며 범위 밖이다.
- `gh api repos/lablup/mlxcel/actions/variables/ROCM_CI_ENABLED`는 404를, `gh api repos/lablup/mlxcel/actions/runners`는 러너 0개를 반환하여 "대기 중" ROCm 게이트 서술을 확인했다.
- 전체 리뷰 사이클: `pr-reviewer`가 HIGH 등급 발견 하나(위의 `cuda-sm70-compile`/`cuda-blockfloat` 트리거)를 찾아 수정했고, `pr-security-checker`는 발견 사항 없이 통과했다(순수 산문 diff, 비밀정보·링크·게이트 완화 주장 없음), `pr-finalizer`는 `CONTRIBUTING.md`의 비영어 대응 파일이 없고 다른 어떤 문서도 오래된 주장을 반복하지 않음을 확인했다.
- Rust 빌드나 테스트 스위트는 해당하지 않는다; PR 자체의 CI 실행은 모든 툴체인 불필요 검사가 통과하고 모든 Rust/WebUI/CUDA/ROCm 잡이 `SKIPPED`로 표시되는 것을 보여주며, 이는 Rust를 건드리지 않는 PR에 대해 수정된 문서가 이제 예측하는 바와 일치한다.

## 5. 관련 작업

- 이슈 #1702: 이 변경의 근거가 된 이슈.
- PR #1285: PR 시점 CI에 `clippy` 잡을 다시 추가한 PR로, 이번 문서 업데이트가 반영하는 사건.
- 이슈 #21 / #23: PR 시점 CI에서 clippy와 테스트 스위트를 원래 제거했던 사건으로, 수정된 문단에서 역사적 맥락으로 참조한다.
- 이슈 #1992: 나열된 여러 잡이 의존하는 `changes` 잡의 `predicate-quantifier: some-with-excludes` 경로 필터 동작을 도입한 이슈.
