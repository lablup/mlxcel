# 기술 보고서: PR #1282 - CI에서 OpenXLA feature 조합 컴파일

## 요약

어떤 CI 잡도 XLA feature를 컴파일하지 않았고, 그 공백으로 결함 두 개가 `main`에 도달했다. self-hosted GB10 러너에서 `cuda,xla-iree`와 `xla-diagnostics`를 컴파일하는 게이트를 추가한다.

보고서를 지탱하는 판단은 둘이다. 게이트는 모든 경고가 아니라 `unused_imports`만 막는다. 현재 트리에서 측정한 결과 전면 `-D warnings`는 즉시 실패하고, **도착하자마자 빨간불인 잡은 게이트이기를 멈추기** 때문이다. 그리고 머지 전에 의도적으로 재현한 결함에서 게이트가 실패하는 것을 관측했다. 통과만 하는 잡은 무언가를 잡아낸다는 증거가 되지 않기 때문이다.

## 1. 문제

`ci.yml`은 `deny`, `fmt`, crate-version, kernel-dtype-key, MLX-pin, cross-repo-ref 잡을 돌리며 크레이트를 빌드하지 않는다. `pipeline-parallel-ci.yml`은 기본 feature로 clippy를 돌린다. `nightly-verify.yml`은 `metal,accelerate`로 clippy를 돌린다. OpenXLA serve 워커는 `#[cfg(feature = "xla-iree")]` 뒤에 있으므로 어떤 체크런도 이를 컴파일한 적이 없다.

| 결함 | 부류 | 발견 경로 |
| --- | --- | --- |
| OpenXLA 워커의 `ModelRequest::PromptCacheWarmup` 미처리 | E0004 | 무관한 리베이스 중 수작업 |
| 죽은 `load_weights_from_dir_with_filter` re-export | `unused_imports` | 같은 리베이스 중 수작업 |
| `cuda,xla-iree`에서 통합 테스트 링크 불가 | 링크 에러 | 다른 PR 검증 중 수작업 |

## 2. 기술적 판단

### 2.1 이슈가 제기한 프로비저닝 문제는 여기 존재하지 않는다

이슈는 IREE 프로비저닝을 진짜 결정 사안으로 봤다. 고정 리비전을 키로 하는 액션 캐시, 컨테이너 이미지, 또는 이미 갖춘 self-hosted 러너 중 선택이다. 조사로 정리됐다. GB10 러너가 `~/.cache/mlxcel/iree-cuda-<버전>`을 가진 바로 그 호스트이고, `scripts/iree/setup-cuda.sh`는 멱등이라 트리가 있으면 "reusing runtime build"를 로그한다. 그래서 잡은 스크립트를 호출해 프로비저닝한다. 따뜻한 러너는 비용이 없고 새 러너만 일회성 빌드를 치른다. `xla-diagnostics`는 `cuda`를 함의하므로 어차피 같은 러너를 가리킨다.

### 2.2 모든 lint가 아니라 결함을 통과시킨 lint를 막는다

선택 전에 측정했다.

```text
cargo clippy --features cuda,xla-iree --lib --tests -- -D warnings
  -> mlxcel에 닿기도 전에 mlxcel-xla에서만 에러 4건
cargo check --features cuda,xla-iree --all-targets
  -> 두 크레이트에 걸쳐 dead-code 경고 약 10건
```

즉 `-D warnings`는 빨간불로 착륙했을 것이다. 도착하자마자 빨간불인 게이트는 우회되고 곧 무시되는데, 이는 게이트가 없는 것보다 나쁘다. 다른 곳의 초록 요약에 거짓 신뢰까지 얹기 때문이다.

`unused_imports`는 죽은 re-export의 정확한 부류이고 컴파일 에러가 다른 결함을 덮으므로, 오늘 초록불인 정책으로 역사적 파손 두 건을 모두 잡는다. 정책 확대는 dead-code 백로그 정리가 필요하고, 여기에 끼워 넣지 않고 별도 작업으로 남긴다.

### 2.3 기록해둘 `$GITHUB_ENV` 형식 버그

`setup-cuda.sh --env`는 셸 `export VAR=값` 줄을 내보낸다. `$GITHUB_ENV`는 벌거벗은 `VAR=값`을 기대하므로, 출력을 그대로 붙이면 문자 그대로 `export IREE_CUDA_HOME`이라는 이름의 변수가 정의된다. 실패 양상이 명백하지 않고 오히려 오도한다. IREE 배포판을 가진 러너에서 `build.rs`가 배포판이 설정되지 않았다며 중단한다. `sed`로 접두어를 벗기고 이유를 호출 지점에 기록했다.

### 2.4 게이트가 덮지 않는 것을 워크플로 파일에 밝힌다

공백의 일부를 메우는 게이트는 공백이 닫혔다는 가정을 부른다. 이 잡을 편집할 사람이 보게 될 자리에 세 가지 제외를 적었다. 테스트 실행 없음. XLA 스위트 실행은 같은 호스트의 개발 작업과 경합하지 않는 GPU가 필요하다. 전면 경고 정책 없음. 위의 이유다. macOS와 `IREE_DIST` 빌드 없음. 어느 쪽도 러너가 없다. 세 번째가 위의 링크 실패가 여전히 안 잡히는 이유다. `cargo check`는 링크를 하지 않는다.

## 3. 변경 요약

| 파일 | 변경 |
| --- | --- |
| `.github/workflows/ci.yml` | GB10의 `xla-compile` 잡, `RUSTFLAGS: -D unused_imports`, 분리된 영속 `CARGO_TARGET_DIR`, `MLX_CUDA_ARCHITECTURES: 121`, 경로 필터에 `build.rs`와 `scripts/iree/**` 추가 |
| `tests/molmo2_xla_vision_parity.rs` | 이 저장소의 파리티 테스트가 non-diagnostics feature에서 만든 dead-code 경고 2건 정리, 게이트가 깨끗한 트리에서 출발하도록 |

## 4. 리뷰 지적사항

2.3의 `$GITHUB_ENV` 버그는 실행이 아니라 출력 형식을 읽어서 잡았다. 중요한 이유는, 그대로 뒀어도 잡은 실패했겠지만 **엉뚱한 서브시스템을 가리키는 이유로** 실패했을 것이기 때문이다.

`MLX_CUDA_ARCHITECTURES`는 자동 감지에 맡기지 않고 `121`로 고정했다. 이 호스트에서 자동 감지는 `121a`를 낸다. 둘은 여기서 수치적으로 동일하지만 릴리스는 `121`을 빌드하고, 게이트는 실제 출하되는 것을 컴파일해야 한다.

## 5. 검증

로컬에서 잡과 동일한 명령·동일한 `RUSTFLAGS`로:

```text
RUSTFLAGS="-D unused_imports" cargo check --features cuda,xla-iree --all-targets                        -> 0
RUSTFLAGS="-D unused_imports" cargo check --no-default-features --features xla-diagnostics --all-targets -> 0
```

PR에서는 통과만 관측하지 않고 세 상태를 모두 거치게 했다.

| 커밋 | 변경 | 판정 |
| --- | --- | --- |
| `763bc8a0` | 제안한 잡 | SUCCESS |
| `6f193333` | `PromptCacheWarmup` arm 제거 | FAILURE |
| `f88709bc` | 되돌림 | SUCCESS |

실패한 실행은 `error[E0004]: non-exhaustive patterns: ModelRequest::PromptCacheWarmup { .. } not covered`를 보고했다. 공백이 통과시킨 바로 그 에러다. 중간 커밋과 되돌림은 브랜치 이력에 일부러 남긴다. squash 머지에서 상쇄되고, 그때까지는 증거다.

## 6. 관련 작업

이슈 #1270이 이 PR로 닫힌다. 원래 컴파일 파손을 제목으로 한 버그로 제기됐고 그 파손의 수정으로 닫혔다가, 네 개 수용 기준 중 하나만 충족됐음이 분명해지자 다시 열리고 제목이 바뀌었다. 남은 셋이 이 잡, 문서화된 feature 매트릭스와 프로비저닝 전략, 그리고 5절의 실증이었다.

2.4에 적은 미커버 부류가 정직한 나머지다. 특히 링크 실패는 실제로 바이너리를 링크하는 잡이 필요하고, 이는 컴파일 검사와 비용 구조가 달라서 여기 끼워 넣지 않고 별도 판단을 받아야 한다.
