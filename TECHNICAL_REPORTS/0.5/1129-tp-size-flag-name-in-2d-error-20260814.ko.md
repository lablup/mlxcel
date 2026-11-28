# 기술 보고서: PR #1129 - fix(cli): name the real --tp-size flag in the 2D-parallelism error

**작성일**: 2026-08-14
**상태**: 완료
**언어**: Rust
**위험도**: Low

---

## 요약

PR #1129는 이슈 #1112를 해결한다. `validate_pipeline_parallel_args`의 2D(파이프라인 x 텐서) 병렬 가드가 담고 있던 메시지는 mlxcel 바이너리가 받지 않는 이름인 `--tensor-parallel-size`를 가리켰다. 이제 실제 플래그인 `--tp-size`를 가리킨다.

리뷰에서 나온 두 가지 발견이 이 PR의 형태를 바꿨고, 둘 다 원래 결함보다 중요하다.

**그 가드는 도달할 수 없다.** `ensure!` 조건이 여덟 줄 위 조기 반환 조건의 정확한 부정이라서 실패할 수 없고, 메시지는 출력될 수 없다. 잘못된 플래그 이름을 본 사용자는 없었다. 사용자에게 보이는 버그가 아니라 잠재된 텍스트 결함이고, 그래서 CHANGELOG 항목 없이 나간다.

**이슈가 존재하지 않는다고 한 `docs/en/` 경로는 실재한다.** `mkdocs.yml`은 `docs_dir: docs/en`을 설정하고 `nav:`에 주석이 인용한 두 파일을 모두 나열한다. 별도 문서 저장소에서 소스를 관리하는 발행 운영 매뉴얼의 페이지들이고, `docs/README.md`는 이 분리를 "의도이지 drift가 아니다"라고 명시한다. 이 PR의 중간 초안은 이슈의 잘못된 전제를 따라 그 참조를 지웠는데, 이는 PR #1122가 위험 요소로 기록해 둔 바로 그 실패 양상이다. 지금은 삭제 대신 유지하고 설명한다.

---

## 1. 문제 정의

### 1.1 배경

`src/commands/generate.rs`의 `validate_pipeline_parallel_args`는 2D 조합을 검사한다. 구조는 이렇다.

```rust
if pp.pp_layers.is_none() && pp.pp_size <= 1 {
    return Ok(());
}
// ...
if tp_size > 1 {
    ensure!(
        pp.pp_size >= 2 || pp.pp_layers.is_some(),
        "2D parallelism requires --pp-size >= 2 (or an explicit --pp-layers spec) \
         alongside --tensor-parallel-size > 1"
    );
```

텐서 병렬 랭크 수는 `--tp-size`이고 `src/main.rs`와 `src/bin/mlx_server.rs`에 정의되어 있다. `--tensor-parallel-size`는 clap 플래그로도 별칭으로도 존재하지 않는다.

### 1.2 기존 문제점

- **메시지가 바이너리가 거부하는 플래그를 지목했다.** 같은 문장의 `--pp-size`와 `--pp-layers`는 둘 다 실재하고, 그 패턴이 세 번째 이름을 믿게 만든다.
- **가드가 발동할 수 없다.** 조기 반환을 지난 시점에는 `pp_layers.is_some() || pp_size >= 2`가 구조적으로 성립하고, 그것이 정확히 `ensure!` 조건이다. 저장소의 단위 테스트가 그 결과를 기록하고 있다. `validate_pipeline_parallel_args_rejects_2d_without_pp_enabled`(`src/commands/generate_tests.rs`)는 `pp_size = 1, tp_size = 2`를 두고 `is_ok()`를 단언하며, 주석은 검증기가 조기 반환한다고 적어 두었다. 이름은 "rejects"인데 단언은 반대다.
- **통합 테스트가 같은 플래그를 넘기면서도 통과했다.** `tests/pp_tp_2d_real_models.rs`의 두 테스트 모두 `--tensor-parallel-size`를 사용했다. 패리티 테스트는 `#[ignore]`다. `pp_tp_2d_validator_accepts_combination`은 아니었고, 공허하게 통과했다. 유일한 단언이 특정 옛 거부 문자열의 부재였는데, clap이 인자 파싱에서 죽인 프로세스에 대해 그 조건은 언제나 참이다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 향후 변경으로 가드가 도달 가능해지면서 잘못된 플래그 이름이 사용자에게 노출 | Medium | Medium |
| 2D 검증기 변경이 검증기에 닿지도 않는 테스트로 "검증"됨 | High | Medium |
| 누군가 교차 트리 매뉴얼 참조를 끊어진 링크로 보고 삭제 | Medium | Medium |

---

## 2. 기술 검토

### 2.1 메시지 변경의 범위

`ensure!` 조건, `tp_size > 1` 분기, `total_ranks` 정합성 검사, 함수의 나머지 모든 갈래가 바이트 단위로 동일하다. 바뀐 것은 문자열 리터럴뿐이며 이슈의 네 번째 수용 기준을 그대로 만족한다.

### 2.2 전수 확인

`src/commands/generate.rs`에 등장하는 모든 `--flag` 토큰을 눈으로 읽는 대신 빌드된 바이너리가 광고하는 롱 플래그 목록과 대조했다.

```
$ grep -o -- '--[a-z][a-z0-9-]*' src/commands/generate.rs | sort -u \
    | comm -23 - <(mlxcel generate --help | grep -o -- '--[a-z][a-z0-9-]*' | sort -u)
--tensor-parallel-size
```

광고된 64개 중 한 줄만 남았다. `--pp-micro-batch-size`, `--pp-size`, `--pp-layers`, `--estimate-memory`, `--no-memory-check`, `--max-tokens`, `--recommend-quant`, `--surgery`가 모두 해석된다.

### 2.3 통합 테스트의 범위를 파서로 좁힌 이유

`run_generate`는 검증기보다 **먼저** `-m`을 해석한다. 검증기가 해석된 모델 디렉터리를 읽기 때문이다.

```rust
args.model.model =
    resolve_model_source_with_override(&args.model.model, args.model.models_dir.as_deref())?;

validate_tensor_parallel_args(&args)?;
validate_pipeline_parallel_args(&args)?;
```

따라서 서브프로세스 호출은 디스크에 실제 모델 없이는 `validate_pipeline_parallel_args`에 닿을 수 없다. 더 나쁜 점은 테스트가 쓰던 값 `nonexistent-model-path-for-validator-only-check`가 유효한 bare repo 세그먼트라서, 해석기가 `$MLXCEL_DEFAULT_ORG`로 확장해 네트워크로 나간다는 것이다. 테스트의 argv를 그대로 실행한 결과다.

```
[mlxcel] 'nonexistent-model-path-for-validator-only-check' -> mlx-community/nonexistent-...
[mlxcel] model '...' not found locally; downloading into the mlxcel store...
Error: failed to download model '...': authentication failed (HTTP 401).
```

두 검증기에 닿기 전에 exit 1이다. 이 PR의 중간 초안은 플래그 이름을 고치고 파싱 거부 가드를 추가하면서 이 argv를 그대로 뒀는데, 그렇게 되면 ignore가 아닌 CI 테스트가 매 실행마다 HuggingFace로 요청을 보낸다. 오프라인 러너에서는 실패가 아니라 타임아웃으로 이어진다. 그 초안은 틀렸고 최종본이 아니다.

최종 테스트는 `--help`를 붙여 파싱이 성공하는 즉시 clap이 종료하게 한다. hermetic하고(런타임 초기화 없음, 네트워크 없음) 종료 코드에 대해 **긍정** 단언을 하므로 일찍 죽어서 통과할 수 없다.

- `--pp-size 2 --tp-size 2 --help`는 exit 0.
- `--pp-size 2 --tensor-parallel-size 2 --help`는 clap 사용법 오류로 exit 2.

범위를 좁혀도 잃는 것은 없다. 검증기 자체는 `src/commands/generate_tests.rs`의 `validate_pipeline_parallel_args_accepts_2d_pp_tp`가 직접 덮는다. 그리고 이 변경은 테스트 주석이 원래부터 주장하던 내용("`mlxcel generate --help`에 2D 플래그를 붙여 호출한다")을 되살린다. 코드가 자기 주석에서 멀어져 있었다.

### 2.4 `docs/en/` 참조는 끊어져 있지 않았다

이슈는 `docs/en/`이 "이 저장소에 있던 적이 없다"고 단언하고, "`src/`에 `docs/en/` 참조가 남지 않을 것"을 수용 기준으로 삼았다. 앞 절반은 이 git 트리에 대해 참이지만 거기서 끌어낸 결론은 아니다.

```
mkdocs.yml:8:docs_dir: docs/en
mkdocs.yml:160:      - Tensor Parallelism: distributed/tensor-parallelism.md
mkdocs.yml:161:      - Pipeline Parallelism: distributed/pipeline-parallelism.md
```

이 두 nav 항목은 `docs_dir: docs/en` 아래에서 주석이 인용한 두 경로로 정확히 해석된다. `docs/README.md`는 `docs/en`, `docs/ko`, `docs/shared`가 별도 문서 저장소에서 관리되고 루트 mkdocs 설정들이 그 트리를 향한 경로를 담는다고 적으며, 그대로 옮기면 "의도이지 drift가 아니다"라고 한다. `Makefile`의 `docs-guard` 타깃이 같은 사실 위에 서 있고, 이를 추가한 PR #1122의 보고서는 "누군가 dangling nav를 `docs/*.md`로 돌려 고쳐서 소유 트리를 망가뜨림"을 영향도 High 위험으로 나열한다.

즉 주석은 실재하는 매뉴얼 페이지를 가리키고 있었다. 따라서 그 수용 기준은 의도적으로 문자 그대로 지키지 않으며, 조용히 충족시키는 대신 이슈와 PR 설명에 사유를 남긴다.

### 2.5 호환성

동작 변경이 없다. 검증기가 받아들이고 거부하는 입력 집합이 동일하고, 문구가 바뀐 그 메시지는 애초에 출력될 수 없다. CLI 표면, 직렬화, API, 의존성 변경이 모두 없다.

---

## 3. 기술적 결정

### 3.1 CHANGELOG 항목을 넣지 않는다

중간 초안은 사용자가 메시지를 따랐다가 `error: unexpected argument '--tensor-parallel-size' found`를 만난다고 적은 항목을 추가했다. 그런 일은 일어날 수 없다. 파이프라인 토폴로지 없는 `--tp-size > 1` 실행은 조기 반환에서 `Ok(())`가 되고 가드에 닿지 않는다. 변경 이력은 사용자에게 보이는 변화를 위한 것이고 여기에는 그런 변화가 없으므로, 기술적으로만 참이고 무의미한 문장으로 고쳐 쓰는 대신 항목을 제거했다.

### 3.2 매뉴얼 참조는 유지하고 체크아웃 내 문서를 덧붙인다

| 선택지 | 장점 | 단점 |
|---|---|---|
| `docs/en/` 경로 두 개 삭제 | 이슈 기준을 문자 그대로 충족 | 잘못된 전제로 정식 운영 매뉴얼을 향한 살아 있는 포인터를 삭제 |
| `docs/distributed.md`로만 재지정 | 짧고 모든 경로가 트리 안에 있음 | 같은 정보 손실이고, `docs/distributed.md`에는 2D 절이 없어 대체가 되지 않음 |
| **선택: 매뉴얼 페이지 둘을 유지하고, `docs/distributed.md`를 체크아웃 내 요약으로 병기하며, 소스가 여기 없는 이유를 밝힘** | 잃는 것이 없고, 다음 독자에게 교차 트리 경로가 설계임을 알림 | `src/`에 `docs/en` 문자열이 남아 이슈 기준에 어긋남 |

주석이 분리 구조를 설명하므로 같은 종류의 점검이 이것을 다시 끊어진 링크로 발견하지 않는다.

### 3.3 메시지만이 아니라 테스트까지 고친다

이 테스트는 이슈와 무관한 부수물이 아니다. 잘못된 플래그 이름을 잡았어야 하는데 잡을 수 없던 산출물이고, 실패 원인도 동일하다. 메시지를 고치면서 메시지가 다시 틀리는 것을 감지하지 못하는 테스트를 남기면 이슈는 닫히지만 구멍은 닫히지 않는다.

### 3.4 도달 불가 가드는 건드리지 않는다

죽은 `ensure!`는 실재하는 결함이지만, 제거하거나 조기 반환 조건을 넓히는 것은 로직 변경이고 "메시지 텍스트만, 검증 로직은 불변"이 이 이슈의 명시적 수용 기준이다. 같은 항등식을 기록하고 있는 잘못 명명된 단위 테스트와 함께 후속 과제로 넘긴다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 2 |
| 동작 변경 | 0 |
| 사용자 가시 변경 | 0 |

### 영역별 변경

| 영역 | 파일 | 요약 |
|---|---|---|
| CLI | `src/commands/generate.rs` | `ensure!` 메시지가 `--tp-size`를 가리킨다. 검증기 주석은 운영 매뉴얼 페이지 둘을 유지하고, 그 소스가 별도 문서 저장소에 있는 것이 설계임을 밝히며, 체크아웃 내 요약으로 `docs/distributed.md`를 병기한다 |
| 테스트 | `tests/pp_tp_2d_real_models.rs` | 패리티 테스트는 `--tp-size` 사용. ignore가 아닌 테스트는 인자 파서로 범위를 좁히고 `--help`로 hermetic하게 만들었으며 종료 코드에 긍정 단언을 한다. 모듈 문서 줄바꿈 정리 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `1593eb8a` | fix | fix(cli): name the real --tp-size flag in the 2D-parallelism error |

---

## 5. 검증 및 후속 과제

### 통과

- `MLX_CUDA_ARCHITECTURES=121 cargo test --profile test-fast --features cuda --test pp_tp_2d_real_models`: 통과, 1 ignored.
- hermetic: ignore가 아닌 테스트는 네트워크 요청을 하지 않는다. argv가 clap 안에서 종료된다.
- 공허하지 않음. 빌드된 바이너리로 양방향 확인했다. `--pp-size 2 --tp-size 2 --help`는 exit 0이고, `--tensor-parallel-size`로 바꾸면 `error: unexpected argument '--tensor-parallel-size' found`와 함께 exit 2다.
- `grep -rn -- "--tensor-parallel-size" src/` 비어 있음.
- `cargo fmt --all -- --check` 통과.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings` 통과.

### 후속 후보

- **도달 불가 가드.** `tp_size > 1` 분기 첫머리의 `ensure!`는 실패할 수 없고, `validate_pipeline_parallel_args_rejects_2d_without_pp_enabled`는 거부한다는 이름 아래 `is_ok()`를 단언한다. 조기 반환이 너무 넓은 것인지 가드가 잉여인지 결정하는 일은 로직 변경이라 여기서는 범위 밖이다.
- **플래그 이름 기계 검증.** 2.2절의 전수 확인은 셸 파이프라인으로만 존재한다. `tests/cli_help_consistency.rs`가 CLI 표면 불변식의 기존 거처이고, 에러 메시지 리터럴에 등장하는 모든 `--flag`가 `--help`에 있는지 단언하는 테스트는 이 사례가 아니라 이 부류를 닫는다.
- **공허한 부정 단언.** 프로세스 출력에 특정 문자열이 없다는 것만 단언하는 테스트는 무관한 이유로 프로세스가 일찍 죽어도 통과한다. `tests/` 전체를 이 형태로 훑는 작업은 독립적이다.
- **2D 조합이 문서화되어 있지 않다.** `docs/distributed.md`에도, nav 기준으로 매뉴얼에도 PP x TP 절이 없다. 그래서 주석이 스스로를 한정해야 한다.
