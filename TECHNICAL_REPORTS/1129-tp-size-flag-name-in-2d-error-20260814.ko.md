# 기술 보고서: PR #1129 - fix(cli): name the real --tp-size flag in the 2D-parallelism error

**작성일**: 2026-08-14
**상태**: 완료
**언어**: Rust, Markdown
**위험도**: Low

---

## 요약

PR #1129는 이슈 #1112를 해결한다. 2D(파이프라인 x 텐서) 병렬 검증기는 토폴로지가 덜 지정된 실행을 거부하면서 `--tensor-parallel-size > 1`을 넘기라고 안내했다. mlxcel 바이너리는 그 이름을 받아 본 적이 없다. 안내를 그대로 따른 사용자가 받는 답은 `error: unexpected argument '--tensor-parallel-size' found`였다. 이제 메시지는 실제 플래그인 `--tp-size`를 가리킨다.

무게가 실리는 발견은 메시지 자체가 아니다. 같은 오타가 `tests/pp_tp_2d_real_models.rs`에도 있었고, `#[ignore]`가 붙지 않아 실제로 도는 테스트인 `pp_tp_2d_validator_accepts_combination`이 **공허하게 통과**하고 있었다. clap이 인자 파싱 단계에서 플래그를 거부하니 프로세스는 검증기에 닿기 전에 죽었고, 테스트의 유일한 단언(특정 옛 거부 문자열이 출력에 없을 것)은 자동으로 성립했다. 이 코드 경로를 지키라고 만든 테스트가 바로 그 결함을 잡을 수 없는 상태였고, 초록불은 아무것도 보증하지 않았다.

---

## 1. 문제 정의

### 1.1 배경

`src/commands/generate.rs`의 `validate_pipeline_parallel_args`는 2D 조합을 검사한다. `tp_size > 1`인 텐서 병렬 실행은 `--pp-size >= 2`이거나 명시적인 `--pp-layers` 명세로 파이프라인 토폴로지를 함께 가져야 한다. 검사 자체는 옳다. 메시지가 틀렸다.

```rust
"2D parallelism requires --pp-size >= 2 (or an explicit --pp-layers spec) \
 alongside --tensor-parallel-size > 1"
```

텐서 병렬 랭크 수는 `--tp-size`이고 `src/main.rs`와 `src/bin/mlx_server.rs`에 정의되어 있다. `--tensor-parallel-size`는 clap 플래그로도 별칭으로도 존재하지 않는다.

### 1.2 기존 문제점

- **하나만 틀린 이름이 모호한 메시지보다 더 해로웠다.** 같은 문장의 `--pp-size`와 `--pp-layers`는 둘 다 실재한다. 메시지를 확인해 본 독자는 셋 중 둘이 맞다는 것을 발견하게 되고, 그 패턴이 나머지 하나를 믿게 만든다.
- **테스트 표면이 같은 오류를 재현하면서 그것을 가렸다.** `tests/pp_tp_2d_real_models.rs`의 두 테스트 모두 명령줄에 `--tensor-parallel-size`를 넘겼다. 패리티 테스트는 `#[ignore]`이고 실제 가중치가 필요해 돌지 않는다. `pp_tp_2d_validator_accepts_combination`은 ignore가 아니어서 돌았고, 잘못된 이유로 통과했다. 이 테스트는 stdout과 stderr에 `pipeline parallelism does not support tensor parallelism` 문자열이 없다는 것만 단언하는데, clap 안에서 죽은 프로세스에 대해 그 조건은 언제나 참이다.
- **검증기 위의 주석이 존재한 적 없는 파일을 가리켰다.** `docs/en/distributed/pipeline-parallelism.md`와 `docs/en/distributed/tensor-parallelism.md`를 인용했다. `git log --all -- docs/en`은 비어 있고, 이 디렉터리는 저장소에 있던 적이 없다. `src/` 전체에서 `docs/en/` 참조는 이 둘뿐이었다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 사용자가 에러 메시지를 따랐다가 무관한 두 번째 에러를 만남 | Medium | High |
| 향후 2D 검증기 변경이 검증기에 닿지도 않는 테스트로 "검증"됨 | High | Medium |
| 독자가 주석을 따라 존재하지 않는 운영 문서를 찾아감 | Low | Medium |

---

## 2. 기술 검토

### 2.1 메시지 변경의 범위

`ensure!` 조건, 이를 감싼 `tp_size > 1` 분기, `total_ranks` 정합성 검사, `validate_pipeline_parallel_args`의 나머지 모든 갈래가 바이트 단위로 동일하다. 바뀐 것은 문자열 리터럴뿐이며, 이슈의 네 번째 수용 기준을 그대로 만족한다.

### 2.2 전수 확인

이슈는 이름 하나가 틀렸다면 나머지도 확인해 볼 만하다는 근거로 인접 메시지 점검을 요청했다. 눈으로 읽는 것보다 바이너리에 직접 묻는 쪽이 강한 증거다. `src/commands/generate.rs`에 등장하는 모든 `--flag` 토큰을 빌드된 바이너리가 실제로 광고하는 롱 플래그 목록과 대조했다.

```
$ grep -o -- '--[a-z][a-z0-9-]*' src/commands/generate.rs | sort -u \
    | comm -23 - <(mlxcel generate --help | grep -o -- '--[a-z][a-z0-9-]*' | sort -u)
--tensor-parallel-size
```

한 줄만 남았다. `--pp-micro-batch-size`, `--pp-size`, `--pp-layers`, `--estimate-memory`, `--no-memory-check`, `--max-tokens`, `--recommend-quant`, `--surgery`를 포함해 광고된 64개 플래그가 모두 해석된다. 표본 점검이 아니라 이 파일에 대한 전수 확인이다.

### 2.3 공허한 테스트

이 실패 양상은 반복되는 형태라 정확히 적어 둘 값어치가 있다. 테스트가 단언하는 것은 **부정문**, 즉 특정 거부 문자열이 나타나지 않는다는 것이다. 프로세스 출력에 대한 부정 단언은 다른 출력을 내는 모든 경로에서 성립하고, 여기에는 대상 코드에 아예 닿지 않는 경로가 전부 포함된다. 명령줄에 플래그 이름 결함이 들어가면서 프로세스가 죽는 지점이 검증기에서 인자 파서로 옮겨 갔고, 단언은 그 차이를 알아채지 못했다.

수정 방향은 전제를 명시하는 것이다. 기존 검사보다 앞에서, 인자가 파싱 단계에서 거부되지 않았음을 단언한다.

```rust
assert!(
    !stderr.contains("unexpected argument") && !stderr.contains("unrecognized"),
    "the 2D flags were rejected by the argument parser, so the validator \
     was never reached:\nstdout={stdout}\nstderr={stderr}"
);
```

이 문자열은 가정이 아니라 실제 바이너리로 확인했다. clap은 `error: unexpected argument '--tensor-parallel-size' found`를 출력하므로 가드의 부분 문자열이 관측된 텍스트와 일치한다. `unrecognized`는 방어적으로 함께 둔 두 번째 표기다.

### 2.4 호환성

이전에 동작하던 명령줄의 동작은 하나도 바뀌지 않는다. 검증기가 받아들이고 거부하는 입력 집합이 동일하고, 거부 하나의 문구만 다르다. CLI 표면 변경, 직렬화 변경, API 변경이 모두 없다.

---

## 3. 기술적 결정

### 3.1 메시지만이 아니라 테스트까지 고친다

| 선택지 | 장점 | 단점 |
|---|---|---|
| `ensure!` 문자열만 수정 | diff가 가장 작다. 이슈의 기준이 `src/` 범위이므로 문자 그대로는 충족된다 | 바이너리가 거부하는 플래그를 넘기는 비-ignore 테스트가 남고, 공허한 통과가 다음 사람에게 그대로 넘어간다 |
| **선택: 메시지, 테스트의 플래그, 테스트의 단언을 함께 수정** | 테스트가 이름 붙인 코드 경로를 실제로 실행한다. 같은 결함이 재유입되면 이제 실패한다 | 이슈의 문자적 범위보다 diff가 약간 넓다 |

이 테스트는 이슈와 무관한 부수물이 아니다. 결함을 잡았어야 하는데 잡지 못한 산출물이고, 실패 원인도 동일하다. 메시지를 고치면서 메시지가 다시 틀리는 것을 감지하지 못하는 테스트를 그대로 두면, 이슈는 닫히지만 구멍은 닫히지 않는다.

### 3.2 주석은 삭제가 아니라 재지정

이슈는 두 가지를 모두 허용했다. `docs/distributed.md`로 재지정하거나, 그 문서가 2D 조합을 다루지 않으면 참조를 삭제하는 것이다. 판단을 위해 `docs/distributed.md`를 읽었다. 텐서 병렬과 파이프라인 병렬을 각각 별도 절로 문서화하고 `--tp-size`, `--pp-size`, `--pp-layers`, `--pp-micro-batch-size`가 모두 등장하지만, 둘을 조합하는 절은 없다.

삭제하면 쓸모 있는 포인터를 잃고, 조용히 재지정하면 문서를 과장하게 된다. 주석은 이제 `docs/distributed.md`를 가리키면서 2D 조합은 아직 그 문서에 정리되어 있지 않다고 밝힌다. 양쪽 모두에 대해 정확한 서술이다.

### 3.3 옛 플래그 이름을 트리에 남기지 않는다

새 가드의 주석은 존재 이유를 설명하며 옛 표기를 언급한다. 문자열을 그대로 인용하는 대신 "바이너리가 받아 본 적 없는 텐서 병렬 플래그 표기"로 적었다. 덕분에 `grep -rn -- "--tensor-parallel-size"`는 `src/` 아래뿐 아니라 저장소 전체에서 비어 있다. 같은 종류의 점검을 다시 돌려도 이것이 살아 있는 히트로 재발견되지 않는다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경 파일 | 3 |
| 추가 줄 | +27 |
| 삭제 줄 | -14 |
| 동작 변경 | 0 |

### 영역별 변경

| 영역 | 파일 | 요약 |
|---|---|---|
| CLI | `src/commands/generate.rs` | `ensure!` 메시지가 `--tp-size`를 가리킨다. 검증기 주석을 존재하지 않는 `docs/en/` 경로 두 개에서 `docs/distributed.md`로 재지정하고 2D 수록 범위를 정확히 명시 |
| 테스트 | `tests/pp_tp_2d_real_models.rs` | 두 테스트 모두 `--tp-size` 사용. `pp_tp_2d_validator_accepts_combination`에 파싱 거부 가드를 추가해 검증기에 닿지 않고는 통과할 수 없게 함. 낡은 줄 번호 참조를 함수 이름으로 교체 |
| 문서 | `CHANGELOG.md` | `## [Unreleased]` / `### Fixed` 항목 추가 |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|---|---|---|
| `1593eb8a` | fix | fix(cli): name the real --tp-size flag in the 2D-parallelism error |

---

## 5. 검증 및 후속 과제

### 통과

- `MLX_CUDA_ARCHITECTURES=121 cargo test --profile test-fast --features cuda --test pp_tp_2d_real_models`: 1 passed, 0 failed, 1 ignored.
- 이슈가 예측한 대로 옛 표기는 거부된다. `mlxcel generate -m /nonexistent -p x -n 1 --pp-size 2 --tensor-parallel-size 2`는 `error: unexpected argument '--tensor-parallel-size' found`를 출력한다.
- 새 표기는 통과한다. 같은 명령을 `--tp-size 2`로 바꾸면 파싱을 지나 모델 해석 단계까지 가고, 일부러 존재하지 않게 준 경로에서 실패한다.
- `grep -rn -- "--tensor-parallel-size" src/` 비어 있음. `grep -rn "docs/en/" src/` 비어 있음.
- `cargo fmt --all -- --check` 통과.
- `cargo clippy --profile test-fast --features cuda --lib --tests -- -D warnings` 통과.

### 후속 후보

- 공허한 부정 단언이라는 형태는 이 테스트만의 문제가 아니다. 프로세스 출력에 특정 문자열이 없다는 것만 단언하는 테스트는, 무관한 이유로 프로세스가 일찍 죽어도 통과한다. `tests/` 전체에서 이 패턴을 훑는 작업은 독립적인 후속 과제가 된다.
- `docs/distributed.md`에 2D(PP x TP) 조합 절이 없어서 재지정한 주석이 스스로를 한정해야 한다. 그 절을 쓰면 한정 문구를 걷어낼 수 있다.
