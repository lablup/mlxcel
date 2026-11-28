# 기술 보고서: PR #1118 - docs: explain CLI DRY runs without sequence breakers

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust (주석과 clap help만)
**위험도**: Low

---

## 요약

PR #1118은 이슈 #1108을 해결한다. CLI는 `--dry-multiplier`를 노출하므로 명령행에서 DRY를 켤 수 있지만, `resolved_cli_sampling_params`는 `dry_sequence_breakers: Vec::new()`를 하드코딩한다. breaker가 없으면 역방향 매칭이 개행이나 구두점 경계에서 멈추지 않아 `match_len`이 계속 자라고, 같은 수치를 서버에 넣었을 때보다 페널티가 강해진다. 코드에도 help 문구에도 그 사실이 적혀 있지 않았다.

이 PR은 이슈가 조사 끝에 도달한 문서 수정이며, 이슈가 처음 제안했던 기능 추가가 아니다. 추가된 13줄은 전부 주석 또는 doc comment다. `--dry-sequence-breakers` CLI 플래그는 추가하지 않았고 동작 변경도 없다.

---

## 1. 문제 정의

### 1.1 배경

이슈 #1108은 플래그 parity 요청으로 시작했다. 서버에는 `--dry-sequence-breakers`가 있고 CLI에는 없으니 추가하자는 것이었다. 이슈에 기록된 조사가 그 프레이밍을 폐기했다. CLI의 `SamplingOptions`를 서버의 요청별 sampling 필드와 대조하면 CLI가 빠뜨린 knob은 하나가 아니라 아홉 개다. `frequency_penalty`, `presence_penalty`, `xtc_probability`, `xtc_threshold`, `logit_bias`, `repetition_context_size`, `stop`, 루프 탐지 필드들이 여기 포함된다. 즉 CLI의 좁은 sampling 표면은 의도된 것이고, "서버에 있는 플래그가 CLI에 없다"는 사실만으로는 아무것도 논증하지 못한다.

### 1.2 기존 문제점

- **하드코딩된 다섯 필드 중 하나만 성격이 다르다.** `resolved_cli_sampling_params`는 sampling 필드 다섯 개를 연속된 블록으로 하드코딩한다. 그중 넷(`frequency_penalty: 0.0`, `presence_penalty: 0.0`, `xtc_probability: 0.0`, `xtc_threshold`)은 정직한 "기능 꺼짐" 기본값이다. CLI에 켤 수 있는 플래그가 없으니 실제로 꺼져 있다. `dry_sequence_breakers: Vec::new()`만 예외인데, `--dry-multiplier`로 DRY를 켤 수 있기 때문이다. 켜진 순간 빈 벡터는 꺼진 기능이 아니라, 사용자가 바꿀 수 없는 설정으로 돌아가는 기능이 된다.
- **그 결과로 생기는 동작 차이가 보이지 않는다.** breaker가 없으면 `src/lib/mlxcel-core/src/sampling.rs`의 역방향 매칭이 경계에서 종료되지 않아 `match_len`이 계속 자라고, 같은 설정에서 서버가 만드는 것보다 페널티 항이 커진다. `mlxcel run`으로 `--dry-multiplier`를 튜닝한 뒤 그 수치를 `mlxcel-server`에 배포한 사용자는 동일한 값에서 다른 출력을 얻고, 어느 help 문구에도 이유가 없다.
- **조사 자체가 유실될 위험이 있었다.** 빈 벡터가 범위 결정의 결과라는 사실이 코드 어디에도 없었다. 다음 독자는 같은 결론에 이르기 위해 아홉 knob 비교를 처음부터 다시 해야 했을 것이다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| CLI에서 튜닝한 sampling 값이 프로덕션에서 다르게 동작하고, 그 차이를 모델이나 seed 탓으로 돌림 | Medium | Medium |
| 이후 기여자가 빈 벡터를 누락으로 읽고 범위 논의 없이 플래그를 추가함 | Low | Medium |
| 독자가 DRY 줄을 아래의 "기능 꺼짐" 네 줄과 묶어 읽고 CLI에서 DRY가 꺼져 있다고 결론 내림 | Low | Medium |

---

## 2. 기술적 검토 사항

### 2.1 diff가 실행 코드가 아님을 증명 가능

추가된 13줄은 `src/commands/generate.rs`의 `//` 주석 10줄과 `src/main.rs`의 `///` doc comment 3줄(2줄 추가, 기존 1줄의 문장부호 조정)이다. 실행 문장, 필드 값, clap 속성 중 바뀐 것이 없다. `cargo fmt --check`는 clean이고, 프로젝트 설정에서 `rustfmt`는 주석을 재배치하지 않는다.

### 2.2 help 한 문장의 적용 범위

`SamplingOptions`는 `GenerateArgs`와 `RunArgs` 양쪽에 flatten되고, chat 경로도 같은 `SamplingConfig` 조립을 공유한다. 따라서 `--dry-multiplier` help에 문장 하나를 넣으면 `mlxcel generate`, `mlxcel run`, `mlxcel chat`이 중복 없이 모두 덮인다. help 문자열 하나로 충분했던 이유이자, `docs/` 아래에 별도 안내를 병기할 필요가 없던 이유다.

### 2.3 두 번째 `dry_multiplier`는 의도적으로 손대지 않음

`src/main.rs` 1245행 부근에 두 번째 `dry_multiplier` 필드가 있는데, `ServeArgs`에 속한다. `ServeArgs`는 그 16줄 아래에서 이미 `--dry-sequence-breakers`를 노출한다. 이 필드는 지금 문서화하고 있는 비대칭의 서버 쪽 절반이므로, 여기에 CLI 단서를 덧붙였다면 서버 help가 서버에 없는 한계를 설명하게 된다. 그대로 두었다.

---

## 3. 기술적 선택과 그 이유

### 3.1 비대칭을 없애는 대신 문서화

| 옵션 | 장점 | 단점 |
|---|---|---|
| `SamplingOptions`에 `--dry-sequence-breakers` 추가 | 비대칭 자체를 제거 | 의도적으로 부분집합인 sampling 표면을 넓힘. tokenizer 시점 토큰화와 새 단위 게이트가 필요. #1102 이전에는 CLI 기본값 `--temp 0.0`에서 무효(`build_sampling_config`의 greedy 갈래가 breaker를 버림) |
| CLI 기본값을 llama.cpp의 breaker 집합으로 변경 | 널리 기대되는 동작과 일치 | 기존 CLI 호출의 동작이 조용히 바뀜. 문서 이슈가 절대 내보내면 안 되는 종류의 변경 |
| **선택: 주석 + 사용자 대상 문장 하나** | 동작 위험 0으로 놀라움을 제거하고, 범위 결정의 이유를 기록 | 비대칭은 남으므로 CLI는 여전히 서버와 동등한 DRY를 표현하지 못함 |

이슈는 플래그 추가를 별도 기능 이슈로 열어두었고, 필요한 수용 형태까지 적어두었다. 파싱된 플래그가 `ResolvedSamplingParams.dry_sequence_breakers`에 토큰 ID로 도달하는지 단위 검증으로 확인하는 방식이며, 모델과 seed에 의존해 flaky한 출력 비교 검증은 쓰지 않는다. 그 경로는 열려 있고 이 PR은 그것을 막지 않는다.

### 3.2 사용자 대상 문장을 `docs/`가 아니라 clap help에 배치

clap help는 사용자가 `--dry-multiplier` 값을 고르는 바로 그 순간에 보이는 텍스트이고, 놀라움이 발생하는 시점도 그 순간이다. `docs/` 페이지는 그보다 이르게 읽히거나 아예 읽히지 않는다. 변경을 `src/` 안에 두면 플래그와 그 단서가 같은 파일에 남으므로, 이후 플래그가 바뀔 때 단서만 다른 파일에 남아 어긋나는 일도 없다.

### 3.3 이 필드가 이웃들과 왜 다른지를 함께 적음

주석은 "CLI가 breaker 없이 DRY를 돌린다"만 적지 않는다. 그 줄이 아래 네 줄과 왜 다른지를 적는다. 넷은 켤 수단이 없어서 꺼져 있고, 이 줄은 `--dry-multiplier`로 켤 수 있으므로 그렇지 않다. 이 대비가 없으면 다음 독자는 빈 벡터가 의도인지 알기 위해 비교를 다시 해야 한다. 주석의 존재 이유는 그 조사를 반복하지 않게 만드는 것이다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 2 |
| 추가된 라인 | +13 |
| 삭제된 라인 | -1 |
| 변경된 실행 라인 | 0 |
| 테스트 추가 | 0 |

### 영역별 변경

| 영역 | 파일 | 주요 내용 |
|---|---|---|
| 코드 주석 | `src/commands/generate.rs` | `dry_sequence_breakers: Vec::new()` 위에 10줄 주석. 범위 결정임을 기록하고, 아래 네 개의 "기능 꺼짐" 이웃과 성격이 다른 이유와 그로 인한 `match_len` 동작을 설명 |
| 사용자 대상 help | `src/main.rs` | `SamplingOptions`의 `--dry-multiplier` help에 한 문장 추가. CLI DRY는 모든 경계를 가로질러 매칭하며 서버의 `--dry-sequence-breakers`에 대응하는 CLI 플래그가 없다는 내용 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `33322d66` | docs | docs: explain CLI DRY runs without sequence breakers |

---

## 5. 검증 및 후속 조치

### 통과

- `cargo fmt --check` clean.
- `python3 scripts/ci/check_cross_repo_refs.py` 통과(새로 추가된 bare `#NNN` 없음).
- diff를 줄 단위로 확인해 주석과 help 문구뿐임을 검증. 추가된 모든 줄이 `^\s*(///|//)`에 일치.

### 다루지 못한 것

구현 worktree에는 `target/` 디렉터리가 없어 컴파일을 돌리지 않았다. 실행 라인을 하나도 바꾸지 않는 변경을 위해 MLX 전체 소스 빌드를 유발할 이유가 없었다. clap 필드의 여러 줄 doc comment는 같은 구조체 안에서 이미 쓰이고 있으므로(예: `seed`), 새로운 패턴이 아니라 기존 패턴이다.

### 후속 후보

- 플래그 자체를 기능 이슈로 진행하는 것. 필요한 수용 형태는 3.1에 기록되어 있다. #1102 이전에는 CLI 기본값 `--temp 0.0`에서 무효다.
- CLI가 빠뜨린 나머지 여덟 개 sampling knob도 같은 방식으로 문서화되어 있지 않지만, 어느 것도 이 실패 양상을 공유하지 않는다. 각각은 CLI에서 애초에 도달 불가능하므로 사용자가 켠 뒤 고정된 설정에 놀랄 일이 없다. 스위치만 있고 짝이 되는 설정이 없는 쌍은 `--dry-multiplier`와 `dry_sequence_breakers`뿐이다.
