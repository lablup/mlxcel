# PR #2532: validate_pipeline_parallel_args의 도달 불가 가드 제거

**작성일**: 2026-10-22
**상태**: 완료
**위험도**: Low
**언어**: Rust

---

## 요약

이슈 #1131은 `src/commands/generate.rs`의 `validate_pipeline_parallel_args`에 절대 실행되지 않는 에러 분기가 있고, 이름은 거부를 말하지만 실제로는 성공을 단언하는 테스트가 있다고 보고했다. 이 PR은 메인테이너 코멘트의 (a) 해석을 구현한다: TP-only 실행(`--tp-size > 1`이며 pipeline parallelism 꺼짐)은 출시된 경로이므로 계속 허용한다. 도달 불가능한 `ensure!` 두 개를 제거하고, 세 번째는 동등한 형태로 단순화했으며, 이름이 잘못된 테스트 두 개를 개명하고, 남은 모든 검사에 테스트를 추가했다. 허용/거부되는 입력 집합은 변하지 않는다.

---

## 1. 문제 정의

### 1.1 배경

검증 함수는 PP가 꺼져 있으면(`pp_layers`가 `None`이고 `pp_size <= 1`) 일찍 반환한다. 그 이후 코드는 `pp_layers.is_some() || pp_size >= 2`일 때만 실행된다. 이후의 검사 두 개가 이 불변식과 모순되어 에러 메시지가 절대 출력될 수 없었다.

### 1.2 기존 문제점

- "2D parallelism requires --pp-size >= 2" 검사는 early return 조건의 정확한 부정이어서 실패할 수 없었다.
- 마지막 `--pp-size must be at least 2` 분기는 `pp_layers == None`일 때만 실행되는데, 이때 early return이 이미 `pp_size >= 2`를 보장한다. 이슈에 명시되지 않았지만 "도달 불가 분기 없음" 기준에 해당한다.
- `..._rejects_2d_without_pp_enabled`와 `..._requires_two_stages_without_manual_ranges`는 이름이 거부를 약속하지만 `is_ok()`를 단언했다.
- `--pp-micro-batch-size 0`, 빈 `--pp-layers`, PP에서의 멀티모달 입력 등 살아 있는 검사 일부에 테스트가 없었다.

### 1.3 위험성

| 위험 | 영향 | 가능성 |
|------|------|--------|
| 죽은 메시지가 CLI가 거부하는 대상을 오해하게 함 | Low (유지보수성) | 결정론적 |
| 실제로 도달 가능한 가드를 제거 | Low (잘못된 토폴로지가 이후 코드에 도달) | 도달성 분석과 테스트로 배제 |

---

## 2. 기술적 검토 사항

### 2.1 보안 관점

새로운 입력 경로는 없다. 검증 함수는 동일한 입력을 거부하며 도달 불가 코드만 삭제된다. unsafe 코드 없음.

### 2.2 성능 관점

해당 없음. 검증은 CLI 시작 시 한 번 실행된다.

### 2.3 호환성 및 의존성

- **Breaking Changes**: 없음. 허용/거부되는 입력이 전후로 동일하다.
- **새 의존성**: 없음.

### 2.4 코드 품질

- `total_ranks > 0` 검사는 같은 메시지의 동등한 `pp_size > 0`으로 단순화해 유지했다. 이 검사는 도달 가능하다: `--pp-size 0 --pp-layers "0-3,4-7" --tp-size 2`는 early return을 통과한다. 이슈 본문은 여기서 `pp_size >= 1`을 가정했지만 `--pp-size`는 제한 없는 `usize`이다.
- 주석이 early return 이후 성립하는 불변식과 토폴로지 가드가 여전히 도달 가능한 이유를 설명한다.
- 에러 문자열 속 플래그 이름 불변식은 이미 main의 `tests/cli_help_consistency.rs`에 `every_flag_named_in_command_source_exists_in_some_help`로 존재하므로 중복 테스트를 추가하지 않았다. 더 엄격한 명령별 변형을 임시 스크립트로 평가했으나 누락은 없었다.

---

## 3. 기술적 결정

### 3.1 TP-only 허용 유지

메인테이너가 #1131에서 (a) 해석을 선택했다: TP-only는 다른 곳에서 `tp_size`로 디스패치되는 지원 경로이다. 거부하면 동작 변경이므로 죽은 검사를 살리지 않고 삭제했다.

### 3.2 토폴로지 가드는 좁혀서 유지

다른 둘과 함께 삭제하면 레이어 스펙과 TP가 있는 `--pp-size 0`이 샤딩 코드에 도달한다. `pp_size > 0` 형태는 같은 에러와 메시지를 더 낮은 비용으로 유지한다.

---

## 4. 구현 상세

- `src/commands/generate.rs`: `pp_size >= 2 || pp_layers.is_some()` ensure와 마지막 `else` ensure를 제거; `total_ranks`를 `pp.pp_size > 0`으로 교체; 설명 주석 추가.
- `src/commands/generate_tests.rs`: 테스트 두 개를 `..._accepts_tp_only_when_pp_disabled`, `..._accepts_single_stage_and_manual_ranges`로 개명; `..._rejects_zero_pp_size_with_layers_and_tp`, `..._rejects_zero_micro_batch_empty_layers_and_media` 추가.

---

## 5. 검증

| 검사 | 결과 |
|---|---|
| `cargo test -p mlxcel --bin mlxcel validate_pipeline_parallel` (test-fast, 기본 features) | 7 통과 |
| `cargo test --test cli_help_consistency` | 27 통과 |
| `cargo clippy -p mlxcel --bin mlxcel --tests -- -D warnings` | 깨끗함 |
| 변경 파일 `cargo fmt --check` | 깨끗함 |
| CI | 보고서 작성 시점에 미확인 |

---

## 6. 변경 요약

### 통계

| 항목 | 값 |
|------|-----|
| 변경 파일 | 2 |
| 라인 | +61 / -22 |
| 추가 테스트 | 2 (개명 2 별도) |

### 관련 커밋

| 해시 | 유형 | 메시지 |
|------|------|--------|
| `b71368c3` | fix | fix(cli): drop unreachable guards in validate_pipeline_parallel_args |
| `84776b93` | style | style: rustfmt the new reachability test |
| `68f95915` | test | test(cli): rename pipeline validator test that named a removed check |
| `2664098f` | test | test(cli): cover the untested pipeline-parallel validator checks |

---

## 7. 후속 조치

### 향후 개선

- 필요한 항목 없음. 원격 토폴로지의 `pp_size * tp_size == nodes` 검사는 클러스터 TOML 검증 계층에 남는다.

### 모니터링 필요 사항

- 없음.
