# 기술 보고서: PR #1125 - fix(cli): make the drifted server flag spellings work on both binaries

**작성일**: 2026-08-14
**작성자**: Jeongkyu Shin
**상태**: 완료
**언어**: Rust (clap 속성, 헬프 텍스트, 테스트), Markdown
**위험도**: Low

---

## 요약

PR #1125는 이슈 #1109를 닫는다. `mlxcel serve`와 `mlxcel-server`는 같은 서버를 손으로 각각 정의한 두 벌의 clap 정의이고, 플래그 철자 112개를 공유한다. 그중 넷이 갈라져 있었다. 최악의 경우는 공유 철자가 아예 없었다. `--n-parallel`은 `serve`에서만, `--parallel`은 `mlxcel-server`에서만 동작했다. 둘 다 같은 `LLAMA_ARG_N_PARALLEL` 환경변수를 읽는데도 두 바이너리 사이에서 복사한 명령줄이 파싱되지 않았다.

넷 중 셋은 빠진 `visible_alias`를 추가해 해결한다. 주 철자는 하나도 바뀌지 않는다. 넷째인 DRY sequence breaker는 두 바이너리가 이름 자체를 다르게 쓰고 있었으므로 주 철자를 정해야 했다. 지속성 있는 절반은 `tests/cli_help_consistency.rs`다. 두 바이너리의 long 이름 표면 전체를 의도적 예외 세 개짜리 허용 목록과 대조하므로, 한쪽에만 추가되고 다른 쪽에서 잊힌 플래그는 즉시 실패한다.

---

## 1. 문제 정의

### 1.1 배경

두 서버 표면은 하나의 flatten된 clap 그룹이 아니다. `ServeArgs`는 `src/main.rs`에, `ServerArgs`는 `src/bin/mlx_server.rs`에 있고 서로를 보며 손으로 유지된다. 공유 플래그 그룹(`TurboKvCacheArgs`, `SpeculativeArgs`)은 flatten되어 있어 갈라질 수 없고, `tests/cli_help_consistency.rs`가 이미 그쪽을 고정하고 있었다. flatten된 그룹 밖에 있는 것들은 정렬을 붙잡아 주는 장치가 없었다.

이 저장소는 같은 문제를 이미 한 번 만나 제대로 풀었다. drafter 플래그는 어느 철자든 어느 바이너리에서든 통하도록 대칭으로 alias가 걸려 있고, `src/main.rs`의 문서 주석이 이유를 그대로 적어 두었다. "so commands copied between the two binaries work unchanged." `--draft-max` / `--draft`도 같은 처리를 받았다. 빠져 있던 것은, 같은 처리를 받지 못한 다음 플래그를 알아채는 장치였다.

### 1.2 기존 문제점

| 개념 | `mlxcel serve` 수용 | `mlxcel-server` 수용 | llama-server |
|---|---|---|---|
| 병렬 슬롯 | `--n-parallel`만 | `--parallel`만 | `--parallel` |
| 예측 상한 | `--n-predict`만 | `--predict`, `--n-predict` | `--n-predict` |
| LoRA 어댑터 | `--adapter`, `--lora` | `--lora`만 | `--lora` |
| DRY breaker | `--dry-sequence-breakers` | `--dry-sequence-breaker` | `--dry-sequence-breaker` |

- **병렬 행만이 어느 철자도 양쪽에서 통하지 않는 경우다.** 복사한 명령줄은 `error: unexpected argument '--parallel' found`로 실패한다. `LLAMA_ARG_N_PARALLEL`은 양쪽에서 동작하므로, 이를 우회책으로 쓴 운영자는 증상만 없애고 원인은 기록하지 않는다. 그 스크립트를 다음에 읽는 사람에게는 이유 없이 플래그를 피하는 배포로 보인다.
- **헬프 텍스트가 이미 분기를 따라 갈라져 있었다.** `mlxcel serve`는 "Use `--n-parallel 1` (or `--no-batch`) for single-slot serving", `mlxcel-server`는 "Use `--parallel 1` (or `--no-batch`) to restore single-slot sequential serving"라고 적혀 있었다. 같은 문단인데 표현이 둘이고, 각자 자기 바이너리 철자에 맞춰 손질돼 있었다. 이 어긋남이 플래그 이름 분기를 버그가 아니라 정상처럼 보이게 만들었다. 두 헬프 출력을 비교하는 사람 눈에는 여러 군데가 다른 두 문단이 보일 뿐, 플래그 이름만 따로 집어낼 근거가 없다.
- **`README.md`는 llama-server 호환을 마이그레이션 수단으로 내세운다.** 넷 중 셋이 특히 `mlxcel serve`에서 그 약속을 깨고 있었다.
- **다섯 번째를 잡을 장치가 없었다.** 기존 일관성 테스트는 flatten된 두 그룹만 다룬다. 두 구조체에 직접 붙은 개념은 스위트의 어떤 불변식에도 들어 있지 않았다.

### 1.3 위험성

| 위험 | 영향도 | 발생 가능성 |
|---|---|---|
| 바이너리 간 복사한 명령줄이 실패하고, 그 원인이 버전 차이나 오타로 돌려짐 | Medium | High |
| 운영자가 `LLAMA_ARG_N_PARALLEL`로 우회하고 플래그 공백이 발견되지 않음 | Low | High |
| 다섯 번째 플래그가 갈라진 채로 머지됨 | Medium | Medium |

---

## 2. 기술적 검토 사항

### 2.1 무엇이 동작을 바꾸고 무엇이 바꾸지 않는가

넷 중 셋은 순수한 추가다. `ServeArgs::n_parallel`에 `visible_alias = "parallel"`, `ServerArgs::parallel`에 `visible_alias = "n-parallel"`, `ServeArgs::n_predict`에 `visible_alias = "predict"`, `ServerArgs::lora`에 `visible_alias = "adapter"`. `visible_alias`는 수용 철자를 하나 늘리고 `--help`에 표시할 뿐, 무엇도 제거하지 않는다. 전에 파싱되던 명령줄은 전부 그대로 파싱된다.

DRY breaker 행만이 주 철자를 바꾸며, 그것도 `mlxcel serve`에서만이다. 이 필드의 clap 이름은 `dry_sequence_breakers`이므로 `#[arg(long)]`이 Rust 식별자에서 `--dry-sequence-breakers`를 유도했다. 단수를 주 철자로 만들려면 `long = "dry-sequence-breaker"`를 명시해야 한다. Rust 필드 이름은 그대로이므로 파싱 이후는 아무 영향이 없다. `src/commands/serve.rs`, `src/server/cli_input.rs`, `src/server/startup.rs`는 계속 `dry_sequence_breakers`를 읽는다.

### 2.2 철자 변경의 파급 범위

복수형은 양쪽 바이너리에서 `visible_alias`로 유지되므로 명령줄 관점에서는 변화가 보이지 않는다. 트리에서 복수형을 산문으로 언급하던 두 곳은 주 철자를 가리키도록 갱신했다. #1118이 `SamplingOptions::dry_multiplier`에 넣은 헬프 문장과 `src/commands/generate.rs:921`의 범위 주석이다. 두 진술은 어느 쪽이든 참이지만(복수형도 여전히 파싱된다), 주 철자를 적어야 헬프 텍스트가 바이너리가 문서화하는 철자를 가리킨다.

### 2.3 단언이 공허하지 않음을 증명해야 했다

통과하는 교차 바이너리 계약 테스트는 실제 회귀에서 실패하지 않으면 가치가 없다. 이 테스트에는 조용히 통과할 경로가 셋 있다. 엔트리 매처가 아무것도 못 찾는 경우, alias 파서가 빈 목록을 돌려주는 경우, 설명 추출기가 양쪽 모두에 빈 문자열을 돌려주는 경우다. 각각을 확인한다.

1. `signature_long_name_accepts_clap_signatures_and_rejects_prose`가 clap이 실제로 렌더링하는 서명 형태 네 가지(long만, long + 값 이름, short + long + 값 이름, 반복 표시가 붙은 값 이름)와 거부해야 할 네 가지(그중에는 `--parallel`로 시작하는 설명 줄과 하이픈 불릿 산문 줄이 있다)를 고정한다.
2. `accepted_spellings_and_description_split_an_entry_correctly`가 렌더링된 엔트리와 그 거울상(주 철자가 다르고, clap이 유도한 값 이름이 다르고, alias 주석이 반대이며, 설명은 같다)에 대해 분리기를 고정한다. 이 둘은 철자 집합과 설명 양쪽에서 같아야 하며, 그것이 바로 계약이 바이너리 간에 수행하는 비교다.
3. `dropping_a_shared_flag_alias_would_fail_the_spelling_parity_assertion`가 실제 `mlxcel-server --help`를 가져와 `--parallel` 엔트리에서 렌더링된 `[alias: --n-parallel]` 주석을 잘라내고, 수용 집합이 `["--parallel"]`로 줄어드는지 단언한다. 기존 `removing_the_rendered_alias_annotation_leaves_the_alias_undocumented`가 drafter 플래그에 쓰는 것과 같은 관용구다.

일회성 실제 변이로 종단 경로도 확인했다. clap 정의에서 `visible_alias = "n-parallel"`을 지우고 다시 빌드하자 `shared_server_flags_accept_the_same_spellings_on_both_binaries`가 `left: ["--parallel"], right: ["--n-parallel", "--parallel"]`로 실패했고, 되돌리자 스위트가 다시 초록이 됐다.

### 2.4 명시적 목록만으로는 이슈가 요구한 것을 주지 못한다

이 변경의 첫 초안은 개념 여섯 개짜리 명시적 목록만 단언했다. 그것은 누군가 이미 적어 둔 짝의 회귀를 잡는다. 이슈가 실제로 요구한 것, 즉 미래의 분기는 잡지 못한다. 내일 한쪽 바이너리에만 추가된 플래그는 그 목록에 없을 뿐이고 스위트는 초록으로 남는다.

이슈가 명시적 목록을 대안으로 제시한 근거는 전체 표면 비교가 너무 시끄러우리라는 가정이었다. 두 바이너리가 정당하게 다르기 때문이다. 재어 보니 그 가정이 깨졌다. 이번 변경 후 두 표면은 철자 134개를 공유하고 정확히 세 개만 다르다. `mlxcel serve` 쪽의 `--estimate-memory`와 `--force`(둘 다 서브커맨드 성격의 일회성 동작), 그리고 `mlxcel-server` 쪽의 `--version`(`mlxcel`은 최상위에 갖고 있다). 세 개짜리 허용 목록은 읽을 수 있는 크기이므로, 실제로 들어가는 것은 전체 표면 단언이다.

두 불변식을 모두 유지하는 이유는 실패하는 대상이 다르기 때문이다. 표면 비교는 이름의 집합을 볼 뿐, 어떤 이름들이 한 개념의 두 철자인지는 보지 못한다. 한 변경에서 `mlxcel-server`의 `--parallel`과 `mlxcel serve`의 `--n-parallel`이 함께 사라지면 두 표면은 여전히 같고, 알아채는 것은 `SHARED_SERVER_FLAG_GROUPS`뿐이다. 명시적 목록은 회귀 가드이고, 표면 비교는 신규 분기 가드다.

---

## 3. 기술적 선택과 그 이유

### 3.1 양쪽 바이너리의 주 철자를 단수 `--dry-sequence-breaker`로

| 옵션 | 장점 | 단점 |
|---|---|---|
| **선택: 양쪽 단수 주 철자, 양쪽 복수 alias** | llama-server와 일치. 애초에 이 플래그들이 `LLAMA_ARG_*` 환경변수를 갖는 이유가 그것이다. `mlxcel-server`는 이미 단수를 쓰고 있어 바뀌는 바이너리는 하나뿐. 결과적으로 두 바이너리가 주 철자 하나를 공유하며, 이는 drafter 플래그가 도달한 상태보다 강하다 | `mlxcel serve` 사용자는 `--help`에서 이전과 다른 주 철자를 본다. 기존 명령줄은 그대로 동작한다 |
| 양쪽 복수 주 철자, 단수 alias | `mlxcel-server`만 바뀜 | 존재 이유 자체가 llama-server 호환인 플래그에서 llama-server와 어긋남 |
| drafter 플래그처럼 주 철자를 반대로 두고 양방향 alias | 가장 가까운 선례와 일관. 주 철자 변경 없음 | drafter의 타협은 mlx-lm과 llama-server 각각에 확립된 철자가 있어 어느 쪽도 강등할 수 없기 때문에 존재한다. 여기에는 mlx-lm 철자가 없으므로 균형 잡을 대상이 없고, 타협은 이유 없이 차이를 보존하게 된다 |

논쟁할 가치가 있는 것은 세 번째다. 바로 옆 코드가 그렇게 하고 있기 때문이다. 따르지 않은 이유는 이렇다. drafter 플래그에는 상충하는 업스트림 관례가 둘이고, `--dry-sequence-breaker`에는 하나다. 충돌이 없는 곳에 타협을 복사하면 두 바이너리가 `--help`에서 영구히 다른 주 철자를 보여줄 뿐, 얻는 것이 없다.

### 3.2 철자만이 아니라 산문도 단언한다

철자 일치만으로는 갈라진 `--n-parallel` 문단이 그대로 남는다. 그리고 그 문단이야말로 분기를 보이지 않게 만든 장치다. 같은 내용의 표현이 둘이면 읽는 사람은 비교를 멈춘다. 그래서 `SHARED_SERVER_FLAG_DESCRIPTIONS`는 drafter가 아닌 네 개념에 대해 동일한 산문을 요구한다.

drafter 두 그룹은 억지로 일치시키는 대신 의도적으로 제외한다. 그 설명은 각 바이너리 자신의 주/alias 역할을 적고 있으며 그 역할은 설계상 반대다. 동일한 산문을 강요하면 둘 중 하나가 거짓이 된다. 제외를 이유와 함께 기록하는 것이 요점이다. 말없이 빠뜨리면 실수처럼 읽힌다.

### 3.3 서명과 주석은 구조적으로 제외한다

`flag_description`은 서명 줄과 `[`로 시작하는 모든 줄을 버린다. 편의가 아니다. 서명 줄은 필연적으로 다르고(주 철자가 다르고, clap이 Rust 필드 이름에서 값 이름을 유도하므로 `--parallel <PARALLEL>` 대 `--n-parallel <N_PARALLEL>`), `[env:]` / `[default:]` / `[alias:]` 주석도 필연적으로 다르다. 구조적으로 제외하면 손으로 유지하는 제외 목록이 낡아서 비교가 무력해지는 일이 없고, 각 바이너리는 계약을 약화시키지 않은 채 자기 주 철자와 alias를 유지한다.

### 3.4 엔트리 앵커를 전체 서명이 아니라 long 이름으로

기존 `flag_help_entry`는 `--draft-model <PATH>` 같은 완전한 렌더 서명을 매칭한다. 값 이름이 계약의 일부일 때는 옳다. 여기서는 재사용할 수 없다. 값 이름이 Rust 필드 이름에서 유도되므로 같은 개념이라도 바이너리마다 다르기 때문이다. `flag_entry_by_long_name`은 long 이름만으로 앵커하되 줄 전체가 서명 형태일 것을 요구하므로 산문이 엔트리를 앵커할 수 없다. 공유되는 본문 절단 루프는 `entry_body`로 뽑아, 두 탐색기가 엔트리의 끝을 다르게 판단할 수 없게 했다.

---

## 4. 변경 요약

### 통계

| 항목 | 값 |
|---|---|
| 변경된 파일 수 | 6 |
| 추가된 라인 | +587 |
| 삭제된 라인 | -16 |
| 변경된 clap 속성 | 5 |
| 테스트 추가 | 13 |

### 영역별 변경

| 영역 | 파일 | 내용 |
|---|---|---|
| clap 정의 | `src/main.rs` | `n_parallel`, `n_predict`에 `visible_alias`; DRY breaker에 `long = "dry-sequence-breaker"`와 복수 alias; `--n-parallel` 헬프 문단을 `mlxcel-server` 쪽과 일치; #1118 문장이 새 주 철자를 가리킴 |
| clap 정의 | `src/bin/mlx_server.rs` | `parallel`, `lora`에 `visible_alias`; DRY breaker에 복수 alias와 공통 문서 주석; 파싱 수준 alias 테스트 셋과 clap 이름 유일성 가드 |
| 코드 주석 | `src/commands/generate.rs` | DRY 범위 주석이 새 주 철자를 가리킴 |
| 단위 테스트 | `src/main_tests.rs` | `mlxcel serve`에 파싱 수준 alias 테스트 넷(이미 대칭이던 `--adapter` / `--lora` 포함)과 clap 이름 유일성 가드 |
| 통합 테스트 | `tests/cli_help_consistency.rs` | 허용 목록 둘을 낀 전체 표면 비교, `SHARED_SERVER_FLAG_GROUPS`, `SHARED_SERVER_FLAG_DESCRIPTIONS`, 계약 테스트 셋, 가드 테스트 다섯, 헬퍼 여섯, `entry_body` 리팩터 |
| 문서 | `docs/CONTINUOUS_BATCHING.md` | `--n-parallel`과 `--parallel`을 바이너리당 하나의 철자처럼 제시하지 않음 |

### 관련 커밋

| Hash | Type | Message |
|---|---|---|
| `36b617b8` | fix | fix(cli): make the drifted server flag spellings work on both binaries |

---

## 5. 검증과 후속

### 통과

- `cargo test --profile test-fast --features cuda --bin mlxcel tests::serve_`: 13 passed.
- `cargo test --profile test-fast --features cuda --bin mlxcel-server tests::`: 13 passed.
- `cargo test --profile test-fast --features cuda --test cli_help_consistency`: 25 passed (기준선 17에서 증가).
- `cargo clippy --profile test-fast --features cuda --lib --bins --tests -- -D warnings`: clean. 두 바이너리 크레이트가 모두 바뀌므로 `--bins`를 포함했다.
- `cargo fmt --all -- --check`: clean.
- 실제 변이 두 건. 각각 되돌린 뒤 다시 초록임을 확인했다. clap 정의에서 `visible_alias = "n-parallel"`을 지우면 명시적 목록 parity 단언이 `left: ["--parallel"], right: ["--n-parallel", "--parallel"]`로 실패한다. `SERVE_ONLY_FLAGS`에서 `--force`를 지우면 전체 표면 단언이 `left: {"--estimate-memory", "--force"}, right: {"--estimate-memory"}`로 실패하는데, 이것이 실제로 한쪽에만 생긴 신규 플래그가 만들 모양이다.

### 다루지 않은 것

각 철자로 서버가 실제로 뜨는지 확인하는 종단 검사는 없다. 파싱 수준 테스트가 해석된 필드 값을 단언하며, 그것이 플래그 이름이 지배하는 경계다. `clap::Parser` 이후는 전부 Rust 필드 이름을 읽으므로 어느 철자가 그 값을 만들었는지 알 수 없다.

short form은 범위 밖이고 여전히 갈라져 있다. `mlxcel-server`에는 `--ctx-size`의 `-c`와 `--predict`의 `-n`이 있고 `mlxcel serve`에는 둘 다 없으므로, `mlxcel-server -m X -c 4096 -n 256`은 아직 그대로 복사되지 않는다. `mlxcel serve`에서 두 글자 모두 비어 있어 닫는 비용은 낮지만, 이슈 #1109가 열거한 표와는 다른 표이고 헬프 텍스트도 더 이상 그렇게 주장하지 않는다. 계약은 long 이름만 비교한다.

설명 일치는 공유 철자 134개 중 4개를 덮는다. 그 밖의 산문 어긋남은 존재하며 가드되지 않는다. 예로 `--prompt-cache-enabled`는 `mlxcel serve`에서 "when the CLI flag is absent", `mlxcel-server`에서 "is not explicitly provided"라고 적는다. 이번에 생긴 것은 아니고, 설명 부분집합이 남기는 공백의 구체적 크기다.

### 후속

이슈 #1103이 DRY breaker 플래그의 효과를 배선한다. 이 PR에 막혀 있었다기보다, 주 철자를 다시 정하지 말라는 의미로 순서가 걸려 있었다. 양쪽 단수로 정리됐으므로 남은 작업은 토큰화와 서버 측 기본값이다. `src/commands/generate.rs`의 범위 주석은 CLI 페널티가 서버보다 강하다고 적고 있는데, 그 진술은 #1103이 서버 플래그를 실제로 동작하게 만든 뒤에야 온전히 참이 되므로 그때 다시 볼 필요가 있다.
