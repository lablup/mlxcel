# 기술 보고서: PR #2019 - embed의 공백 전용 프롬프트 거부 처리

**날짜**: 2026-09-28
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

## 요약

PR #2019는 `mlxcel embed`가 공백만으로 이루어진 프롬프트(`-p "   "`)를 그대로 통과시키던 문제를 수정한다. 같은 종류의 입력에 대해 `mlxcel rerank`는 이미 올바르게 거부하고 있었다. `run_embed`의 검사 로직을 `String::is_empty` 대신 `p.trim().is_empty()`로 바꾸어 `run_rerank`의 기존 검사와 맞추었고, 인덱스를 담고 있으면서도 이름이 `empty`였던 바인딩을 `index`로 정정했다. 단위 테스트 하나로 거부 동작과 보고되는 인덱스 값을 함께 고정했다.

## 1. 문제 정의

### 1.1 배경

`src/commands/`의 `run_embed`와 `run_rerank`는 입력 검증 앞부분이 거의 동일하다. 둘 다 입력 목록이 비어 있는 경우와 이미지 파일이 없거나 존재하지 않는 경우를 같은 문구로 거부한다. 다만 한 가지 검사에서 차이가 있었다. `run_rerank`는 `d.trim().is_empty()`로 공백 전용 문서를 거부하는 반면, `run_embed`는 `String::is_empty`로 말 그대로 빈 문자열만 거부했다.

### 1.2 기존 문제

- **검증 불일치**: `mlxcel embed -p "   "`는 검사를 통과해 모델을 로드하고 사실상 빈 입력에 대해 순전파를 수행했지만, 같은 입력을 `mlxcel rerank -d "   "`에 주면 즉시 명확한 오류로 실패했다.
- **오해를 부르는 바인딩 이름**: `args.prompts.iter().position(String::is_empty)`의 결과를 `empty`라는 이름으로 받았다. `Iterator::position`은 불리언이 아니라 인덱스를 반환하므로, 이어지는 `bail!("prompt {empty} is empty")`는 마치 `empty`가 플래그인 것처럼 읽혔다.

### 1.3 위험도 평가

낮음. 이 변경은 모델 로드나 순전파보다 먼저 실행되는 CLI 검증 조건을 더 엄격하게 만들 뿐이다. 기존에 유효했던 입력의 동작을 느슨하게 만들 수 없으며, 이전에 거부되던 입력은 여전히 거부되고 공백 전용 입력만 새로 거부 대상에 추가된다.

## 2. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경 파일 | 1개 (`src/commands/embed.rs`) |
| 커밋 | 2개 |
| 추가/삭제 줄 수 | 38 / 2 |

- `run_embed`의 빈 프롬프트 검사를 `args.prompts.iter().position(String::is_empty)`에서 `args.prompts.iter().position(|p| p.trim().is_empty())`로 바꾸어, `src/commands/rerank.rs`의 `d.trim().is_empty()`와 맞췄다.
- 바인딩 이름을 `empty`에서 `index`로 바꿨다. `bail!("prompt {index} is empty")` 문구 자체는 변경하지 않았다.
- `embed.rs`의 기존 인라인 `#[cfg(test)] mod tests { ... }` 블록에 `run_embed_rejects_whitespace_only_prompts_like_rerank` 테스트를 추가했다. 공백 전용 프롬프트 하나(인덱스 0)와, 공백 항목이 첫 번째가 아닌 혼합 목록(인덱스 1) 두 경우를 모두 검증한다.
- 후속 커밋에서 테스트가 쓰던 `EmbedArgs.model`의 자리표시자 값을 HuggingFace 저장소 ID로 해석될 수 있는 `"unused"` 문자열에서 존재하지 않는 절대 경로로 바꿔, 검사 로직이 훗날 퇴행하더라도 네트워크 조회 없이 오프라인에서 곧바로 실패하도록 했다.

## 3. 기술적 선택과 그 이유

### 3.1 새 `embed_tests.rs` 대신 인라인 테스트 모듈 사용

**배경**: `src/commands/` 디렉터리는 테스트 구성 방식이 정확히 두 갈래로 나뉜다. 일곱 개 파일(`embed.rs`, `rerank.rs` 포함)은 인라인 `#[cfg(test)] mod tests { ... }` 블록을 쓰고, 다른 일곱 개는 `#[path = "..."] mod tests;`로 연결한 별도의 `<이름>_tests.rs` 파일을 쓴다.

**근거**: 이번 수정의 직접적인 참고 구현인 `rerank.rs`가 이미 인라인 방식을 쓰고 있다. `embed.rs`의 기존 인라인 블록을 확장하면 두 형제 파일이 서로 일관성을 유지하게 되며, 이는 디렉터리 나머지 절반과 맞추는 것보다 이 경우에는 더 중요한 기준이다.

### 3.2 순수 헬퍼를 뽑아내는 대신 `run_embed`를 직접 테스트

**배경**: `run_embed`의 검증 로직(빈 입력 검사, 공백 검사, 이미지 존재 검사)은 모델 리졸브나 MLX 런타임 초기화보다 먼저 실행된다.

**근거**: bail이 가장 먼저 일어나기 때문에, 테스트는 직접 만든 `EmbedArgs`로 `run_embed`를 호출하고 반환된 `Err`를 검증한다. 이는 별도로 뽑아낸 헬퍼 함수가 아니라 실제 CLI 진입점을 그대로 행사하는 방식이며, 임시 디렉터리나 네트워크 접근, 모델 체크포인트가 전혀 필요 없어 테스트가 사실상 0ms에 끝난다.

## 4. 검증

- `cargo test --release --bin mlxcel commands::embed::tests`: 새 테스트를 포함해 4개 모두 통과.
- `cargo check --lib --tests`: 이상 없음.
- `cargo clippy --release --bin mlxcel --tests -- -D warnings`: 이상 없음 (`mlxcel-core`의 기존 무관한 C++ 빌드 경고 1건 제외).
- `cargo fmt --check -- src/commands/embed.rs`: 이상 없음.
- 독립적인 `pr-reviewer`와 `pr-security-checker` 검토에서 CRITICAL, HIGH, MEDIUM 등급 문제는 발견되지 않았다. LOW 등급 제안 1건(테스트의 자리표시자 모델 경로)은 반영했다.
- 실행하지 않은 항목: 실제 체크포인트를 사용한 `mlxcel embed` 종단 간 실행. 이번 변경은 모델 로드 이전에 실행되는 검증 로직에 한정되기 때문이다.

## 5. 관련 작업

- 이슈 #1664: 이번 변경의 출처 이슈.
- `src/commands/rerank.rs`: `trim().is_empty()` 검사와 인라인 테스트 관례의 참고 구현.
