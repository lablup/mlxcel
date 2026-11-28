# 기술 리포트: PR #2016 - --prompt 토큰화 실패 시 패닉 대신 에러 반환

**작성일**: 2026-09-28
**상태**: 완료
**언어**: Rust
**위험도**: 낮음

## 요약

PR #2016은 `src/bin/speculative_bench.rs`의 `encode_prompt`를 패닉하는 헬퍼에서 `anyhow::Result<Vec<i32>>`를 반환하는 헬퍼로 변경했다. 이제 토크나이저가 거부하는 `--prompt` 값이 들어오면 패닉 백트레이스 대신 `--prompt`를 명시한 읽을 수 있는 에러 메시지가 나온다. 이 수정은 형제 바이너리인 `bench_decode.rs`의 기존 `tokenize_prompt` 헬퍼를 그대로 따르며, 이 바이너리에서 이전에 수정된 `--target` unwrap 패닉(#1242)과 같은 종류의 결함이다.

## 1. 문제 정의

### 1.1 배경

`speculative_bench`는 드래프터 없는 기준 디코드와 MTP(멀티 토큰 예측) 추측 디코드를 비교하는 벤치마크 하네스다. 두 경로 모두 추론을 실행하기 전에 운영자가 지정한 `--prompt`를 `encode_prompt`를 통해 토큰화한다.

### 1.2 기존 문제점

- **토크나이저 거부 시 패닉**: `encode_prompt`는 `.expect("tokenizer.encode must succeed on a valid utf-8 prompt")`로 끝났다. 토크나이저의 `encode` 호출이 거부하는 프롬프트가 들어오면, 두 호출 지점(`run_baseline`, `run_mtp`)이 이미 `anyhow::Result`를 반환하고 있어 에러를 정상적으로 전파할 수 있었음에도, 전체 프로세스가 문제된 플래그를 알려주는 진단 메시지 없이 패닉 백트레이스로 중단됐다.

### 1.3 위험성

낮음. `speculative_bench`는 추론 요청 경로나 배포되는 CLI 표면의 일부가 아닌 내부 벤치마크 바이너리다. 방치하더라도 비용은 토크나이저가 인코딩할 수 없는 프롬프트를 운영자가 입력했을 때 혼란스러운 실패 형태(진단 대신 패닉)로 나타나는 정도였다.

## 2. 변경 요약

| 항목 | 값 |
|------|-----|
| 변경된 파일 | 1 |
| 추가된 줄 | +9 |
| 삭제된 줄 | -9 |

- `encode_prompt(tokenizer: &MlxcelTokenizer, prompt: &str) -> Vec<i32>`가 `-> Result<Vec<i32>>`로 바뀌었고, `.expect(...)`는 `.map_err(|err| anyhow::anyhow!("--prompt failed to tokenize: {err}"))?`로 교체됐다.
- 두 호출 지점인 `run_baseline`(기준 경로)과 `run_mtp`(추측 경로)에 `?`를 추가해 새로운 `Result`를 전파한다. 두 함수 모두 이미 `Result`를 반환하고 있었으므로 추가적인 시그니처 변경은 필요 없었다.
- `encode_prompt` 위의 문서 주석을 새로운 에러 경로를 설명하도록 갱신했고, 여전히 패닉하는 테스트 헬퍼를 가리키던 부정확한 참조 대신 이 수정이 따르는 패턴인 `bench_decode.rs`의 `tokenize_prompt`를 가리키도록 바꿨다.
- 성공 경로(BOS 처리, `encode` 호출, `u32 -> i32` 변환)는 변경되지 않았다.

## 3. 기술적 선택과 그 이유

### 3.1 실패한 플래그 이름을 명시하는 에러 메시지

**컨텍스트**: `bench_decode.rs`의 `tokenize_prompt`는 동일한 토크나이저 에러를 `anyhow::anyhow!("tokenization failed: {err}")`로 감싸는데, 어떤 CLI 플래그가 거부된 텍스트를 제공했는지는 명시하지 않는다.

**선택 이유**: 이슈의 완료 기준은 에러가 `--prompt`를 명시해야 한다고 명확히 요구했다. `speculative_bench`는 텍스트를 다루는 입력이 하나 이상(드래프트/타겟 모델 경로, 프롬프트)이기 때문에, 단순히 "tokenization failed"라는 메시지만으로는 어떤 인자가 문제인지 운영자가 추측해야 하는 상황이 남는다. 메시지는 `"--prompt failed to tokenize: {err}"`가 되어 기존 토크나이저 에러는 그대로 유지하면서 플래그 이름을 추가했다.

**트레이드오프**: `bench_decode.rs`의 메시지 문구와 완전히 동일하지는 않지만, 동일한 `Result` + `map_err` + `anyhow!` 형태를 유지하므로 같은 계열의 수정임을 여전히 알아볼 수 있다.

## 4. 검증

- `cargo check --bin speculative_bench --features metal,accelerate`(빌드 코어 2개로 제한): 깨끗함. 마지막 수정이 반영되기 전에 시작된 백그라운드 빌드로 인한 오래된 검사 결과가 아님을 확인하기 위해, 최종 파일 상태에 대해 증분 재실행(0.23초)도 수행했다.
- `cargo clippy --bin speculative_bench --features metal,accelerate -- -D warnings`: 깨끗함(유일하게 출력된 경고는 `mlxcel-core`의 빌드 스크립트에서 나오는 기존의 무관한 C++ `-Wunused-variable` 경고이며 `-D warnings`의 대상이 아님).
- `cargo fmt --check`: 깨끗함.
- `python3 scripts/ci/check_cross_repo_refs.py`: 깨끗함.
- 독립적으로 수행된 `pr-reviewer`와 `pr-security-checker` 검토에서 CRITICAL/HIGH/MEDIUM 지적 사항이 전혀 발견되지 않았고, `pr-finalizer`는 채워야 할 테스트나 문서 공백이 없음을 확인했다.
- 실행하지 않음: 디스크에 저장된 실제 모델 체크포인트를 사용한 벤치마크. 이 호스트의 GPU는 동시에 진행 중인 다른 작업과 공유되며, 이번 변경은 추론 경로가 아닌 토큰화 단계의 에러 처리에 한정된다.
- 유닛 테스트를 추가하지 않음: 인트리 `MlxcelTokenizer` 백엔드(HuggingFace, SentencePiece, Tiktoken) 중 어느 것도 순수한 유효 UTF-8 텍스트를 거부하지 않으므로, 새로운 `Err` 분기를 유도할 자연스러운 픽스처가 없다. 이 PR이 따르는 형제 함수인 `bench_decode.rs`의 `tokenize_prompt` 역시 유닛 테스트가 없다. 타입 변경 자체는 두 호출 지점에서 컴파일러가 검증한다.

## 5. 관련 작업

- 이슈 #1667: 이 변경의 근거가 된 이슈.
- 이슈 #1242: 동일 바이너리에서 이전에 수정된 `--target` unwrap 패닉, 같은 종류의 결함.
- `src/bin/bench_decode.rs::tokenize_prompt`: 이 수정이 따르는 참조 구현.
