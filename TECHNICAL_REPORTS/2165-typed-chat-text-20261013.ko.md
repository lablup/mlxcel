# 기술 보고서: PR #2165 - Typed Chat Text

**작성일**: 2026-10-13
**상태**: 리뷰 준비 완료
**언어**: Rust
**위험도**: 중간

## 요약

typed text part는 처리하지만 문자열 content를 조용히 버리는 checkpoint 선택 템플릿을 모델 로드 시 한 번 감지하도록 수정했습니다. 해당 템플릿의 typed `ChatMessage` 렌더만 빌려온 text-part view를 사용하며, raw JSON 렌더와 문자열 호환 템플릿의 기존 계약은 그대로 유지합니다.

## 1. 문제 정의

Granite Vision 3.2 checkpoint는 이미지 요청에 필요하여 우선 선택되는 standalone multimodal 템플릿을 제공하지만, 이 템플릿은 typed content part에서만 텍스트를 읽습니다. CLI와 서버의 text-only 경로는 문자열을 전달했으므로 질문이 prompt에서 사라졌고, 실제 checkpoint는 47-token prompt와 질문에 무관한 인사말을 생성했습니다.

tokenizer configuration의 문자열 템플릿으로 fallback하면 이미지와 텍스트 요청이 서로 다른 템플릿을 사용하게 되고 CLI, 서버, prompt-cache history 렌더가 달라집니다.

## 3. 기술적 선택과 그 이유

### 선택 시점에 호환성을 한 번 감지

checkpoint 템플릿을 선택한 뒤 변경하지 않은 raw renderer를 통해 문자열 probe와 typed-text probe를 각각 한 번 렌더합니다. typed probe만 marker를 보존할 때 `wrap_string_content = true`를 저장하며, 어느 probe에서든 렌더 오류가 발생하면 변환을 활성화하지 않습니다.

### 텍스트 복사 없이 typed message만 변환

`apply_inner`는 원래 role과 하나의 typed text part를 담은 작은 borrowed wrapper를 직렬화합니다. wrapper는 `&str` 참조를 유지한 채 MiniJinja에 직접 전달되므로 중간 JSON tree와 문자열 중복 할당이 없습니다. `ChatMessage`를 빠짐없이 destructuring하여 향후 필드가 추가되면 렌더 view가 해당 필드를 보존하도록 갱신하기 전까지 컴파일이 실패합니다.

### 기존 계약 유지

`apply_raw*`, 명시적인 `with_template` processor, default processor, 문자열 호환 템플릿, typed image 순서, probe 오류 동작은 바뀌지 않습니다. CLI와 서버 호출부를 수정하지 않았고 새 의존성도 없습니다.

## 7. 변경 요약

| 분류 | 파일 수 | 내용 |
|---|---:|---|
| Production | 1 | 선택 시점 호환성 flag와 borrowed `apply_inner` 렌더 view |
| 회귀 테스트 | 1 | 정확한 Granite prompt, role, 빈 문자열, raw 동작, 호환 템플릿, constructor, probe 오류를 검증하는 8개 테스트 |
| 고정 fixture | 5 | 출처가 기록된 Granite Vision 3.2 및 Granite 4 템플릿 |
| 보고서 | 2 | 영문 및 국문 구현 기록 |

## 부록

### 테스트 결과

- `cargo test --locked -p mlxcel -p mlxcel-surgery --lib --profile test-fast --features cuda -j4 text_wrap_tests -- --test-threads=1`: 8개 통과, 실패 0개이며 관련 없는 mlxcel 테스트 9,105개와 surgery 테스트 160개는 filter되었습니다.
- 저장소 contract gate를 통과했습니다: workspace version, kernel dtype key, kernel port dispatch, llama compatibility, ROCm overlay 검사, Python tooling 테스트 123개, WebUI contract fixture 49개.
- NVIDIA GB10/CUDA에서 `mlx-community/granite-vision-3.2-2b-4bit` revision `351831ab5218b1f66bbfdfd57bd8f20d81ce8baf`을 실행한 결과, CLI는 8 token으로 “The capital of France is Paris.”라고 답했습니다. 서버의 문자열 및 typed-text 요청도 모두 55 prompt token과 8 completion token으로 같은 답을 반환했습니다.
- 동결된 baseline은 두 서버 요청 형식 모두에서 47 prompt token과 질문에 무관한 20-token 인사말을 반환하여 결함을 재현했습니다.
- 구현 리뷰, 보안 재검토, 정적 최종화에서 남은 문제가 발견되지 않았습니다.

### Provenance

- 동결 source diff SHA-256: `1d088cbdc926b50a1af5fc8810435dd2cccb3b846bf5477a00b363ea6d9ceaa4`.
- Candidate binary SHA-256: CLI `706fcf0864c6adbd251e2725db1f623044b49e6a33436617925dc9b1da4d09f8`; server `496534a54f6ea92ceaea5e3cfaec20237b11f753bd5fef114927208607a2998c`.
- Baseline binary SHA-256: CLI `592c33a141ccd7a497732e132ac6da302661e22400e27fbcdb2cb3ca858ae8ff`; server `bc076e4d0b2fde769eb15ababe241e72828c5f9c0b3ec2142137cef688c5a656`.

### 검증 한계

이 이슈 단위에서는 focused CUDA 테스트와 Linux/GB10의 실제 checkpoint 하나를 검증했습니다. 전체 workspace test 및 Clippy suite, ROCm inference, macOS inference, 추가 모델 family는 실행하지 않았으며 broad check는 상위 integration gate와 hosted CI가 담당합니다.
