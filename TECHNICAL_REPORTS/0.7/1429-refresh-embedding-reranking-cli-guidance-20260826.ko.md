# 기술 보고서: PR #1429 - 임베딩·리랭킹 CLI 안내 갱신

**작성일**: 2026-08-26

**상태**: 완료

**언어**: Rust, Markdown

**위험도**: Low

## 요약

PR #1429는 최근 구현된 임베딩·리랭킹 인터페이스를 `mlxcel`, `mlxcel-server`, README와 유지보수 문서에서 쉽게 찾을 수 있도록 정리한다. 또한 `mlxcel arch`에 공개 임베딩 제품명과 Qwen 3.6·3.8 별칭을 표시하고, Muse 체크포인트 세부사항은 아키텍처 이름에서 분리해 목록의 역할을 명확히 한다.

## 1. 문제 정의

임베딩과 리랭킹 구현은 여러 PR을 통해 완료됐지만 공개 안내가 코드와 어긋나 있었다. 상세 임베딩 문서는 일부 family forward port가 아직 미완성인 것처럼 설명했고, README에는 실행 가능한 retrieval 예제가 없었으며, reranker 환경변수가 빠져 있었고, `mlxcel-server --help`는 단독 모델과 side model 실행 방식을 설명하지 않았다. 반면 최상위 도움말에는 빠르게 낡을 수 있는 특정 Muse 체크포인트와 고정된 tensor-parallel family 목록이 포함돼 있었다.

아키텍처 목록은 enum 기준으로는 완전했지만 일부 공개 제품명이 드러나지 않았다. 특히 Qwen 3.6과 Qwen 3.8은 각각 `qwen3_5_moe`, `qwen3_5` 구현 경로를 재사용하므로 별도 항목이 없다는 이유만으로 사용자가 미지원 모델로 오해할 수 있었다.

이 불일치는 inference correctness보다 운영상 위험을 만든다. 사용자가 잘못된 server startup mode를 선택하거나 `/v1/rerank`를 발견하지 못하거나 내부 구현 별칭을 지원 누락으로 해석할 수 있다.

## 2. 변경 요약

- `mlxcel embed`와 `mlxcel rerank` quick start를 추가하고 문서 index, contributor guide, environment variable 표, supported-model 문서와 Unreleased changelog를 동기화했다.
- `mlxcel-server --help`에 embedding-only, cross-encoder-only, chat과 side model 병행 구성을 추가하고 두 worker가 공유하는 queue·timeout 제어를 설명했다.
- 특정 모델 중심의 최상위 도움말을 supported-model 및 distributed-runtime 문서 링크로 교체했다.
- `mlxcel arch`에 serving interface footer를 추가하고 Qwen 3.6을 `qwen3_5_moe`, Qwen 3.8을 `qwen3_5`에 매핑했으며 LFM2.5-Embedding, Nemotron-3-Embed처럼 인식하기 쉬운 공개 이름을 사용했다.
- Muse 아키텍처 이름은 `Muse Glimmer 30B VLM`으로 줄였고 precision, cache, feature, checkpoint 제약은 supported-model 문서에 유지했다.
- 두 바이너리의 렌더링된 Clap help와 architecture catalog를 검증하는 regression test를 추가했다.

## 3. 기술적 선택과 그 이유

### 3.1 최상위 도움말은 안정적으로 유지하고 변동이 큰 정보는 관리 문서로 연결

Model qualification, 지원 precision, distributed constraint와 benchmark 결과는 command surface보다 자주 바뀐다. 최상위 도움말에 특정 시점의 체크포인트 정보나 tensor-parallel family 목록을 복제하지 않고 해당 정보를 관리하는 문서 위치를 안내하며, `mlxcel arch`는 로컬 architecture inventory 역할을 유지한다.

### 3.2 Runtime variant를 복제하지 않고 공개 버전 별칭을 표시

실제 구현 경로마다 하나의 `ModelType`을 유지하는 것이 runtime 구조에 맞다. Catalog display name에는 같은 경로를 공유하는 공개 버전을 함께 표시하고 footer에는 해당 `model_type` key를 명시해 구현 구조를 왜곡하지 않으면서 지원 여부를 직접 답한다.

### 3.3 Argument parsing뿐 아니라 렌더링된 도움말을 테스트

기존 parser test는 flag가 허용되는지는 검증하지만 설명 누락이나 오래된 after-help 문구는 찾지 못한다. 새 테스트는 long help와 architecture catalog를 렌더링해 필요한 command, flag, product name, endpoint, documentation link를 검증하고 이번에 제거한 checkpoint-specific guidance가 다시 나타나지 않도록 한다.

## 4. 호환성과 위험

- **Breaking change**: 없음. Command name, flag, environment-variable 동작, endpoint, request schema와 inference path는 바뀌지 않는다.
- **새 의존성**: 없음.
- **Runtime 영향**: 사람이 읽는 help와 architecture display string 외에는 없다.
- **잔여 위험**: 앞으로 기존 architecture를 재사용하는 새 모델 버전이 나오면 metadata와 문서를 함께 갱신해야 하지만, 현재 mapping과 모든 등록 `ModelType`의 노출 여부는 regression test가 보호한다.

## 5. 변경 통계

| 항목 | 값 |
|------|----|
| 변경된 파일 수 | 11 |
| 추가된 라인 | 292 |
| 삭제된 라인 | 72 |
| 관련 커밋 | `e05bb5e` |

## 6. 검증

- `cargo fmt --all -- --check`
- `cargo clippy --fix --workspace --all-targets --allow-dirty -- -D warnings`
- `cargo test --bin mlxcel --bin mlxcel-server`: 214 passed
- `cargo test --test cli_help_consistency`: 25 passed
- `python3 scripts/ci/check_crate_versions.py`
- `python3 scripts/ci/check_kernel_dtype_keys.py`
- 빌드된 `mlxcel --help`, `mlxcel embed --help`, `mlxcel rerank --help`, `mlxcel arch`, `mlxcel-server --help` 출력 수동 확인

## 7. 후속 조치

이번 문서 갱신 범위에서 필수 후속 조치는 없다. 기존 architecture를 재사용하는 새 model generation을 추가할 때는 같은 변경에서 `ModelType::metadata()`의 공개 별칭, 필요한 경우 `mlxcel arch` 별칭 안내, supported-model 문서를 함께 갱신해야 한다.
