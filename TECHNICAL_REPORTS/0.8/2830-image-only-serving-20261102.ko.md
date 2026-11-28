# PR #2830: 채팅 모델 없이 이미지 생성 서빙

**Date**: 2026-11-02
**Status**: Completed (merged; closes #2825)

## 요약

PR #2830은 `mlxcel-server --image-model <dir>`와 `mlxcel serve --image-model <dir>`가 `-m` 없이 시작해 이미지 라우트만 서빙하게 한다. 모드는 두 번째 enum을 만들지 않고 `EmbeddingServingMode`에 `ImageOnly`를 더했으므로 health, props, capability 보고가 기존 `blocks_generation` 경로를 그대로 따른다. 채팅 표면은 기존 구조화 `501`을 돌려주되 이제 `--image-model`을 이름에 넣고, `/tokenize`, `/detokenize`, `/apply-template`도 같은 게이트를 탄다. 빈 토크나이저로 실행될 자리이기 때문이다. 이미지 워커는 공용 시작 경로와 종료 조인 경로를 유지하므로 #2745의 teardown 계약이 그대로 성립한다.

## 1. 문제 정의

`mlxcel-server --image-model <dir>`를 `-m` 없이 실행하면 `--model/-m is required to start the server`로 종료됐다. `mlxcel generate-image -m <dir>`는 채팅 모델이 필요 없는데도 그렇다. 이미지 모델 하나를 띄우는 데스크톱 클라이언트가 무관한 채팅 모델까지 올려야 했다. 의존성은 생성이 아니라 시작 구조에 있었다. `load_chat_front`가 체크포인트에서 토크나이저를 만들어야 하고, 배포되는 `MlxcelTokenizer::stub`은 테스트 전용이다.

## 2. 변경 요약

| 영역 | 변경 |
|---|---|
| 모드 | `EmbeddingServingMode::ImageOnly`(`--image-model`, `image_only`), 이미지 경로가 있고 모델 소스가 없을 때 선택, 충돌은 `resolve_image_only_mode`가 거부 |
| 두 바이너리 | `model_path`를 비우고 `ensure_inert`를 건너뛰며 `--image-model` 또는 `MLXCEL_IMAGE_MODEL`을 해석하고, 라우터 선택자, `--embedding-model`, `--reranker-model`, `--embeddings`, `--reranking`과 결합하면 두 플래그를 모두 이름에 넣어 거부 |
| 시작 | `load_chat_front`, 워밍업, 프롬프트 캐시를 건너뛰고 `new_without_chat_model_with_id`로 `model-worker-image-only` 스레드의 프로바이더를 만들며, #2745 이후의 사이드 워커 및 종료 조인 경로에 이미지 워커를 등록 |
| 빈 경로 가드 | `model_facts`, `detect_model_media_support`, embedding 및 rerank 감지, generation-config 기본값, context-window 읽기, Gemma4 패밀리 감지, props JSON 읽기, tensor 및 pipeline parallel 플래그 거부 |
| 라우트 | `chat_unavailable_message`에 `ImageOnly` 문구 추가, `/tokenize`, `/detokenize`, `/apply-template`는 이 모드에서만 게이트, `side_model_ready`는 `image_model.is_some()`, `check_serving_mode`는 `has_image`를 받음 |
| `/v1/models` | `data` 항목과 `models` 블록 모두 이미지 id를 `["image"]`로 보고하고 `completion`은 넣지 않음 |
| 문서 | `docs/image-generation.md`, `docs/environment-variables.md`, 두 바이너리의 `--image-model` 도움말, WebUI 시작 라벨과 설정 픽스처 |

## 3. 기술적 선택

### 3.1 기존 모드 enum 확장

`ImageOnly`는 병렬 enum을 만들지 않고 `EmbeddingServingMode`에 들어간다. `blocks_generation`이 이미 health, props, capability 보고를 결정하므로 두 번째 enum은 같은 결정을 복제하고 어긋날 것이다.

### 3.2 서빙 id는 별칭 또는 디렉터리 이름

프로바이더, `display_model_id`, `ImageModelProvider::model_id`가 모두 첫 `--alias`가 있으면 그것을, 없으면 `side_model_id_for_path`를 반환한다. 따라서 `/v1/models`에서 복사한 `model` 값이 별칭 유무와 관계없이 이미지 라우트의 `check_model`을 통과한다.

### 3.3 모호한 명령줄은 무시하지 않고 거부

라우터 선택자나 embedding, rerank, generation 플래그가 image-only와 결합하면 조용히 무시되거나 모드와 모순된다. 시작 단계에서 두 플래그를 이름에 넣어 실패하므로, 명령줄이 프로세스가 서빙하지 않는 능력을 주장할 수 없다.

### 3.4 채팅 게이트는 이 모드에 한정

`/tokenize`, `/detokenize`, `/apply-template`에는 채팅 게이트가 없었다. image-only에서는 빈 토크나이저로 실행되므로 `chat_not_available`을 호출한다. 호출은 `ImageOnly`에 조건부라 다른 모드, 즉 `-m --image-model`에서는 라우트가 그대로다.

### 3.5 종료 경로는 복사하지 않고 상속

이미지 워커는 #2745가 도입한 같은 사이드 워커 수집기와 `join_model_workers_before_exit`에 등록된다. 새 모드는 두 번째 워커나 두 번째 종료 경로를 추가하지 않는다.

## 4. 검증

| 검사 | 결과 |
|---|---|
| `cargo check --lib --tests`, 두 바이너리 | 통과 |
| `cargo fmt --all -- --check` | 통과 |
| `cargo clippy -p mlxcel --lib --tests -- -D warnings` | 통과 |
| `cargo test --lib server::routes` | 525 passed |
| `cargo test --lib server::startup` | 105 passed |
| `cargo test --lib server::cli_input` | 164 passed |
| `cargo test --lib server::model_source` | 28 passed |
| `cargo test --lib server::runtime_settings` | 14 passed |
| `cargo test --lib tokenizer::` | 138 passed |
| `cargo test --bin mlxcel commands::serve` | 16 passed |
| `cargo test --bin mlxcel-server image_` | 3 passed |
| 두 바이너리 빌드 | 통과 |

두 바이너리를 빈 `--image-model` 디렉터리로 스모크 실행하면 이미지 워커에 도달해 image-only 이유를 로그하고 `failed to load --image-model ...`로 치명 실패하며 워커 조인을 거쳐 종료했다. `--image-model --embeddings`와 `--image-model --reranker-model`은 두 플래그를 모두 이름에 넣었고, 인자 없는 출력은 `--image-model`을 안내했다.

이 호스트에서 기존 테스트 두 개가 기본 2 MiB 테스트 스레드 스택을 넘치므로(#1384) 스위트는 `RUST_MIN_STACK=16777216`으로 실행했다.

## 5. 미검증

이 호스트의 `models/`에 FLUX 이미지 체크포인트가 없어 이슈의 수동 시퀀스(health, `/v1/models`, 디코드 가능한 PNG, 채팅의 `501`)를 끝에서 끝까지 실행하지 못했다. 라우트, 시작, CLI 스모크 테스트가 그 자리를 대신하며 PR 본문에 그 사실을 기록했다.

## 6. 학습 포인트

채팅 모델의 부재는 생성 의존성이 아니라 시작 구조의 문제였다. 이미지 워커는 이미 `image_model_path`만 읽었고, 단단한 의존성은 `load_chat_front`와 그것이 필요로 하는 토크나이저였다. 그 경계를 먼저 이름 붙인 것이 넓은 기능을 일련의 가드된 조기 반환으로 바꿨다.

## 7. 위험

중간. 이 모드는 두 바이너리의 시작을 바꾸고 모든 `startup.model_path` 소비자를 건드린다. 빈 경로 가드가 넓고, 가드 없이 추가되는 향후 소비자는 이 모드에서 패닉할 수 있다. 충돌 거부가 모호한 명령줄을 막지만 이 호스트에서 실제 이미지 생성 수동 실행은 아직 없다.
