# PR #2460: OpenAI Privacy Filter (openai_privacy_filter) PII tagger

**작성일**: 2026-10-19
**상태**: mlx-community bf16과 8-bit 체크포인트로 CUDA (GB10)에서 구현 및 검증 완료, PR 열림

## 요약

에픽 #2391에 속한 #2387을 닫는다. OpenAI Privacy Filter를 `mlxcel tag`에 추가한다. 이 모델은 양방향 sparse-MoE 토큰 분류기(레이어 8개, expert 128개, top-4)로, PII를 BIOES label로 태깅한다. #2383(PR #2456)의 `TokenClassifierModel`을 구현하며, 체크포인트별 calibration을 쓰는 제약 Viterbi 디코더를 추가한다. calibration의 operating point는 `--operating-point`로 고른다.

## 문제 정의

`model_type: openai_privacy_filter`를 처리하는 arm이 없어 모든 진입점이 `Unsupported model type`으로 실패했다. 이 모델은 #2383 파이프라인에 없던 요소도 필요로 한다. 긴 입력에 대한 sink 포함 banded attention, GPT-OSS expert layout과 YaRN RoPE, 그리고 calibration된 transition bias를 적용해 올바른 BIOES 경로만 내는 디코더다.

## 기술적 결정

- **Encoder.** `src/models/openai_privacy_filter.rs`가 config를 읽는다(transformers 5의 `rope_parameters` 끌어올리기 포함). HF fused layout을 정리하는 규칙은 다음과 같다. `gate_up_proj`는 나눈 뒤 전치한다. `down_proj`는 전치하고 이름을 바꾼다. sink는 cast하거나 양자화하지 않는다. 체크포인트가 router를 양자화했다면 router는 8 bit / group 64로 양자화한다. `openai_privacy_filter_layers.rs`에는 band mask, query 단위로 나눈 sink 포함 banded attention, 토큰 chunk 단위의 f32 top-4 MoE가 있다.
- **GPT-OSS helper 공유.** `yarn_rope_freqs`(`truncate` 스위치 포함), `ExpertLinear`, `gpt_oss_swiglu`를 `src/models/gpt_oss_common.rs`로 옮겼고, `gpt_oss.rs`도 이를 호출한다. gpt-oss-20b의 teacher-forced trace로 이 이동이 아무것도 바꾸지 않음을 확인했다.
- **Viterbi 디코더.** `src/token_classification/viterbi.rs`는 `mlx_vlm.privacy_filter.ViterbiDecoder`와 `TransitionBiases`를 이식한다. calibration은 `viterbi_calibration.json`(`operating_points.<name>.biases`, 키 6개)에서 읽는다. 파일이 없으면 모든 bias를 0으로 둔다. 알 수 없는 operating point나 누락된 키는 로드 시점에 실패한다. `decode_modes()`가 Viterbi를 먼저 나열하므로 기본값은 Viterbi이고, `--argmax`도 계속 동작한다.
- **계약 유지.** `TokenModelKind::PrivacyFilter` variant 하나와 탐지 arm 하나만 추가했다. `TokenClassifierLoadOptions`와 `load_token_classifier_with`는 추가만 한 것이고, `load_token_classifier`가 이를 감싼다. trait 시그니처는 하나도 바뀌지 않았다.
- **탐지와 거부.** `get_model_type`은 "openai_privacy_filter is a token classifier; use `mlxcel tag`"로 실패한다. 이 type은 `ENCODER_ONLY_MODEL_TYPES`에 넣지 않았으므로 embedding 탐지가 이를 가져가지 않는다. argmax만 쓰는 계열에 `--operating-point`를 주면 weight를 로드하기 전에 거부한다.
- **dense plane은 정렬 없이 gather.** f32 activation과 bf16 weight로 `sorted_indices = true`인 CUDA `gather_mm`을 쓰면 결과가 할당 이력에 따라 달라졌다. 같은 프로세스에서 8-bit 모델을 먼저 실행하면 bf16 출력이 일부 토큰에서 어긋났고, 어떤 모델을 먼저 실행했는지에 따라 결정적으로 같은 차이가 났다. 정렬하지 않은 gather는 정확하다. 정렬된 `gather_qmm`(8-bit와 GPT-OSS 경로)은 어떤 순서에서도 정확하다. 그래서 dense plane은 정렬 없이 gather하며, 비용은 7.5k 토큰에서 8-bit forward의 약 1.5배다.
- **리뷰 후 보강.** i32를 넘치게 하거나 범위 밖을 참조하게 만드는 config와 plane은 로드 시점에 거부한다.

## 변경 사항

- `src/models/openai_privacy_filter.rs`, `openai_privacy_filter_layers.rs`, `openai_privacy_filter_tests.rs`(신규).
- `src/models/gpt_oss_common.rs`, `gpt_oss_common_tests.rs`(신규), `gpt_oss.rs`, `detection.rs`, `mod.rs`.
- `src/token_classification/viterbi.rs`, `privacy_filter.rs`와 그 테스트(신규), `loader.rs`, `mod.rs`.
- `src/commands/tag.rs`(`--operating-point`), `src/main.rs`(도움말 예시).
- `tests/fixtures/token_classification/`: Privacy Filter config, mlx-vlm fixture(mlx-vlm 0.7.8 git `73d366ec`, mlx CPU)와 그 생성 스크립트.
- `docs/token-classification.md`, `docs/supported-models.md`, `CHANGELOG.md`.

## 검증

GB10, `--profile test-fast --features cuda`, `MLX_ENABLE_TF32=0`. 테스트 바이너리는 `gpu-lock` 아래에서 `--test-threads=1`로 실행했다. `privacy_filter` 20, `viterbi` 13, `gpt_oss` 11, `token_classification` 64, `models::openai_privacy_filter` 10개가 모두 통과했다. bin 테스트 `extract_tests` 5개와 `tag` 4개도 통과하고, clippy와 fmt도 깨끗하다.

| 검사 | 결과 |
|---|---|
| `mlxcel tag -m models/mlx/openai-privacy-filter-bf16 "Alice Smith can be reached at alice@example.com."` | `private_person` (0..11), `private_email` (30..47), `<PRIVATE_PERSON> can be reached at <PRIVATE_EMAIL>.` |
| `--argmax`, `--operating-point default` | 같은 span |
| `--replacement "[REDACTED]"` / `--keep private_email` | `[REDACTED] can be reached at [REDACTED].` / `<PRIVATE_PERSON> can be reached at alice@example.com.` |
| `--operating-point nope` | 정의된 `default` 지점을 알려주며 거부 |
| 8-bit 체크포인트 | 같은 span |
| `mlxcel generate -m models/mlx/openai-privacy-filter-bf16` | 토큰 분류기 안내와 함께 거부 |
| mlx-vlm 대비 span: 문장 6개와 replacement, keep, Viterbi와 argmax, bf16과 8-bit | 동일 |
| 4536 토큰 입력, 기본 chunk 대 chunk 없음(bf16) | 최대 dp 절댓값 0.00000, argmax 뒤집힘 0개 |
| GPT-OSS trace, gpt-oss-20b-mxfp4, `5 60 8 512`, main 대 브랜치 | top-1 0/300, decided 0/100, logit 차이 최대 0.0000, 행이 바이트 단위로 동일 |

acceptance 문장(10x33)의 log-prob parity는 다음과 같다. dp 절댓값의 최댓값은 6e-5이고, dlogp 절댓값의 최댓값은 -10 nat 위 label에서 0.19, -18 nat 아래 꼬리 label에서 5.4다. oracle 자신의 f32 forward도 그 꼬리에서는 bf16 실행과 4.5 nat 차이가 난다. 그래서 이슈의 2e-2 기준은 확률 공간에 적용하고, -10 nat 위 label에는 0.5 nat 기준을 더했다. 이 차이는 이슈 본문에 기록했다. 처음 쓴 독일어 테스트 문장은 bf16 CUDA와 bf16 CPU 사이에서 0.19 nat 차이로 argmax가 뒤집혔다. 그래서 fixture에 0.5 nat의 margin을 요구하고 그 문장을 교체했다.

## 후속 작업

- 할당 이력에 따라 결과가 달라지는 CUDA 정렬 `gather_mm`(f32 activation, bf16 weight, overlay `matmul.cpp`의 `M == 1 && right_sorted_` 분기)은 별도 이슈로 다룰 만하다. 다른 bf16 MoE 모델도 영향을 받을 수 있다.
- `-4bit`, `-5bit`, `-6bit`, `-mxfp4`, `-mxfp8`, `-nvfp4` export는 이 호스트에 없다. GPT-OSS처럼 `ExpertLinear`를 거쳐 로드된다.
- 낮은 등급의 리뷰 항목은 다음과 같다. Viterbi 합은 f32로 계산한다(upstream은 f64 bias를 더한다). band mask를 chunk와 레이어마다 다시 만든다. 형태가 잘못된 텐서는 로드 시점의 오류 대신 MLX 안에서 abort된다.

## 통계

| 항목 | 값 |
|------|-------|
| 변경 파일 | 24 |
| 추가 줄 | 5071 |
| 삭제 줄 | 286 |

| 커밋 | 메시지 |
|--------|---------|
| `7b362477` | feat(models): support OpenAI Privacy Filter PII tagger in mlxcel tag |
| `35078b81` | fix(models): satisfy clippy on the Privacy Filter port |
| `87c041ae` | docs: state the Viterbi tie order as the candidate order, not label order |
| `2a088636` | fix(models): refuse Privacy Filter configs and planes that would wrap or index out of bounds |
| `7b7ecd99` | docs: document Privacy Filter public items and the unsorted expert gather |

## 관련 항목

- 에픽 #2391. #2383(PR #2456) 위에 구현했고, 형제 이슈는 #2390(GLiNER2.5)이다.
